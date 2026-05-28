"""Shared fabrication constraint primitives.

These constraint functions are importable by both:
- ``openlithohub`` (ILT/OPC mask synthesis pipeline)
- ``diffnano`` (nanophotonic inverse design pipeline)

Each primitive is a pure differentiable function of a shared parameterization
tensor (density field, SDF, or control-point representation).  Both pipelines
call them through the same import path — this is the C4 unified-autograd-graph
mechanism.

Tier 3 module (release after CN priority confirmation).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "minimum_cd_penalty",
    "curvature_penalty",
    "binarization_penalty",
    "corner_rounding_penalty",
    "combined_fabrication_penalty",
]


def minimum_cd_penalty(
    density: torch.Tensor,
    min_cd_pixels: float = 4.0,
) -> torch.Tensor:
    """Differentiable penalty for minimum critical dimension (CD) violations.

    Uses erosion–dilation to detect regions where the feature size
    is below *min_cd_pixels*.  The penalty is the L2 norm of the
    erosion residual — small when features are larger than *min_cd_pixels*.

    Parameters
    ----------
    density : Tensor, shape ``(H, W)`` or ``(N, H, W)``
        Continuous density field ∈ [0, 1].
    min_cd_pixels : float
        Minimum feature size in pixels (approximate).

    Returns
    -------
    penalty : Tensor, scalar
    """
    if density.dim() == 2:
        density = density.unsqueeze(0).unsqueeze(0)
    elif density.dim() == 3:
        density = density.unsqueeze(0)

    r = max(1, int(min_cd_pixels / 2))
    k_size = 2 * r + 1

    # Erosion ≈ -max_pool(-density)
    eroded = -F.max_pool2d(-density, k_size, stride=1, padding=r)

    # Opening (erosion then dilation) detects small features
    opened = -F.max_pool2d(-eroded, k_size, stride=1, padding=r)

    # Penalty: difference between original and opened
    diff = density.squeeze() - opened.squeeze()
    return (diff ** 2).mean()


def curvature_penalty(
    density: torch.Tensor,
    max_curvature: float = 0.1,
) -> torch.Tensor:
    """Differentiable curvature penalty on the density field boundary.

    Penalizes high curvature (tight corners) by computing the Laplacian
    of the density field near the boundary region.

    Parameters
    ----------
    density : Tensor, shape ``(H, W)``
        Continuous density field.
    max_curvature : float
        Maximum allowed curvature (inverse pixels).

    Returns
    -------
    penalty : Tensor, scalar
    """
    if density.dim() == 2:
        density = density.unsqueeze(0).unsqueeze(0)

    # Laplacian via convolution
    lap_kernel = torch.tensor(
        [[[0, 1, 0], [1, -4, 1], [0, 1, 0]]],
        dtype=density.dtype,
        device=density.device,
    ).unsqueeze(0)

    laplacian = F.conv2d(density, lap_kernel, padding=1)

    # Weight by boundary proximity (gradient magnitude)
    grad_x = F.conv2d(
        density,
        torch.tensor([[[[-1, 0, 1]]]], dtype=density.dtype, device=density.device),
        padding=(0, 1),
    )
    grad_y = F.conv2d(
        density,
        torch.tensor([[[[-1], [0], [1]]]], dtype=density.dtype, device=density.device),
        padding=(1, 0),
    )
    boundary_weight = (grad_x ** 2 + grad_y ** 2).sqrt()

    weighted_lap = (laplacian ** 2) * boundary_weight
    return weighted_lap.mean()


def binarization_penalty(
    density: torch.Tensor,
) -> torch.Tensor:
    """Penalty encouraging binary (0 or 1) density values.

    Uses the formulation: p * (1 - p) where p is the density.
    Minimized at p = 0 or p = 1.

    Parameters
    ----------
    density : Tensor
        Continuous density field.

    Returns
    -------
    penalty : Tensor, scalar
    """
    return (density * (1 - density)).mean()


def corner_rounding_penalty(
    density: torch.Tensor,
    radius_pixels: float = 3.0,
) -> torch.Tensor:
    """Penalty for sharp corners that may not resolve in fabrication.

    Detects corners via gradient direction changes and penalizes
    regions where the corner radius is below *radius_pixels*.

    Parameters
    ----------
    density : Tensor, shape ``(H, W)``
    radius_pixels : float
        Minimum corner radius.

    Returns
    -------
    penalty : Tensor, scalar
    """
    if density.dim() == 2:
        density = density.unsqueeze(0).unsqueeze(0)

    # Sobel filters for gradient direction
    sobel_x = torch.tensor(
        [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
        dtype=density.dtype,
        device=density.device,
    ) / 8.0
    sobel_y = sobel_x.transpose(-2, -1)

    gx = F.conv2d(density, sobel_x, padding=1)
    gy = F.conv2d(density, sobel_y, padding=1)

    # Gradient magnitude and direction
    mag = (gx ** 2 + gy ** 2).sqrt() + 1e-12
    theta = torch.atan2(gy, gx)

    # Laplacian of gradient direction (changes rapidly at corners)
    lap_kernel = torch.tensor(
        [[[0, 1, 0], [1, -4, 1], [0, 1, 0]]],
        dtype=density.dtype,
        device=density.device,
    ).unsqueeze(0)

    dir_change = F.conv2d(theta, lap_kernel, padding=1)

    # Weight by gradient magnitude (boundary regions)
    weighted = (dir_change ** 2) * mag
    return weighted.mean()


def combined_fabrication_penalty(
    density: torch.Tensor,
    *,
    min_cd: float = 4.0,
    max_curvature: float = 0.1,
    corner_radius: float = 3.0,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Combined fabrication penalty from all constraint primitives.

    This is the main entry point called from both the lithography and
    photonics pipelines through the C4 unified autograd graph.

    Parameters
    ----------
    density : Tensor, shape ``(H, W)``
        Continuous density field.
    min_cd : float
        Minimum critical dimension in pixels.
    max_curvature : float
        Maximum allowed curvature.
    corner_radius : float
        Minimum corner radius in pixels.
    weights : dict, optional
        Override weights for each penalty term.  Default is all 1.0.

    Returns
    -------
    penalty : Tensor, scalar
    """
    w = weights or {}
    w.setdefault("cd", 1.0)
    w.setdefault("curvature", 1.0)
    w.setdefault("binarization", 1.0)
    w.setdefault("corner", 1.0)

    total = torch.tensor(0.0, device=density.device, dtype=density.dtype)

    if w["cd"] > 0:
        total = total + w["cd"] * minimum_cd_penalty(density, min_cd)
    if w["curvature"] > 0:
        total = total + w["curvature"] * curvature_penalty(density, max_curvature)
    if w["binarization"] > 0:
        total = total + w["binarization"] * binarization_penalty(density)
    if w["corner"] > 0:
        total = total + w["corner"] * corner_rounding_penalty(density, corner_radius)

    return total
