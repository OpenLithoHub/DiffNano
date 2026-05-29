"""Fabricable subspace projection and multi-axis perturbation kernels.

Extends the v0.1 C5 robustness module with:
- Correlated multi-axis perturbation (linewidth x sidewall x thickness)
- Sidewall angle drift perturbation (differentiable via spatial transformer)
- Layer thickness variation perturbation
- Corner rounding perturbation kernel
- Joint Gaussian model with Cholesky decomposition

References
----------
- Ma et al. (2024), BOSON-1: arXiv:2411.08210
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = [
    "MultiAxisPerturbation",
    "sidewall_angle_perturbation",
    "thickness_perturbation",
    "corner_rounding_perturbation",
]


def _differentiable_shift_1d(
    signal: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Shift each row of a 2D signal by a differentiable amount.

    Uses linear interpolation via F.grid_sample for differentiable shifting.

    Parameters
    ----------
    signal : Tensor, shape ``(H, W)``
    shifts : Tensor, shape ``(H,)``
        Per-row shift in pixels (positive = shift right).

    Returns
    -------
    shifted : Tensor, shape ``(H, W)``
    """
    H, W = signal.shape
    device = signal.device
    dtype = signal.dtype

    x_base = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    y_base = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_base, x_base, indexing="ij")

    shift_normalized = shifts.unsqueeze(1) / (W / 2.0)
    grid_x_shifted = grid_x - shift_normalized

    grid = torch.stack([grid_x_shifted, grid_y], dim=-1).unsqueeze(0)
    input_4d = signal.unsqueeze(0).unsqueeze(0)
    output = F.grid_sample(
        input_4d,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return output.squeeze(0).squeeze(0)


def sidewall_angle_perturbation(
    density: torch.Tensor,
    angle_delta_deg: torch.Tensor,
    pixel_size_nm: float = 5.0,
) -> torch.Tensor:
    """Apply sidewall angle drift perturbation (fully differentiable).

    Uses spatial transformer with linear interpolation for gradient flow.

    Parameters
    ----------
    density : Tensor, shape ``(H, W)``
    angle_delta_deg : Tensor, scalar
        Sidewall angle deviation in degrees.
    pixel_size_nm : float

    Returns
    -------
    perturbed : Tensor, shape ``(H, W)``
    """
    tan_angle = torch.tan(angle_delta_deg * math.pi / 180.0)

    H, W = density.shape
    y_coords = torch.linspace(0, 1, H, device=density.device, dtype=density.dtype)
    shift = tan_angle * y_coords * W * pixel_size_nm / (2 * pixel_size_nm)
    shift_pixels = shift / pixel_size_nm

    return _differentiable_shift_1d(density, shift_pixels)


def thickness_perturbation(
    density: torch.Tensor,
    delta_nm: torch.Tensor,
    pixel_size_nm: float = 5.0,
) -> torch.Tensor:
    """Apply layer thickness variation perturbation.

    Parameters
    ----------
    density : Tensor, shape ``(H, W)``
    delta_nm : Tensor, scalar
    pixel_size_nm : float

    Returns
    -------
    perturbed : Tensor, shape ``(H, W)``
    """
    physical_thickness_nm = density.shape[0] * pixel_size_nm
    scale = 1.0 + delta_nm / physical_thickness_nm
    return (density * scale).clamp(0.0, 1.0)


def corner_rounding_perturbation(
    density: torch.Tensor,
    radius_nm: torch.Tensor,
    pixel_size_nm: float = 5.0,
) -> torch.Tensor:
    """Apply corner rounding perturbation via Gaussian smoothing.

    Fully differentiable — uses fixed kernel size with sigma-controlled blur.

    Parameters
    ----------
    density : Tensor, shape ``(H, W)``
    radius_nm : Tensor, scalar
    pixel_size_nm : float

    Returns
    -------
    perturbed : Tensor, shape ``(H, W)``
    """
    radius_px = (radius_nm / pixel_size_nm).abs().clamp(max=10.0)
    sigma = radius_px / 2.0

    if sigma < 0.5:
        return density

    k_size = 21
    k_half = k_size // 2

    # Clamp kernel size to input dimensions
    H, W = density.shape
    k_size = min(k_size, min(H, W))
    if k_size % 2 == 0:
        k_size -= 1
    if k_size < 3:
        return density
    k_half = k_size // 2
    x = torch.arange(k_size, device=density.device, dtype=density.dtype) - k_half
    kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / (kernel_1d.sum() + 1e-12)

    padded = F.pad(
        density.unsqueeze(0).unsqueeze(0),
        [k_half] * 4,
        mode="reflect",
    )
    h_kernel = kernel_1d.reshape(1, 1, 1, -1)
    h_blurred = F.conv2d(padded, h_kernel, padding=0)
    v_kernel = kernel_1d.reshape(1, 1, -1, 1)
    result = F.conv2d(h_blurred, v_kernel, padding=0)

    return result.squeeze(0).squeeze(0)


class MultiAxisPerturbation:
    """Joint multi-axis perturbation with correlated sampling.

    Combines linewidth, sidewall angle, thickness, and corner rounding
    perturbations with a correlated joint Gaussian model.

    Parameters
    ----------
    sigma_linewidth_nm : float
        Standard deviation of linewidth perturbation (nm).
    sigma_sidewall_deg : float
        Standard deviation of sidewall angle perturbation (degrees).
    sigma_thickness_nm : float
        Standard deviation of thickness perturbation (nm).
    sigma_corner_nm : float
        Standard deviation of corner rounding perturbation (nm).
    correlation_matrix : Tensor, shape ``(4, 4)``, optional
        Correlation matrix for joint sampling. Identity if None.
    pixel_size_nm : float
        Pixel size in nm.
    """

    def __init__(
        self,
        sigma_linewidth_nm: float = 5.0,
        sigma_sidewall_deg: float = 2.0,
        sigma_thickness_nm: float = 3.0,
        sigma_corner_nm: float = 4.0,
        correlation_matrix: torch.Tensor | None = None,
        pixel_size_nm: float = 5.0,
    ):
        self.sigmas = torch.tensor(
            [sigma_linewidth_nm, sigma_sidewall_deg, sigma_thickness_nm, sigma_corner_nm],
            dtype=torch.float64,
        )
        self.pixel_size_nm = pixel_size_nm

        if correlation_matrix is not None:
            cov = torch.diag(self.sigmas) @ correlation_matrix @ torch.diag(self.sigmas)
            self.cov_chol = torch.linalg.cholesky(cov)
        else:
            self.cov_chol = torch.diag(self.sigmas)

    def sample(self, n_samples: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Draw correlated perturbation samples.

        Returns
        -------
        samples : Tensor, shape ``(n_samples, 4)``
            Columns: [linewidth_nm, sidewall_deg, thickness_nm, corner_nm]
        """
        eps = torch.randn(n_samples, 4, device=device, dtype=torch.float64)
        return eps @ self.cov_chol.mT

    def apply(
        self,
        density: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply all four perturbations to a density field.

        Parameters
        ----------
        density : Tensor, shape ``(H, W)``
        delta : Tensor, shape ``(4,)``
            [linewidth_nm, sidewall_deg, thickness_nm, corner_nm]

        Returns
        -------
        perturbed : Tensor, shape ``(H, W)``
        """

        # Apply in sequence: linewidth → sidewall → thickness → corner rounding
        result = density.clone()

        # Linewidth via SDF shift (simplified: threshold shift on density)
        delta_lw = delta[0] / (self.pixel_size_nm * 10)
        result = torch.sigmoid(10.0 * (result - 0.5 + delta_lw))

        # Sidewall angle
        result = sidewall_angle_perturbation(result, delta[1], self.pixel_size_nm)

        # Thickness
        result = thickness_perturbation(result, delta[2], self.pixel_size_nm)

        # Corner rounding
        result = corner_rounding_perturbation(result, delta[3], self.pixel_size_nm)

        return result
