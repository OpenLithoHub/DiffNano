"""Projection and smoothing filters for topology optimization.

Provides Heaviside projection and density-based smoothing filters
compatible with binarization continuation schedules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["heaviside_projection", "smooth_filter", "beta_continuation_schedule"]


def heaviside_projection(
    density: torch.Tensor,
    beta: float = 10.0,
    eta: float = 0.5,
) -> torch.Tensor:
    """Smoothed Heaviside projection (sigmoid-based).

    Parameters
    ----------
    density : Tensor
        Continuous density field ∈ [0, 1].
    beta : float
        Sharpness parameter (→ ∞ gives step function).
    eta : float
        Threshold parameter.

    Returns
    -------
    projected : Tensor
        Projected density ∈ (0, 1).
    """
    return torch.sigmoid(beta * (density - eta))


def smooth_filter(
    density: torch.Tensor,
    radius: float = 2.0,
) -> torch.Tensor:
    """Apply a smoothing (blurring) filter to the density field.

    Uses a 2D Gaussian kernel via convolution.

    Parameters
    ----------
    density : Tensor, shape ``(H, W)`` or ``(N, H, W)``
        Density field(s).
    radius : float
        Smoothing kernel radius in pixels.

    Returns
    -------
    smoothed : Tensor
    """
    if density.dim() == 2:
        density = density.unsqueeze(0).unsqueeze(0)
    elif density.dim() == 3:
        density = density.unsqueeze(0)

    k_size = int(2 * radius + 1)
    sigma = radius / 2.0

    x = torch.arange(k_size, dtype=density.dtype, device=density.device) - k_size // 2
    kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2 + 1e-12))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0)

    pad = k_size // 2
    smoothed = F.conv2d(density, kernel, padding=pad)

    # Remove only the dimensions we added (0 and 1 for 2D input, 0 for 3D)
    if smoothed.dim() == 4 and smoothed.shape[0] == 1 and smoothed.shape[1] == 1:
        smoothed = smoothed.squeeze(0).squeeze(0)
    elif smoothed.dim() == 4 and smoothed.shape[0] == 1:
        smoothed = smoothed.squeeze(0)
    else:
        smoothed = smoothed.squeeze(0) if smoothed.shape[0] == 1 else smoothed

    return smoothed


def beta_continuation_schedule(
    step: int,
    total_steps: int = 500,
    beta_start: float = 1.0,
    beta_end: float = 64.0,
) -> float:
    """Exponential β-continuation schedule.

    Progressively sharpens the projection from *beta_start* to *beta_end*
    over *total_steps* iterations.

    Parameters
    ----------
    step : int
        Current optimization step.
    total_steps : int
        Total number of steps.
    beta_start : float
        Initial β (soft projection).
    beta_end : float
        Final β (near-binary).

    Returns
    -------
    beta : float
    """
    t = min(step / total_steps, 1.0)
    return beta_start * (beta_end / beta_start) ** t
