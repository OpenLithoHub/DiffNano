"""Fabricable subspace projection and multi-axis perturbation kernels.

Extends the v0.1 C5 robustness module with:
- Correlated multi-axis perturbation (linewidth × sidewall × thickness)
- Sidewall angle drift perturbation
- Layer thickness variation perturbation
- Corner rounding perturbation kernel
- Joint Gaussian model with Cholesky decomposition

References
----------
- Ma et al. (2024), BOSON-1: arXiv:2411.08210
"""

from __future__ import annotations

import torch

__all__ = [
    "MultiAxisPerturbation",
    "sidewall_angle_perturbation",
    "thickness_perturbation",
    "corner_rounding_perturbation",
]


def sidewall_angle_perturbation(
    density: torch.Tensor,
    angle_delta_deg: torch.Tensor,
    pixel_size_nm: float = 5.0,
) -> torch.Tensor:
    """Apply sidewall angle drift perturbation.

    Simulates the effect of sidewall angle variation on a density field
    by shifting the density profile vertically, approximating a trapezoidal
    cross-section from an angled sidewall.

    Parameters
    ----------
    density : Tensor, shape ``(H, W)``
        Density field.
    angle_delta_deg : Tensor, scalar
        Sidewall angle deviation in degrees.
    pixel_size_nm : float
        Pixel size in nm.

    Returns
    -------
    perturbed : Tensor, shape ``(H, W)``
    """
    # Convert angle to a vertical shift per horizontal pixel
    tan_angle = torch.tan(angle_delta_deg * 3.14159265 / 180.0)

    H, W = density.shape
    # Create a gradient that shifts density based on height
    y_coords = torch.linspace(0, 1, H, device=density.device, dtype=density.dtype)
    shift = tan_angle * y_coords.unsqueeze(1) * W * pixel_size_nm / (2 * pixel_size_nm)

    # Apply as a horizontal shift varying with height
    perturbed = density.clone()
    shift_pixels = (shift / pixel_size_nm).round().long().clamp(-W // 2, W // 2)

    for i in range(H):
        s = shift_pixels[i].item()
        if s > 0:
            perturbed[i, s:] = density[i, :-s]
            perturbed[i, :s] = density[i, 0]
        elif s < 0:
            perturbed[i, :s] = density[i, -s:]
            perturbed[i, s:] = density[i, -1]

    return perturbed


def thickness_perturbation(
    density: torch.Tensor,
    delta_nm: torch.Tensor,
    pixel_size_nm: float = 5.0,
) -> torch.Tensor:
    """Apply layer thickness variation perturbation.

    Scales the density field to simulate thickness variation by adjusting
    the effective fill factor uniformly.

    Parameters
    ----------
    density : Tensor, shape ``(H, W)``
    delta_nm : Tensor, scalar
        Thickness perturbation in nm.
    pixel_size_nm : float

    Returns
    -------
    perturbed : Tensor, shape ``(H, W)``
    """
    scale = 1.0 + delta_nm / (density.shape[0] * pixel_size_nm)
    perturbed = density * scale
    return perturbed.clamp(0.0, 1.0)


def corner_rounding_perturbation(
    density: torch.Tensor,
    radius_nm: torch.Tensor,
    pixel_size_nm: float = 5.0,
) -> torch.Tensor:
    """Apply corner rounding perturbation via Gaussian smoothing.

    Simulates the effect of fabrication corner rounding by applying
    a Gaussian blur with radius proportional to the rounding amount.

    Parameters
    ----------
    density : Tensor, shape ``(H, W)``
    radius_nm : Tensor, scalar
        Corner rounding radius in nm.
    pixel_size_nm : float

    Returns
    -------
    perturbed : Tensor, shape ``(H, W)``
    """
    radius_px = (radius_nm / pixel_size_nm).abs().clamp(max=10.0)
    sigma = radius_px / 2.0

    if sigma < 0.5:
        return density

    # Create 1D Gaussian kernel
    k_size = int(6 * sigma.item()) + 1
    if k_size % 2 == 0:
        k_size += 1
    x = torch.arange(k_size, device=density.device, dtype=density.dtype) - k_size // 2
    kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Separable 2D convolution
    padded = torch.nn.functional.pad(
        density.unsqueeze(0).unsqueeze(0),
        [k_size // 2] * 4,
        mode="reflect",
    )
    # Horizontal pass
    h_kernel = kernel_1d.reshape(1, 1, 1, -1)
    h_blurred = torch.nn.functional.conv2d(padded, h_kernel, padding=0)
    # Vertical pass
    v_kernel = kernel_1d.reshape(1, 1, -1, 1)
    result = torch.nn.functional.conv2d(h_blurred, v_kernel, padding=0)

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
            [sigma_linewidth_nm, sigma_sidewall_deg,
             sigma_thickness_nm, sigma_corner_nm],
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
        return eps @ self.cov_chol.T

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
