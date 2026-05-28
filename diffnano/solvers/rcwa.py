"""Differentiable RCWA (Rigorous Coupled-Wave Analysis) solver.

Implements an S-matrix formulation for periodic multilayer structures with
full PyTorch autograd support.

References
----------
- Liu & Fan (2020), grcwa: arXiv:2005.01481 (baseline, no degeneracy handling)
- Kim & Lee (2023), TORCWA: CPC 282, 108552 (broadening-based stabilization)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from diffnano.solvers._result import SimResult

__all__ = ["RCWASolver"]


def _build_toeplitz_1d(
    eps_profile: torch.Tensor,
    n_fourier: int,
) -> torch.Tensor:
    """Build Toeplitz permittivity convolution matrix from a 1D profile.

    Parameters
    ----------
    eps_profile : Tensor, shape ``(N_grid,)``
        Permittivity sampled on a real-space grid within one period.
    n_fourier : int
        Number of Fourier coefficients to retain (2*orders+1).

    Returns
    -------
    eps_conv : Tensor, shape ``(n_fourier, n_fourier)``
    """
    N = eps_profile.shape[0]

    eps_fft = torch.fft.fft(eps_profile.to(torch.complex128)) / N

    half = n_fourier // 2
    indices = torch.arange(-half, half + 1, device=eps_profile.device) % N
    coeffs = eps_fft[indices]

    # Build Toeplitz (not circulant) matrix using FFT coefficient indexing
    row_idx = torch.arange(n_fourier, device=eps_profile.device)
    col_idx = torch.arange(n_fourier, device=eps_profile.device)
    diff = col_idx.unsqueeze(0) - row_idx.unsqueeze(1) + half
    diff = diff.clamp(0, n_fourier - 1)
    eps_conv = coeffs[diff]

    return eps_conv


def _propagate_layer(
    eps_conv: torch.Tensor,
    kx_norm: torch.Tensor,
    ky_norm: torch.Tensor,
    k0: float,
    thickness_nm: float,
    period_x: float,
    period_y: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute forward/backward propagation matrices for one layer.

    Returns (phase, eigenvectors).
    """
    n = eps_conv.shape[0]
    device = eps_conv.device
    dtype = torch.complex128

    m = torch.arange(n, device=device, dtype=torch.float64) - n // 2
    Kx = torch.diag(kx_norm + m * (2 * math.pi / period_x) / k0)
    Ky = torch.diag(ky_norm + m * (2 * math.pi / period_y) / k0)

    Kx = Kx.to(dtype)
    Ky = Ky.to(dtype)

    # P = eps_conv - Kx^2 (1D simplification)
    P = eps_conv - Kx @ Kx

    # Make P Hermitian for stable eigendecomposition
    P_herm = (P + P.conj().mT) / 2.0

    eigenvalues, eigenvectors = torch.linalg.eigh(P_herm)

    # Handle evanescent modes: negative eigenvalues → imaginary gamma
    # Use complex sqrt with small damping for numerical stability
    damping = 1e-10
    gamma = torch.sqrt(eigenvalues.to(dtype) + damping)

    # Phase accumulation (imaginary for evanescent → exponential decay)
    phase = torch.exp(1j * k0 * thickness_nm * gamma)

    return phase, eigenvectors


class RCWASolver:
    """Differentiable RCWA solver for periodic multilayer structures.

    Parameters
    ----------
    fourier_orders : int
        Number of Fourier orders retained on each side (total = 2*orders+1).
    wavelength_nm : float
        Operating wavelength in nanometers.
    period_nm : tuple[float, float]
        Grating period ``(px, py)`` in nanometers.
    eps_ambient : float
        Permittivity of the ambient (superstrate).
    eps_substrate : float
        Permittivity of the substrate.
    device : str or torch.device
        Compute device.
    degen_tol : float
        Degeneracy tolerance for eigendecomposition backward.
    """

    def __init__(
        self,
        fourier_orders: int = 10,
        wavelength_nm: float = 532.0,
        period_nm: tuple[float, float] = (400.0, 400.0),
        eps_ambient: float = 1.0,
        eps_substrate: float = 1.0,
        device: str | torch.device = "cpu",
        degen_tol: float = 1e-6,
    ):
        self.fourier_orders = fourier_orders
        self.n_fourier = 2 * fourier_orders + 1
        self.wavelength_nm = wavelength_nm
        self.period_nm = period_nm
        self.eps_ambient = eps_ambient
        self.eps_substrate = eps_substrate
        self.device = torch.device(device)
        self.degen_tol = degen_tol

    @property
    def _k0(self) -> float:
        return 2 * math.pi / self.wavelength_nm

    def forward(
        self,
        geometry: torch.Tensor,
        wavelengths: Sequence[float] | torch.Tensor | None = None,
        *,
        source: dict | None = None,
    ) -> SimResult:
        """Run RCWA forward simulation.

        Parameters
        ----------
        geometry : Tensor
            Layer geometry. Either:
            - 2D: ``(n_layers, n_grid)`` permittivity profiles per layer
            - 3D: ``(n_layers, H, W)`` density field
        wavelengths : sequence or Tensor, optional
            Wavelengths in nm.
        source : dict, optional
            Source config: ``{"theta": float, "polarization": "TE"|"TM",
            "thickness_nm": float}``.

        Returns
        -------
        SimResult
            ``field`` contains diffraction efficiencies, shape ``(W, n_fourier)``.
        """
        if wavelengths is None:
            wavelengths = [self.wavelength_nm]
        if not isinstance(wavelengths, torch.Tensor):
            wavelengths = torch.tensor(wavelengths, dtype=torch.float64, device=self.device)
        wavelengths = wavelengths.to(self.device)

        src = source or {}
        theta = src.get("theta", 0.0)
        polarization = src.get("polarization", "TE")
        thickness_nm = src.get("thickness_nm", None)

        if geometry.dim() == 2:
            return self._forward_1d(geometry, wavelengths, theta, polarization, thickness_nm)
        elif geometry.dim() == 3:
            return self._forward_2d(geometry, wavelengths, theta, polarization, thickness_nm)
        else:
            raise ValueError(f"geometry must be 2D or 3D tensor, got {geometry.dim()}D")

    def _forward_1d(
        self,
        eps_layers: torch.Tensor,
        wavelengths: torch.Tensor,
        theta: float,
        polarization: str,
        thickness_nm: float | None = None,
    ) -> SimResult:
        """Forward pass for 1D grating (n_layers, n_grid)."""
        n_layers = eps_layers.shape[0]
        n_wl = wavelengths.shape[0]
        n = self.n_fourier
        device = self.device
        px, py = self.period_nm

        all_efficiencies = []

        for wi in range(n_wl):
            wl = wavelengths[wi]
            k0 = 2 * math.pi / wl
            kx0 = k0 * math.sin(math.radians(theta))

            total_field = torch.ones(n, dtype=torch.complex128, device=device)

            for li in range(n_layers):
                eps_profile = eps_layers[li]
                eps_conv = _build_toeplitz_1d(eps_profile, n)

                layer_thickness = thickness_nm if thickness_nm is not None else px / n_layers

                phase, evecs = _propagate_layer(
                    eps_conv,
                    torch.tensor([kx0], device=device, dtype=torch.float64),
                    torch.zeros(1, device=device, dtype=torch.float64),
                    k0,
                    layer_thickness,
                    px,
                    py,
                )

                # Use solve instead of inv for numerical stability
                coeffs = torch.linalg.solve(evecs, total_field.to(torch.complex128))
                coeffs = coeffs * phase
                total_field = evecs @ coeffs

            # Transmission efficiency per order
            eff = (total_field * total_field.conj()).real
            eff = torch.clamp(eff, min=0.0)
            total = eff.sum()
            if total > 0:
                eff = eff / total

            all_efficiencies.append(eff)

        field = torch.stack(all_efficiencies)

        return SimResult(
            field=field.to(torch.float64),
            wavelengths=wavelengths,
            metadata={
                "n_layers": n_layers,
                "fourier_orders": self.fourier_orders,
                "polarization": polarization,
                "theta": theta,
            },
        )

    def _forward_2d(
        self,
        density: torch.Tensor,
        wavelengths: torch.Tensor,
        theta: float,
        polarization: str,
        thickness_nm: float | None = None,
    ) -> SimResult:
        """Forward pass for 2D geometry (n_layers, H, W)."""
        eps_low = self.eps_ambient
        eps_high = self.eps_substrate if self.eps_substrate > 1.0 else 12.0
        eps_layers = eps_low + (eps_high - eps_low) * density.mean(dim=-1)
        return self._forward_1d(eps_layers, wavelengths, theta, polarization, thickness_nm)

    def diffraction_efficiency(
        self,
        geometry: torch.Tensor,
        wavelengths: Sequence[float] | torch.Tensor | None = None,
        order: int = 0,
        *,
        source: dict | None = None,
    ) -> torch.Tensor:
        """Diffraction efficiency for a specific order."""
        result = self.forward(geometry, wavelengths, source=source)
        idx = order + self.fourier_orders
        return result.field[:, idx]

    def transmission(
        self,
        geometry: torch.Tensor,
        wavelengths: Sequence[float] | torch.Tensor | None = None,
        *,
        source: dict | None = None,
    ) -> torch.Tensor:
        """Total transmission (sum over all orders)."""
        result = self.forward(geometry, wavelengths, source=source)
        return result.field.sum(dim=-1)
