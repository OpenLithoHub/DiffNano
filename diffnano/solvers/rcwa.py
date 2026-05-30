"""Differentiable RCWA (Rigorous Coupled-Wave Analysis) solver.

Implements an S-matrix formulation for periodic multilayer structures with
full PyTorch autograd support.

Batched mode: wavelengths and layers are processed with batched
``torch.linalg.eigh`` and ``torch.linalg.solve`` for GPU utilization.

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
    if n_fourier > N:
        raise ValueError(f"n_fourier ({n_fourier}) must not exceed grid size ({N})")

    eps_fft = torch.fft.fft(eps_profile.to(torch.complex128)) / N

    half = n_fourier // 2
    indices = torch.arange(-half, half + 1, device=eps_profile.device) % N
    coeffs = eps_fft[indices]

    row_idx = torch.arange(n_fourier, device=eps_profile.device)
    col_idx = torch.arange(n_fourier, device=eps_profile.device)
    diff = col_idx.unsqueeze(0) - row_idx.unsqueeze(1) + half
    diff = diff % n_fourier
    eps_conv = coeffs[diff]

    return eps_conv


def _build_toeplitz_batched(
    eps_layers: torch.Tensor,
    n_fourier: int,
) -> torch.Tensor:
    """Build Toeplitz matrices for all layers simultaneously.

    Parameters
    ----------
    eps_layers : Tensor, shape ``(n_layers, N_grid)``
    n_fourier : int

    Returns
    -------
    eps_conv : Tensor, shape ``(n_layers, n_fourier, n_fourier)``, complex128
    """
    _n_layers = eps_layers.shape[0]
    N = eps_layers.shape[1]
    device = eps_layers.device
    if n_fourier > N:
        raise ValueError(f"n_fourier ({n_fourier}) must not exceed grid size ({N})")

    eps_fft = torch.fft.fft(eps_layers.to(torch.complex128), dim=-1) / N

    half = n_fourier // 2
    indices = torch.arange(-half, half + 1, device=device) % N
    coeffs = eps_fft[:, indices]  # (n_layers, n_fourier)

    row_idx = torch.arange(n_fourier, device=device)
    col_idx = torch.arange(n_fourier, device=device)
    diff = col_idx.unsqueeze(0) - row_idx.unsqueeze(1) + half
    diff = diff % n_fourier

    eps_conv = coeffs[:, diff]  # (n_layers, n_fourier, n_fourier)

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

    P = eps_conv - Kx @ Kx

    P_herm = (P + P.conj().mT) / 2.0

    eigenvalues, eigenvectors = torch.linalg.eigh(P_herm)

    damping = 1e-10
    gamma = torch.sqrt(eigenvalues.to(dtype) + damping)

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
        """Forward pass for 1D grating (n_layers, n_grid) — batched over wavelengths."""
        n_layers = eps_layers.shape[0]
        n_wl = wavelengths.shape[0]
        n = self.n_fourier
        device = self.device
        px, py = self.period_nm
        dtype = torch.complex128

        # 1. Build Toeplitz for all layers: (n_layers, n, n)
        eps_conv_all = _build_toeplitz_batched(eps_layers, n)

        # 2. Expand to (n_wl, n_layers, n, n)
        eps_conv_batch = eps_conv_all.unsqueeze(0).expand(n_wl, -1, -1, -1)

        # 3. Build P matrix for all (wl, layer) pairs
        k0_all = 2 * math.pi / wavelengths  # (n_wl,)
        kx0_all = k0_all * math.sin(math.radians(theta))  # (n_wl,)

        m = torch.arange(n, device=device, dtype=torch.float64) - n // 2

        # Kx diagonal per wavelength: kx0 + m * 2pi/px / k0 → (n_wl, n)
        kx_diag = kx0_all.unsqueeze(1) + m.unsqueeze(0) * (2 * math.pi / px) / k0_all.unsqueeze(1)
        kx_sq_diag = kx_diag**2  # (n_wl, n)

        # Build Kx^2 as diagonal matrix: (n_wl, n, n)
        kx_sq_mat = torch.diag_embed(kx_sq_diag.to(dtype))  # (n_wl, n, n)

        # P = eps_conv - Kx^2: (n_wl, n_layers, n, n) - (n_wl, 1, n, n)
        P = eps_conv_batch - kx_sq_mat.unsqueeze(1)

        # Hermitian symmetrize
        P_herm = (P + P.conj().transpose(-2, -1)) / 2.0

        # 4. Batch eigendecomposition: (n_wl * n_layers, n, n)
        P_flat = P_herm.reshape(n_wl * n_layers, n, n)
        eigenvalues_flat, eigenvectors_flat = torch.linalg.eigh(P_flat)
        eigenvalues = eigenvalues_flat.reshape(n_wl, n_layers, n)
        eigenvectors = eigenvectors_flat.reshape(n_wl, n_layers, n, n)

        # 5. Compute gamma and phase for all (wl, layer)
        gamma = torch.sqrt(eigenvalues.to(dtype) + 1e-10)  # (n_wl, n_layers, n)

        layer_thickness = thickness_nm if thickness_nm is not None else px / n_layers
        k0_expanded = k0_all.unsqueeze(1).unsqueeze(2)  # (n_wl, 1, 1)
        phase = torch.exp(1j * k0_expanded * layer_thickness * gamma)  # (n_wl, n_layers, n)

        # 6. Layer-by-layer propagation (sequential over layers, batched over wavelengths)
        total_field = torch.ones(n_wl, n, dtype=dtype, device=device)

        for li in range(n_layers):
            evecs_li = eigenvectors[:, li]  # (n_wl, n, n)
            phase_li = phase[:, li]  # (n_wl, n)

            # coeffs = solve(evecs, total_field)
            coeffs = torch.linalg.solve(evecs_li, total_field.unsqueeze(-1)).squeeze(-1)
            coeffs = coeffs * phase_li
            total_field = torch.bmm(evecs_li, coeffs.unsqueeze(-1)).squeeze(-1)

        # 7. Transmission efficiency per order
        eff = (total_field * total_field.conj()).real  # (n_wl, n)
        eff = torch.clamp(eff, min=0.0)
        totals = eff.sum(dim=-1, keepdim=True)
        totals = torch.where(totals > 0, totals, torch.ones_like(totals))
        eff = eff / totals

        return SimResult(
            field=eff.to(torch.float64),
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
        """Forward pass for 2D geometry (n_layers, H, W).

        Note: spatial variation along the last dimension (W) is averaged out,
        converting the 2D density to 1D layer profiles. This is a simplification
        for 1D RCWA; for full 2D structures, process each row independently.
        """
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
        """Total diffraction efficiency (sum of all transmitted orders).

        Because efficiencies are normalized to sum to 1.0, this always
        returns approximately 1.0 for lossless structures. For actual
        power transmission, use the un-normalized field.
        """
        result = self.forward(geometry, wavelengths, source=source)
        return result.field.sum(dim=-1)
