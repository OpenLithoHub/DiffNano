"""Differentiable 2D FDFD (Frequency-Domain Finite-Difference) solver.

Solves the 2D Helmholtz equation on a Yee grid for TE and TM polarization.
Uses dense linear algebra (``torch.linalg.solve``) so autograd flows through
the entire solve — GPU-native, no sparse-library dependency.

References
----------
- Hughes et al. (2019), ceviche: ACS Photonics (baseline FDFD approach)
- Clean-room reimplementation; GPU-native via PyTorch, not derived from ceviche source.
"""

from __future__ import annotations

import math

import torch

from diffnano.solvers._result import SimResult

__all__ = ["FDFDSolver2D"]


def _build_helmholtz_tm(
    omega: float,
    eps_r: torch.Tensor,
    dl: float,
    pml_params: tuple[int, float, float] | None,
) -> torch.Tensor:
    """Build the Helmholtz operator matrix for TM polarization (Ez).

    Solves:  (D_x^T D_x + D_y^T D_y + k0^2 eps) Ez = -i omega mu0 Jz

    where D_x, D_y are finite-difference derivative matrices and eps is
    the relative permittivity on the grid.

    Parameters
    ----------
    omega : float
        Angular frequency.
    eps_r : Tensor, shape ``(H, W)``
        Relative permittivity.
    dl : float
        Grid spacing.
    pml_params : tuple or None
        ``(n_layers, sigma_max, kappa_max)`` for PML. None = no PML.

    Returns
    -------
    A : Tensor, shape ``(N, N)``
        Dense Helmholtz operator matrix (complex).
    """
    H, W = eps_r.shape
    N = H * W
    device = eps_r.device
    dtype = torch.complex128

    # Diagonal permittivity term
    eps_diag = eps_r.reshape(N)

    # Build 2D Laplacian via finite differences
    # Using Kronecker-product structure: L = I_W ⊗ D_HH + D_WW ⊗ I_H
    # where D_HH is the 1D second-derivative matrix of size H

    # 1D second derivative: D2 = tridiag(1, -2, 1) / dl^2
    def _d2_matrix(n: int) -> torch.Tensor:
        diag_main = torch.full((n,), -2.0, device=device, dtype=torch.float64)
        diag_off = torch.ones(n - 1, device=device, dtype=torch.float64)
        D2 = torch.diag(diag_main) + torch.diag(diag_off, 1) + torch.diag(diag_off, -1)
        return D2 / (dl * dl)

    D2H = _d2_matrix(H)  # (H, H)
    D2W = _d2_matrix(W)  # (W, W)

    IH = torch.eye(H, device=device, dtype=torch.float64)
    IW = torch.eye(W, device=device, dtype=torch.float64)

    # Laplacian = I_W ⊗ D2H + D2W ⊗ I_H
    L = torch.kron(IW, D2H) + torch.kron(D2W, IH)  # (N, N)

    # Add PML conductivity profile
    if pml_params is not None:
        n_pml, sigma_max, kappa_max = pml_params
        sigma_x = _pml_profile_1d(W, n_pml, sigma_max, device)
        sigma_y = _pml_profile_1d(H, n_pml, sigma_max, device)

        # Build sigma at each grid point (H, W) -> (N,)
        sigma_2d = sigma_y.unsqueeze(1).expand(H, W) + sigma_x.unsqueeze(0).expand(H, W)
        s = sigma_2d.reshape(N)

        # Absorbing factor: integrate sigma profile
        # For FDFD: modify the Laplacian with complex stretching
        s_complex = s.to(dtype)
        absorber = -1j * omega * s_complex + s_complex**2 / (abs(omega) + 1e-12)
    else:
        absorber = 0.0

    # Full operator: A = L + k0^2 * eps_diag + absorber
    k0_real = omega
    k0_sq_real = k0_real**2

    A = L.to(dtype) + torch.diag(k0_sq_real * eps_diag.to(dtype) + absorber)

    return A


def _build_helmholtz_te(
    omega: float,
    eps_r: torch.Tensor,
    dl: float,
    pml_params: tuple[int, float, float] | None,
) -> torch.Tensor:
    """Build the Helmholtz operator matrix for TE polarization (Hz).

    For TE mode: curl(1/eps * curl(H)) + k0^2 H = source
    Simplified to: D^T (1/eps) D H + k0^2 H = source

    where D is the finite-difference curl operator.
    """
    H, W = eps_r.shape
    N = H * W
    device = eps_r.device
    dtype = torch.complex128

    k0_real = omega

    # Inverse permittivity at grid points
    inv_eps = (1.0 / (eps_r + 1e-12)).reshape(N)

    def _d2_matrix(n: int) -> torch.Tensor:
        diag_main = torch.full((n,), -2.0, device=device, dtype=torch.float64)
        diag_off = torch.ones(n - 1, device=device, dtype=torch.float64)
        D2 = torch.diag(diag_main) + torch.diag(diag_off, 1) + torch.diag(diag_off, -1)
        return D2 / (dl * dl)

    D2H = _d2_matrix(H)
    D2W = _d2_matrix(W)

    IH = torch.eye(H, device=device, dtype=torch.float64)
    IW = torch.eye(W, device=device, dtype=torch.float64)

    # Modified Laplacian with 1/eps weighting
    L_basic = torch.kron(IW, D2H) + torch.kron(D2W, IH)
    inv_eps_diag = torch.diag(inv_eps.to(dtype))
    L_weighted = inv_eps_diag @ L_basic.to(dtype)

    # PML
    if pml_params is not None:
        n_pml, sigma_max, _ = pml_params
        sigma_x = _pml_profile_1d(W, n_pml, sigma_max, device)
        sigma_y = _pml_profile_1d(H, n_pml, sigma_max, device)
        sigma_2d = sigma_y.unsqueeze(1).expand(H, W) + sigma_x.unsqueeze(0).expand(H, W)
        s = sigma_2d.reshape(N)
        s_complex = s.to(dtype)
        absorber = -1j * omega * s_complex + s_complex**2 / (abs(omega) + 1e-12)
    else:
        absorber = 0.0

    k0_diag = (k0_real**2 + absorber) * torch.ones(
        N,
        device=device,
        dtype=dtype,
    )
    A = L_weighted + torch.diag(k0_diag)

    return A


def _pml_profile_1d(
    size: int,
    n_pml: int,
    sigma_max: float,
    device: torch.device,
) -> torch.Tensor:
    """Quadratic PML conductivity profile for one axis.

    Returns sigma values at each grid point along one dimension.
    """
    sigma = torch.zeros(size, device=device, dtype=torch.float64)
    if n_pml <= 0:
        return sigma
    for i in range(n_pml):
        # Quadratic grading
        val = sigma_max * ((n_pml - i) / n_pml) ** 2
        sigma[i] = val
        sigma[size - 1 - i] = val
    return sigma


class FDFDSolver2D:
    """Differentiable 2D FDFD solver.

    Solves the frequency-domain Maxwell's equations on a Yee grid using
    dense linear algebra.  Full autograd support through ``torch.linalg.solve``.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        ``(H, W)`` grid dimensions.
    dl : float
        Grid spacing in nm.
    wavelength_nm : float
        Operating wavelength.
    polarization : str
        "TE" (Hz) or "TM" (Ez).
    pml_layers : int
        Number of PML layers on each boundary.
    eps_background : float
        Background permittivity.
    device : str or torch.device
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (50, 50),
        dl: float = 20.0,
        wavelength_nm: float = 1550.0,
        polarization: str = "TM",
        pml_layers: int = 10,
        eps_background: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        self.grid_shape = grid_shape
        self.dl = dl
        self.wavelength_nm = wavelength_nm
        self.polarization = polarization.upper()
        self.pml_layers = pml_layers
        self.eps_background = eps_background
        self._device = torch.device(device)

        self.omega = 2 * math.pi * 3e17 / (wavelength_nm * 1e-9)  # rad/s
        # Simplified normalized omega for FDFD
        self.omega_norm = 2 * math.pi / wavelength_nm

    def forward(
        self,
        geometry: torch.Tensor,
        wavelengths: list[float] | torch.Tensor | None = None,
        *,
        source: dict | None = None,
    ) -> SimResult:
        """Run FDFD forward simulation.

        Parameters
        ----------
        geometry : Tensor, shape ``(H, W)``
            Relative permittivity map (``eps_r``).
        wavelengths : ignored
            FDFD is single-frequency; wavelength is set at construction.
        source : dict, optional
            ``{"type": "point"|"line", "pos": [y, x], "amplitude": float}``.
            Default: point source at grid center.

        Returns
        -------
        SimResult
            ``field`` contains the Ez (TM) or Hz (TE) field, shape ``(1, H*W)``.
        """
        if wavelengths is None:
            wavelengths = [self.wavelength_nm]
        if not isinstance(wavelengths, torch.Tensor):
            wavelengths = torch.tensor(wavelengths, dtype=torch.float64, device=self.device)

        H, W = self.grid_shape
        N = H * W
        device = self.device
        dtype = torch.complex128

        # Ensure geometry is on the right device
        eps_r = geometry.to(device)

        if eps_r.dim() == 3:
            eps_r = eps_r.squeeze(0)

        # Build source vector
        src = source or {}
        src_type = src.get("type", "point")
        b = torch.zeros(N, dtype=dtype, device=device)

        if src_type == "point":
            pos = src.get("pos", [H // 2, W // 2])
            idx = pos[0] * W + pos[1]
            amp = src.get("amplitude", 1.0)
            b[idx] = amp
        elif src_type == "line":
            row = src.get("row", H // 2)
            amp = src.get("amplitude", 1.0)
            for j in range(W):
                b[row * W + j] = amp

        # PML parameters
        pml_params = None
        if self.pml_layers > 0:
            sigma_max = 0.5 * (self.pml_layers + 1) / self.dl
            pml_params = (self.pml_layers, sigma_max, 1.0)

        # Build operator matrix
        omega = self.omega_norm
        if self.polarization == "TM":
            A = _build_helmholtz_tm(omega, eps_r, self.dl, pml_params)
        else:
            A = _build_helmholtz_te(omega, eps_r, self.dl, pml_params)

        # Solve A x = b
        b_2d = b.unsqueeze(1)  # (N, 1)
        x = torch.linalg.solve(A, b_2d).squeeze(1)  # (N,)

        # Field reshaped to (1, H*W) for consistency — keep complex for phase
        field = x.unsqueeze(0)

        return SimResult(
            field=field,
            wavelengths=wavelengths,
            metadata={
                "polarization": self.polarization,
                "grid_shape": self.grid_shape,
                "dl": self.dl,
                "omega_norm": omega,
            },
        )

    def solve(
        self,
        eps_r: torch.Tensor,
        source: dict | None = None,
    ) -> torch.Tensor:
        """Convenience: return the field reshaped as ``(H, W)``.

        Parameters
        ----------
        eps_r : Tensor, shape ``(H, W)``
        source : dict, optional

        Returns
        -------
        field : Tensor, shape ``(H, W)``
        """
        result = self.forward(eps_r, source=source)
        H, W = self.grid_shape
        return result.field.reshape(H, W)

    @property
    def device(self) -> torch.device:
        return self._device
