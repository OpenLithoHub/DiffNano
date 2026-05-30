"""Sparse FDFD solver with O(N) memory and adjoint-based gradients.

Replaces the dense Helmholtz matrix (O(N^2) memory, O(N^3) solve) with a
sparse representation (O(N) memory, O(N) per nonzero) using ``scipy.sparse``.
Gradients are computed via the adjoint state method inside a custom
``torch.autograd.Function``, avoiding the need to differentiate through the
sparse LU factorization.

Memory scaling: 100x100 grid (N=10000) dense = 800 MB → sparse = ~1.6 MB.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from diffnano.solvers._result import SimResult

__all__ = ["SparseFDFDSolver2D"]


# ---------------------------------------------------------------------------
# Sparse Helmholtz builders
# ---------------------------------------------------------------------------


def _pml_sigma(omega: float, sigma: np.ndarray) -> np.ndarray:
    """PML absorber term added to the Helmholtz diagonal."""
    return -1j * omega * sigma + sigma**2 / (abs(omega) + 1e-12)


def _build_pml_profile(
    H: int,
    W: int,
    pml_params: tuple[int, float, float] | None,
) -> np.ndarray:
    """Return the PML absorber profile as a 1-D array of shape (H*W,)."""
    N = H * W
    if pml_params is None:
        return np.zeros(N, dtype=np.complex128)

    n_pml, sigma_max, _ = pml_params
    sigma_x = np.zeros(W, dtype=np.float64)
    sigma_y = np.zeros(H, dtype=np.float64)
    for i in range(n_pml):
        val = sigma_max * ((n_pml - i) / n_pml) ** 2
        sigma_x[i] = val
        sigma_x[W - 1 - i] = val
        sigma_y[i] = val
        sigma_y[H - 1 - i] = val
    sigma_2d = sigma_y[:, None] + sigma_x[None, :]
    return sigma_2d.ravel()


def build_sparse_helmholtz_tm(
    eps_r: np.ndarray,
    omega: float,
    dl: float,
    pml_params: tuple[int, float, float] | None,
) -> sp.csc_matrix:
    """Build the sparse Helmholtz operator for TM polarization (Ez).

    A = L_5pt + k0^2 * diag(eps_r) + PML_diag
    """
    H, W = eps_r.shape
    N = H * W
    inv_dl2 = 1.0 / (dl * dl)
    k0_sq = omega * omega

    eps_flat = eps_r.ravel().astype(np.float64)
    rows_r = np.arange(N) // W
    cols_c = np.arange(N) % W

    # Diagonal: -4/dl^2 + k0^2 * eps + PML
    diag_val = -4.0 * inv_dl2 + k0_sq * eps_flat
    pml = _build_pml_profile(H, W, pml_params)
    if pml_params is not None:
        diag_val = diag_val + _pml_sigma(omega, pml.real)

    row_idx = [np.arange(N)]
    col_idx = [np.arange(N)]
    val_list = [diag_val.astype(np.complex128)]

    off = inv_dl2

    # Right neighbour (i, i+1): c < W-1
    m = cols_c < W - 1
    row_idx.append(np.arange(N)[m])
    col_idx.append((np.arange(N) + 1)[m])
    val_list.append(np.full(m.sum(), off, dtype=np.complex128))

    # Left neighbour (i, i-1): c > 0
    m = cols_c > 0
    row_idx.append(np.arange(N)[m])
    col_idx.append((np.arange(N) - 1)[m])
    val_list.append(np.full(m.sum(), off, dtype=np.complex128))

    # Down neighbour (i, i+W): r < H-1
    m = rows_r < H - 1
    row_idx.append(np.arange(N)[m])
    col_idx.append((np.arange(N) + W)[m])
    val_list.append(np.full(m.sum(), off, dtype=np.complex128))

    # Up neighbour (i, i-W): r > 0
    m = rows_r > 0
    row_idx.append(np.arange(N)[m])
    col_idx.append((np.arange(N) - W)[m])
    val_list.append(np.full(m.sum(), off, dtype=np.complex128))

    rows = np.concatenate(row_idx)
    cols = np.concatenate(col_idx)
    vals = np.concatenate(val_list)

    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()


def build_sparse_helmholtz_te(
    eps_r: np.ndarray,
    omega: float,
    dl: float,
    pml_params: tuple[int, float, float] | None,
) -> sp.csc_matrix:
    """Build the sparse Helmholtz operator for TE polarization (Hz).

    A = diag(1/eps) @ L_5pt + k0^2 * I + PML_diag
    """
    H, W = eps_r.shape
    N = H * W
    inv_dl2 = 1.0 / (dl * dl)
    k0_sq = omega * omega

    inv_eps = 1.0 / (eps_r.ravel().astype(np.float64) + 1e-12)
    rows_r = np.arange(N) // W
    cols_c = np.arange(N) % W

    diag_val = inv_eps * (-4.0 * inv_dl2) + k0_sq
    if pml_params is not None:
        pml = _build_pml_profile(H, W, pml_params)
        diag_val = diag_val + _pml_sigma(omega, pml.real)

    row_idx = [np.arange(N)]
    col_idx = [np.arange(N)]
    val_list = [diag_val.astype(np.complex128)]

    # Off-diagonals weighted by inv_eps
    m = cols_c < W - 1
    row_idx.append(np.arange(N)[m])
    col_idx.append((np.arange(N) + 1)[m])
    val_list.append((inv_eps[m] * inv_dl2).astype(np.complex128))

    m = cols_c > 0
    row_idx.append(np.arange(N)[m])
    col_idx.append((np.arange(N) - 1)[m])
    val_list.append((inv_eps[m] * inv_dl2).astype(np.complex128))

    m = rows_r < H - 1
    row_idx.append(np.arange(N)[m])
    col_idx.append((np.arange(N) + W)[m])
    val_list.append((inv_eps[m] * inv_dl2).astype(np.complex128))

    m = rows_r > 0
    row_idx.append(np.arange(N)[m])
    col_idx.append((np.arange(N) - W)[m])
    val_list.append((inv_eps[m] * inv_dl2).astype(np.complex128))

    rows = np.concatenate(row_idx)
    cols = np.concatenate(col_idx)
    vals = np.concatenate(val_list)

    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()


# ---------------------------------------------------------------------------
# Custom autograd: sparse forward solve + adjoint-state backward
# ---------------------------------------------------------------------------


class _SparseHelmholtzSolve(torch.autograd.Function):
    """Solve A(eps_r) x = b in sparse format with adjoint-state backward.

    Forward: build sparse A, solve with ``scipy.sparse.linalg.spsolve``.
    Backward: solve the adjoint system A^H lambda = grad_x, then compute
    dL/d(eps_r) analytically.
    """

    @staticmethod
    def forward(
        ctx,
        eps_r: torch.Tensor,
        b: torch.Tensor,
        omega: float,
        dl: float,
        pml_params: tuple | None,
        polarization: str,
        grid_shape: tuple[int, int],
    ) -> torch.Tensor:
        device = eps_r.device
        H, W = grid_shape

        eps_np = eps_r.detach().cpu().numpy().astype(np.float64)
        b_np = b.detach().cpu().numpy()

        if polarization == "TM":
            A = build_sparse_helmholtz_tm(eps_np, omega, dl, pml_params)
        else:
            A = build_sparse_helmholtz_te(eps_np, omega, dl, pml_params)

        x_np = spla.spsolve(A, b_np)
        x = torch.from_numpy(x_np.copy()).to(device=device, dtype=torch.complex128)

        ctx.save_for_backward(eps_r)
        ctx.A_csc = A
        ctx.x_np = x_np.copy()
        ctx.omega = omega
        ctx.dl = dl
        ctx.pml_params = pml_params
        ctx.polarization = polarization
        ctx.grid_shape = grid_shape
        ctx.device = device

        return x

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor):
        eps_r, = ctx.saved_tensors
        device = ctx.device
        H, W = ctx.grid_shape
        omega = ctx.omega
        k0_sq = omega * omega

        # Adjoint solve: A^H lambda = grad_x
        A_H = ctx.A_csc.conjugate().transpose().tocsc()
        grad_x_np = grad_x.detach().cpu().numpy()
        lambda_np = spla.spsolve(A_H, grad_x_np)

        x_np = ctx.x_np

        if ctx.polarization == "TM":
            # dL/d(eps_k) = -k0^2 * Re(conj(lambda_k) * x_k)
            grad_np = -k0_sq * np.real(np.conj(lambda_np) * x_np)
        else:
            # TE: A = diag(1/eps) @ L + k0^2 I + PML
            # ∂A/∂(eps_k) only affects row k: -1/eps_k^2 * (row k of L)
            # Lx = diag(eps) * (A@x - (k0^2 + PML)*x)
            eps_np = eps_r.detach().cpu().numpy().ravel()
            Ax = ctx.A_csc.dot(x_np)

            pml_diag = np.zeros(len(x_np), dtype=np.complex128)
            if ctx.pml_params is not None:
                pml = _build_pml_profile(H, W, ctx.pml_params)
                pml_diag = _pml_sigma(omega, pml.real)

            Lx = eps_np * (Ax - (k0_sq + pml_diag) * x_np)
            inv_eps_sq = 1.0 / (eps_np + 1e-12) ** 2
            grad_np = inv_eps_sq * np.real(np.conj(lambda_np) * Lx)

        grad_eps = torch.from_numpy(grad_np).to(device=device, dtype=torch.float64).reshape(H, W)
        return grad_eps, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public solver class
# ---------------------------------------------------------------------------


class SparseFDFDSolver2D:
    """Differentiable 2D FDFD solver using sparse Helmholtz matrices.

    Drop-in replacement for ``FDFDSolver2D`` that uses O(N) memory instead
    of O(N^2).  Gradients are computed via the adjoint state method, so the
    sparse LU factorization does not need to be differentiable.

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

        self.omega_norm = 2 * math.pi / wavelength_nm

    @property
    def device(self) -> torch.device:
        return self._device

    def _pml_params(self) -> tuple[int, float, float] | None:
        if self.pml_layers <= 0:
            return None
        sigma_max = 0.5 * (self.pml_layers + 1) / self.dl
        return (self.pml_layers, sigma_max, 1.0)

    def forward(
        self,
        geometry: torch.Tensor,
        wavelengths: list[float] | torch.Tensor | None = None,
        *,
        source: dict | None = None,
    ) -> SimResult:
        """Run sparse FDFD forward simulation.

        Parameters
        ----------
        geometry : Tensor, shape ``(H, W)``
            Relative permittivity map (``eps_r``).
        wavelengths : ignored
            FDFD is single-frequency; wavelength is set at construction.
        source : dict, optional
            ``{"type": "point"|"line", "pos": [y, x], "amplitude": float}``.

        Returns
        -------
        SimResult
            ``field`` contains the Ez (TM) or Hz (TE) field, shape ``(1, N)``.
        """
        if wavelengths is None:
            wavelengths = [self.wavelength_nm]
        if not isinstance(wavelengths, torch.Tensor):
            wavelengths = torch.tensor(wavelengths, dtype=torch.float64, device=self._device)

        H, W = self.grid_shape
        N = H * W
        device = self._device
        dtype_c = torch.complex128

        eps_r = geometry.to(device)
        if eps_r.dim() == 3:
            eps_r = eps_r.squeeze(0)

        # Source vector
        src = source or {}
        src_type = src.get("type", "point")
        b = torch.zeros(N, dtype=dtype_c, device=device)

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

        # Sparse solve with adjoint backward
        omega = self.omega_norm
        x = _SparseHelmholtzSolve.apply(
            eps_r, b, omega, self.dl, self._pml_params(),
            self.polarization, self.grid_shape,
        )

        field = x.unsqueeze(0)

        return SimResult(
            field=field,
            wavelengths=wavelengths,
            metadata={
                "polarization": self.polarization,
                "grid_shape": self.grid_shape,
                "dl": self.dl,
                "omega_norm": omega,
                "sparse": True,
            },
        )

    def solve(
        self,
        eps_r: torch.Tensor,
        source: dict | None = None,
    ) -> torch.Tensor:
        """Convenience: return the field reshaped as ``(H, W)``."""
        result = self.forward(eps_r, source=source)
        H, W = self.grid_shape
        return result.field.reshape(H, W)
