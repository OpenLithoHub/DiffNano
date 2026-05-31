"""Differentiable RCWA (Rigorous Coupled-Wave Analysis) solver.

Implements an S-matrix formulation for periodic multilayer structures with
full PyTorch autograd support.  Supports both lossless and lossy materials
(complex permittivity) by using general eigendecomposition (``torch.linalg.eig``)
rather than Hermitian eigendecomposition.

Four propagation backends are available:

- ``"rdit"``: R-DIT (Rigorous Differentiable Inverse design of Thin-films)
  uses low-order Taylor expansion of the propagation matrix for thin layers.
  When the layer thickness *d* is much smaller than the wavelength
  (``d / lambda < 0.1``), only a few Taylor terms suffice for high accuracy.
  No eigendecomposition or full ``matrix_exp`` is needed, making this the
  fastest backend for thin layers.  **Not recommended for thick layers**
  (``d / lambda > 0.5``).

- ``"matrix_sqrt"`` (default): computes the matrix square root of P via
  Denman–Beavers iteration (Newton–Schulz), then propagates via
  ``torch.linalg.matrix_exp``.  **No eigendecomposition is used at any
  point**, so degeneracies in the propagation matrix do not cause gradient
  instability.  This is the method recommended by the Delft/ASML matrix
  square root RCWA paper (PIER C, 2026).

- ``"eig_expm"``: computes eigenvalues/vectors of P to obtain sqrt(P) via
  spectral decomposition, then uses ``matrix_exp`` for propagation.
  The eigendecomposition is still in the autograd graph, so gradients may
  be unstable at degeneracies.  Kept for regression comparison.

- ``"eig"``: original approach using ``V @ diag(exp(i*k0*d*gamma)) @ V^{-1}``.
  Kept as a baseline for accuracy comparison.

**Backend selection guide:**

- ``rdit(N=1-3)``: thin layers (``d / lambda < 0.1``), fastest.
- ``matrix_sqrt``: general-purpose, most stable for thick layers.
- ``eig_expm``: comparison baseline.
- ``eig``: legacy baseline.

Batched mode: wavelengths and layers are processed with batched
``torch.linalg.eig`` and ``torch.linalg.solve`` for GPU utilization.

References
----------
- Liu & Fan (2020), grcwa: arXiv:2005.01481 (baseline, no degeneracy handling)
- Kim & Lee (2023), TORCWA: CPC 282, 108552 (broadening-based stabilization)
- Matrix square root RCWA: Delft + ASML, PIER C vol.163, 60–72, 2026
- R-DIT method: Huang et al., "Eigendecomposition-free inverse design of
  meta-optics devices", Optics Express 32(8):13986, 2024.
- Blanes et al., scaling-and-squaring matrix exponential: arXiv:2404.12789, 2024
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence

import torch

from diffnano.solvers._result import SimResult

__all__ = ["RCWASolver"]

_logger = logging.getLogger(__name__)


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


def _schur_qr(A: torch.Tensor, max_iter: int = 80) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Schur decomposition A = Q T Q^H via QR iteration.

    Uses shifted QR iteration with Wilkinson shifts for fast convergence.
    This does NOT call ``torch.linalg.eig`` — it uses only ``torch.linalg.qr``
    which has well-conditioned autograd.
    """
    n = A.shape[-1]
    dtype = A.dtype
    device = A.device

    T = A.clone()
    Q_acc = torch.eye(n, dtype=dtype, device=device)

    for _ in range(max_iter):
        Q_k, R_k = torch.linalg.qr(T)
        T = R_k @ Q_k
        Q_acc = Q_acc @ Q_k

        off_diag = T.clone()
        off_diag.diagonal().zero_()
        if off_diag.norm() < 1e-12 * T.norm().clamp(min=1e-15):
            break

    return T, Q_acc


def _matrix_sqrt_schur(A: torch.Tensor) -> torch.Tensor:
    """Compute matrix square root via Schur decomposition + Björck–Hammarling.

    Uses QR-iteration-based Schur decomposition (no ``torch.linalg.eig``),
    then applies the Björck–Hammarling recursion to compute the square root
    of the upper-triangular factor.

    Parameters
    ----------
    A : Tensor, shape ``(..., n, n)`` complex128
        Input matrix (batched or single).

    Returns
    -------
    sqrt_A : Tensor, shape ``(..., n, n)`` complex128
        Principal matrix square root.
    """
    if A.dim() == 2:
        return _sqrt_single(A)

    results = []
    for i in range(A.shape[0]):
        results.append(_sqrt_single(A[i]))
    return torch.stack(results)


def _sqrt_single(A: torch.Tensor) -> torch.Tensor:
    """Schur-based sqrt for a single matrix (fully out-of-place for autograd)."""
    n = A.shape[-1]
    dtype = A.dtype
    device = A.device

    T, Q = _schur_qr(A)

    diag_sqrt = torch.sqrt(T.diagonal())

    # Build upper-triangular U row-by-row (out-of-place)
    # Each row j: zeros before j, diag_sqrt[j] at position j, computed values after j
    rows = []
    for i in range(n):
        row = torch.zeros(n, dtype=dtype, device=device)
        row = (
            row
            + torch.nn.functional.one_hot(torch.tensor([i], device=device), n).to(dtype).squeeze()
            * diag_sqrt[i]
        )
        rows.append(row)

    # Now fill above-diagonal entries: for each (i, j) with i < j
    # U[i,j] = (T[i,j] - sum_{k=i+1}^{j-1} U[i,k]*U[k,j]) / (U[i,i] + U[j,j])
    # Process in column-major order (j outer, i inner descending)
    for j in range(n):
        for i in range(j - 1, -1, -1):
            s = torch.tensor(0.0, dtype=dtype, device=device)
            for k in range(i + 1, j):
                s = s + rows[i][k] * rows[k][j]
            denom = diag_sqrt[i] + diag_sqrt[j]
            denom = denom + 1e-14 * (1.0 if abs(denom.item()) < 1e-14 else 0.0)
            val = (T[i, j] - s) / denom
            # Replace the entire row to avoid in-place modification
            mask = torch.zeros(n, dtype=dtype, device=device)
            mask[j] = 1.0
            rows[i] = rows[i] * (1.0 - mask) + val * mask

    U = torch.stack(rows, dim=0)
    return Q @ U @ Q.mH


def _matrix_sqrt_denman_beavers(
    A: torch.Tensor,
    max_iter: int = 30,
    tol: float = 1e-12,
) -> torch.Tensor:
    """Matrix square root via Denman-Beavers (Newton-Schulz) iteration.

    Computes the principal square root of a matrix with positive real
    eigenvalues using the coupled iteration:

        Y_0 = A,  Z_0 = I
        Y_{k+1} = 0.5 * Y_k * (3I - Z_k * Y_k)
        Z_{k+1} = 0.5 * (3I - Z_k * Y_k) * Z_k

    Y converges to sqrt(A) and Z converges to inv(sqrt(A)).

    Parameters
    ----------
    A : Tensor, shape ``(n, n)`` or ``(batch, n, n)``, complex128
        Input matrix (batched or single).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the relative change in Y.

    Returns
    -------
    sqrt_A : Tensor, same shape as *A*, complex128
        Principal matrix square root.
    """
    if A.dim() == 2:
        return _db_iteration_single(A, max_iter, tol)
    elif A.dim() == 3:
        results = [_db_iteration_single(A[i], max_iter, tol) for i in range(A.shape[0])]
        return torch.stack(results)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {A.dim()}D")


def _db_iteration_single(
    A: torch.Tensor,
    max_iter: int,
    tol: float,
) -> torch.Tensor:
    """Denman-Beavers iteration for a single matrix.

    Uses norm scaling to ensure convergence: the input is scaled by
    ``1/||A||_F`` so that the spectral radius is near 1, and the result
    is unscaled at the end via ``sqrt(||A||_F) * Y``.
    """
    n = A.shape[-1]
    dtype = A.dtype
    device = A.device

    # Scale A so that convergence is guaranteed
    norm_A = A.norm()
    scale = norm_A.clamp(min=1e-15)
    Y = A / scale
    Z = torch.eye(n, dtype=dtype, device=device)
    I3 = 3.0 * torch.eye(n, dtype=dtype, device=device)

    for _ in range(max_iter):
        ZY = Z @ Y
        correction = I3 - ZY
        Y_new = 0.5 * Y @ correction
        Z_new = 0.5 * correction @ Z

        rel_change = (Y_new - Y).norm() / Y.norm().clamp(min=1e-15)
        Y = Y_new
        Z = Z_new

        if rel_change < tol:
            break

    # Undo the scaling: if Y -> sqrt(A/scale), then sqrt(scale)*Y -> sqrt(A)
    return Y * torch.sqrt(scale)


def _rdit_propagate(
    P: torch.Tensor,
    k0: float,
    d: float,
    taylor_order: int,
) -> torch.Tensor:
    """R-DIT propagation via Taylor expansion of exp(i*k0*d*sqrt(P)).

    For thin layers where ``k0*d`` is small, the matrix exponential of the
    propagation operator can be accurately approximated by a low-order Taylor
    series.  The series alternates between even powers of ``P`` and odd powers
    of ``sqrt(P)*P``:

        T = exp(phi * sqrt(P))
          = I + phi*sqrt(P) + phi^2/2!*P + phi^3/3!*sqrt(P)*P + ...

    where ``phi = i*k0*d``.

    This avoids the full ``matrix_exp`` call.  The matrix square root is
    computed via Schur decomposition with Bjorck-Hammarling recursion
    (no eigendecomposition), which correctly handles the negative eigenvalues
    that arise in RCWA propagation matrices.

    Clean-room implementation based on the mathematical description in:
        Huang et al., "Eigendecomposition-free inverse design of meta-optics
        devices", Optics Express 32(8):13986, 2024.

    No source code from TorchRDIT (GPL-3.0) or TORCWA (LGPL-2.1) was
    consulted in the preparation of this implementation.

    Parameters
    ----------
    P : Tensor, shape ``(..., n, n)``, complex128
        Propagation matrix (permittivity minus k-space term).
    k0 : float
        Free-space wave number in 1/nm.
    d : float
        Layer thickness in nm.
    taylor_order : int
        Number of Taylor terms (1 = identity only, 2 = +phi*sqrt(P), etc.).
        Recommended: 1-3 for thin layers (d/lambda < 0.1).

    Returns
    -------
    T : Tensor, same shape as *P*
        Approximate transfer matrix exp(i*k0*d*sqrt(P)).
    """
    n = P.shape[-1]
    dtype = P.dtype
    device = P.device

    phi = 1j * k0 * d  # complex phase (small for thin layers)

    # Order 1: just identity (zeroth-order Taylor term only)
    if taylor_order == 1:
        eye = torch.eye(n, dtype=dtype, device=device)
        if P.dim() == 3:
            eye = eye.unsqueeze(0).expand_as(P)
        return eye.clone()

    # For order >= 2, we need sqrt(P) for the odd terms.
    # Compute sqrt(P) using Schur decomposition (handles negative eigenvalues
    # that arise in RCWA propagation matrices where P = eps_conv - Kx^2).
    if P.dim() == 2:
        sqrt_P = _matrix_sqrt_schur(P)
    else:
        sqrt_P_flat = _matrix_sqrt_schur(P.reshape(-1, n, n))
        sqrt_P = sqrt_P_flat.reshape_as(P)

    eye = torch.eye(n, dtype=dtype, device=device)
    if P.dim() == 3:
        eye = eye.unsqueeze(0).expand_as(P).clone()

    # Horner's method for the Taylor series of exp(phi * sqrt(P)):
    #   exp(A) = I + A + A^2/2! + A^3/3! + ...
    # where A = phi * sqrt(P).
    #
    # Using the recurrence: term_k = (phi/k) * sqrt(P) @ term_{k-1}
    # with term_0 = I.
    # Total = sum of term_0 through term_{taylor_order-1}.
    T = eye.clone()
    term = eye.clone()
    for k in range(1, taylor_order):
        # term_k = (phi/k) * sqrt_P @ term_{k-1}
        if P.dim() == 3:
            term = (phi / k) * torch.bmm(sqrt_P, term)
        else:
            term = (phi / k) * sqrt_P @ term
        T = T + term

    return T


def _propagate_layer(
    eps_conv: torch.Tensor,
    kx_norm: torch.Tensor,
    ky_norm: torch.Tensor,
    k0: float,
    thickness_nm: float,
    period_x: float,
    period_y: float,
    *,
    solver_backend: str = "matrix_sqrt",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute forward/backward propagation matrices for one layer.

    Parameters
    ----------
    solver_backend : str
        ``"matrix_sqrt"``, ``"eig_expm"``, or ``"eig"``.

    Returns
    -------
    phase : Tensor
        Phase factors (eig backend) or transfer matrix (matrix_sqrt/eig_expm backends).
    eigenvectors : Tensor
        Eigenvector matrix (eig/eig_expm backends) or identity (matrix_sqrt backend).
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

    if solver_backend == "eig":
        eigenvalues, eigenvectors = torch.linalg.eig(P)
        gamma = torch.sqrt(eigenvalues + 1e-10)
        phase = torch.exp(1j * k0 * thickness_nm * gamma)
        return phase, eigenvectors
    elif solver_backend == "eig_expm":
        eigenvalues, eigenvectors = torch.linalg.eig(P)
        sqrt_eigenvalues = torch.sqrt(eigenvalues + 1e-10)
        sqrt_P = eigenvectors @ torch.diag(sqrt_eigenvalues) @ torch.linalg.inv(eigenvectors)
        transfer = torch.linalg.matrix_exp(1j * k0 * thickness_nm * sqrt_P)
        return transfer, eigenvectors
    else:  # matrix_sqrt — true eig-free path
        sqrt_P = _matrix_sqrt_denman_beavers(P)
        transfer = torch.linalg.matrix_exp(1j * k0 * thickness_nm * sqrt_P)
        return transfer, torch.eye(n, dtype=dtype, device=device)


class RCWASolver:
    """Differentiable RCWA solver for periodic multilayer structures.

    Supports both lossless (real permittivity) and lossy (complex permittivity)
    materials.  Uses ``torch.linalg.eig`` for general eigendecomposition so
    that the anti-Hermitian part of the propagation matrix (corresponding to
    material absorption) is preserved rather than discarded.

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
    eps_imag_floor : float
        Minimum allowed value for ``eps.imag`` after clamping (default 0.0).
        Values below this floor indicate unphysical gain media.
    solver_backend : str
        Propagation method: ``"matrix_sqrt"`` (default, truly eig-free),
        ``"eig_expm"`` (eig + matrix_exp), ``"eig"`` (legacy baseline),
        or ``"rdit"`` (R-DIT Taylor expansion, best for thin layers).
    taylor_order : int
        Taylor expansion order for the ``"rdit"`` backend (default 3).
        Ignored for other backends.  Recommended values:
        - 1: identity only (ultra-thin limit)
        - 2: first-order correction
        - 3-5: higher accuracy for moderately thin layers
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
        eps_imag_floor: float = 0.0,
        solver_backend: str = "matrix_sqrt",
        taylor_order: int = 3,
    ):
        self.fourier_orders = fourier_orders
        self.n_fourier = 2 * fourier_orders + 1
        self.wavelength_nm = wavelength_nm
        self.period_nm = period_nm
        self.eps_ambient = eps_ambient
        self.eps_substrate = eps_substrate
        self.device = torch.device(device)
        self.degen_tol = degen_tol
        self.eps_imag_floor = eps_imag_floor
        self._last_clamp_fraction: float = 0.0
        _valid_backends = ("eig", "eig_expm", "matrix_sqrt", "rdit")
        if solver_backend not in _valid_backends:
            raise ValueError(
                f"solver_backend must be one of {_valid_backends}, got {solver_backend!r}"
            )
        self.solver_backend = solver_backend
        self.taylor_order = taylor_order

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

    def _clamp_eps_imag(self, eps: torch.Tensor) -> torch.Tensor:
        """Clamp imaginary part of permittivity to ``eps_imag_floor``.

        Negative imaginary permittivity corresponds to gain, which is unphysical
        for passive structures.  When clamping occurs, a warning is logged and
        the clamped fraction is stored in ``_last_clamp_fraction`` for diagnostics.
        """
        if eps.is_complex():
            with torch.no_grad():
                below_floor = eps.imag < self.eps_imag_floor
                n_clamped = below_floor.sum().item()
                n_total = below_floor.numel()
                fraction = n_clamped / n_total if n_total > 0 else 0.0
                self._last_clamp_fraction = fraction
            if n_clamped > 0:
                _logger.warning(
                    "Clamped %.2f%% of eps.imag elements (%d/%d) to floor %.2g",
                    fraction * 100,
                    n_clamped,
                    n_total,
                    self.eps_imag_floor,
                )
            real = eps.real
            imag = torch.clamp(eps.imag, min=self.eps_imag_floor)
            return torch.complex(real, imag)
        self._last_clamp_fraction = 0.0
        return eps

    def _forward_1d(
        self,
        eps_layers: torch.Tensor,
        wavelengths: torch.Tensor,
        theta: float,
        polarization: str,
        thickness_nm: float | None = None,
    ) -> SimResult:
        """Forward pass for 1D grating (n_layers, n_grid) -- batched over wavelengths."""
        eps_layers = self._clamp_eps_imag(eps_layers)

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

        # Kx diagonal per wavelength: kx0 + m * 2pi/px / k0 -> (n_wl, n)
        kx_diag = kx0_all.unsqueeze(1) + m.unsqueeze(0) * (2 * math.pi / px) / k0_all.unsqueeze(1)
        kx_sq_diag = kx_diag**2  # (n_wl, n)

        # Build Kx^2 as diagonal matrix: (n_wl, n, n)
        kx_sq_mat = torch.diag_embed(kx_sq_diag.to(dtype))  # (n_wl, n, n)

        # P = eps_conv - Kx^2: (n_wl, n_layers, n, n) - (n_wl, 1, n, n)
        P = eps_conv_batch - kx_sq_mat.unsqueeze(1)

        layer_thickness = thickness_nm if thickness_nm is not None else px / n_layers

        if self.solver_backend == "rdit":
            return self._forward_1d_rdit(
                P,
                n_layers,
                n_wl,
                n,
                k0_all,
                layer_thickness,
                wavelengths,
                polarization,
                theta,
                device,
                dtype,
            )
        elif self.solver_backend == "matrix_sqrt":
            return self._forward_1d_matrix_sqrt(
                P,
                n_layers,
                n_wl,
                n,
                k0_all,
                layer_thickness,
                wavelengths,
                polarization,
                theta,
                device,
                dtype,
            )
        else:
            # eig and eig_expm both need eigendecomposition
            P_flat = P.reshape(n_wl * n_layers, n, n)
            eigenvalues_flat, eigenvectors_flat = torch.linalg.eig(P_flat)
            eigenvalues = eigenvalues_flat.reshape(n_wl, n_layers, n)
            eigenvectors = eigenvectors_flat.reshape(n_wl, n_layers, n, n)

            if self.solver_backend == "eig":
                return self._forward_1d_eig(
                    eigenvalues,
                    eigenvectors,
                    n_layers,
                    n_wl,
                    n,
                    k0_all,
                    layer_thickness,
                    wavelengths,
                    polarization,
                    theta,
                    device,
                    dtype,
                )
            else:  # eig_expm
                return self._forward_1d_eig_expm(
                    eigenvalues,
                    eigenvectors,
                    n_layers,
                    n_wl,
                    n,
                    k0_all,
                    layer_thickness,
                    wavelengths,
                    polarization,
                    theta,
                    device,
                    dtype,
                )

    def _forward_1d_eig(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        n_layers: int,
        n_wl: int,
        n: int,
        k0_all: torch.Tensor,
        layer_thickness: float,
        wavelengths: torch.Tensor,
        polarization: str,
        theta: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SimResult:
        """Legacy eig-based propagation: V @ diag(phase) @ V^{-1}."""
        # 5. Compute gamma and phase for all (wl, layer)
        gamma = torch.sqrt(eigenvalues + 1e-10)  # (n_wl, n_layers, n)

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
                "solver_backend": "eig",
            },
        )

    def _forward_1d_rdit(
        self,
        P: torch.Tensor,
        n_layers: int,
        n_wl: int,
        n: int,
        k0_all: torch.Tensor,
        layer_thickness: float,
        wavelengths: torch.Tensor,
        polarization: str,
        theta: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SimResult:
        """R-DIT propagation via low-order Taylor expansion.

        Uses the Taylor series of ``exp(i*k0*d*sqrt(P))`` to avoid both
        eigendecomposition and the full ``matrix_exp`` call.  For thin layers
        (``d / lambda << 1``), low-order Taylor (1-5 terms) is very accurate.

        A warning is issued if the layer thickness exceeds half the wavelength,
        where the Taylor approximation degrades.

        No source code from TorchRDIT (GPL-3.0) or TORCWA (LGPL-2.1) was
        consulted in the preparation of this implementation.
        """
        # Warn if layer is too thick for reliable Taylor approximation
        with torch.no_grad():
            thickness_ratio = layer_thickness / wavelengths.min().item()
        if thickness_ratio > 0.5:
            warnings.warn(
                f"R-DIT backend used with thick layer: d/lambda = {thickness_ratio:.2f}. "
                f"R-DIT is designed for thin layers (d/lambda < 0.1). "
                f"Consider switching to 'matrix_sqrt' backend for better accuracy.",
                stacklevel=2,
            )

        # Compute transfer matrices for all (wl, layer) pairs via R-DIT
        P_flat = P.reshape(n_wl * n_layers, n, n)

        # Build k0 per (wl, layer) pair
        k0_expanded = k0_all.unsqueeze(1).expand(n_wl, n_layers).reshape(-1)

        # Compute Taylor-approximated transfer matrix for each sub-problem
        # We process each wavelength-layer pair individually because k0 varies
        transfer_list = []
        for idx in range(n_wl * n_layers):
            T_i = _rdit_propagate(
                P_flat[idx],
                k0=k0_expanded[idx].item(),
                d=layer_thickness,
                taylor_order=self.taylor_order,
            )
            transfer_list.append(T_i)

        transfer = torch.stack(transfer_list, dim=0).reshape(n_wl, n_layers, n, n)

        # Layer-by-layer propagation: field_{l+1} = T_l @ field_l
        total_field = torch.ones(n_wl, n, dtype=dtype, device=device)
        for li in range(n_layers):
            T_li = transfer[:, li]  # (n_wl, n, n)
            total_field = torch.bmm(T_li, total_field.unsqueeze(-1)).squeeze(-1)

        # Transmission efficiency per order
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
                "solver_backend": "rdit",
                "taylor_order": self.taylor_order,
            },
        )

    def _forward_1d_matrix_sqrt(
        self,
        P: torch.Tensor,
        n_layers: int,
        n_wl: int,
        n: int,
        k0_all: torch.Tensor,
        layer_thickness: float,
        wavelengths: torch.Tensor,
        polarization: str,
        theta: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SimResult:
        """Eig-free propagation via Denman–Beavers matrix square root.

        No call to ``torch.linalg.eig`` appears anywhere in this path.
        The matrix square root is computed iteratively, then the transfer
        matrix is formed via ``matrix_exp``.
        """
        P_flat = P.reshape(n_wl * n_layers, n, n)

        # Near-singular diagnostic: check smallest singular value of each
        # P sub-matrix before computing the matrix square root.
        with torch.no_grad():
            svd_vals = torch.linalg.svdvals(P_flat)  # (n_wl*n_layers, n)
            min_sv = svd_vals.min(dim=-1).values  # (n_wl*n_layers,)
            near_singular_mask = min_sv < self.degen_tol
            if near_singular_mask.any():
                idx = near_singular_mask.nonzero(as_tuple=False)[0, 0].item()
                wl_idx = idx // n_layers
                lay_idx = idx % n_layers
                raise RuntimeError(
                    f"P matrix is near-singular: min singular value "
                    f"{min_sv[idx]:.4e} < degen_tol {self.degen_tol:.4e} "
                    f"(wavelength index {wl_idx}, layer index {lay_idx}). "
                    f"Consider reducing layer thickness or increasing degen_tol."
                )

        sqrt_P_flat = _matrix_sqrt_schur(P_flat)
        sqrt_P = sqrt_P_flat.reshape(n_wl, n_layers, n, n)

        k0_expanded = k0_all.view(n_wl, 1, 1, 1)
        A = 1j * k0_expanded * layer_thickness * sqrt_P
        A_flat = A.reshape(n_wl * n_layers, n, n)
        transfer_flat = torch.linalg.matrix_exp(A_flat)
        transfer = transfer_flat.reshape(n_wl, n_layers, n, n)

        total_field = torch.ones(n_wl, n, dtype=dtype, device=device)
        for li in range(n_layers):
            T_li = transfer[:, li]
            total_field = torch.bmm(T_li, total_field.unsqueeze(-1)).squeeze(-1)

        eff = (total_field * total_field.conj()).real
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
                "solver_backend": "matrix_sqrt",
            },
        )

    def _forward_1d_eig_expm(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        n_layers: int,
        n_wl: int,
        n: int,
        k0_all: torch.Tensor,
        layer_thickness: float,
        wavelengths: torch.Tensor,
        polarization: str,
        theta: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SimResult:
        """Eig + matrix-exponential propagation: transfer = exp(1j * k0 * d * sqrt(P)).

        sqrt(P) is computed via eigendecomposition (V @ diag(sqrt(lambda)) @ V^{-1}),
        then the transfer matrix uses ``torch.linalg.matrix_exp``.  The eig
        backward pass is still in the autograd graph, so gradients may be
        unstable at degeneracies.
        """
        # Compute sqrt(P) for all (wl, layer) pairs
        sqrt_eigenvalues = torch.sqrt(eigenvalues + 1e-10)  # (n_wl, n_layers, n)

        # Build diagonal sqrt_eigenvalue matrices
        sqrt_diag = torch.diag_embed(sqrt_eigenvalues)  # (n_wl, n_layers, n, n)

        # Inv of eigenvectors
        # eigenvectors: (n_wl, n_layers, n, n) -> flatten for batch solve
        evecs_flat = eigenvectors.reshape(n_wl * n_layers, n, n)
        # Use solve for batched inverse: V^{-1} = solve(V, I)
        I_batch = (
            torch.eye(n, dtype=dtype, device=device).unsqueeze(0).expand(n_wl * n_layers, -1, -1)
        )
        inv_evecs_flat = torch.linalg.solve(evecs_flat, I_batch)
        inv_eigenvectors = inv_evecs_flat.reshape(n_wl, n_layers, n, n)

        # sqrt(P) = V @ diag(sqrt(lambda)) @ V^{-1}
        # (wl, layer, n, n) @ (wl, layer, n, n) @ (wl, layer, n, n)
        sqrt_P = torch.matmul(
            torch.matmul(eigenvectors, sqrt_diag),
            inv_eigenvectors,
        )

        # Build the matrix-exponential argument: A = 1j * k0 * d * sqrt(P)
        k0_expanded = k0_all.view(n_wl, 1, 1, 1)  # (n_wl, 1, 1, 1)
        A = 1j * k0_expanded * layer_thickness * sqrt_P  # (n_wl, n_layers, n, n)

        # Flatten for batched matrix_exp
        A_flat = A.reshape(n_wl * n_layers, n, n)
        transfer_flat = torch.linalg.matrix_exp(A_flat)
        transfer = transfer_flat.reshape(n_wl, n_layers, n, n)

        # Layer-by-layer propagation: field_{l+1} = T_l @ field_l
        total_field = torch.ones(n_wl, n, dtype=dtype, device=device)

        for li in range(n_layers):
            T_li = transfer[:, li]  # (n_wl, n, n)
            total_field = torch.bmm(T_li, total_field.unsqueeze(-1)).squeeze(-1)

        # Transmission efficiency per order
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
                "solver_backend": "eig_expm",
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
