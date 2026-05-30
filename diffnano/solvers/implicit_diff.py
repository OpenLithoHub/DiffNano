"""Matrix-free GMRES and implicit differentiation for the FDFD solver.

Provides memory-efficient adjoint-based gradients through the converged FDFD
field without materialising the full Jacobian.  The adjoint equation
  (∂R/∂E)^T λ = ∂L/∂E
is solved with restarted GMRES using only matvec products obtained from
torch.func.vjp, so peak memory is O(N · restart) rather than O(N²).

Ported from DiffCFD's diffcfd.utils.linalg (GMRES) and
diffcfd.solvers.implicit_diff (fixed-point gradient), adapted for the
Helmholtz residual used by DiffNano's FDFDSolver2D.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

__all__ = ["gmres_matfree", "fdfd_fixed_point_gradient"]


# ---------------------------------------------------------------------------
# Matrix-free restarted GMRES (GMRES-m / Arnoldi)
# ---------------------------------------------------------------------------


def gmres_matfree(
    matvec: Callable[[Tensor], Tensor],
    b: Tensor,
    x0: Tensor | None = None,
    tol: float = 1e-6,
    max_iter: int = 200,
    restart: int = 30,
) -> tuple[Tensor, int]:
    """Solve A x = b with matrix-free restarted GMRES.

    Uses Modified Gram-Schmidt Arnoldi with Givens rotations for numerical
    stability.  All computation is pure PyTorch --- no scipy or numpy
    dependency.

    Parameters
    ----------
    matvec : callable
        Function computing y = A @ v for an input vector v.
    b : Tensor, shape (N,)
        Right-hand side vector.
    x0 : Tensor or None
        Initial guess; defaults to the zero vector.
    tol : float
        Relative residual tolerance: convergence when ||r|| / ||b|| < tol.
    max_iter : int
        Maximum total Arnoldi iterations across all restart cycles.
    restart : int
        Krylov subspace dimension per restart cycle.

    Returns
    -------
    x : Tensor, shape (N,)
        Approximate solution.
    total_iters : int
        Number of Arnoldi iterations performed.
    """
    dtype = b.dtype
    device = b.device
    b_norm: Tensor = b.norm()
    tol_bnorm = (tol * b_norm).item()
    if b_norm == 0:
        return torch.zeros_like(b), 0

    x = x0.clone() if x0 is not None else torch.zeros_like(b)
    total_iters = 0
    converged = False

    for _ in range(max(1, max_iter // restart + 1)):
        # Residual for current x
        r = b - matvec(x)
        r_norm = r.norm()
        if r_norm < tol * b_norm:
            converged = True
            break

        m = min(restart, max_iter - total_iters)
        if m <= 0:
            break

        # Arnoldi basis (stored as list for incremental growth)
        Q: list[Tensor] = [r / r_norm]
        H = torch.zeros(m + 1, m, dtype=dtype, device=device)
        cs = torch.zeros(m, dtype=dtype, device=device)
        sn = torch.zeros(m, dtype=dtype, device=device)
        # RHS of the least-squares problem in the Krylov subspace
        e1 = torch.zeros(m + 1, dtype=dtype, device=device)
        e1[0] = r_norm

        j_used = 0
        for j in range(m):
            # Matrix-vector product with latest basis vector
            w = matvec(Q[j])

            # Modified Gram-Schmidt orthogonalisation.
            # torch.dot computes sum(a*b) without conjugation, which is wrong
            # for the complex inner product.  Use torch.vdot (conjugates the
            # first argument) or fall back to dot for real tensors.
            for i in range(j + 1):
                if b.is_complex():
                    H[i, j] = torch.vdot(Q[i], w)
                else:
                    H[i, j] = torch.dot(w, Q[i])
                w = w - H[i, j] * Q[i]
            H[j + 1, j] = w.norm()
            if H[j + 1, j].abs() > 1e-14:
                Q.append(w / H[j + 1, j])
            else:
                Q.append(torch.zeros_like(w))

            # Apply previously computed Givens rotations to column j.
            #
            # We use a unitary Givens rotation G such that
            #   G [h_i; h_{i+1}] = [r; 0]   with r real >= 0
            #
            # G = [[c, s], [-conj(s), conj(c)]]  where
            #   c = conj(h_jj)/d,  s = conj(h_j1j)/d,  d = sqrt(|h_jj|^2+|h_j1j|^2)
            #
            # For real-valued tensors this reduces to the standard real Givens.
            is_complex = b.is_complex()
            for i in range(j):
                if is_complex:
                    tmp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                    H[i + 1, j] = -torch.conj(sn[i]) * H[i, j] + torch.conj(cs[i]) * H[i + 1, j]
                else:
                    tmp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                    H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = tmp

            # Compute new Givens rotation for (H[j,j], H[j+1,j])
            h_jj = H[j, j]
            h_j1j = H[j + 1, j]
            if is_complex:
                denom = torch.sqrt(
                    torch.conj(h_jj) * h_jj + torch.conj(h_j1j) * h_j1j,
                )
            else:
                denom = torch.sqrt(h_jj * h_jj + h_j1j * h_j1j)

            if denom.abs() < 1e-14:
                cs_j = torch.tensor(1.0, dtype=dtype, device=device)
                sn_j = torch.tensor(0.0, dtype=dtype, device=device)
            else:
                if is_complex:
                    cs_j = torch.conj(h_jj) / denom
                    sn_j = torch.conj(h_j1j) / denom
                else:
                    cs_j = h_jj / denom
                    sn_j = h_j1j / denom

            cs[j] = cs_j
            sn[j] = sn_j

            # Apply new rotation to H and the RHS
            H[j, j] = cs_j * H[j, j] + sn_j * H[j + 1, j]
            H[j + 1, j] = 0.0
            if is_complex:
                e1[j + 1] = -torch.conj(sn_j) * e1[j]
                e1[j] = cs_j * e1[j]
            else:
                e1[j + 1] = -sn_j * e1[j]
                e1[j] = cs_j * e1[j]

            total_iters += 1
            j_used = j + 1

            # Check convergence via residual estimate in transformed basis
            if e1[j + 1].abs() < tol_bnorm:
                converged = True
                break

        if j_used == 0:
            break

        # Back-solve the upper-triangular system H y = e1
        H_sq = H[:j_used, :j_used]
        e_sq = e1[:j_used]
        y = torch.linalg.solve_triangular(
            H_sq,
            e_sq.unsqueeze(1),
            upper=True,
        ).squeeze(1)

        # Update solution: x += Q y
        Q_mat = torch.stack(Q[:j_used], dim=1)  # (N, j_used)
        x = x + Q_mat @ y

        if converged:
            break

    return x, total_iters


# ---------------------------------------------------------------------------
# Implicit differentiation through the FDFD converged field
# ---------------------------------------------------------------------------


def _fdfd_residual(
    E: Tensor,
    eps_r: Tensor,
    A_fn: Callable[[Tensor], Tensor],
    b: Tensor,
) -> Tensor:
    """Evaluate the FDFD residual R(E, eps_r) = A(eps_r) @ E - b.

    The Helmholtz operator A depends on the permittivity eps_r.  At the
    converged solution of the forward solve we have R(E*, eps_r) = 0.

    Parameters
    ----------
    E : Tensor, shape (N,)
        Field vector (complex).
    eps_r : Tensor, shape (H, W)
        Relative permittivity map (real-valued, detached copy with grad).
    A_fn : callable
        Closure that builds the Helmholtz operator matrix A from eps_r and
        returns A.  Signature: A_fn(eps_r) -> Tensor (N, N).
    b : Tensor, shape (N,)
        Source vector (complex).

    Returns
    -------
    residual : Tensor, shape (N,)
    """
    A = A_fn(eps_r)
    return A @ E - b


def fdfd_fixed_point_gradient(
    solver,
    geometry: Tensor,
    loss_grad: Tensor,
    tol: float = 1e-6,
    max_iter: int = 200,
    E_star: Tensor | None = None,
    source: dict | None = None,
) -> Tensor:
    """Compute dL/d(eps_r) via the implicit function theorem at the FDFD solution.

    At the converged field E* of the Helmholtz equation
        R(E, eps_r) = A(eps_r) E - b = 0
    the implicit function theorem gives
        dL/d(eps_r) = -(∂R/∂eps_r)^T λ
    where λ solves the adjoint equation
        (∂R/∂E)^T λ = ∂L/∂E.

    The Jacobian-transpose matvec is evaluated via ``torch.func.vjp``
    so the full Jacobian is never materialised.  The adjoint equation is
    solved with :func:`gmres_matfree`.

    Parameters
    ----------
    solver : FDFDSolver2D
        A configured FDFD solver instance.  Its forward-pass parameters
        (grid shape, dl, polarization, PML, omega_norm) are used to rebuild
        the Helmholtz operator.
    geometry : Tensor, shape (H, W)
        Relative permittivity map (``eps_r``).  Must match the grid used to
        construct *solver*.
    loss_grad : Tensor, shape (H, W) or (H*W,)
        Gradient of the scalar loss with respect to the field E, in PyTorch's
        autograd convention (complex-valued).  For a loss L computed from the
        field via standard PyTorch operations, this is the cotangent that
        ``loss.backward()`` would propagate to the field tensor.  If 2-D it
        will be flattened to (H*W,).
    tol : float
        GMRES convergence tolerance for the adjoint solve.
    max_iter : int
        Maximum GMRES iterations.

    Returns
    -------
    grad_eps_r : Tensor, shape (H, W)
        Gradient of the loss with respect to the permittivity map.
        Returned on the same device and with ``torch.float64`` dtype
        (DiffNano convention).
    """
    device = solver.device
    H, W = solver.grid_shape
    N = H * W

    # Ensure geometry is on the correct device
    eps_r = geometry.to(device)
    if eps_r.dim() == 3:
        eps_r = eps_r.squeeze(0)

    # --- Run the forward solve to get E* (detached) -----------------------
    if E_star is None:
        result = solver.forward(eps_r)
        E_star = result.field.reshape(-1).detach()  # (N,), complex128, no grad
    else:
        E_star = E_star.to(device).reshape(-1).detach()

    # Flatten loss_grad to (N,) complex to match field dtype
    dL_dE = loss_grad.to(device).reshape(-1)
    if dL_dE.is_floating_point():
        dL_dE = dL_dE.to(torch.complex128)

    # --- Build the operator-assembly closure -------------------------------
    # PML parameters (mirrors FDFDSolver2D.forward)
    pml_params: tuple[int, float, float] | None = None
    if solver.pml_layers > 0:
        sigma_max = 0.5 * (solver.pml_layers + 1) / solver.dl
        pml_params = (solver.pml_layers, sigma_max, 1.0)

    omega = solver.omega_norm

    # Import the operator builders locally to avoid circular top-level deps
    from diffnano.solvers.fdfd2d import _build_helmholtz_te, _build_helmholtz_tm

    if solver.polarization == "TM":
        build_A = _build_helmholtz_tm
    else:
        build_A = _build_helmholtz_te

    def A_fn(eps_r_arg: Tensor) -> Tensor:
        return build_A(omega, eps_r_arg, solver.dl, pml_params)

    # --- Build the source vector b (mirrors FDFDSolver2D.forward) ---------
    src = source or {}
    dtype_c = torch.complex128
    b = torch.zeros(N, dtype=dtype_c, device=device)
    pos = src.get("pos", [H // 2, W // 2])
    idx = pos[0] * W + pos[1]
    b[idx] = 1.0

    # --- Adjoint solve: (∂R/∂E)^T λ = ∂L/∂E ----------------------------
    # Use a detached copy of eps_r that tracks gradients for the vjp below
    eps_r_d = eps_r.detach().clone().requires_grad_(True)

    def matvec_Jt_E(v: Tensor) -> Tensor:
        """Compute (∂R/∂E)^T v via vjp."""
        _, vjp_fn = torch.func.vjp(
            lambda E: _fdfd_residual(E, eps_r_d, A_fn, b),
            E_star,
        )
        return vjp_fn(v)[0]

    # The adjoint equation is (∂R/∂E)^T λ = dL/dE
    lambda_vec, _ = gmres_matfree(
        matvec_Jt_E,
        dL_dE.detach(),
        tol=tol,
        max_iter=max_iter,
    )

    # --- Compute dL/d(eps_r) = -(∂R/∂eps_r)^T λ ------------------------
    _, vjp_fn_eps = torch.func.vjp(
        lambda eps: _fdfd_residual(E_star, eps, A_fn, b),
        eps_r_d,
    )
    dL_deps = -vjp_fn_eps(lambda_vec.detach())[0]

    return dL_deps
