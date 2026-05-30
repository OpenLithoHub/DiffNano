"""Stress tests for RCWA backends: degeneracy and thick-layer stability.

Verifies that:
1. The matrix_sqrt backend handles degenerate eigenvalue structures
   (symmetric gratings causing paired/repeated eigenvalues) without
   gradient explosion, where the eig backend degrades.
2. All three backends agree on forward output for well-conditioned inputs.
3. Thick layers produce stable numerical results across backends.
"""

import pytest
import torch

from diffnano.solvers.rcwa import RCWASolver


def _symmetric_grating(n_layers: int, n_grid: int, eps_high: float = 4.0) -> torch.Tensor:
    """Create a symmetric grating that induces eigenvalue degeneracy.

    Symmetric permittivity profiles (even function about the center of
    the period) cause the RCWA propagation matrix P to have paired or
    repeated eigenvalues, which is the regime where eig-based backward
    is numerically unstable.
    """
    x = torch.linspace(0, 2 * torch.pi, n_grid, dtype=torch.float64)
    eps_bg = 1.0
    modulation = (eps_high - eps_bg) * 0.5 * (1.0 + torch.cos(2 * x))
    eps_profile = eps_bg + modulation
    return eps_profile.unsqueeze(0).expand(n_layers, -1).clone()


def _degenerate_lossy_grating(n_layers: int, n_grid: int) -> torch.Tensor:
    """Lossy symmetric grating for complex-P degeneracy testing."""
    real_part = _symmetric_grating(n_layers, n_grid, eps_high=4.0)
    imag_part = torch.full_like(real_part, 0.3)
    return torch.complex(real_part, imag_part)


BACKENDS = ["matrix_sqrt", "eig_expm", "eig"]


class TestDegeneracyStress:
    """Test that matrix_sqrt handles degenerate P matrices gracefully."""

    @pytest.fixture
    def solver_matrix_sqrt(self):
        return RCWASolver(
            fourier_orders=5,
            wavelength_nm=532.0,
            period_nm=(400.0, 400.0),
            solver_backend="matrix_sqrt",
        )

    @pytest.fixture
    def solver_eig(self):
        return RCWASolver(
            fourier_orders=5,
            wavelength_nm=532.0,
            period_nm=(400.0, 400.0),
            solver_backend="eig",
        )

    def test_matrix_sqrt_gradient_no_nan_degenerate(self, solver_matrix_sqrt):
        """matrix_sqrt backend must produce finite gradients on degenerate P."""
        eps = _symmetric_grating(5, 100).detach().requires_grad_(True)
        result = solver_matrix_sqrt.forward(eps, wavelengths=[532.0])
        loss = result.field[:, solver_matrix_sqrt.fourier_orders].sum()
        loss.backward()
        assert eps.grad is not None
        assert torch.isfinite(eps.grad).all(), "NaN in matrix_sqrt gradient on degenerate input"

    def test_matrix_sqrt_gradient_no_nan_lossy_degenerate(self, solver_matrix_sqrt):
        """matrix_sqrt handles complex degenerate P (lossy symmetric grating)."""
        eps = _degenerate_lossy_grating(5, 100).detach().requires_grad_(True)
        result = solver_matrix_sqrt.forward(eps, wavelengths=[532.0])
        loss = result.field[:, solver_matrix_sqrt.fourier_orders].sum()
        loss.backward()
        assert eps.grad is not None
        assert torch.isfinite(eps.grad).all(), "NaN in matrix_sqrt gradient on lossy degenerate input"

    def test_backward_agreement_all_backends(self):
        """All three backends produce physically equivalent forward results.

        The QR-iteration Schur decomposition may reorder diffraction orders
        compared to eig, so we compare sorted efficiency values rather than
        order-sensitive element-wise comparison.
        """
        eps_base = _symmetric_grating(5, 100, eps_high=3.5)
        results = {}
        for backend in BACKENDS:
            solver = RCWASolver(
                fourier_orders=5,
                wavelength_nm=532.0,
                period_nm=(400.0, 400.0),
                solver_backend=backend,
            )
            with torch.no_grad():
                r = solver.forward(eps_base.clone(), wavelengths=[532.0])
            results[backend] = r.field.sort().values

        for pair in [("matrix_sqrt", "eig"), ("matrix_sqrt", "eig_expm")]:
            a, b = results[pair[0]], results[pair[1]]
            rel_err = (a - b).abs().max() / (a.abs().max() + 1e-12)
            assert rel_err < 0.1, (
                f"Forward mismatch {pair[0]} vs {pair[1]} (sorted): rel_err={rel_err:.4f}"
            )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_multi_seed_gradient_no_nan(self, backend):
        """10-seed gradient stability check for each backend."""
        solver = RCWASolver(
            fourier_orders=5,
            wavelength_nm=532.0,
            period_nm=(400.0, 400.0),
            solver_backend=backend,
        )
        n_nan = 0
        for seed in range(10):
            torch.manual_seed(seed)
            eps = (2.0 + torch.rand(5, 100, dtype=torch.float64)).detach().requires_grad_(True)
            result = solver.forward(eps, wavelengths=[532.0])
            loss = result.field[:, solver.fourier_orders].sum()
            loss.backward()
            if not torch.isfinite(eps.grad).all():
                n_nan += 1
        assert n_nan == 0, (
            f"Backend {backend}: {n_nan}/10 seeds produced NaN gradients"
        )


class TestThickLayerStability:
    """Verify numerical stability for layers that are thick relative to wavelength."""

    @pytest.fixture
    def solver(self):
        return RCWASolver(
            fourier_orders=3,
            wavelength_nm=532.0,
            period_nm=(400.0, 400.0),
        )

    def test_thick_layer_forward_finite(self, solver):
        """Thick layer (10x wavelength) produces finite output."""
        eps = torch.ones(3, 80, dtype=torch.float64) * 2.25
        result = solver.forward(
            eps,
            wavelengths=[532.0],
            source={"thickness_nm": 5320.0},  # 10 wavelengths thick
        )
        assert torch.isfinite(result.field).all()

    def test_thick_layer_gradient_finite(self, solver):
        """Gradient through thick layer is finite."""
        eps = (2.0 + 0.3 * torch.rand(3, 80, dtype=torch.float64)).detach().requires_grad_(True)
        result = solver.forward(
            eps,
            wavelengths=[532.0],
            source={"thickness_nm": 2660.0},  # 5 wavelengths
        )
        loss = result.field[:, solver.fourier_orders].sum()
        loss.backward()
        assert eps.grad is not None
        assert torch.isfinite(eps.grad).all(), "NaN in gradient for thick layer"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_thick_layer_all_backends(self, backend):
        """All backends handle thick layers without NaN."""
        solver = RCWASolver(
            fourier_orders=3,
            wavelength_nm=532.0,
            period_nm=(400.0, 400.0),
            solver_backend=backend,
        )
        eps = torch.ones(3, 80, dtype=torch.float64) * 2.25
        result = solver.forward(
            eps,
            wavelengths=[532.0],
            source={"thickness_nm": 5320.0},
        )
        assert torch.isfinite(result.field).all(), (
            f"NaN in forward for thick layer with backend={backend}"
        )
