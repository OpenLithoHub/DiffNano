"""Tests for the Solver Protocol, RCWA, FDFD, and FDTD solvers."""

import pytest
import torch

from diffnano.solvers import FDFDSolver2D, FDTDSolver2D, FDTDSolver3D, RCWASolver, SimResult, Solver


class TestSimResult:
    def test_basic(self):
        field = torch.randn(3, 21)
        wl = torch.tensor([500.0, 532.0, 600.0])
        r = SimResult(field, wl)
        assert r.field.shape == (3, 21)
        assert r.wavelengths.shape == (3,)
        assert r.metadata == {}
        assert r.device == field.device

    def test_metadata(self):
        r = SimResult(torch.randn(10), torch.tensor([500.0]), {"key": "val"})
        assert r.metadata["key"] == "val"


class TestSolverProtocol:
    def test_rcwa_is_solver(self):
        solver = RCWASolver(fourier_orders=3)
        assert isinstance(solver, Solver)


class TestRCWASolver:
    @pytest.fixture
    def solver(self):
        return RCWASolver(
            fourier_orders=3,
            wavelength_nm=532.0,
            period_nm=(400.0, 400.0),
            device="cpu",
        )

    def test_init(self, solver):
        assert solver.fourier_orders == 3
        assert solver.n_fourier == 7
        assert solver.wavelength_nm == 532.0

    def test_forward_1d_shape(self, solver):
        eps = torch.ones(5, 100, dtype=torch.float64) * 1.5  # 5 uniform layers
        result = solver.forward(eps, wavelengths=[532.0])
        assert isinstance(result, SimResult)
        assert result.field.shape[0] == 1  # 1 wavelength
        assert result.field.shape[1] == solver.n_fourier

    def test_forward_2d_shape(self, solver):
        density = torch.rand(3, 20, 20, dtype=torch.float64)
        result = solver.forward(density, wavelengths=[532.0, 600.0])
        assert result.field.shape[0] == 2  # 2 wavelengths
        assert result.field.shape[1] == solver.n_fourier

    def test_gradient_flows(self, solver):
        eps = torch.full((3, 50), 2.0, dtype=torch.float64, requires_grad=True)
        result = solver.forward(eps, wavelengths=[532.0])
        loss = result.field.sum()
        loss.backward()
        assert eps.grad is not None
        assert eps.grad.shape == eps.shape

    def test_diffraction_efficiency(self, solver):
        eps = torch.ones(5, 80, dtype=torch.float64) * 2.0
        eff = solver.diffraction_efficiency(eps, order=0)
        assert eff.numel() == 1
        assert eff.item() >= 0

    def test_transmission(self, solver):
        eps = torch.ones(4, 60, dtype=torch.float64) * 1.5
        t = solver.transmission(eps)
        assert t.numel() == 1
        assert t.item() >= 0

    def test_multi_wavelength(self, solver):
        eps = torch.ones(4, 60, dtype=torch.float64) * 2.0
        result = solver.forward(eps, wavelengths=[500.0, 532.0, 600.0])
        assert result.field.shape[0] == 3

    def test_source_config(self, solver):
        eps = torch.ones(4, 60, dtype=torch.float64) * 2.0
        result = solver.forward(
            eps,
            wavelengths=[532.0],
            source={"theta": 0.1, "polarization": "TE"},
        )
        assert result.field.shape[0] == 1

    def test_invalid_geometry_dim(self, solver):
        with pytest.raises(ValueError, match="2D or 3D"):
            solver.forward(torch.ones(10, dtype=torch.float64))


# -----------------------------------------------------------------------
# FDFD 2D Solver
# -----------------------------------------------------------------------


class TestFDFDSolver2D:
    @pytest.fixture
    def solver(self):
        return FDFDSolver2D(
            grid_shape=(12, 12),
            dl=20.0,
            wavelength_nm=1550.0,
            polarization="TM",
            pml_layers=2,
            device="cpu",
        )

    def test_init(self, solver):
        assert solver.grid_shape == (12, 12)
        assert solver.polarization == "TM"

    def test_forward_shape(self, solver):
        eps = torch.ones(12, 12, dtype=torch.float64) * 2.25
        result = solver.forward(eps)
        assert isinstance(result, SimResult)
        assert result.field.shape[0] == 1
        assert result.field.shape[1] == 12 * 12

    def test_gradient_flows(self, solver):
        eps = torch.full((12, 12), 2.25, dtype=torch.float64, requires_grad=True)
        result = solver.forward(eps)
        loss = result.field.sum()
        loss.backward()
        assert eps.grad is not None
        assert eps.grad.shape == (12, 12)

    def test_solve_method(self, solver):
        eps = torch.ones(12, 12, dtype=torch.float64) * 2.25
        field = solver.solve(eps)
        assert field.shape == (12, 12)

    def test_point_source(self, solver):
        eps = torch.ones(12, 12, dtype=torch.float64) * 2.25
        result = solver.forward(eps, source={"type": "point", "pos": [6, 6]})
        assert result.field.shape[0] == 1

    def test_line_source(self, solver):
        eps = torch.ones(12, 12, dtype=torch.float64) * 2.25
        result = solver.forward(eps, source={"type": "line", "row": 6})
        assert result.field.shape[0] == 1

    def test_te_polarization(self):
        solver = FDFDSolver2D(
            grid_shape=(10, 10),
            dl=20.0,
            polarization="TE",
            pml_layers=0,
        )
        eps = torch.ones(10, 10, dtype=torch.float64) * 2.25
        result = solver.forward(eps)
        assert result.field.shape[0] == 1

    def test_solver_protocol(self, solver):
        assert isinstance(solver, Solver)

    def test_3d_geometry_input(self, solver):
        eps = torch.ones(1, 12, 12, dtype=torch.float64) * 2.25
        result = solver.forward(eps)
        assert result.field.shape[0] == 1


# -----------------------------------------------------------------------
# FDTD 2D Solver
# -----------------------------------------------------------------------


class TestFDTDSolver2D:
    @pytest.fixture
    def solver(self):
        return FDTDSolver2D(
            grid_shape=(20, 20),
            dl=20.0,
            wavelength_nm=1550.0,
            polarization="TM",
            pml_layers=0,
            n_steps=10,
            device="cpu",
        )

    def test_init(self, solver):
        assert solver.grid_shape == (20, 20)
        assert solver.n_steps == 10
        assert solver.courant == 0.5

    def test_forward_shape(self, solver):
        eps = torch.ones(20, 20, dtype=torch.float64) * 2.25
        result = solver.forward(eps)
        assert isinstance(result, SimResult)
        assert result.field.shape == (1, 20, 20)

    def test_gradient_flows(self, solver):
        eps = torch.full((20, 20), 2.25, dtype=torch.float64, requires_grad=True)
        result = solver.forward(eps)
        loss = result.field.sum()
        loss.backward()
        assert eps.grad is not None
        assert eps.grad.shape == (20, 20)

    def test_point_source(self, solver):
        eps = torch.ones(20, 20, dtype=torch.float64) * 2.25
        result = solver.forward(eps, source={"type": "gaussian_pulse", "pos": [10, 10]})
        assert result.field.shape == (1, 20, 20)

    def test_line_source(self, solver):
        eps = torch.ones(20, 20, dtype=torch.float64) * 2.25
        result = solver.forward(eps, source={"type": "gaussian_pulse", "row": 10})
        assert result.field.shape == (1, 20, 20)

    def test_continuous_source(self, solver):
        eps = torch.ones(20, 20, dtype=torch.float64) * 2.25
        result = solver.forward(eps, source={"type": "continuous"})
        assert result.field.shape == (1, 20, 20)

    def test_te_polarization(self):
        solver = FDTDSolver2D(
            grid_shape=(15, 15),
            polarization="TE",
            n_steps=5,
            pml_layers=0,
        )
        eps = torch.ones(15, 15, dtype=torch.float64) * 2.25
        result = solver.forward(eps)
        assert result.field.shape == (1, 15, 15)

    def test_checkpoint_mode(self):
        solver = FDTDSolver2D(
            grid_shape=(15, 15),
            n_steps=10,
            use_checkpoint=True,
            checkpoint_segments=2,
            pml_layers=0,
        )
        eps = torch.ones(15, 15, dtype=torch.float64) * 2.25
        result = solver.forward(eps)
        assert result.field.shape == (1, 15, 15)

    def test_time_series(self, solver):
        eps = torch.ones(20, 20, dtype=torch.float64) * 2.25
        ts = solver.time_series(eps, probe=(15, 15), n_steps=5)
        assert ts.shape == (5,)

    def test_solver_protocol(self, solver):
        assert isinstance(solver, Solver)

    def test_metadata(self, solver):
        eps = torch.ones(20, 20, dtype=torch.float64) * 2.25
        result = solver.forward(eps)
        assert "polarization" in result.metadata
        assert result.metadata["n_steps"] == 10


# -----------------------------------------------------------------------
# FDTD 3D Solver
# -----------------------------------------------------------------------


class TestFDTDSolver3D:
    @pytest.fixture
    def solver(self):
        return FDTDSolver3D(
            grid_shape=(10, 10, 10),
            dl=20.0,
            wavelength_nm=1550.0,
            pml_layers=0,
            n_steps=5,
            device="cpu",
        )

    def test_init(self, solver):
        assert solver.grid_shape == (10, 10, 10)
        assert solver.n_steps == 5
        assert solver.courant == 0.4

    def test_forward_shape(self, solver):
        eps = torch.ones(10, 10, 10, dtype=torch.float64) * 2.25
        result = solver.forward(eps)
        assert isinstance(result, SimResult)
        assert result.field.shape == (3, 10, 10, 10)

    def test_gradient_flows(self, solver):
        eps = torch.full((10, 10, 10), 2.25, dtype=torch.float64, requires_grad=True)
        result = solver.forward(eps)
        loss = result.field.sum()
        loss.backward()
        assert eps.grad is not None
        assert eps.grad.shape == (10, 10, 10)

    def test_point_source(self, solver):
        eps = torch.ones(10, 10, 10, dtype=torch.float64) * 2.25
        result = solver.forward(eps, source={"type": "gaussian_pulse", "pos": [5, 5, 5]})
        assert result.field.shape == (3, 10, 10, 10)

    def test_plane_source(self, solver):
        eps = torch.ones(10, 10, 10, dtype=torch.float64) * 2.25
        result = solver.forward(eps, source={"type": "gaussian_pulse", "plane": "xy", "z": 3})
        assert result.field.shape == (3, 10, 10, 10)

    def test_continuous_source(self, solver):
        eps = torch.ones(10, 10, 10, dtype=torch.float64) * 2.25
        result = solver.forward(eps, source={"type": "continuous"})
        assert result.field.shape == (3, 10, 10, 10)

    def test_checkpoint_mode(self):
        solver = FDTDSolver3D(
            grid_shape=(8, 8, 8),
            n_steps=6,
            use_checkpoint=True,
            checkpoint_segments=2,
            pml_layers=0,
        )
        eps = torch.ones(8, 8, 8, dtype=torch.float64) * 2.25
        result = solver.forward(eps)
        assert result.field.shape == (3, 8, 8, 8)

    def test_time_series(self, solver):
        eps = torch.ones(10, 10, 10, dtype=torch.float64) * 2.25
        ts = solver.time_series(eps, probe=(5, 5, 5), n_steps=5)
        assert ts.shape == (5,)

    def test_solver_protocol(self, solver):
        assert isinstance(solver, Solver)

    def test_4d_geometry_input(self, solver):
        eps = torch.ones(1, 10, 10, 10, dtype=torch.float64) * 2.25
        result = solver.forward(eps)
        assert result.field.shape == (3, 10, 10, 10)

    def test_metadata(self, solver):
        eps = torch.ones(10, 10, 10, dtype=torch.float64) * 2.25
        result = solver.forward(eps)
        assert "grid_shape" in result.metadata
        assert result.metadata["n_steps"] == 5
