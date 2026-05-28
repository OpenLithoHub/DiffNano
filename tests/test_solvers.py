"""Tests for the Solver Protocol and RCWA solver."""

import pytest
import torch

from diffnano.solvers import RCWASolver, SimResult, Solver


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
