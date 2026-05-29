"""Tests for the metalens, PhC, and waveguide workflows."""

import pytest
import torch

from diffnano.solvers.rcwa import RCWASolver
from diffnano.workflows.broadband import BroadbandOptimizer
from diffnano.workflows.end_to_end import EndToEndPipeline
from diffnano.workflows.metalens import MetalensDesigner
from diffnano.workflows.multi_objective import MultiObjectiveExplorer
from diffnano.workflows.phc import PhCDesigner
from diffnano.workflows.waveguide import WaveguideDesigner


class TestMetalensDesigner:
    @pytest.fixture
    def designer(self):
        return MetalensDesigner(
            wavelength_nm=532.0,
            numerical_aperture=0.5,
            diameter_um=10.0,  # small for fast tests
            pixel_size_nm=200.0,
            fourier_orders=3,
            device="cpu",
        )

    def test_init(self, designer):
        assert designer.wavelength_nm == 532.0
        assert designer.na == 0.5
        assert designer.target_phase.shape == designer.grid_shape

    def test_phase_matching_loss(self, designer):
        h = torch.rand(*designer.grid_shape, dtype=torch.float64)
        loss = designer.phase_matching_loss(h)
        assert loss.numel() == 1
        assert loss.item() >= 0

    def test_strehl_ratio(self, designer):
        h = torch.rand(*designer.grid_shape, dtype=torch.float64)
        strehl = designer.strehl_ratio(h)
        assert 0.0 <= strehl.item() <= 1.0

    def test_gradient_flow(self, designer):
        h = torch.rand(*designer.grid_shape, dtype=torch.float64, requires_grad=True)
        loss = designer.phase_matching_loss(h)
        loss.backward()
        assert h.grad is not None

    def test_optimize_short(self, designer):
        h, history = designer.optimize(
            n_steps=5,
            verbose=False,
        )
        assert h.shape == designer.grid_shape
        assert len(history) == 5

    def test_optimize_robust(self, designer):
        h, history = designer.optimize(
            n_steps=3,
            robust=True,
            sigma_nm=5.0,
            n_mc_samples=4,
            verbose=False,
        )
        assert h.shape == designer.grid_shape
        assert len(history) == 3


# -----------------------------------------------------------------------
# Photonic Crystal Workflow
# -----------------------------------------------------------------------


class TestPhCDesigner:
    @pytest.fixture
    def designer(self):
        return PhCDesigner(
            lattice="square",
            lattice_constant_nm=400.0,
            n_air=1.0,
            n_material=3.5,
            grid_resolution=8,
            n_g=2,
            n_bands=4,
            polarization="TM",
            target_band_gap=(1, 2),
            device="cpu",
        )

    def test_init(self, designer):
        assert designer.grid_shape == (8, 8)
        assert designer.lattice == "square"

    def test_k_path(self, designer):
        k = designer.k_points
        assert k.shape[1] == 2
        assert k.shape[0] > 0

    def test_band_structure_shape(self, designer):
        density = torch.rand(8, 8, dtype=torch.float64)
        bands = designer.band_structure(density)
        assert bands.shape[0] == designer.k_points.shape[0]
        # n_bands is capped by min(6, N_G)
        assert bands.shape[1] > 0

    def test_bandgap_ratio(self, designer):
        # Uniform density: no bandgap expected
        density = torch.ones(8, 8, dtype=torch.float64) * 0.5
        ratio = designer.bandgap_ratio(density)
        assert ratio.numel() == 1
        assert ratio.item() >= 0

    def test_bandgap_loss(self, designer):
        density = torch.rand(8, 8, dtype=torch.float64, requires_grad=True)
        loss = designer.bandgap_loss(density)
        assert loss.numel() == 1
        loss.backward()
        assert density.grad is not None

    def test_maximize_bandgap_short(self, designer):
        density, history = designer.maximize_bandgap(
            n_steps=3,
            verbose=False,
        )
        assert density.shape == (8, 8)
        assert len(history) == 3


# -----------------------------------------------------------------------
# Waveguide Workflow
# -----------------------------------------------------------------------


class TestWaveguideDesigner:
    @pytest.fixture
    def designer(self):
        return WaveguideDesigner(
            wavelength_nm=1550.0,
            grid_shape=(30, 30),
            dl=20.0,
            n_core=2.5,
            n_clad=1.0,
            waveguide_width_nm=200.0,
            device="cpu",
        )

    def test_init(self, designer):
        assert designer.wavelength_nm == 1550.0
        assert designer.grid_shape == (30, 30)

    def test_waveguide_eps(self, designer):
        eps = designer.waveguide_eps()
        assert eps.shape == (30, 30)
        assert eps.min().item() == pytest.approx(1.0)  # clad
        assert eps.max().item() == pytest.approx(2.5**2)  # core

    def test_fundamental_mode(self, designer):
        n_eff, mode = designer.fundamental_mode()
        assert n_eff.numel() == 1
        assert n_eff.item() > 1.0  # guided
        assert mode.shape == (30, 30)

    def test_mode_overlap(self, designer):
        field = torch.randn(30, 30, dtype=torch.float64)
        target = torch.randn(30, 30, dtype=torch.float64)
        overlap = designer.mode_overlap(field, target)
        assert 0.0 <= overlap.item() <= 1.0

    def test_mode_overlap_gradient(self, designer):
        field = torch.randn(30, 30, dtype=torch.float64, requires_grad=True)
        target = torch.randn(30, 30, dtype=torch.float64)
        overlap = designer.mode_overlap(field, target)
        overlap.backward()
        assert field.grad is not None

    def test_transmission_loss(self, designer):
        field = torch.randn(30, 30, dtype=torch.float64)
        target = torch.randn(30, 30, dtype=torch.float64)
        loss = designer.transmission_loss(field, target)
        assert loss.numel() == 1
        assert loss.item() >= 0

    def test_self_overlap(self, designer):
        mode = torch.randn(30, 30, dtype=torch.float64)
        overlap = designer.mode_overlap(mode, mode)
        assert overlap.item() == pytest.approx(1.0, abs=0.01)


# -----------------------------------------------------------------------
# Broadband Optimization
# -----------------------------------------------------------------------


class TestBroadbandOptimizer:
    @pytest.fixture
    def optimizer(self):
        solver = RCWASolver(
            fourier_orders=3,
            wavelength_nm=532.0,
            period_nm=(400.0, 400.0),
        )
        return BroadbandOptimizer(
            solver=solver,
            wavelengths_nm=[500.0, 532.0, 600.0],
            grid_shape=(15, 15),
            n_layers=3,
        )

    def test_init(self, optimizer):
        assert optimizer.n_wl == 3
        assert optimizer.weights.shape == (3,)

    def test_objective(self, optimizer):
        density = torch.rand(15, 15, dtype=torch.float64)
        loss = optimizer.objective(density, target_order=0)
        assert loss.numel() == 1

    def test_objective_gradient(self, optimizer):
        density = torch.rand(15, 15, dtype=torch.float64, requires_grad=True)
        loss = optimizer.objective(density, target_order=0)
        loss.backward()
        assert density.grad is not None

    def test_optimize_short(self, optimizer):
        density, history = optimizer.optimize(
            n_steps=3,
            target_order=0,
            verbose=False,
        )
        assert density.shape == (15, 15)
        assert len(history) == 3


# -----------------------------------------------------------------------
# Multi-Objective Explorer (C8)
# -----------------------------------------------------------------------


class TestMultiObjectiveExplorer:
    @pytest.fixture
    def explorer(self):
        return MultiObjectiveExplorer(
            objectives={
                "transmission": lambda d: -(d.sum()),
                "binarization": lambda d: ((d - 0.5) ** 2).mean(),
            },
            grid_shape=(10, 10),
            n_pareto_points=3,
        )

    def test_init(self, explorer):
        assert explorer.n_objectives == 2
        assert explorer.n_pareto_points == 3

    def test_scalarized_loss(self, explorer):
        density = torch.rand(10, 10, dtype=torch.float64)
        weights = {"transmission": 0.5, "binarization": 0.5}
        loss = explorer._scalarized_loss(density, weights)
        assert loss.numel() == 1

    def test_explore(self, explorer):
        pareto = explorer.explore(n_steps=3, lr=0.01, verbose=False)
        assert len(pareto) >= 1
        density, obj_values = pareto[0]
        assert density.shape == (10, 10)
        assert "transmission" in obj_values
        assert "binarization" in obj_values


# -----------------------------------------------------------------------
# End-to-End Pipeline (C8)
# -----------------------------------------------------------------------


class TestEndToEndPipeline:
    @pytest.fixture
    def pipeline(self):
        solver = RCWASolver(
            fourier_orders=3,
            wavelength_nm=532.0,
            period_nm=(400.0, 400.0),
        )
        return EndToEndPipeline(
            solver=solver,
            grid_shape=(15, 15),
            wavelengths_nm=[532.0],
        )

    def test_init(self, pipeline):
        assert pipeline.grid_shape == (15, 15)

    def test_forward_pass(self, pipeline):
        density = torch.rand(15, 15, dtype=torch.float64)
        results = pipeline.forward_pass(density)
        assert "total_loss" in results
        assert "optical_loss" in results
        assert "fab_loss" in results
        assert "constraint_loss" in results

    def test_forward_pass_gradient(self, pipeline):
        density = torch.rand(15, 15, dtype=torch.float64, requires_grad=True)
        results = pipeline.forward_pass(density)
        results["total_loss"].backward()
        assert density.grad is not None

    def test_optimize_short(self, pipeline):
        density, history = pipeline.optimize(
            n_steps=3,
            lr=0.01,
            verbose=False,
        )
        assert density.shape == (15, 15)
        assert len(history["total"]) == 3
