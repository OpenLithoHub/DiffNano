"""Tests for the metalens, PhC, and waveguide workflows."""

import pytest
import torch

from diffnano.workflows.metalens import MetalensDesigner
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
            n_steps=3, verbose=False,
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
        assert eps.max().item() == pytest.approx(2.5 ** 2)  # core

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
