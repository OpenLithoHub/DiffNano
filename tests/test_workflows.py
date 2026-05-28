"""Tests for the metalens workflow."""

import pytest
import torch

from diffnano.workflows.metalens import MetalensDesigner


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
