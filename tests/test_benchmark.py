"""Tests for benchmark metrics and datasets."""

import pytest
import torch

from diffnano.benchmark.datasets import metalens_devlin2016, silicon_grating_1d
from diffnano.benchmark.metrics import (
    bandgap_ratio,
    strehl_ratio_from_phase,
    transmission_efficiency,
)


class TestMetrics:
    def test_transmission_efficiency(self):
        t = torch.tensor([[0.3, 0.5, 0.2]], dtype=torch.float64)
        eff = transmission_efficiency(t)
        assert eff.item() == pytest.approx(1.0, abs=0.01)

    def test_strehl_zero_error(self):
        phase_err = torch.zeros(10, dtype=torch.float64)
        s = strehl_ratio_from_phase(phase_err)
        assert s.item() == pytest.approx(1.0, abs=0.01)

    def test_strehl_ratio_from_field(self):
        """strehl_ratio_from_field returns 1.0 for identical real fields."""
        from diffnano.benchmark.metrics import strehl_ratio_from_field

        H, W = 32, 32
        field = torch.ones(1, H, W, dtype=torch.float64)
        target = torch.ones(1, H, W, dtype=torch.float64)
        sr = strehl_ratio_from_field(field, target)
        assert sr.item() > 0.99

    def test_strehl_large_error(self):
        phase_err = torch.ones(10, dtype=torch.float64) * 3.0
        s = strehl_ratio_from_phase(phase_err)
        assert s.item() < 0.01

    def test_bandgap_ratio(self):
        freq = torch.linspace(1.0, 10.0, 100, dtype=torch.float64)
        # Create a gap between freq 4 and 6
        trans = torch.ones_like(freq)
        trans[30:60] = 0.01
        ratio = bandgap_ratio(freq, trans, threshold=0.1)
        assert ratio.item() > 0

    def test_no_bandgap(self):
        freq = torch.linspace(1.0, 10.0, 50, dtype=torch.float64)
        trans = torch.ones_like(freq)
        ratio = bandgap_ratio(freq, trans, threshold=0.1)
        assert ratio.item() == 0.0


class TestDatasets:
    def test_silicon_grating(self):
        eps = silicon_grating_1d(n_grid=100)
        assert eps.shape == (100,)
        assert eps.min().item() == 1.0  # air
        assert eps.max().item() == 12.0  # Si

    def test_metalens_devlin(self):
        data = metalens_devlin2016(n_pixels=50)
        assert "target_phase" in data
        assert data["target_phase"].shape == (50, 50)
        assert data["pixel_size_nm"] > 0
        assert data["focal_length_um"] > 0
