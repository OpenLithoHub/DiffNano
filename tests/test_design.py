"""Tests for design modules: parameterization, projection, constraints."""

import pytest
import torch

from diffnano.design.constraints_shared import (
    binarization_penalty,
    combined_fabrication_penalty,
    corner_rounding_penalty,
    curvature_penalty,
    minimum_cd_penalty,
)
from diffnano.design.parameterization import (
    BSplineCurve,
    DensityField,
    HeightMap,
)
from diffnano.design.projection import (
    beta_continuation_schedule,
    heaviside_projection,
    smooth_filter,
)
from diffnano.design.curvilinear import CurvilinearMask, dvas_boundary


class TestHeightMap:
    def test_output_shape(self):
        hm = HeightMap((10, 10), wavelength_nm=532.0)
        h = torch.rand(10, 10, dtype=torch.float64)
        phase = hm(h)
        assert phase.shape == (10, 10)

    def test_gradient(self):
        hm = HeightMap((10, 10))
        h = torch.rand(10, 10, dtype=torch.float64, requires_grad=True)
        phase = hm(h)
        phase.sum().backward()
        assert h.grad is not None

    def test_callable(self):
        hm = HeightMap((5, 5))
        h = torch.rand(5, 5, dtype=torch.float64)
        assert hm(h).shape == (5, 5)


class TestDensityField:
    def test_output_range(self):
        df = DensityField((10, 10))
        density = torch.rand(10, 10, dtype=torch.float64)
        eps = df(density)
        assert eps.min() >= 1.0
        assert eps.max() <= 12.0

    def test_binary_limits(self):
        df = DensityField((5, 5), beta=100.0)
        eps0 = df(torch.zeros(5, 5, dtype=torch.float64))
        eps1 = df(torch.ones(5, 5, dtype=torch.float64))
        assert eps0.min().item() == pytest.approx(1.0, abs=0.01)
        assert eps1.max().item() == pytest.approx(12.0, abs=0.01)

    def test_gradient(self):
        df = DensityField((5, 5))
        density = torch.rand(5, 5, dtype=torch.float64, requires_grad=True)
        eps = df(density)
        eps.sum().backward()
        assert density.grad is not None

    def test_beta_override(self):
        df = DensityField((5, 5), beta=1.0)
        density = torch.ones(5, 5, dtype=torch.float64) * 0.5
        soft = df(density, beta=1.0)
        hard = df(density, beta=100.0)
        # Harder projection should be closer to 0 or 1
        assert (hard - 0.5).abs().max() >= (soft - 0.5).abs().max()


class TestBSplineCurve:
    def test_rasterize_shape(self):
        bs = BSplineCurve((20, 20), pixel_size_nm=5.0, n_eval=20, beta=10.0)
        cp = torch.tensor([[5.0, 10.0], [10.0, 5.0], [15.0, 10.0], [10.0, 15.0]],
                          dtype=torch.float64)
        mask = bs(cp)
        assert mask.shape == (20, 20)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_gradient(self):
        bs = BSplineCurve((15, 15), pixel_size_nm=5.0, n_eval=20, beta=5.0)
        cp = torch.tensor([[5.0, 7.5], [7.5, 5.0], [10.0, 7.5], [7.5, 10.0]],
                          dtype=torch.float64, requires_grad=True)
        mask = bs(cp)
        mask.sum().backward()
        assert cp.grad is not None

    def test_sdf_output(self):
        bs = BSplineCurve((15, 15), pixel_size_nm=5.0, n_eval=20)
        cp = torch.tensor([[5.0, 7.5], [7.5, 5.0], [10.0, 7.5], [7.5, 10.0]],
                          dtype=torch.float64)
        sdf = bs.sdf(cp)
        assert sdf.shape == (15, 15)

    def test_shift_parameter(self):
        bs = BSplineCurve((30, 30), pixel_size_nm=1.0, n_eval=30)
        cp = torch.tensor([[10.0, 15.0], [15.0, 10.0], [20.0, 15.0], [15.0, 20.0]],
                          dtype=torch.float64)
        mask_default = bs(cp, beta=5.0)
        mask_shifted = bs(cp, shift=torch.tensor(2.0, dtype=torch.float64), beta=5.0)
        # Shifted mask should differ (feature size changed)
        assert not torch.allclose(mask_default, mask_shifted, atol=1e-3)


class TestProjection:
    def test_heaviside_binary(self):
        d = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        p = heaviside_projection(d, beta=100.0)
        assert p[0].item() == pytest.approx(0.0, abs=0.01)
        assert p[1].item() == pytest.approx(0.5, abs=0.01)
        assert p[2].item() == pytest.approx(1.0, abs=0.01)

    def test_heaviside_gradient(self):
        d = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64, requires_grad=True)
        p = heaviside_projection(d, beta=10.0)
        p.sum().backward()
        assert d.grad is not None

    def test_smooth_filter(self):
        d = torch.rand(20, 20, dtype=torch.float64)
        s = smooth_filter(d, radius=2.0)
        assert s.shape == (20, 20)

    def test_beta_schedule(self):
        assert beta_continuation_schedule(0, 500) == pytest.approx(1.0, abs=0.01)
        assert beta_continuation_schedule(500, 500) == pytest.approx(64.0, abs=1.0)
        mid = beta_continuation_schedule(250, 500)
        assert 1.0 < mid < 64.0


class TestConstraints:
    @pytest.fixture
    def density(self):
        return torch.rand(20, 20, dtype=torch.float64)

    def test_min_cd_penalty(self, density):
        p = minimum_cd_penalty(density, min_cd_pixels=4.0)
        assert p.numel() == 1
        assert p.item() >= 0

    def test_curvature_penalty(self, density):
        p = curvature_penalty(density)
        assert p.numel() == 1
        assert p.item() >= 0

    def test_binarization_penalty(self):
        d = torch.tensor([0.0, 1.0], dtype=torch.float64)
        p = binarization_penalty(d)
        assert p.item() == pytest.approx(0.0, abs=1e-6)

        d2 = torch.tensor([0.5], dtype=torch.float64)
        p2 = binarization_penalty(d2)
        assert p2.item() > 0

    def test_corner_rounding(self, density):
        p = corner_rounding_penalty(density)
        assert p.numel() == 1
        assert p.item() >= 0

    def test_combined(self, density):
        p = combined_fabrication_penalty(density)
        assert p.numel() == 1
        assert p.item() >= 0

    def test_combined_gradient(self):
        d = torch.rand(15, 15, dtype=torch.float64, requires_grad=True)
        p = combined_fabrication_penalty(d)
        p.backward()
        assert d.grad is not None

    def test_combined_weights(self):
        d = torch.rand(10, 10, dtype=torch.float64)
        p1 = combined_fabrication_penalty(
            d,
            weights={
                "cd": 0.0, "curvature": 0.0,
                "binarization": 1.0, "corner": 0.0,
            },
        )
        p2 = binarization_penalty(d)
        assert p1.item() == pytest.approx(p2.item(), rel=1e-3)


# -----------------------------------------------------------------------
# Curvilinear Mask (C8)
# -----------------------------------------------------------------------


class TestCurvilinearMask:
    @pytest.fixture
    def cmask(self):
        return CurvilinearMask(
            grid_shape=(20, 20),
            pixel_size_nm=5.0,
            n_eval=30,
            beta=10.0,
        )

    def test_rasterize_shape(self, cmask):
        cp = torch.tensor([
            [25.0, 50.0], [50.0, 25.0], [75.0, 50.0], [50.0, 75.0],
        ], dtype=torch.float64)
        mask = cmask.forward(cp)
        assert mask.shape == (20, 20)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_gradient_flow(self, cmask):
        cp = torch.tensor([
            [25.0, 50.0], [50.0, 25.0], [75.0, 50.0], [50.0, 75.0],
        ], dtype=torch.float64, requires_grad=True)
        mask = cmask.forward(cp)
        mask.sum().backward()
        assert cp.grad is not None
        assert not torch.isnan(cp.grad).any(), "NaN gradient detected in CurvilinearMask"

    def test_sdf_output(self, cmask):
        cp = torch.tensor([
            [25.0, 50.0], [50.0, 25.0], [75.0, 50.0], [50.0, 75.0],
        ], dtype=torch.float64)
        sdf = cmask.sdf(cp)
        assert sdf.shape == (20, 20)

    def test_shift_parameter(self, cmask):
        cp = torch.tensor([
            [25.0, 50.0], [50.0, 25.0], [75.0, 50.0], [50.0, 75.0],
        ], dtype=torch.float64)
        mask1 = cmask.forward(cp, beta=5.0)
        mask2 = cmask.forward(cp, shift=torch.tensor(2.0, dtype=torch.float64), beta=5.0)
        assert not torch.allclose(mask1, mask2, atol=1e-3)

    def test_optimize(self, cmask):
        def loss_fn(mask):
            return -mask.mean() + ((mask - 0.5) ** 2).mean()

        cp, history = cmask.optimize(
            n_control_points=4, loss_fn=loss_fn, n_steps=5, lr=0.01, verbose=False,
        )
        assert cp.shape == (4, 2)
        assert len(history) == 5


class TestDVASBoundary:
    def test_basic(self):
        dist = torch.tensor([3.0, 4.0, 3.0, 4.0], dtype=torch.float64)
        pts = dvas_boundary(4, dist, center_angle=None)
        assert pts.shape == (4, 2)

    def test_gradient(self):
        dist = torch.tensor([3.0, 4.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
        pts = dvas_boundary(4, dist, center_angle=None)
        pts.sum().backward()
        assert dist.grad is not None
