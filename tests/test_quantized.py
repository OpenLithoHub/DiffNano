"""Tests for quantization-aware inverse design with STE."""

import pytest
import torch

from diffnano.design.projection import heaviside_projection, smooth_filter
from diffnano.design.quantized import (
    BinarySTE,
    QuantizationNoiseGuardrail,
    QuantizedOptimizer,
    StraightThroughQuantize,
)


class TestStraightThroughQuantize:
    def test_binary_forward(self):
        q = StraightThroughQuantize(n_levels=2)
        x = torch.tensor([0.1, 0.4, 0.6, 0.9], dtype=torch.float64, requires_grad=True)
        out = q(x)
        assert set(out.tolist()) <= {0.0, 1.0}
        assert out.tolist() == [0.0, 0.0, 1.0, 1.0]

    def test_ste_gradient_flows(self):
        q = StraightThroughQuantize(n_levels=2)
        x = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
        out = q(x)
        out.sum().backward()
        assert x.grad is not None
        assert (x.grad != 0).any()

    def test_k_level_quantization(self):
        q = StraightThroughQuantize(n_levels=4)
        levels = q.levels
        assert levels.shape == (4,)
        x = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0], dtype=torch.float64)
        out = q(x)
        for val in out:
            dists = (val - levels).abs()
            assert dists.min().item() < 1e-6

    def test_deterministic_with_seed(self):
        q = StraightThroughQuantize(n_levels=2)
        torch.manual_seed(42)
        x1 = torch.rand(10, dtype=torch.float64, requires_grad=True)
        out1 = q(x1)

        torch.manual_seed(42)
        x2 = torch.rand(10, dtype=torch.float64, requires_grad=True)
        out2 = q(x2)
        assert torch.equal(out1, out2)

    def test_invalid_levels(self):
        with pytest.raises(ValueError):
            StraightThroughQuantize(n_levels=1)


class TestBinarySTE:
    def test_output_is_binary(self):
        b = BinarySTE()
        x = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        out = b(x)
        assert set(out.tolist()) <= {0.0, 1.0}

    def test_sigmoid_gradient(self):
        b = BinarySTE()
        x = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
        out = b(x)
        out.sum().backward()
        assert x.grad is not None
        assert (x.grad > 0).all()
        expected = torch.sigmoid(x.detach()) * (1 - torch.sigmoid(x.detach()))
        assert torch.allclose(x.grad, expected, atol=1e-6)


class TestQuantizationNoiseGuardrail:
    def test_consistent_gradient_direction(self):
        guard = QuantizationNoiseGuardrail()
        x = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
        def loss_fn(t):
            return -(t**2).sum()
        loss = loss_fn(x)
        loss.backward()
        cosine, at_boundary = guard.check(x, x.grad, loss_fn)
        assert cosine.item() > 0.5

    def test_boundary_relaxed_tolerance(self):
        guard = QuantizationNoiseGuardrail(boundary_tolerance=0.3)
        x = torch.tensor([0.49, 0.51], dtype=torch.float64, requires_grad=True)
        def loss_fn(t):
            return -(t**2).sum()
        loss = loss_fn(x)
        loss.backward()
        _, at_boundary = guard.check(x, x.grad, loss_fn)
        assert at_boundary.item() is True


class TestQuantizedOptimizer:
    def test_runs_end_to_end(self):
        def fom_fn(x):
            return -(x - 0.7).pow(2).sum()

        opt = QuantizedOptimizer(grid_shape=(4, 4), n_steps=5, lr=0.1)
        result, fom = opt._run_single(fom_fn, seed=0, quantize_aware=True)
        assert result.shape == (4, 4)
        assert isinstance(fom, float)

    def test_comparison_structure(self):
        def fom_fn(x):
            return x.mean()

        opt = QuantizedOptimizer(grid_shape=(4, 4), n_steps=5, lr=0.1)
        results = opt.compare_approaches(fom_fn, n_seeds=3)
        assert "quantization_aware_foms" in results
        assert "post_hoc_foms" in results
        assert "p_value" in results
        assert "qa_win_rate" in results
        assert len(results["quantization_aware_foms"]) == 3
        assert len(results["post_hoc_foms"]) == 3

    def test_aware_differs_from_posthoc(self):
        def fom_fn(x):
            return -(x - 0.3).pow(2).sum() - (x - 0.8).pow(2).sum()

        opt = QuantizedOptimizer(grid_shape=(4, 4), n_steps=10, lr=0.05)
        results = opt.compare_approaches(fom_fn, n_seeds=4)
        assert results["quantization_aware_foms"] != results["post_hoc_foms"]


class TestIntegration:
    def test_quantized_with_heaviside_projection(self):
        q = StraightThroughQuantize(n_levels=2)
        BinarySTE()
        x = torch.rand(8, 8, dtype=torch.float64, requires_grad=True)
        xq = q(x)
        projected = heaviside_projection(xq, beta=10.0)
        assert projected.shape == (8, 8)
        assert projected.min() >= 0.0
        assert projected.max() <= 1.0
        projected.sum().backward()
        assert x.grad is not None

    def test_smooth_then_quantize_chain(self):
        q = StraightThroughQuantize(n_levels=2)
        x = torch.rand(10, 10, dtype=torch.float64, requires_grad=True)
        smoothed = smooth_filter(x, radius=1.5)
        quantized = q(smoothed)
        quantized.sum().backward()
        assert x.grad is not None
