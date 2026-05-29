"""Tests for the robustness (C5 + C7) module."""

import pytest
import torch

from diffnano.design.robustness import (
    AdaptiveRobustOptimizer,
    FabricableSubspaceProjection,
    MultiAxisPerturbation,
    antithetic_sampler,
    apply_perturbation_to_density,
    axial_samples,
    corner_rounding_perturbation,
    correlated_perturbation,
    linewidth_perturbation,
    relaxed_heaviside_perturbation,
    reparameterize_sample,
    robust_gradient_step,
    sidewall_angle_perturbation,
    thickness_perturbation,
)


class TestReparameterizeSample:
    def test_shape(self):
        mu = torch.zeros(3, dtype=torch.float64)
        sigma = torch.ones(3, dtype=torch.float64)
        deltas = reparameterize_sample(mu, sigma, n_samples=8)
        assert deltas.shape == (8, 3)

    def test_gradient_flow(self):
        mu = torch.zeros(5, dtype=torch.float64, requires_grad=True)
        sigma = torch.ones(5, dtype=torch.float64, requires_grad=True)
        deltas = reparameterize_sample(mu, sigma, n_samples=4)
        deltas.sum().backward()
        assert mu.grad is not None
        assert sigma.grad is not None


class TestLinewidthPerturbation:
    def test_sdf_shift(self):
        sdf = torch.randn(10, 10, dtype=torch.float64)
        delta = torch.tensor(5.0, dtype=torch.float64)
        perturbed = linewidth_perturbation(sdf, delta, pixel_size_nm=5.0)
        assert perturbed.shape == sdf.shape
        # Shifting by 5nm with 5nm pixels = 1 pixel shift
        expected = sdf - 1.0
        assert torch.allclose(perturbed, expected)

    def test_gradient(self):
        sdf = torch.randn(10, 10, dtype=torch.float64, requires_grad=True)
        delta = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        out = linewidth_perturbation(sdf, delta)
        out.sum().backward()
        assert sdf.grad is not None
        assert delta.grad is not None


class TestDensityPerturbation:
    def test_output_shape(self):
        density = torch.rand(10, 10, dtype=torch.float64)
        delta = torch.tensor(2.0, dtype=torch.float64)
        out = apply_perturbation_to_density(density, delta)
        assert out.shape == density.shape

    def test_gradient(self):
        density = torch.rand(8, 8, dtype=torch.float64, requires_grad=True)
        delta = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        out = apply_perturbation_to_density(density, delta)
        out.sum().backward()
        assert density.grad is not None
        assert delta.grad is not None


class TestRelaxedHeaviside:
    def test_output_range(self):
        sdf = torch.randn(10, 10, dtype=torch.float64)
        delta = torch.tensor(0.0, dtype=torch.float64)
        mask = relaxed_heaviside_perturbation(sdf, delta, beta=10.0)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


class TestAntitheticSampler:
    def test_paired(self):
        samples = antithetic_sampler(1.0, (3,), dtype=torch.float64)
        assert samples.shape == (8, 3)  # 4 pairs = 8 samples
        # Check antithetic pairing
        for i in range(4):
            assert torch.allclose(samples[i], -samples[i + 4])


class TestRobustGradientStep:
    def test_basic(self):
        params = torch.randn(10, dtype=torch.float64, requires_grad=True)

        def forward_fn(p):
            return (p**2).sum()

        loss = robust_gradient_step(
            params,
            forward_fn,
            sigma_nm=1.0,
            n_samples=4,
            antithetic=True,
        )
        assert loss.numel() == 1
        loss.backward()
        assert params.grad is not None

    def test_custom_perturbation(self):
        params = torch.randn(5, dtype=torch.float64, requires_grad=True)

        def forward_fn(p):
            return p.sum()

        def perturb_fn(p, delta):
            return p + delta * 0.1

        loss = robust_gradient_step(
            params,
            forward_fn,
            sigma_nm=2.0,
            n_samples=4,
            perturbation_fn=perturb_fn,
        )
        loss.backward()
        assert params.grad is not None


# -----------------------------------------------------------------------
# C7: Adaptive Robust Optimization
# -----------------------------------------------------------------------


class TestAxialSamples:
    def test_count(self):
        samples = axial_samples(3, 5.0)
        assert samples.shape == (7, 3)  # 2*3 + 1

    def test_nominal_is_zero(self):
        samples = axial_samples(2, 3.0)
        assert torch.allclose(samples[0], torch.zeros(2, dtype=torch.float64))

    def test_axial_structure(self):
        n = 4
        samples = axial_samples(n, 2.0)
        for i in range(n):
            pos = samples[1 + 2 * i]
            neg = samples[2 + 2 * i]
            assert torch.allclose(pos, -neg)
            assert pos.abs().sum().item() == pytest.approx(2.0)


class TestFabricableSubspaceProjection:
    def test_output_shape(self):
        proj = FabricableSubspaceProjection(n_levels=4)
        density = torch.rand(10, 10, dtype=torch.float64)
        result = proj.project(density)
        assert result.shape == density.shape

    def test_gradient_flows(self):
        proj = FabricableSubspaceProjection(n_levels=4, temperature=0.5)
        density = torch.rand(8, 8, dtype=torch.float64, requires_grad=True)
        result = proj.project(density)
        result.sum().backward()
        assert density.grad is not None

    def test_projection_loss(self):
        proj = FabricableSubspaceProjection(n_levels=4)
        density = torch.rand(10, 10, dtype=torch.float64, requires_grad=True)
        loss = proj.projection_loss(density)
        assert loss.numel() == 1
        loss.backward()
        assert density.grad is not None


class TestAdaptiveRobustOptimizer:
    def test_basic_optimization(self):
        params = torch.randn(10, dtype=torch.float64)

        def forward_fn(p, delta):
            return (p**2).sum()

        def perturb_fn(p, delta):
            return p + delta.sum() * 0.01

        opt = AdaptiveRobustOptimizer(n_variation_dims=2, sigma=1.0)
        result, history = opt.optimize(
            params,
            forward_fn,
            perturb_fn,
            n_steps=10,
            lr=0.01,
            verbose=False,
        )
        assert result.shape == params.shape
        assert len(history) == 10

    def test_curriculum(self):
        opt = AdaptiveRobustOptimizer(n_variation_dims=2, sigma=1.0)
        params = torch.randn(5, dtype=torch.float64, requires_grad=True)

        def forward_fn(p, delta):
            return (p**2).sum()

        def perturb_fn(p, delta):
            return p

        loss = opt.compute_robust_loss(
            params,
            forward_fn,
            perturb_fn,
            curriculum_frac=0.5,
        )
        assert loss.numel() == 1
        loss.backward()
        assert params.grad is not None


# -----------------------------------------------------------------------
# C5 full: Multi-axis perturbations
# -----------------------------------------------------------------------


class TestSidewallAnglePerturbation:
    def test_output_shape(self):
        density = torch.rand(10, 10, dtype=torch.float64)
        angle = torch.tensor(2.0, dtype=torch.float64)
        result = sidewall_angle_perturbation(density, angle)
        assert result.shape == density.shape

    def test_zero_angle(self):
        density = torch.rand(10, 10, dtype=torch.float64)
        angle = torch.tensor(0.0, dtype=torch.float64)
        result = sidewall_angle_perturbation(density, angle)
        assert torch.allclose(result, density)


class TestThicknessPerturbation:
    def test_output_shape(self):
        density = torch.rand(10, 10, dtype=torch.float64)
        delta = torch.tensor(5.0, dtype=torch.float64)
        result = thickness_perturbation(density, delta)
        assert result.shape == density.shape

    def test_zero_delta(self):
        density = torch.rand(10, 10, dtype=torch.float64)
        delta = torch.tensor(0.0, dtype=torch.float64)
        result = thickness_perturbation(density, delta)
        assert torch.allclose(result, density)

    def test_clamped(self):
        density = torch.ones(10, 10, dtype=torch.float64)
        delta = torch.tensor(10000.0, dtype=torch.float64)
        result = thickness_perturbation(density, delta)
        assert result.max() <= 1.0


class TestCornerRoundingPerturbation:
    def test_output_shape(self):
        density = torch.rand(10, 10, dtype=torch.float64)
        radius = torch.tensor(10.0, dtype=torch.float64)
        result = corner_rounding_perturbation(density, radius)
        assert result.shape == density.shape

    def test_zero_radius(self):
        density = torch.rand(10, 10, dtype=torch.float64)
        radius = torch.tensor(0.0, dtype=torch.float64)
        result = corner_rounding_perturbation(density, radius)
        assert torch.allclose(result, density)


class TestMultiAxisPerturbation:
    def test_sample_shape(self):
        pert = MultiAxisPerturbation()
        samples = pert.sample(8)
        assert samples.shape == (8, 4)

    def test_apply(self):
        pert = MultiAxisPerturbation()
        density = torch.rand(20, 20, dtype=torch.float64)
        delta = torch.tensor([2.0, 0.5, 1.0, 3.0], dtype=torch.float64)
        result = pert.apply(density, delta)
        assert result.shape == density.shape

    def test_correlated_sampling(self):
        corr = torch.tensor(
            [
                [1.0, 0.5, 0.3, 0.0],
                [0.5, 1.0, 0.2, 0.0],
                [0.3, 0.2, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        pert = MultiAxisPerturbation(correlation_matrix=corr)
        samples = pert.sample(100)
        assert samples.shape == (100, 4)


class TestCorrelatedPerturbation:
    def test_shape(self):
        params = torch.randn(10, dtype=torch.float64)
        chol = torch.eye(3, dtype=torch.float64)
        deltas = correlated_perturbation(params, chol, n_samples=16)
        assert deltas.shape == (16, 3)
