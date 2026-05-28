"""Tests for the robustness (C5) module."""

import torch

from diffnano.design.robustness import (
    antithetic_sampler,
    apply_perturbation_to_density,
    linewidth_perturbation,
    relaxed_heaviside_perturbation,
    reparameterize_sample,
    robust_gradient_step,
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
            return (p ** 2).sum()

        loss = robust_gradient_step(
            params, forward_fn,
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
            params, forward_fn,
            sigma_nm=2.0,
            n_samples=4,
            perturbation_fn=perturb_fn,
        )
        loss.backward()
        assert params.grad is not None
