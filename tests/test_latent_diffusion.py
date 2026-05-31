"""Tests for physics-guided latent diffusion inverse design."""

import torch
import pytest

from diffnano.design.latent_diffusion import (
    LatentEncoder,
    LatentDecoder,
    PhysicsGuidance,
    ConditionedDiffusion,
    LatentDiffusionDesigner,
    LatentDiffusionBenchmark,
)
from diffnano.design.quantized import StraightThroughQuantize


def _make_encoder_decoder(grid_size=16, latent_dim=8):
    enc = LatentEncoder(grid_size=grid_size, latent_dim=latent_dim, hidden_channels=16)
    dec = LatentDecoder(latent_dim=latent_dim, grid_size=grid_size, hidden_channels=16)
    return enc, dec


def _make_diffusion(latent_dim=8, cond_dim=16):
    return ConditionedDiffusion(latent_dim=latent_dim, cond_dim=cond_dim, n_blocks=2, n_steps=100)


def _make_designer(grid_size=16, latent_dim=8, cond_dim=16, use_quantizer=False):
    enc, dec = _make_encoder_decoder(grid_size, latent_dim)
    diff = _make_diffusion(latent_dim, cond_dim)
    quantizer = StraightThroughQuantize(n_levels=2) if use_quantizer else None
    return LatentDiffusionDesigner(enc, dec, diff, quantizer=quantizer)


class TestLatentEncoder:
    def test_latent_encoder_shape(self):
        enc = LatentEncoder(grid_size=16, latent_dim=8, hidden_channels=16)
        x = torch.randn(4, 16, 16)
        mu, log_var = enc(x)
        assert mu.shape == (4, 8)
        assert log_var.shape == (4, 8)

    def test_encoder_accepts_4d_input(self):
        enc = LatentEncoder(grid_size=16, latent_dim=8)
        x = torch.randn(2, 1, 16, 16)
        mu, log_var = enc(x)
        assert mu.shape == (2, 8)


class TestLatentDecoder:
    def test_latent_decoder_shape(self):
        dec = LatentDecoder(latent_dim=8, grid_size=16, hidden_channels=16)
        z = torch.randn(4, 8)
        out = dec(z)
        assert out.shape == (4, 16, 16)

    def test_decoder_output_range(self):
        dec = LatentDecoder(latent_dim=8, grid_size=16, hidden_channels=16)
        z = torch.randn(4, 8)
        out = dec(z)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestAutoencoder:
    def test_autoencoder_roundtrip(self):
        enc, dec = _make_encoder_decoder(grid_size=16, latent_dim=8)
        x = torch.rand(3, 16, 16)
        mu, _ = enc(x)
        recon = dec(mu)
        assert recon.shape == x.shape


class TestPhysicsGuidance:
    def test_physics_guidance_gradient(self):
        dec = LatentDecoder(latent_dim=8, grid_size=16, hidden_channels=16)
        target = torch.randn(2, 4)
        dummy_model = lambda d: d.reshape(d.shape[0], -1)[:, :4]
        guidance = PhysicsGuidance(forward_model=dummy_model, decoder=dec, target_response=target)
        z = torch.randn(2, 8)
        grad = guidance.guide_score(z, target, guidance_scale=1.0)
        assert grad.shape == (2, 8)
        assert not torch.all(grad == 0)


class TestConditionedDiffusion:
    def test_conditioned_diffusion_forward(self):
        diff = _make_diffusion(latent_dim=8, cond_dim=16)
        batch = 4
        noisy_z = torch.randn(batch, 8)
        t = torch.randint(0, 100, (batch,))
        cond = torch.randn(batch, 16)
        out = diff(noisy_z, t, cond)
        assert out.shape == (batch, 8)

    def test_conditioned_diffusion_sampling(self):
        diff = _make_diffusion(latent_dim=8, cond_dim=16)
        cond = torch.randn(2, 16)
        samples = diff.sample(cond, n_steps=5, n_samples=1)
        assert samples.shape == (2, 8)


class TestLatentDiffusionDesigner:
    def test_designer_produces_candidates(self):
        designer = _make_designer(grid_size=16, latent_dim=8, cond_dim=16)
        target = torch.randn(16)
        result = designer.design(target, n_candidates=4, n_diffusion_steps=5)
        assert "candidates" in result
        assert "latent_samples" in result
        assert result["candidates"].shape[0] == 4
        assert result["candidates"].shape[-1] == 16

    def test_designer_quantize_refine(self):
        designer = _make_designer(grid_size=16, latent_dim=8, cond_dim=16, use_quantizer=True)
        candidates = torch.rand(4, 16, 16, dtype=torch.float64)
        quantized = designer._quantize_refine(candidates)
        assert quantized.shape == candidates.shape
        unique_vals = torch.unique(quantized)
        assert len(unique_vals) <= 2

    def test_designer_with_robust_scoring(self):
        designer = _make_designer(grid_size=16, latent_dim=8, cond_dim=16)
        candidates = torch.rand(4, 16, 16)
        condition = torch.randn(8)
        scorer = lambda design, cond: design.mean()
        result = designer._score_with_robust(candidates, condition, scorer)
        assert "scores" in result
        assert "best" in result
        assert result["scores"].shape == (4,)

    def test_designer_with_decision_gate(self):
        designer = _make_designer(grid_size=16, latent_dim=8, cond_dim=16)
        target = torch.randn(16)
        result = designer.design(target, n_candidates=4, n_diffusion_steps=5)
        candidates = result["candidates"]

        threshold = candidates.reshape(4, -1).mean(dim=-1).median()
        mask = candidates.reshape(4, -1).mean(dim=-1) >= threshold
        accepted = candidates[mask]

        assert accepted.shape[0] >= 1
        assert accepted.shape[-1] == 16


class TestLatentDiffusionBenchmark:
    def test_benchmark_runs(self):
        designer = _make_designer(grid_size=16, latent_dim=8, cond_dim=16)

        def warm_start_fn(target):
            candidates = torch.rand(4, 16, 16)
            scores = torch.tensor([c.mean() for c in candidates])
            best_idx = scores.argmax()
            return {"candidates": candidates, "best": candidates[best_idx], "best_score": scores[best_idx]}

        scorer = lambda design, cond: design.mean()
        bench = LatentDiffusionBenchmark(designer, warm_start_fn, scorer, grid_size=16)
        result = bench.run(torch.randn(16), n_seeds=2, n_candidates=4)

        assert "diffusion" in result
        assert "warm_start" in result
        assert "best_fom" in result["diffusion"]
        assert "mean_fom" in result["diffusion"]
        assert "diversity" in result["diffusion"]
        assert "best_fom" in result["warm_start"]


class TestTraining:
    def test_loss_decreases(self):
        enc, dec = _make_encoder_decoder(grid_size=16, latent_dim=8)
        diff = _make_diffusion(latent_dim=8, cond_dim=8)
        designer = LatentDiffusionDesigner(enc, dec, diff)

        opt = torch.optim.Adam(designer.parameters(), lr=1e-3)
        designs = torch.rand(8, 16, 16)
        responses = torch.rand(8, 8)

        losses = []
        for _ in range(20):
            opt.zero_grad()
            result = designer.train_step(designs, responses)
            result["total_loss"].backward()
            opt.step()
            losses.append(result["total_loss"].item())

        assert losses[-1] < losses[0]
