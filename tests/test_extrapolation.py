"""Tests for extrapolative inverse design with current-diffusion conditioning (N11.2)."""

import torch

from diffnano.design.extrapolation import (
    CurrentDiffusionConditioner,
    ExtrapolationBenchmark,
    ExtrapolationDesigner,
)
from diffnano.design.latent_diffusion import (
    ConditionedDiffusion,
    LatentDecoder,
    LatentDiffusionDesigner,
    LatentEncoder,
)


def _make_encoder_decoder(grid_size=16, latent_dim=8):
    enc = LatentEncoder(grid_size=grid_size, latent_dim=latent_dim, hidden_channels=16)
    dec = LatentDecoder(latent_dim=latent_dim, grid_size=grid_size, hidden_channels=16)
    return enc, dec


def _make_diffusion(latent_dim=8, cond_dim=16):
    return ConditionedDiffusion(latent_dim=latent_dim, cond_dim=cond_dim, n_blocks=2, n_steps=100)


def _make_base_designer(grid_size=16, latent_dim=8, cond_dim=16):
    enc, dec = _make_encoder_decoder(grid_size, latent_dim)
    diff = _make_diffusion(latent_dim, cond_dim)
    return LatentDiffusionDesigner(enc, dec, diff)


def _make_conditioner(latent_dim=8, n_freq_features=4):
    return CurrentDiffusionConditioner(
        latent_dim=latent_dim,
        n_freq_features=n_freq_features,
        extrapolation_strength=1.0,
    )


def _make_extrapolation_designer(grid_size=16, latent_dim=8, cond_dim=16):
    enc, dec = _make_encoder_decoder(grid_size, latent_dim)
    diff = _make_diffusion(latent_dim, cond_dim)
    base = LatentDiffusionDesigner(enc, dec, diff)
    conditioner = _make_conditioner(latent_dim, n_freq_features=4)
    return ExtrapolationDesigner(base, conditioner, dec, extrapolation_scale=1.5)


class TestCurrentDiffusionConditioner:
    def test_frequency_features_shape(self):
        """FFT-based features have correct shape for batch and single inputs."""
        conditioner = _make_conditioner(latent_dim=8, n_freq_features=4)
        response = torch.randn(3, 32)
        features = conditioner.compute_frequency_features(response)
        assert features.shape == (3, 4)

    def test_frequency_features_single_input(self):
        """Single 1-D response vector produces shape (1, n_freq_features)."""
        conditioner = _make_conditioner(latent_dim=8, n_freq_features=4)
        response = torch.randn(32)
        features = conditioner.compute_frequency_features(response)
        assert features.shape == (1, 4)

    def test_condition_with_dynamics_modifies_latents(self):
        """condition_with_dynamics produces latents different from the input."""
        conditioner = _make_conditioner(latent_dim=8, n_freq_features=4)
        z = torch.randn(4, 8)
        freq = torch.randn(4, 4)
        z_out = conditioner.condition_with_dynamics(z, freq)
        assert z_out.shape == z.shape
        assert not torch.allclose(z_out, z, atol=1e-6)


class TestExtrapolationDesigner:
    def test_produces_valid_candidates(self):
        """ExtrapolationDesigner.design_extrapolative returns valid candidates."""
        designer = _make_extrapolation_designer(grid_size=16, latent_dim=8, cond_dim=16)
        target = torch.randn(16)
        result = designer.design_extrapolative(
            target,
            held_out_fom_range=(0.8, 1.0),
            n_candidates=4,
            n_diffusion_steps=5,
        )
        assert "candidates" in result
        assert "latent_samples" in result
        assert "conditioned_latents" in result
        assert "freq_features" in result
        assert result["candidates"].shape[0] == 4
        assert result["candidates"].shape[-1] == 16
        # Candidates are in [0, 1] (sigmoid decoder)
        assert result["candidates"].min() >= 0.0
        assert result["candidates"].max() <= 1.0

    def test_evaluate_extrapolation_returns_metrics(self):
        """evaluate_extrapolation returns proper verification metrics."""
        designer = _make_extrapolation_designer(grid_size=16, latent_dim=8, cond_dim=16)

        candidates = torch.rand(4, 16, 16)

        def hf_fn(designs):
            return {"fom": torch.rand(designs.shape[0])}

        metrics = designer.evaluate_extrapolation(candidates, hf_fn)
        assert "foms" in metrics
        assert "best_fom" in metrics
        assert "best_idx" in metrics
        assert "mean_fom" in metrics
        assert metrics["foms"].shape == (4,)
        assert metrics["best_fom"].shape == ()
        assert metrics["best_idx"].shape == ()


class TestExtrapolationBenchmark:
    def test_benchmark_runs_end_to_end(self):
        """ExtrapolationBenchmark.run completes for both modes."""
        base = _make_base_designer(grid_size=16, latent_dim=8, cond_dim=16)
        enc, dec = _make_encoder_decoder(16, 8)
        diff = _make_diffusion(8, 16)
        base_inner = LatentDiffusionDesigner(enc, dec, diff)
        conditioner = _make_conditioner(latent_dim=8, n_freq_features=4)
        ext = ExtrapolationDesigner(base_inner, conditioner, dec)

        def hf_fn(designs):
            return {"fom": torch.rand(designs.shape[0])}

        bench = ExtrapolationBenchmark(ext, base, hf_fn, grid_size=16)
        result = bench.run(
            extrapolation_target=torch.randn(16),
            interpolation_target=torch.randn(16),
            n_candidates=4,
            n_diffusion_steps=5,
            n_seeds=2,
        )
        assert "extrapolation" in result
        assert "interpolation" in result
        for key in ("foms", "best_fom", "mean_fom", "diversity"):
            assert key in result["extrapolation"]
            assert key in result["interpolation"]

    def test_extrapolation_differs_from_interpolation(self):
        """Extrapolation and interpolation produce different latent samples.

        The extrapolation conditioner applies a frequency-based shift that
        must produce distinct latent samples compared to vanilla diffusion.
        """
        enc, dec = _make_encoder_decoder(16, 8)
        diff = _make_diffusion(8, 16)
        base_inner = LatentDiffusionDesigner(enc, dec, diff)
        conditioner = _make_conditioner(latent_dim=8, n_freq_features=4)
        ext = ExtrapolationDesigner(base_inner, conditioner, dec, extrapolation_scale=2.0)

        target = torch.randn(16)
        torch.manual_seed(42)
        ext_result = ext.design_extrapolative(target, n_candidates=4, n_diffusion_steps=5)

        torch.manual_seed(42)
        interp_result = base_inner.design(target, n_candidates=4, n_diffusion_steps=5)

        # Conditioned latents must differ from raw diffusion samples
        assert not torch.allclose(
            ext_result["conditioned_latents"],
            interp_result["latent_samples"],
            atol=1e-4,
        )
