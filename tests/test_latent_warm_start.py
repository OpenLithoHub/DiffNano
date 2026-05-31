"""Tests for latent warm-start module (N8.3)."""

import pytest
import torch

from diffnano.design.latent_warm_start import (
    ConditionalLatentSampler,
    StrehlScorer,
    WilcoxonComparison,
)
from diffnano.design.representation_learning import LearnedRepresentation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trained_vae():
    """Return a small VAE trained on a few synthetic designs."""
    vae = LearnedRepresentation(grid_size=8, latent_dim=4, device="cpu")
    designs = [torch.rand(8, 8, dtype=torch.float64) for _ in range(20)]
    vae.train_vae(designs, n_epochs=5, batch_size=10, verbose=False)
    return vae


@pytest.fixture
def sampler(trained_vae):
    return ConditionalLatentSampler(
        vae=trained_vae, latent_dim=4, device="cpu"
    )


@pytest.fixture
def simple_fom():
    """FOM that prefers higher mean density (trivial, differentiable)."""
    def fom(geom: torch.Tensor) -> torch.Tensor:
        return geom.mean()
    return fom


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConditionalLatentSampler:
    def test_conditional_latent_sampler_shape(self, sampler):
        """sample_candidates returns (n_candidates, H, W)."""
        condition = torch.rand(8, 8, dtype=torch.float64)
        candidates = sampler.sample_candidates(condition, n_candidates=5)
        assert candidates.shape == (5, 8, 8)

    def test_sample_candidates_diverse(self, sampler):
        """Candidates should differ from each other."""
        condition = torch.rand(8, 8, dtype=torch.float64)
        candidates = sampler.sample_candidates(condition, n_candidates=6)
        # At least one pair must differ
        all_same = all(
            torch.allclose(candidates[0], candidates[i], atol=1e-6)
            for i in range(1, 6)
        )
        assert not all_same

    def test_sample_candidates_values_bounded(self, sampler):
        """Decoded geometries should be in [0, 1] (sigmoid output)."""
        condition = torch.rand(8, 8, dtype=torch.float64)
        candidates = sampler.sample_candidates(condition, n_candidates=4)
        assert candidates.min() >= 0.0
        assert candidates.max() <= 1.0

    def test_batch_refine_improves_fom(self, sampler, simple_fom):
        """After refinement, mean FOM should not decrease."""
        condition = torch.rand(8, 8, dtype=torch.float64)
        candidates = sampler.sample_candidates(condition, n_candidates=3)

        initial_foms = torch.tensor([simple_fom(c).item() for c in candidates])

        refined, histories = sampler.batch_refine(
            candidates, simple_fom, n_steps=10, lr=0.01
        )

        refined_foms = torch.tensor([simple_fom(c).item() for c in refined])

        # Mean FOM should not get worse (allow equality for edge cases)
        assert refined_foms.mean() >= initial_foms.mean() - 1e-4

    def test_score_and_select_returns_best(self, sampler, simple_fom):
        """Top-k selection picks the highest-scoring candidates."""
        candidates = torch.rand(5, 8, 8, dtype=torch.float64)

        best, scores, indices = sampler.score_and_select(
            candidates, simple_fom, top_k=2
        )

        assert best.shape == (2, 8, 8)
        assert scores.shape == (5,)
        assert indices.shape == (2,)

        # indices should correspond to the top-2 scores
        sorted_indices = torch.argsort(scores, descending=True)[:2]
        assert set(indices.tolist()) == set(sorted_indices.tolist())

    def test_warm_start_pipeline_runs(self, sampler, simple_fom):
        """Full warm_start_optimize pipeline should run end-to-end."""
        condition = torch.rand(8, 8, dtype=torch.float64)
        result = sampler.warm_start_optimize(
            condition,
            simple_fom,
            n_candidates=4,
            top_k=2,
            refine_steps=5,
        )

        assert "best_geometry" in result
        assert "all_candidates" in result
        assert "all_scores" in result
        assert result["best_geometry"].shape == (8, 8)
        assert result["all_candidates"].shape == (4, 8, 8)

    def test_satisfies_candidate_sampler_protocol(self, sampler):
        """ConditionalLatentSampler should satisfy CandidateSampler protocol."""
        from diff_surrogate.generative import CandidateSampler
        assert isinstance(sampler, CandidateSampler)


class TestStrehlScorer:
    def test_strehl_scorer_returns_scalar(self):
        """StrehlScorer.score should return a scalar tensor."""
        from diffnano.solvers.rcwa import RCWASolver

        rcwa = RCWASolver(
            fourier_orders=2,
            wavelength_nm=1550.0,
            period_nm=(400.0, 400.0),
        )
        scorer = StrehlScorer(rcwa, wavelength=1550.0)

        geometry = torch.ones(3, 32, dtype=torch.float64) * 2.25
        score = scorer.score(geometry)

        assert score.dim() == 0
        assert score.item() >= 0.0


class TestWilcoxonComparison:
    def test_wilcoxon_comparison_returns_p_value(self, sampler, simple_fom):
        """WilcoxonComparison.compare should return a valid p-value."""
        result = WilcoxonComparison.compare(
            sampler,
            simple_fom,
            n_seeds=5,
            n_candidates=3,
            grid_size=8,
        )

        assert "p_value" in result
        assert "warm_start_foms" in result
        assert "random_foms" in result
        assert "warm_start_wins" in result
        assert 0.0 <= result["p_value"] <= 1.0
        assert len(result["warm_start_foms"]) == 5
        assert len(result["random_foms"]) == 5
        assert 0.0 <= result["warm_start_wins"] <= 1.0
