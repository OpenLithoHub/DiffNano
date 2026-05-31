"""Tests for robust posterior warm start (N9.3)."""

import pytest
import torch

from diffnano.design.latent_warm_start import ConditionalLatentSampler
from diffnano.design.quantized import BinarySTE, StraightThroughQuantize
from diffnano.design.robust_warm_start import (
    AngleSweepScorer,
    ProcessCornerWarmStart,
    RobustPosteriorWarmStart,
)
from diffnano.design.representation_learning import LearnedRepresentation


def _make_base_scorer(angle_sensitive: bool = False):
    """Create a simple FoM scorer for testing.

    If angle_sensitive, FoM degrades at large angles (simulates realistic behavior).
    """
    def scorer(design, condition):
        fom = design.sum() * 0.01
        if angle_sensitive and condition.numel() > 0:
            angle_param = condition[-1]
            fom = fom * (1.0 - 0.3 * angle_param.abs())
        return fom
    return scorer


def _make_latent_sampler(grid_size: int = 8, latent_dim: int = 4) -> ConditionalLatentSampler:
    vae = LearnedRepresentation(grid_size=grid_size, latent_dim=latent_dim)
    return ConditionalLatentSampler(vae=vae, latent_dim=latent_dim)


class TestAngleSweepScorer:
    def test_produces_scalar_score(self):
        scorer_fn = _make_base_scorer()
        sweep = AngleSweepScorer(scorer_fn, angle_range=(-30, 30), n_angles=7)
        design = torch.rand(8, 8, dtype=torch.float64)
        condition = torch.zeros(5, dtype=torch.float64)
        score = sweep.score(design, condition)
        assert score.dim() == 0
        assert score.dtype == torch.float64

    def test_score_with_bands_returns_tuple(self):
        scorer_fn = _make_base_scorer()
        sweep = AngleSweepScorer(scorer_fn, angle_range=(-30, 30), n_angles=5)
        design = torch.rand(8, 8, dtype=torch.float64)
        condition = torch.zeros(5, dtype=torch.float64)
        torch.manual_seed(0)
        worst, lower, upper = sweep.score_with_bands(design, condition, n_mc=8)
        assert worst.dim() == 0
        assert lower.dim() == 0
        assert upper.dim() == 0
        assert lower <= worst
        assert worst <= upper

    def test_worst_case_leq_any_single_angle(self):
        scorer_fn = _make_base_scorer(angle_sensitive=True)
        sweep = AngleSweepScorer(scorer_fn, angle_range=(-30, 30), n_angles=7)
        design = torch.ones(8, 8, dtype=torch.float64)
        condition = torch.zeros(5, dtype=torch.float64)

        worst = sweep.score(design, condition)

        lo, hi = sweep.angle_range
        angles = torch.linspace(lo, hi, sweep.n_angles, dtype=torch.float64)
        single_scores = []
        for angle in angles:
            cond = condition.clone()
            cond[-1] = angle / hi
            single_scores.append(scorer_fn(design, cond))

        for s in single_scores:
            assert worst.item() <= s.item() + 1e-10


class TestProcessCornerWarmStart:
    def test_produces_scalar_score(self):
        fom_fn = lambda x: x.sum() * 0.01
        pc = ProcessCornerWarmStart(fom_fn=fom_fn, n_corners=4)
        design = torch.rand(8, 8, dtype=torch.float64)
        score = pc.score(design)
        assert score.dim() == 0

    def test_score_varies_with_design(self):
        fom_fn = lambda x: x.sum()
        pc = ProcessCornerWarmStart(fom_fn=fom_fn, n_corners=4)
        d1 = torch.ones(8, 8, dtype=torch.float64) * 0.3
        d2 = torch.ones(8, 8, dtype=torch.float64) * 0.9
        s1 = pc.score(d1)
        s2 = pc.score(d2)
        assert s2 > s1


class TestRobustPosteriorWarmStart:
    def test_sample_robust_correct_shapes(self):
        scorer_fn = _make_base_scorer()
        sweep = AngleSweepScorer(scorer_fn, n_angles=3)
        sampler = _make_latent_sampler(grid_size=8, latent_dim=4)
        robust_ws = RobustPosteriorWarmStart(
            latent_sampler=sampler,
            robust_scorer=sweep.score,
            n_candidates=4,
        )
        condition = torch.rand(8, 8, dtype=torch.float64)
        result = robust_ws.sample_robust(condition, n_candidates=4)

        assert result["candidates"].shape == (4, 8, 8)
        assert result["robust_scores"].shape == (4,)
        assert result["best"].shape == (8, 8)
        assert result["best_score"].dim() == 0

    def test_scores_by_worst_case(self):
        scorer_fn = _make_base_scorer(angle_sensitive=True)
        sweep = AngleSweepScorer(scorer_fn, angle_range=(-30, 30), n_angles=5)
        sampler = _make_latent_sampler(grid_size=8, latent_dim=4)
        robust_ws = RobustPosteriorWarmStart(
            latent_sampler=sampler,
            robust_scorer=sweep.score,
            n_candidates=6,
        )
        condition = torch.rand(8, 8, dtype=torch.float64)
        result = robust_ws.sample_robust(condition, n_candidates=6)

        for i in range(6):
            assert result["robust_scores"][i] <= result["best_score"] + 1e-10

    def test_with_decision_gate(self):
        scorer_fn = _make_base_scorer()
        sweep = AngleSweepScorer(scorer_fn, n_angles=3)
        sampler = _make_latent_sampler(grid_size=8, latent_dim=4)
        robust_ws = RobustPosteriorWarmStart(
            latent_sampler=sampler,
            robust_scorer=sweep.score,
            n_candidates=4,
        )

        threshold = 0.0

        def gate(scores, candidates):
            mask = scores > threshold
            if mask.any():
                accepted = candidates[mask]
            else:
                accepted = candidates[:0]
            return mask, accepted

        condition = torch.rand(8, 8, dtype=torch.float64)
        result = robust_ws.sample_with_decision_gate(condition, n_candidates=4, decision_gate=gate)
        assert result["accepted_mask"].shape == (4,)
        assert result["accepted_mask"].dtype == torch.bool

    def test_compare_robust_vs_nominal(self):
        scorer_fn = _make_base_scorer(angle_sensitive=True)
        sweep = AngleSweepScorer(scorer_fn, angle_range=(-10, 10), n_angles=3)
        sampler = _make_latent_sampler(grid_size=8, latent_dim=4)
        robust_ws = RobustPosteriorWarmStart(
            latent_sampler=sampler,
            robust_scorer=sweep.score,
            n_candidates=4,
        )

        def fom_fn(x):
            return x.sum() * 0.01

        condition = torch.rand(8, 8, dtype=torch.float64)
        result = robust_ws.compare_robust_vs_nominal(condition, fom_fn, n_seeds=3, n_candidates=4)
        assert len(result["robust_foms"]) == 3
        assert len(result["nominal_foms"]) == 3
        assert 0.0 <= result["robust_wins"] <= 1.0


class TestQuantizedRobustIntegration:
    def test_quantized_design_with_robust_scoring(self):
        scorer_fn = _make_base_scorer()
        sweep = AngleSweepScorer(scorer_fn, n_angles=3)
        quantizer = StraightThroughQuantize(n_levels=2)
        sampler = _make_latent_sampler(grid_size=8, latent_dim=4)

        robust_ws = RobustPosteriorWarmStart(
            latent_sampler=sampler,
            robust_scorer=sweep.score,
            n_candidates=4,
            quantize_fn=quantizer,
        )

        condition = torch.rand(8, 8, dtype=torch.float64)
        result = robust_ws.sample_robust(condition, n_candidates=4)

        for i in range(4):
            quantized = quantizer(result["candidates"][i])
            assert set(quantized.unique().tolist()) <= {0.0, 1.0}

    def test_binary_ste_with_process_corner(self):
        fom_fn = lambda x: x.sum()
        pc = ProcessCornerWarmStart(fom_fn=fom_fn, n_corners=4)
        ste = BinarySTE()
        design = torch.rand(8, 8, dtype=torch.float64)
        binary_design = ste(design)
        score = pc.score(binary_design)
        assert score.dim() == 0


class TestDeterminism:
    def test_deterministic_with_seed(self):
        scorer_fn = _make_base_scorer()
        sweep = AngleSweepScorer(scorer_fn, n_angles=3)
        sampler = _make_latent_sampler(grid_size=8, latent_dim=4)

        robust_ws = RobustPosteriorWarmStart(
            latent_sampler=sampler,
            robust_scorer=sweep.score,
            n_candidates=4,
        )
        condition = torch.rand(8, 8, dtype=torch.float64)

        torch.manual_seed(99)
        r1 = robust_ws.sample_robust(condition, n_candidates=4)

        torch.manual_seed(99)
        r2 = robust_ws.sample_robust(condition, n_candidates=4)

        assert torch.allclose(r1["robust_scores"], r2["robust_scores"])
        assert torch.equal(r1["best"], r2["best"])
