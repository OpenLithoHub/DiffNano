"""Tests for multi-fidelity inverse design with RCWA/FDTD cost-aware fusion."""

from __future__ import annotations

import torch
import pytest
from torch import Tensor

from diffnano.design.multifidelity import (
    FoundryConstraints,
    FidelityOracle,
    MultiFidelityDesigner,
    MultiFidelityDesignBenchmark,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_oracle(
    grid_size: int = 16,
    resp_dim: int = 8,
) -> FidelityOracle:
    """Create a FidelityOracle with toy RCWA/FDTD evaluators."""

    def rcwa_fn(design: Tensor) -> Tensor:
        d = design.detach().reshape(-1)
        return torch.randn(resp_dim, dtype=torch.float64) + d[:resp_dim] * 0.1

    def fdtd_fn(design: Tensor) -> Tensor:
        d = design.detach().reshape(-1)
        return torch.randn(resp_dim, dtype=torch.float64) + d[:resp_dim] * 0.2

    return FidelityOracle(rcwa_fn=rcwa_fn, fdtd_fn=fdtd_fn)


class _StubCostModel:
    """Minimal CostModel stand-in for unit tests (avoids diff-surrogate import)."""

    def __init__(self, budget: float = 1000.0):
        self.fidelity_levels = {"low": 1.0, "high": 20.0}
        self.total_budget = budget
        self.budget_consumed = 0.0

    def remaining(self) -> float:
        return self.total_budget - self.budget_consumed

    def consume(self, fidelity: str, n: int) -> None:
        self.budget_consumed += self.fidelity_levels[fidelity] * n


# ---------------------------------------------------------------------------
# FoundryConstraints tests
# ---------------------------------------------------------------------------


class TestFoundryConstraints:
    def test_foundry_constraints_pass(self):
        """A large uniform design should pass all constraints."""
        foundry = FoundryConstraints(min_feature_nm=40.0, min_space_nm=40.0, pixel_size_nm=10.0)
        # 8x8 grid, pixel_size=10nm -> 80nm per cell, well above min 40nm
        design = torch.ones(8, 8, dtype=torch.float64)
        report = foundry.check(design)
        assert report["passed"] is True
        assert report["violations"] == 0

    def test_foundry_constraints_fail(self):
        """An alternating 0101 pattern should violate spacing/feature rules."""
        foundry = FoundryConstraints(min_feature_nm=40.0, min_space_nm=40.0, pixel_size_nm=20.0)
        # min_feature_px = ceil(40/20) = 2, min_space_px = 2
        # Alternating pattern creates single-pixel features and spaces
        row = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float64)
        design = row.unsqueeze(0).expand(4, -1).clone()
        report = foundry.check(design)
        assert report["passed"] is False
        assert report["violations"] > 0

    def test_foundry_project_fixes(self):
        """Projection should eliminate short runs via morphological opening."""
        foundry = FoundryConstraints(min_feature_nm=20.0, min_space_nm=20.0, pixel_size_nm=10.0)
        # min_feature_px = 2, min_space_px = 2
        # A single-pixel feature embedded in zeros should be removed by opening
        design = torch.zeros(8, 16, dtype=torch.float64)
        design[:, 7] = 1.0  # single-pixel-wide vertical stripe
        projected = foundry.project(design)
        # The single-pixel-wide feature should be removed
        assert (projected == 0.0).all()
        report = foundry.check(projected)
        assert report["passed"] is True


# ---------------------------------------------------------------------------
# FidelityOracle tests
# ---------------------------------------------------------------------------


class TestFidelityOracle:
    def test_fidelity_oracle_dispatch(self):
        """Oracle should dispatch to the correct evaluator."""
        rcwa_fn = lambda d: d.sum().unsqueeze(0)
        fdtd_fn = lambda d: (d * 2).sum().unsqueeze(0)

        oracle = FidelityOracle(rcwa_fn=rcwa_fn, fdtd_fn=fdtd_fn)
        design = torch.ones(4, 4, dtype=torch.float64)

        low = oracle.evaluate_low(design)
        assert low.item() == pytest.approx(16.0)

        high = oracle.evaluate_high(design)
        assert high.item() == pytest.approx(32.0)

        # Generic dispatch
        assert oracle.evaluate(design, "low").item() == pytest.approx(16.0)
        assert oracle.evaluate(design, "high").item() == pytest.approx(32.0)

        with pytest.raises(ValueError, match="fidelity must be"):
            oracle.evaluate(design, "medium")


# ---------------------------------------------------------------------------
# MultiFidelityDesigner tests
# ---------------------------------------------------------------------------


class TestMultiFidelityDesigner:
    def _make_designer(self) -> MultiFidelityDesigner:
        oracle = _make_oracle()
        cost_model = _StubCostModel(budget=10000.0)
        foundry = FoundryConstraints(min_feature_nm=10.0, min_space_nm=10.0, pixel_size_nm=5.0)
        return MultiFidelityDesigner(
            oracle=oracle,
            cost_model=cost_model,
            foundry=foundry,
        )

    def test_multifidelity_designer_runs(self):
        """Full design pipeline should return expected keys and shapes."""
        designer = self._make_designer()
        target = torch.randn(8, dtype=torch.float64)
        result = designer.design(target, n_initial=20, n_top=3, grid_size=8)

        assert "best" in result
        assert "best_score" in result
        assert "candidates" in result
        assert "screened" in result
        assert "foundry_reports" in result
        assert result["best"].shape == (8, 8)
        assert result["candidates"].shape == (20, 8, 8)
        assert result["screened"].shape[0] == 3

    def test_screen_candidates_selects_top(self):
        """screen_candidates should return exactly n_top candidates."""
        designer = self._make_designer()
        candidates = torch.rand(20, 8, 8, dtype=torch.float64)
        top, scores = designer.screen_candidates(candidates, n_top=5)
        assert top.shape[0] == 5
        assert scores.shape[0] == 5

    def test_verify_candidates_uses_fdtd(self):
        """verify_candidates should call FDTD and return responses."""
        oracle = _make_oracle()
        cost_model = _StubCostModel()
        foundry = FoundryConstraints(min_feature_nm=5.0, min_space_nm=5.0, pixel_size_nm=5.0)
        designer = MultiFidelityDesigner(oracle=oracle, cost_model=cost_model, foundry=foundry)

        candidates = torch.rand(3, 8, 8, dtype=torch.float64)
        target = torch.randn(8, dtype=torch.float64)
        responses, info = designer.verify_candidates(candidates, target)

        assert responses.shape[0] == 3
        assert "hf_foms" in info
        assert len(info["hf_foms"]) == 3

    def test_compare_vs_single_fidelity(self):
        """compare_vs_single_fidelity should return all method results."""
        designer = self._make_designer()
        target = torch.randn(8, dtype=torch.float64)
        result = designer.compare_vs_single_fidelity(
            target, n_seeds=2, n_initial=10, n_top=3, grid_size=8
        )

        assert "mf_foms" in result
        assert "lf_foms" in result
        assert "hf_foms" in result
        assert len(result["mf_foms"]) == 2
        assert len(result["lf_foms"]) == 2
        assert len(result["hf_foms"]) == 2


# ---------------------------------------------------------------------------
# MultiFidelityDesignBenchmark tests
# ---------------------------------------------------------------------------


class TestMultiFidelityDesignBenchmark:
    def test_benchmark_runs(self):
        """Benchmark should produce results for all three methods."""
        oracle = _make_oracle(grid_size=8, resp_dim=4)
        foundry = FoundryConstraints(min_feature_nm=10.0, min_space_nm=10.0, pixel_size_nm=5.0)
        benchmark = MultiFidelityDesignBenchmark(
            oracle=oracle,
            foundry=foundry,
            grid_size=8,
        )
        target = torch.randn(4, dtype=torch.float64)
        results = benchmark.run(
            target, n_seeds=2, hf_budgets=[3, 5], n_initial=10
        )

        assert "multifidelity" in results
        assert "high_only" in results
        assert "low_only" in results

        # Check that foms lists are populated
        for method in ("multifidelity", "high_only", "low_only"):
            assert len(results[method]["foms"]) > 0
            assert "hf_calls" in results[method]
            assert "foundry_pass_rate" in results[method]
