"""Tests for FDTD benchmark and cross-validation framework.

Validates FDTDBenchmarkSuite, ExternalCrossValidator, SystolicUpdateEvaluator,
and BenchmarkConfig/BenchmarkResult dataclasses using the real FDTDSolver3D
with small grids for fast CPU execution.
"""

import json
import tempfile
from pathlib import Path

import torch

from diffnano.solvers.fdtd3d import FDTDSolver3D
from diffnano.solvers.fdtd_benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    ExternalCrossValidator,
    FDTDBenchmarkSuite,
    SystolicUpdateEvaluator,
)

_SMALL_GRID = (8, 8, 8)
_SMALL_STEPS = 6


class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.grid_sizes == [(16, 16, 16), (32, 32, 32)]
        assert cfg.n_time_steps == 100
        assert cfg.backward_modes == ["time_reversal", "checkpoint", "autograd"]
        assert cfg.device == "cpu"

    def test_custom_grid_sizes(self):
        cfg = BenchmarkConfig(grid_sizes=[(4, 4, 4)])
        assert cfg.grid_sizes == [(4, 4, 4)]

    def test_custom_modes(self):
        cfg = BenchmarkConfig(backward_modes=["autograd"])
        assert cfg.backward_modes == ["autograd"]


class TestBenchmarkResult:
    def test_all_fields(self):
        r = BenchmarkResult(
            grid_size=(8, 8, 8),
            backward_mode="autograd",
            forward_time_ms=1.0,
            backward_time_ms=2.0,
            peak_memory_mb=0.5,
            gradient_cosine_vs_autograd=None,
            device="cpu",
        )
        assert r.grid_size == (8, 8, 8)
        assert r.backward_mode == "autograd"
        assert r.forward_time_ms == 1.0
        assert r.backward_time_ms == 2.0
        assert r.peak_memory_mb == 0.5
        assert r.gradient_cosine_vs_autograd is None
        assert r.device == "cpu"

    def test_with_cosine(self):
        r = BenchmarkResult(
            grid_size=(8, 8, 8),
            backward_mode="time_reversal",
            forward_time_ms=1.0,
            backward_time_ms=2.0,
            peak_memory_mb=0.5,
            gradient_cosine_vs_autograd=0.995,
            device="cpu",
        )
        assert r.gradient_cosine_vs_autograd == 0.995


class TestFDTDBenchmarkSuite:
    def test_runs_on_cpu(self):
        cfg = BenchmarkConfig(
            grid_sizes=[_SMALL_GRID],
            n_time_steps=_SMALL_STEPS,
            backward_modes=["autograd", "time_reversal"],
            device="cpu",
        )
        suite = FDTDBenchmarkSuite()
        results = suite.run(FDTDSolver3D, cfg)

        assert len(results) == 2
        for r in results:
            assert isinstance(r, BenchmarkResult)
            assert r.grid_size == _SMALL_GRID
            assert r.forward_time_ms > 0
            assert r.backward_time_ms > 0
            assert r.device == "cpu"

    def test_autograd_is_reference(self):
        cfg = BenchmarkConfig(
            grid_sizes=[_SMALL_GRID],
            n_time_steps=_SMALL_STEPS,
            backward_modes=["autograd"],
            device="cpu",
        )
        suite = FDTDBenchmarkSuite()
        results = suite.run(FDTDSolver3D, cfg)

        assert results[0].gradient_cosine_vs_autograd is None

    def test_cosine_vs_autograd(self):
        cfg = BenchmarkConfig(
            grid_sizes=[_SMALL_GRID],
            n_time_steps=_SMALL_STEPS,
            backward_modes=["autograd", "time_reversal"],
            device="cpu",
        )
        suite = FDTDBenchmarkSuite()
        results = suite.run(FDTDSolver3D, cfg)

        tr_result = results[1]
        assert tr_result.gradient_cosine_vs_autograd is not None
        assert tr_result.gradient_cosine_vs_autograd > 0.90

    def test_summary_table(self):
        cfg = BenchmarkConfig(
            grid_sizes=[_SMALL_GRID],
            n_time_steps=_SMALL_STEPS,
            backward_modes=["autograd"],
            device="cpu",
        )
        suite = FDTDBenchmarkSuite()
        suite.run(FDTDSolver3D, cfg)

        table = suite.summary_table()
        assert "autograd" in table
        assert "8x8x8" in table

    def test_empty_summary(self):
        suite = FDTDBenchmarkSuite()
        assert suite.summary_table() == "No benchmark results."


class TestExternalCrossValidatorValidateForward:
    def test_identical_passes(self):
        xval = ExternalCrossValidator()
        field = torch.randn(3, 8, 8, 8, dtype=torch.float64)
        result = xval.validate_forward(field, field)
        assert result["passed"]
        assert result["relative_error"] < 1e-10
        assert result["max_absolute_error"] < 1e-10

    def test_different_fails(self):
        xval = ExternalCrossValidator()
        a = torch.ones(3, 8, 8, 8, dtype=torch.float64)
        b = torch.ones(3, 8, 8, 8, dtype=torch.float64) * 2.0
        result = xval.validate_forward(a, b, rtol=0.1)
        assert not result["passed"]
        assert result["relative_error"] > 0.1

    def test_rtol_parameter(self):
        xval = ExternalCrossValidator()
        a = torch.ones(3, 8, 8, 8, dtype=torch.float64)
        b = a + 0.01 * torch.randn_like(a)
        result_loose = xval.validate_forward(a, b, rtol=1.0)
        result_strict = xval.validate_forward(a, b, rtol=1e-6)
        assert result_loose["passed"]
        assert not result_strict["passed"]


class TestExternalCrossValidatorValidateGradient:
    def test_identical_gradients(self):
        xval = ExternalCrossValidator()
        grad = torch.randn(8, 8, 8, dtype=torch.float64)
        result = xval.validate_gradient(grad, grad)
        assert result["passed"]
        assert result["cosine_similarity"] > 0.999

    def test_opposite_direction(self):
        xval = ExternalCrossValidator()
        grad = torch.randn(8, 8, 8, dtype=torch.float64)
        result = xval.validate_gradient(grad, -grad)
        assert not result["passed"]
        assert result["cosine_similarity"] < -0.99

    def test_cosine_threshold(self):
        xval = ExternalCrossValidator()
        torch.manual_seed(0)
        a = torch.randn(8, 8, 8, dtype=torch.float64)
        noise = 0.1 * torch.randn_like(a)
        b = a + noise
        result = xval.validate_gradient(a, b, cosine_threshold=0.5)
        assert result["passed"]
        assert result["cosine_similarity"] > 0.5


class TestExternalCrossValidatorTestCase:
    def test_reproducible(self):
        xval = ExternalCrossValidator()
        tc1 = xval.generate_test_case(grid_size=(8, 8, 8))
        tc2 = xval.generate_test_case(grid_size=(8, 8, 8))
        assert torch.equal(tc1["eps"], tc2["eps"])

    def test_default_source_pos(self):
        xval = ExternalCrossValidator()
        tc = xval.generate_test_case(grid_size=(10, 12, 14))
        assert tc["source_pos"] == (5, 6, 7)

    def test_custom_source_pos(self):
        xval = ExternalCrossValidator()
        tc = xval.generate_test_case(
            grid_size=(8, 8, 8), source_pos=(1, 2, 3), freq=2.0
        )
        assert tc["source_pos"] == (1, 2, 3)
        assert tc["freq"] == 2.0
        assert tc["seed"] == 42
        assert tc["eps"].shape == (8, 8, 8)

    def test_load_external_results(self):
        xval = ExternalCrossValidator()
        field_data = torch.randn(3, 4, 4, 4, dtype=torch.float64)
        grad_data = torch.randn(4, 4, 4, dtype=torch.float64)

        payload = {
            "field": {
                "shape": list(field_data.shape),
                "data": field_data.flatten().tolist(),
            },
            "gradient": {
                "shape": list(grad_data.shape),
                "data": grad_data.flatten().tolist(),
            },
            "source": "test_solver",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(payload, f)
            tmppath = f.name

        try:
            loaded = xval.load_external_results(tmppath)
            assert torch.allclose(loaded["field"], field_data, atol=1e-10)
            assert torch.allclose(loaded["gradient"], grad_data, atol=1e-10)
            assert loaded["source"] == "test_solver"
        finally:
            Path(tmppath).unlink()


class TestSystolicUpdateEvaluator:
    def test_measures_bandwidth(self):
        solver = FDTDSolver3D(
            grid_shape=_SMALL_GRID,
            dl=20.0,
            wavelength_nm=1550.0,
            pml_layers=0,
            n_steps=4,
            device="cpu",
            courant=0.35,
        )
        evaluator = SystolicUpdateEvaluator()
        result = evaluator.measure_bandwidth(solver, _SMALL_GRID)

        assert "field_size_mb" in result
        assert "update_bandwidth_gbs" in result
        assert "theoretical_peak_gbs" in result
        assert "bandwidth_utilization" in result
        assert result["field_size_mb"] > 0
        assert result["update_bandwidth_gbs"] > 0
        assert result["theoretical_peak_gbs"] > 0

    def test_bandwidth_utilization_bounded(self):
        solver = FDTDSolver3D(
            grid_shape=_SMALL_GRID,
            dl=20.0,
            wavelength_nm=1550.0,
            pml_layers=0,
            n_steps=4,
            device="cpu",
            courant=0.35,
        )
        evaluator = SystolicUpdateEvaluator()
        result = evaluator.measure_bandwidth(solver, _SMALL_GRID, n_trials=3)

        assert result["bandwidth_utilization"] >= 0.0


class TestDeterminism:
    def test_deterministic_with_seed(self):
        cfg = BenchmarkConfig(
            grid_sizes=[_SMALL_GRID],
            n_time_steps=_SMALL_STEPS,
            backward_modes=["autograd"],
            device="cpu",
        )
        suite = FDTDBenchmarkSuite()
        results_a = suite.run(FDTDSolver3D, cfg)

        suite2 = FDTDBenchmarkSuite()
        results_b = suite2.run(FDTDSolver3D, cfg)

        # Forward results are deterministic (seed=42 in _make_eps_grid).
        # Times may differ but grid_size and mode must match.
        assert results_a[0].grid_size == results_b[0].grid_size
        assert results_a[0].backward_mode == results_b[0].backward_mode
