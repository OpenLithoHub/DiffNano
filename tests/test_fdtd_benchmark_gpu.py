"""Tests for GPU FDTD benchmarking, multi-scale metalens, and FDTDX cross-validation.

Runs on CPU only to ensure portability across CI environments.
"""

from __future__ import annotations

import pytest
import torch

from diffnano.solvers.fdtd_benchmark_gpu import (
    FDTDBenchmarkConfig,
    FDTDGPUBenchmark,
    FDTDXCrossValidator,
    GPUMemoryStrategy,
    MultiScaleMetalens,
    StabilityReport,
)


# ---------------------------------------------------------------------------
# 1. GPUMemoryStrategy enum
# ---------------------------------------------------------------------------


class TestGPUMemoryStrategyEnum:
    def test_enum_values(self):
        assert GPUMemoryStrategy.TIME_REVERSAL.value == "time_reversal"
        assert GPUMemoryStrategy.CHECKPOINT.value == "checkpoint"
        assert GPUMemoryStrategy.FULL_AUTODIFF.value == "full_autodiff"

    def test_enum_members(self):
        members = list(GPUMemoryStrategy)
        assert len(members) == 3
        assert GPUMemoryStrategy.TIME_REVERSAL in members
        assert GPUMemoryStrategy.CHECKPOINT in members
        assert GPUMemoryStrategy.FULL_AUTODIFF in members

    def test_enum_from_string(self):
        assert GPUMemoryStrategy("time_reversal") == GPUMemoryStrategy.TIME_REVERSAL
        assert GPUMemoryStrategy("checkpoint") == GPUMemoryStrategy.CHECKPOINT
        assert GPUMemoryStrategy("full_autodiff") == GPUMemoryStrategy.FULL_AUTODIFF


# ---------------------------------------------------------------------------
# 2. FDTDBenchmarkConfig defaults
# ---------------------------------------------------------------------------


class TestBenchmarkConfigDefaults:
    def test_default_grid_sizes(self):
        cfg = FDTDBenchmarkConfig()
        assert cfg.grid_sizes == [(32, 32, 32), (64, 64, 64)]

    def test_default_strategies(self):
        cfg = FDTDBenchmarkConfig()
        assert len(cfg.memory_strategies) == 3
        assert GPUMemoryStrategy.TIME_REVERSAL in cfg.memory_strategies
        assert GPUMemoryStrategy.CHECKPOINT in cfg.memory_strategies
        assert GPUMemoryStrategy.FULL_AUTODIFF in cfg.memory_strategies

    def test_default_time_steps(self):
        cfg = FDTDBenchmarkConfig()
        assert cfg.n_time_steps == 100

    def test_default_device(self):
        cfg = FDTDBenchmarkConfig()
        assert cfg.device == "cpu"

    def test_custom_config(self):
        cfg = FDTDBenchmarkConfig(
            grid_sizes=[(8, 8, 8)],
            memory_strategies=[GPUMemoryStrategy.FULL_AUTODIFF],
            n_time_steps=10,
            device="cpu",
        )
        assert cfg.grid_sizes == [(8, 8, 8)]
        assert cfg.memory_strategies == [GPUMemoryStrategy.FULL_AUTODIFF]
        assert cfg.n_time_steps == 10


# ---------------------------------------------------------------------------
# 3. FDTDGPUBenchmark CPU fallback
# ---------------------------------------------------------------------------


_SMALL_GRID = (8, 8, 8)
_SMALL_STEPS = 6


class TestGPUBenchmarkCPUFallback:
    def test_detect_gpu_returns_bool(self):
        result = FDTDGPUBenchmark.detect_gpu()
        assert isinstance(result, bool)

    def test_run_single_cpu(self):
        cfg = FDTDBenchmarkConfig(
            grid_sizes=[_SMALL_GRID],
            n_time_steps=_SMALL_STEPS,
            device="cpu",
        )
        bench = FDTDGPUBenchmark(cfg)
        rec = bench.run_single(
            _SMALL_GRID, GPUMemoryStrategy.FULL_AUTODIFF, "cpu"
        )
        assert rec["grid_size"] == _SMALL_GRID
        assert rec["strategy"] == "full_autodiff"
        assert rec["forward_time_ms"] > 0
        assert rec["backward_time_ms"] > 0
        assert rec["peak_memory_mb"] > 0

    def test_cpu_fallback_benchmark(self):
        cfg = FDTDBenchmarkConfig(
            grid_sizes=[_SMALL_GRID],
            n_time_steps=_SMALL_STEPS,
            device="cpu",
        )
        bench = FDTDGPUBenchmark(cfg)
        report = bench.cpu_fallback_benchmark()

        assert isinstance(report, StabilityReport)
        assert report.is_valid is True
        assert len(report.gradient_cosines) >= 1
        for c in report.gradient_cosines:
            assert c >= 0.99

    def test_run_all_cpu(self):
        cfg = FDTDBenchmarkConfig(
            grid_sizes=[_SMALL_GRID],
            memory_strategies=[
                GPUMemoryStrategy.FULL_AUTODIFF,
                GPUMemoryStrategy.TIME_REVERSAL,
            ],
            n_time_steps=_SMALL_STEPS,
            device="cpu",
        )
        bench = FDTDGPUBenchmark(cfg)
        results = bench.run_all()

        assert len(results) == 2
        ad = results[0]
        tr = results[1]
        assert ad["strategy"] == "full_autodiff"
        assert ad["gradient_cosine"] is None  # reference
        assert tr["strategy"] == "time_reversal"
        assert tr["gradient_cosine"] is not None
        assert tr["gradient_cosine"] > 0.90

    def test_memory_scaling(self):
        cfg = FDTDBenchmarkConfig(
            grid_sizes=[(8, 8, 8), (12, 12, 12)],
            memory_strategies=[GPUMemoryStrategy.FULL_AUTODIFF],
            n_time_steps=_SMALL_STEPS,
            device="cpu",
        )
        bench = FDTDGPUBenchmark(cfg)
        scaling = bench.benchmark_memory_scaling()

        assert len(scaling) == 2
        # Larger grid should use more memory.
        assert scaling[1]["peak_memory_mb"] >= scaling[0]["peak_memory_mb"]


# ---------------------------------------------------------------------------
# 4. MultiScaleMetalens
# ---------------------------------------------------------------------------


class TestMultiScaleMetalens:
    def test_coarse_solve(self):
        msl = MultiScaleMetalens(
            coarse_grid=(8, 8, 8),
            fine_grid=(8, 8, 8),
            n_tiles=(1, 1, 1),
        )
        eps = 1.5 * torch.ones(8, 8, 8, dtype=torch.float64)
        result = msl.coarse_solve(eps)
        assert result.shape == (3, 8, 8, 8)

    def test_fine_solve(self):
        msl = MultiScaleMetalens(
            coarse_grid=(8, 8, 8),
            fine_grid=(8, 8, 8),
            n_tiles=(1, 1, 1),
        )
        eps = 1.5 * torch.ones(8, 8, 8, dtype=torch.float64)
        coarse = msl.coarse_solve(eps)
        fine = msl.fine_solve(eps, coarse)
        assert fine.shape == (3, 8, 8, 8)

    def test_forward_runs(self):
        msl = MultiScaleMetalens(
            coarse_grid=(8, 8, 8),
            fine_grid=(8, 8, 8),
            n_tiles=(1, 1, 1),
        )
        design = 1.5 * torch.ones(8, 8, 8, dtype=torch.float64)
        result = msl.forward(design)

        assert "coarse_field" in result
        assert "tile_fields" in result
        assert "n_tiles" in result
        assert len(result["tile_fields"]) == 1
        assert result["n_tiles"] == (1, 1, 1)

    def test_benchmark_scaling(self):
        msl = MultiScaleMetalens(
            coarse_grid=(8, 8, 8),
            fine_grid=(8, 8, 8),
            n_tiles=(1, 1, 1),
        )
        scaling = msl.benchmark_scaling(base_size=8, max_tiles=2)
        assert len(scaling) == 2
        assert scaling[0]["n_tiles_per_axis"] == 1
        assert scaling[1]["n_tiles_per_axis"] == 2
        for entry in scaling:
            assert entry["coarse_time_ms"] > 0
            assert entry["fine_time_ms"] > 0


# ---------------------------------------------------------------------------
# 5. FDTDXCrossValidator (synthetic reference)
# ---------------------------------------------------------------------------


class TestFDTDXCrossValidatorSynthetic:
    def test_generate_synthetic_reference(self):
        xval = FDTDXCrossValidator()
        ref = xval.generate_synthetic_reference((8, 8, 8), n_steps=6)
        assert "field" in ref
        assert "gradient" in ref
        assert ref["field"].shape == (3, 8, 8, 8)
        assert ref["gradient"].shape == (8, 8, 8)

    def test_validate_forward_identical(self):
        xval = FDTDXCrossValidator()
        field = torch.randn(3, 8, 8, 8, dtype=torch.float64)
        report = xval.validate_forward(field, field)
        assert report["passed"]
        assert report["relative_error"] < 1e-10

    def test_validate_forward_different(self):
        xval = FDTDXCrossValidator()
        a = torch.ones(3, 8, 8, 8, dtype=torch.float64)
        b = torch.ones(3, 8, 8, 8, dtype=torch.float64) * 2.0
        report = xval.validate_forward(a, b, rtol=0.1)
        assert not report["passed"]

    def test_validate_gradient_identical(self):
        xval = FDTDXCrossValidator()
        grad = torch.randn(8, 8, 8, dtype=torch.float64)
        report = xval.validate_gradient(grad, grad)
        assert report["passed"]
        assert report["cosine_similarity"] > 0.999

    def test_validate_gradient_opposite(self):
        xval = FDTDXCrossValidator()
        grad = torch.randn(8, 8, 8, dtype=torch.float64)
        report = xval.validate_gradient(grad, -grad)
        assert not report["passed"]
        assert report["cosine_similarity"] < -0.99

    def test_run_cross_validation_synthetic(self):
        xval = FDTDXCrossValidator()
        report = xval.run_cross_validation(
            n_seeds=2,
            grid_size=(8, 8, 8),
            n_steps=6,
        )
        assert isinstance(report, StabilityReport)
        assert report.is_valid is True
        assert len(report.forward_errors) == 2
        assert len(report.gradient_cosines) == 2
        for err in report.forward_errors:
            assert err < 1e-3
        for cos in report.gradient_cosines:
            assert cos >= 0.99


# ---------------------------------------------------------------------------
# 6. StabilityReport
# ---------------------------------------------------------------------------


class TestStabilityReport:
    def test_default_values(self):
        report = StabilityReport()
        assert report.forward_errors == []
        assert report.gradient_cosines == []
        assert report.memory_used_mb == {}
        assert report.is_valid is False

    def test_with_values(self):
        report = StabilityReport(
            forward_errors=[0.001, 0.002],
            gradient_cosines=[0.995, 0.998],
            memory_used_mb={"full_autodiff": 10.5, "time_reversal": 2.3},
            is_valid=True,
        )
        assert len(report.forward_errors) == 2
        assert len(report.gradient_cosines) == 2
        assert report.memory_used_mb["full_autodiff"] == 10.5
        assert report.is_valid is True

    def test_validity_criteria(self):
        # All cosines >= 0.99 and errors < 1e-3 => valid
        valid = StabilityReport(
            forward_errors=[0.0005, 0.0008],
            gradient_cosines=[0.995, 0.997],
            is_valid=True,
        )
        assert valid.is_valid

        # A bad cosine => manually set is_valid=False
        invalid = StabilityReport(
            forward_errors=[0.0005],
            gradient_cosines=[0.85],
            is_valid=False,
        )
        assert not invalid.is_valid

    def test_empty_report_is_invalid(self):
        report = StabilityReport()
        assert not report.is_valid
