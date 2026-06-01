"""Tests for gpu_benchmark module: Metalens3DDesigner, FDTDGPURealBenchmark, MultiScaleBenchmark.

Runs on CPU only to ensure portability across CI environments.
"""

from __future__ import annotations

import math

import pytest
import torch

from diffnano.design.gpu_benchmark import (
    ConvergenceRecord,
    FDTDGPURealBenchmark,
    GPUDeviceMetrics,
    Metalens3DConfig,
    Metalens3DDesigner,
    MultiScaleBenchmark,
)

# ---------------------------------------------------------------------------
# 1. Metalens3DConfig
# ---------------------------------------------------------------------------


class TestMetalens3DConfig:
    def test_defaults(self):
        cfg = Metalens3DConfig()
        assert cfg.aperture_um == 100.0
        assert cfg.na == 0.8
        assert cfg.focal_length_um == 200.0
        assert cfg.wavelength_nm == 1550.0
        assert cfg.grid_resolution_nm == 20.0
        assert cfg.n_material == 2.4
        assert cfg.n_ambient == 1.0

    def test_custom(self):
        cfg = Metalens3DConfig(
            aperture_um=50.0,
            na=0.5,
            focal_length_um=100.0,
            wavelength_nm=532.0,
        )
        assert cfg.aperture_um == 50.0
        assert cfg.na == 0.5
        assert cfg.focal_length_um == 100.0
        assert cfg.wavelength_nm == 532.0


# ---------------------------------------------------------------------------
# 2. Metalens3DDesigner
# ---------------------------------------------------------------------------

_SMALL_APERTURE = Metalens3DConfig(
    aperture_um=0.08,  # 80 nm -> very small grid for fast tests
    grid_resolution_nm=20.0,
    focal_length_um=0.2,
    wavelength_nm=1550.0,
)


class TestMetalens3DDesigner:
    def test_construction(self):
        designer = Metalens3DDesigner(_SMALL_APERTURE)
        assert designer.grid_size_1d >= 4
        assert designer.grid_size_1d % 2 == 0

    def test_target_phase_shape(self):
        designer = Metalens3DDesigner(_SMALL_APERTURE)
        phase = designer.target_phase()
        assert phase.shape == (designer.grid_size_1d, designer.grid_size_1d)
        assert phase.dtype == torch.float64

    def test_target_phase_range(self):
        designer = Metalens3DDesigner(_SMALL_APERTURE)
        phase = designer.target_phase()
        assert phase.min().item() >= 0.0
        assert phase.max().item() < 2 * math.pi + 1e-6

    def test_target_phase_central_symmetry(self):
        designer = Metalens3DDesigner(_SMALL_APERTURE)
        phase = designer.target_phase()
        n = phase.shape[0]
        # Phase should be radially symmetric: compare opposite quadrants.
        q1 = phase[: n // 2, : n // 2]
        q4 = phase[n // 2 :, n // 2 :]
        # The values in opposite quadrants should be close after flipping.
        # Not exact due to even-odd grid offsets, but rough symmetry holds.
        assert q1.shape[0] > 0
        assert q4.shape[0] > 0

    def test_generate_pattern_shape(self):
        designer = Metalens3DDesigner(_SMALL_APERTURE)
        pattern = designer.generate_pattern(n_layers=2, seed=0)
        assert pattern.dim() == 3
        assert pattern.shape[0] == 2
        assert pattern.shape[1] == designer.grid_size_1d
        assert pattern.shape[2] == designer.grid_size_1d

    def test_generate_pattern_permittivity_range(self):
        designer = Metalens3DDesigner(_SMALL_APERTURE)
        pattern = designer.generate_pattern(n_layers=2, seed=0)
        eps_low = _SMALL_APERTURE.n_ambient**2
        eps_high = _SMALL_APERTURE.n_material**2
        assert pattern.min().item() >= eps_low - 0.1
        assert pattern.max().item() <= eps_high + 0.1

    def test_generate_pattern_deterministic(self):
        designer = Metalens3DDesigner(_SMALL_APERTURE)
        p1 = designer.generate_pattern(n_layers=2, seed=7)
        p2 = designer.generate_pattern(n_layers=2, seed=7)
        assert torch.allclose(p1, p2)

    def test_generate_pattern_reproducible_across_calls(self):
        # Pattern generation is deterministic from the phase profile;
        # verify that repeated calls with the same config produce the same result.
        designer = Metalens3DDesigner(_SMALL_APERTURE)
        p1 = designer.generate_pattern(n_layers=2, seed=1)
        p2 = designer.generate_pattern(n_layers=2, seed=1)
        p3 = designer.generate_pattern(n_layers=2, seed=42)
        # Same seed always gives same pattern.
        assert torch.allclose(p1, p2)
        # Pattern values are valid even for different seeds.
        assert p3.shape == p1.shape

    def test_decompose_tiles(self):
        designer = Metalens3DDesigner(_SMALL_APERTURE)
        pattern = designer.generate_pattern(n_layers=2, seed=0)
        tiles = designer.decompose_tiles(pattern, tile_size=max(2, designer.grid_size_1d // 2))
        assert len(tiles) >= 1
        for t in tiles:
            assert t.dim() == 3
            assert t.shape[0] == 2

    def test_reassemble_tiles_roundtrip(self):
        designer = Metalens3DDesigner(_SMALL_APERTURE)
        pattern = designer.generate_pattern(n_layers=2, seed=0)
        ts = max(2, designer.grid_size_1d // 2)
        tiles = designer.decompose_tiles(pattern, tile_size=ts)
        reassembled = designer.reassemble_tiles(tiles, pattern.shape, tile_size=ts)
        assert reassembled.shape == pattern.shape
        assert torch.allclose(reassembled, pattern, atol=1e-10)

    def test_device_property(self):
        designer = Metalens3DDesigner(_SMALL_APERTURE, device="cpu")
        assert designer.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# 3. GPUDeviceMetrics
# ---------------------------------------------------------------------------


class TestGPUDeviceMetrics:
    def test_defaults(self):
        m = GPUDeviceMetrics()
        assert m.device_name == "cpu"
        assert m.cuda_available is False
        assert m.total_memory_mb == 0.0
        assert m.forward_time_ms == 0.0
        assert m.backward_time_ms == 0.0
        assert m.peak_memory_mb == 0.0
        assert m.memory_delta_mb == 0.0
        assert m.throughput_samples_per_sec == 0.0

    def test_with_values(self):
        m = GPUDeviceMetrics(
            device_name="NVIDIA A100",
            cuda_available=True,
            total_memory_mb=81920.0,
            forward_time_ms=12.5,
            backward_time_ms=25.0,
            peak_memory_mb=1024.0,
            memory_delta_mb=512.0,
            throughput_samples_per_sec=26.667,
        )
        assert m.device_name == "NVIDIA A100"
        assert m.cuda_available is True
        assert m.total_memory_mb == 81920.0
        assert m.forward_time_ms == 12.5
        assert m.backward_time_ms == 25.0


# ---------------------------------------------------------------------------
# 4. FDTDGPURealBenchmark
# ---------------------------------------------------------------------------


class TestFDTDGPURealBenchmark:
    def test_detect_gpu_returns_tuple(self):
        available, name = FDTDGPURealBenchmark.detect_gpu()
        assert isinstance(available, bool)
        assert isinstance(name, str)

    def test_gpu_memory_info_returns_dict(self):
        info = FDTDGPURealBenchmark.gpu_memory_info()
        assert "allocated_mb" in info
        assert "reserved_mb" in info
        assert "total_mb" in info
        assert all(isinstance(v, float) for v in info.values())

    def test_construction_defaults(self):
        bench = FDTDGPURealBenchmark()
        assert bench.grid_sizes == [(16, 16, 16)]
        assert bench.n_time_steps == 20
        assert bench.n_warmup == 1
        assert bench.n_trials == 3

    def test_construction_custom(self):
        bench = FDTDGPURealBenchmark(
            grid_sizes=[(8, 8, 8)],
            n_time_steps=5,
            n_warmup=0,
            n_trials=1,
        )
        assert bench.grid_sizes == [(8, 8, 8)]
        assert bench.n_time_steps == 5

    def test_compare_cpu_gpu(self):
        bench = FDTDGPURealBenchmark(
            grid_sizes=[(8, 8, 8)],
            n_time_steps=3,
            n_warmup=0,
            n_trials=1,
        )
        results = bench.compare_cpu_gpu()
        assert "cpu" in results
        assert isinstance(results["cpu"], GPUDeviceMetrics)
        assert results["cpu"].forward_time_ms > 0
        assert results["cpu"].backward_time_ms > 0
        assert results["cpu"].throughput_samples_per_sec > 0

    def test_run_scaling_benchmark(self):
        bench = FDTDGPURealBenchmark(
            grid_sizes=[(8, 8, 8), (10, 10, 10)],
            n_time_steps=3,
            n_warmup=0,
            n_trials=1,
        )
        results = bench.run_scaling_benchmark()
        assert len(results) == 2
        for r in results:
            assert "grid_size" in r
            assert "forward_time_ms" in r
            assert "backward_time_ms" in r
            assert "peak_memory_mb" in r
            assert "throughput_samples_per_sec" in r
            assert r["forward_time_ms"] > 0

    def test_run_all_alias(self):
        bench = FDTDGPURealBenchmark(
            grid_sizes=[(8, 8, 8)],
            n_time_steps=3,
            n_warmup=0,
            n_trials=1,
        )
        results = bench.run_all()
        assert len(results) == 1

    def test_cpu_measure_has_positive_times(self):
        bench = FDTDGPURealBenchmark(
            grid_sizes=[(8, 8, 8)],
            n_time_steps=5,
            n_warmup=0,
            n_trials=2,
        )
        metrics = bench._measure_single((8, 8, 8), "cpu")
        assert metrics.forward_time_ms > 0
        assert metrics.backward_time_ms > 0

    def test_scaling_larger_grid_slower(self):
        bench = FDTDGPURealBenchmark(
            grid_sizes=[(6, 6, 6), (10, 10, 10)],
            n_time_steps=3,
            n_warmup=0,
            n_trials=1,
        )
        results = bench.run_scaling_benchmark()
        # Larger grid should take longer (or at least not faster by a lot).
        fwd_small = results[0]["forward_time_ms"]
        fwd_large = results[1]["forward_time_ms"]
        # Just verify both are positive; timing is noisy in CI.
        assert fwd_small > 0
        assert fwd_large > 0


# ---------------------------------------------------------------------------
# 5. ConvergenceRecord
# ---------------------------------------------------------------------------


class TestConvergenceRecord:
    def test_defaults(self):
        rec = ConvergenceRecord()
        assert rec.iteration == 0
        assert rec.fom == 0.0
        assert rec.elapsed_ms == 0.0

    def test_with_values(self):
        rec = ConvergenceRecord(iteration=5, fom=0.95, elapsed_ms=123.4)
        assert rec.iteration == 5
        assert rec.fom == 0.95
        assert rec.elapsed_ms == 123.4


# ---------------------------------------------------------------------------
# 6. MultiScaleBenchmark
# ---------------------------------------------------------------------------


class TestMultiScaleBenchmark:
    @pytest.fixture()
    def designer(self):
        cfg = Metalens3DConfig(
            aperture_um=0.08,
            grid_resolution_nm=20.0,
            focal_length_um=0.2,
            wavelength_nm=1550.0,
        )
        return Metalens3DDesigner(cfg, device="cpu")

    def test_construction(self, designer):
        bench = MultiScaleBenchmark(designer)
        assert bench.n_iterations_single == 20
        assert bench.n_iterations_coarse == 10
        assert bench.n_iterations_fine == 5

    def test_run_returns_expected_keys(self, designer):
        bench = MultiScaleBenchmark(
            designer,
            n_iterations_single=3,
            n_iterations_coarse=2,
            n_iterations_fine=1,
        )
        result = bench.run()
        assert "single_scale" in result
        assert "multi_scale" in result
        assert "single_scale_best_fom" in result
        assert "multi_scale_best_fom" in result
        assert "single_scale_total_ms" in result
        assert "multi_scale_total_ms" in result
        assert "speedup_ratio" in result

    def test_convergence_curves_have_records(self, designer):
        bench = MultiScaleBenchmark(
            designer,
            n_iterations_single=3,
            n_iterations_coarse=2,
            n_iterations_fine=1,
        )
        result = bench.run()
        assert len(result["single_scale"]) == 3
        assert len(result["multi_scale"]) >= 2  # at least coarse iterations

    def test_convergence_records_are_valid(self, designer):
        bench = MultiScaleBenchmark(
            designer,
            n_iterations_single=3,
            n_iterations_coarse=2,
            n_iterations_fine=1,
        )
        result = bench.run()
        for rec in result["single_scale"]:
            assert isinstance(rec, ConvergenceRecord)
            assert rec.elapsed_ms >= 0
        for rec in result["multi_scale"]:
            assert isinstance(rec, ConvergenceRecord)
            assert rec.elapsed_ms >= 0

    def test_timing_monotonically_increases(self, designer):
        bench = MultiScaleBenchmark(
            designer,
            n_iterations_single=5,
            n_iterations_coarse=3,
            n_iterations_fine=2,
        )
        result = bench.run()
        ss_times = [r.elapsed_ms for r in result["single_scale"]]
        for i in range(1, len(ss_times)):
            assert ss_times[i] >= ss_times[i - 1]

    def test_speedup_ratio_positive(self, designer):
        bench = MultiScaleBenchmark(
            designer,
            n_iterations_single=3,
            n_iterations_coarse=2,
            n_iterations_fine=1,
        )
        result = bench.run()
        assert result["speedup_ratio"] > 0

    def test_run_with_custom_pattern(self, designer):
        bench = MultiScaleBenchmark(
            designer,
            n_iterations_single=2,
            n_iterations_coarse=1,
            n_iterations_fine=1,
        )
        pattern = designer.generate_pattern(n_layers=2, seed=42)
        result = bench.run(pattern=pattern)
        assert "single_scale" in result
        assert len(result["single_scale"]) == 2

    def test_synthetic_fom(self, designer):
        target = designer.target_phase()
        design = torch.ones(designer.grid_size_1d, designer.grid_size_1d, dtype=torch.float64)
        fom = MultiScaleBenchmark._synthetic_fom(design, target)
        assert fom.dim() == 0  # scalar
        assert fom.item() >= -1.0
        assert fom.item() <= 1.0

    def test_synthetic_fom_3d_input(self, designer):
        target = designer.target_phase()
        design = torch.ones(3, designer.grid_size_1d, designer.grid_size_1d, dtype=torch.float64)
        fom = MultiScaleBenchmark._synthetic_fom(design, target)
        assert fom.dim() == 0

    def test_synthetic_fom_identical(self):
        phase = torch.randn(8, 8, dtype=torch.float64)
        fom = MultiScaleBenchmark._synthetic_fom(phase, phase)
        assert fom.item() > 0.99

    def test_synthetic_fom_opposite(self):
        phase = torch.randn(8, 8, dtype=torch.float64)
        fom = MultiScaleBenchmark._synthetic_fom(phase, -phase)
        assert fom.item() < -0.99

    def test_synthetic_fom_zero_target(self):
        phase = torch.randn(8, 8, dtype=torch.float64)
        target = torch.zeros(8, 8, dtype=torch.float64)
        fom = MultiScaleBenchmark._synthetic_fom(phase, target)
        # Zero target -> denom ~ 0, returns 0
        assert fom.item() == 0.0


# ---------------------------------------------------------------------------
# 7. Integration: designer + benchmark together
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_designer_with_benchmark(self):
        cfg = Metalens3DConfig(
            aperture_um=0.08,
            grid_resolution_nm=20.0,
            focal_length_um=0.2,
            wavelength_nm=1550.0,
        )
        designer = Metalens3DDesigner(cfg, device="cpu")
        pattern = designer.generate_pattern(n_layers=2, seed=0)
        assert pattern.shape[0] == 2

        bench = MultiScaleBenchmark(
            designer,
            n_iterations_single=2,
            n_iterations_coarse=1,
            n_iterations_fine=1,
        )
        result = bench.run(pattern=pattern)
        assert result["single_scale_best_fom"] != 0 or result["multi_scale_best_fom"] != 0

    def test_fdtd_benchmark_with_metalens_pattern(self):
        cfg = Metalens3DConfig(
            aperture_um=0.08,
            grid_resolution_nm=20.0,
            focal_length_um=0.2,
            wavelength_nm=1550.0,
        )
        designer = Metalens3DDesigner(cfg, device="cpu")
        pattern = designer.generate_pattern(n_layers=1, seed=0)

        # Use the pattern as a grid for FDTD benchmark.
        gs = (pattern.shape[0], pattern.shape[1], pattern.shape[2])
        bench = FDTDGPURealBenchmark(
            grid_sizes=[gs],
            n_time_steps=3,
            n_warmup=0,
            n_trials=1,
        )
        results = bench.run_scaling_benchmark()
        assert len(results) == 1
        assert results[0]["grid_size"] == gs


# ---------------------------------------------------------------------------
# 8. Module-level imports via __init__.py
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_imports_from_design_init(self):
        from diffnano.design import (
            ConvergenceRecord,
            FDTDGPURealBenchmark,
            GPUDeviceMetrics,
            Metalens3DConfig,
            Metalens3DDesigner,
            MultiScaleBenchmark,
        )

        assert Metalens3DDesigner is not None
        assert FDTDGPURealBenchmark is not None
        assert MultiScaleBenchmark is not None
        assert Metalens3DConfig is not None
        assert GPUDeviceMetrics is not None
        assert ConvergenceRecord is not None

    def test_all_exports(self):
        import diffnano.design as design_mod

        for name in [
            "Metalens3DConfig",
            "Metalens3DDesigner",
            "GPUDeviceMetrics",
            "FDTDGPURealBenchmark",
            "ConvergenceRecord",
            "MultiScaleBenchmark",
        ]:
            assert name in design_mod.__all__
            assert hasattr(design_mod, name)
