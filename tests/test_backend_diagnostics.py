"""Tests for RCWA backend diagnostics: uncertainty, benchmarks, regime tables."""

import pytest
import torch

from diffnano.solvers.backend_diagnostics import (
    BackendBenchmarkTable,
    BackendDiagnostics,
    generate_operating_regime_table,
)

# ---------------------------------------------------------------------------
# Minimal test configs to keep runtime short
# ---------------------------------------------------------------------------

_THIN_CFG = {"d_over_lambda": 0.05, "n_orders": 3, "loss_tangent": 0.0}
_MEDIUM_CFG = {"d_over_lambda": 0.3, "n_orders": 3, "loss_tangent": 0.0}
_LOSSY_CFG = {"d_over_lambda": 0.1, "n_orders": 3, "loss_tangent": 0.2}

_MINIMAL_CONFIGS = [_THIN_CFG, _MEDIUM_CFG, _LOSSY_CFG]


# ---------------------------------------------------------------------------
# Test 1: diagnose runs on a small config set
# ---------------------------------------------------------------------------


class TestDiagnosticsRuns:
    """BackendDiagnostics.diagnose completes on representative configs."""

    def test_diagnostics_runs_on_configs(self):
        """diagnose() returns a result dict with the expected keys for each backend."""
        diag = BackendDiagnostics()
        results = diag.diagnose(_MINIMAL_CONFIGS)

        for backend in ("eig", "eig_expm", "matrix_sqrt", "rdit"):
            assert backend in results, f"missing backend {backend}"
            entry = results[backend]
            for key in (
                "forward_error",
                "gradient_cosine",
                "forward_errors",
                "gradient_cosines",
                "residual_band",
                "regimes",
            ):
                assert key in entry, f"missing key {key} for {backend}"

    def test_reference_backend_has_zero_error(self):
        """Reference backend reports zero forward error and cosine = 1."""
        diag = BackendDiagnostics(reference_backend="eig")
        results = diag.diagnose([_THIN_CFG])
        eig = results["eig"]
        assert eig["forward_error"] == 0.0
        assert eig["gradient_cosine"] == 1.0

    def test_forward_errors_are_non_negative(self):
        """All forward errors must be >= 0."""
        diag = BackendDiagnostics()
        results = diag.diagnose(_MINIMAL_CONFIGS)
        for backend, entry in results.items():
            for err in entry["forward_errors"]:
                assert err >= 0, f"negative forward error for {backend}"

    def test_gradient_cosines_in_range(self):
        """Gradient cosine similarities must be in [-1, 1] or NaN."""
        diag = BackendDiagnostics()
        results = diag.diagnose(_MINIMAL_CONFIGS)
        for backend, entry in results.items():
            for cos in entry["gradient_cosines"]:
                if not torch.isnan(torch.tensor(cos)):
                    assert -1.0 <= cos <= 1.0, f"cosine {cos} out of range for {backend}"

    def test_residual_band_is_max_error(self):
        """residual_band equals the max of forward_errors."""
        diag = BackendDiagnostics()
        results = diag.diagnose(_MINIMAL_CONFIGS)
        for backend, entry in results.items():
            if backend == diag.reference_backend:
                continue
            assert abs(entry["residual_band"] - max(entry["forward_errors"])) < 1e-10


# ---------------------------------------------------------------------------
# Test 2: recommend_backend returns valid structure
# ---------------------------------------------------------------------------


class TestRecommendBackend:
    """BackendDiagnostics.recommend_backend returns a valid recommendation."""

    @pytest.mark.parametrize(
        "d_over_lambda,n_orders,loss_tangent",
        [
            (0.02, 3, 0.0),  # thin, low orders
            (0.3, 5, 0.0),  # medium
            (1.0, 8, 0.1),  # thick, some loss
            (0.05, 10, 0.5),  # thin but high orders + lossy
        ],
    )
    def test_recommend_backend_returns_valid(self, d_over_lambda, n_orders, loss_tangent):
        """recommend_backend returns all required fields with valid values."""
        diag = BackendDiagnostics()
        rec = diag.recommend_backend(d_over_lambda, n_orders, loss_tangent)

        assert "backend" in rec
        assert "confidence" in rec
        assert "fallback" in rec
        assert "residual_estimate" in rec

        assert rec["backend"] in ("eig", "eig_expm", "matrix_sqrt", "rdit")
        assert 0.0 <= rec["confidence"] <= 1.0
        assert 0.0 <= rec["residual_estimate"] <= 1.0
        assert rec["fallback"] in ("eig", "eig_expm", "matrix_sqrt", "rdit")

    def test_thin_layer_recommends_rdit(self):
        """Very thin layers should prefer rdit."""
        diag = BackendDiagnostics()
        rec = diag.recommend_backend(0.02, 3, 0.0)
        assert rec["backend"] == "rdit"

    def test_thick_layer_recommends_matrix_sqrt(self):
        """Thick layers should prefer matrix_sqrt."""
        diag = BackendDiagnostics()
        rec = diag.recommend_backend(2.0, 5, 0.0)
        assert rec["backend"] == "matrix_sqrt"


# ---------------------------------------------------------------------------
# Test 3: benchmark table structure
# ---------------------------------------------------------------------------


class TestBenchmarkTable:
    """BackendBenchmarkTable.run produces correctly structured results."""

    def test_benchmark_table_structure(self):
        """run() returns list of dicts with all expected keys."""
        bench = BackendBenchmarkTable()
        results = bench.run([_THIN_CFG], device="cpu")

        assert isinstance(results, list)
        assert len(results) == 4  # one per backend

        required_keys = {
            "backend",
            "config_idx",
            "forward_time_ms",
            "backward_time_ms",
            "peak_memory_mb",
            "forward_error",
            "gradient_cosine",
        }
        for r in results:
            assert required_keys.issubset(r.keys()), f"missing keys: {required_keys - r.keys()}"
            assert r["backend"] in ("eig", "eig_expm", "matrix_sqrt", "rdit")
            assert isinstance(r["forward_time_ms"], float)
            assert r["forward_time_ms"] >= 0
            assert r["backward_time_ms"] >= 0

    def test_benchmark_multiple_configs(self):
        """Benchmark with multiple configs produces correct number of rows."""
        bench = BackendBenchmarkTable()
        results = bench.run([_THIN_CFG, _MEDIUM_CFG], device="cpu")
        assert len(results) == 2 * 4  # 2 configs * 4 backends

    def test_benchmark_reference_error_zero(self):
        """The eig backend should have near-zero forward error vs itself."""
        bench = BackendBenchmarkTable()
        results = bench.run([_THIN_CFG], device="cpu")
        eig_row = [r for r in results if r["backend"] == "eig"][0]
        assert eig_row["forward_error"] < 1e-10


# ---------------------------------------------------------------------------
# Test 4: operating regime table is complete
# ---------------------------------------------------------------------------


class TestRegimeTable:
    """generate_operating_regime_table returns complete data."""

    def test_regime_table_complete(self):
        """Regime table has entries for all 4 backends."""
        table = generate_operating_regime_table()

        assert isinstance(table, list)
        assert len(table) == 4

        backends_in_table = {e["backend"] for e in table}
        assert backends_in_table == {"eig", "eig_expm", "matrix_sqrt", "rdit"}

        required_keys = {
            "backend",
            "d_over_lambda_range",
            "max_orders",
            "loss_range",
            "accuracy_estimate",
            "failure_boundary",
        }
        for entry in table:
            assert required_keys.issubset(entry.keys()), (
                f"missing keys in {entry['backend']}: {required_keys - entry.keys()}"
            )

    def test_rdit_has_limited_d_over_lambda(self):
        """rdit should have a narrower d/lambda range than other backends."""
        table = generate_operating_regime_table()
        rdit = next(e for e in table if e["backend"] == "rdit")
        eig = next(e for e in table if e["backend"] == "eig")

        rdit_max = rdit["d_over_lambda_range"][1]
        eig_max = eig["d_over_lambda_range"][1]
        assert rdit_max < eig_max, "rdit should have more limited d/lambda range"

    def test_matrix_sqrt_accuracy_high(self):
        """matrix_sqrt should have accuracy >= 0.9."""
        table = generate_operating_regime_table()
        msqrt = next(e for e in table if e["backend"] == "matrix_sqrt")
        assert msqrt["accuracy_estimate"] >= 0.9


# ---------------------------------------------------------------------------
# Test 5: markdown table format
# ---------------------------------------------------------------------------


class TestMarkdownTable:
    """BackendBenchmarkTable.to_markdown_table produces valid markdown."""

    def test_markdown_table_format(self):
        """to_markdown_table returns a properly formatted markdown table."""
        bench = BackendBenchmarkTable()
        bench.run([_THIN_CFG], device="cpu")
        md = bench.to_markdown_table()

        assert isinstance(md, str)
        lines = md.strip().split("\n")
        assert len(lines) >= 3  # header, separator, at least 1 data row

        # Header row should contain column names
        assert "backend" in lines[0]
        assert "fwd_ms" in lines[0]
        assert "fwd_err" in lines[0]

        # Separator row
        assert "---" in lines[1]

        # Data rows should have pipe-separated values
        for data_line in lines[2:]:
            parts = data_line.split("|")
            # Each row starts and ends with |, so split gives empty first/last
            assert len(parts) >= 8  # empty + 7 columns + empty

    def test_markdown_empty_before_run(self):
        """to_markdown_table before run() still returns a valid table."""
        bench = BackendBenchmarkTable()
        md = bench.to_markdown_table()
        assert "backend" in md
        assert "---" in md
