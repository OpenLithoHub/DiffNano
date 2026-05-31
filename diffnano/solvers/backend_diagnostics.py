"""RCWA backend diagnostics: uncertainty quantification and benchmark table.

Provides tools for comparing the four RCWA propagation backends (eig, eig_expm,
matrix_sqrt, rdit) across a range of operating conditions, computing forward
accuracy, gradient fidelity, and valid operating regimes.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Sequence

import torch

from diffnano.solvers.rcwa import RCWASolver

__all__ = [
    "BackendDiagnostics",
    "BackendBenchmarkTable",
    "generate_operating_regime_table",
]

_ALL_BACKENDS = ("eig", "eig_expm", "matrix_sqrt", "rdit")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_solver(
    backend: str,
    fourier_orders: int = 5,
    wavelength_nm: float = 532.0,
    period_nm: tuple[float, float] = (400.0, 400.0),
    taylor_order: int = 5,
) -> RCWASolver:
    return RCWASolver(
        fourier_orders=fourier_orders,
        wavelength_nm=wavelength_nm,
        period_nm=period_nm,
        solver_backend=backend,
        taylor_order=taylor_order,
    )


def _make_geometry(
    n_layers: int = 3,
    n_grid: int = 80,
    eps_high: float = 4.0,
    loss_tangent: float = 0.0,
) -> torch.Tensor:
    """Create a test grating with optional loss."""
    x = torch.linspace(0, 2 * torch.pi, n_grid, dtype=torch.float64)
    eps_real = 1.0 + (eps_high - 1.0) * 0.5 * (1.0 + torch.cos(2 * x))
    if loss_tangent > 0:
        eps_imag = torch.full_like(eps_real, eps_real.mean().item() * loss_tangent)
        return torch.complex(eps_real, eps_imag).unsqueeze(0).expand(n_layers, -1).clone()
    return eps_real.unsqueeze(0).expand(n_layers, -1).clone()


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flattened real tensors."""
    fa = a.real.flatten().to(torch.float64)
    fb = b.real.flatten().to(torch.float64)
    denom = fa.norm() * fb.norm()
    if denom < 1e-30:
        return 0.0
    return torch.dot(fa, fb).item() / denom.item()


# ---------------------------------------------------------------------------
# BackendDiagnostics
# ---------------------------------------------------------------------------

class BackendDiagnostics:
    """Diagnostic tool for RCWA backend selection and uncertainty quantification.

    For each backend, computes:
    - Forward accuracy vs reference (eig backend)
    - Gradient fidelity (cosine similarity vs eig gradients)
    - Valid operating regime (d/lambda, orders, loss ranges where backend is reliable)
    """

    def __init__(
        self,
        backends: Sequence[str] | None = None,
        reference_backend: str = "eig",
    ):
        if backends is None:
            backends = list(_ALL_BACKENDS)
        for b in backends:
            if b not in _ALL_BACKENDS:
                raise ValueError(f"Unknown backend {b!r}; choose from {_ALL_BACKENDS}")
        if reference_backend not in _ALL_BACKENDS:
            raise ValueError(f"Unknown reference backend {reference_backend!r}")
        self.backends = list(backends)
        self.reference_backend = reference_backend

    # ------------------------------------------------------------------
    # Core diagnostic
    # ------------------------------------------------------------------

    def diagnose(self, test_configs: Sequence[dict]) -> dict:
        """Run diagnostics on a range of test configurations.

        Parameters
        ----------
        test_configs : list of dict
            Each dict may contain:
            - d_over_lambda (float): layer thickness / wavelength ratio
            - n_orders (int): Fourier orders retained
            - loss_tangent (float): imaginary eps fraction
            - thickness_nm (float): layer thickness in nm (overrides d_over_lambda)

        Returns
        -------
        dict
            Per-backend results keyed by backend name.  Each value is a dict:
            - forward_error: relative L2 error vs reference (mean across configs)
            - gradient_cosine: cosine similarity with reference gradients (mean)
            - forward_errors: list of per-config errors
            - gradient_cosines: list of per-config cosines
            - residual_band: max residual across test cases
            - regimes: description of valid operating regime
        """
        results: dict[str, dict] = {}

        # Compute reference results
        ref_forward = []
        ref_grads = []
        for cfg in test_configs:
            solver, geometry, wl, thickness = self._config_to_solver(cfg, self.reference_backend)
            geo = geometry.detach().requires_grad_(True)
            out = solver.forward(geo, wavelengths=[wl], source={"thickness_nm": thickness})
            loss = out.field[:, solver.fourier_orders].sum()
            loss.backward()
            ref_forward.append(out.field.detach().clone())
            ref_grads.append(geo.grad.detach().clone() if geo.grad is not None else torch.zeros_like(geo))

        # Evaluate each backend vs reference
        for backend in self.backends:
            if backend == self.reference_backend:
                results[backend] = {
                    "forward_error": 0.0,
                    "gradient_cosine": 1.0,
                    "forward_errors": [0.0] * len(test_configs),
                    "gradient_cosines": [1.0] * len(test_configs),
                    "residual_band": 0.0,
                    "regimes": "reference backend",
                }
                continue

            fwd_errors = []
            grad_cosines = []

            for i, cfg in enumerate(test_configs):
                solver, geometry, wl, thickness = self._config_to_solver(cfg, backend)
                geo = geometry.detach().requires_grad_(True)
                out = solver.forward(geo, wavelengths=[wl], source={"thickness_nm": thickness})
                loss = out.field[:, solver.fourier_orders].sum()
                loss.backward()

                # Forward error: relative L2 of sorted efficiencies
                ref_eff = ref_forward[i].sort().values
                test_eff = out.field.detach().sort().values
                denom = ref_eff.norm().item()
                fwd_err = (ref_eff - test_eff).norm().item() / max(denom, 1e-30)
                fwd_errors.append(fwd_err)

                # Gradient cosine similarity
                grad_test = geo.grad.detach().clone() if geo.grad is not None else torch.zeros_like(geo)
                grad_ref = ref_grads[i]
                cos = _cosine_similarity(grad_ref, grad_test)
                grad_cosines.append(cos)

            mean_fwd = sum(fwd_errors) / len(fwd_errors)
            mean_cos = sum(grad_cosines) / len(grad_cosines)
            residual_band = max(fwd_errors)

            results[backend] = {
                "forward_error": mean_fwd,
                "gradient_cosine": mean_cos,
                "forward_errors": fwd_errors,
                "gradient_cosines": grad_cosines,
                "residual_band": residual_band,
                "regimes": self._describe_regime(backend, fwd_errors, grad_cosines, test_configs),
            }

        return results

    # ------------------------------------------------------------------
    # Backend recommendation
    # ------------------------------------------------------------------

    def recommend_backend(
        self,
        d_over_lambda: float,
        n_orders: int,
        loss_tangent: float,
    ) -> dict:
        """Recommend best backend for given operating conditions.

        Returns
        -------
        dict
            - backend: recommended backend name
            - confidence: confidence score (0-1)
            - fallback: fallback backend if primary fails
            - residual_estimate: estimated accuracy (1 - expected forward error)
        """
        # Decision logic based on known backend characteristics:
        # - rdit: thin layers (d/lambda < 0.1), fast but less accurate for thick
        # - matrix_sqrt: general purpose, stable for thick/degenerate
        # - eig/eig_expm: baseline, may have gradient instability at degeneracies
        if d_over_lambda < 0.1:
            primary = "rdit"
            confidence = max(0.5, 1.0 - d_over_lambda * 3)
            fallback = "matrix_sqrt"
            residual_estimate = max(0.85, 1.0 - d_over_lambda * 5)
        elif d_over_lambda < 0.5:
            primary = "matrix_sqrt"
            confidence = 0.9
            fallback = "eig_expm"
            residual_estimate = 0.95
        else:
            primary = "matrix_sqrt"
            confidence = 0.85
            fallback = "eig"
            residual_estimate = 0.9

        # Loss tangent adjustments
        if loss_tangent > 0.5:
            confidence *= 0.9
            residual_estimate *= 0.95

        # High order count may reduce rdit accuracy
        if primary == "rdit" and n_orders > 8:
            confidence *= 0.85
            primary = "matrix_sqrt"
            fallback = "rdit"

        confidence = min(max(confidence, 0.0), 1.0)
        residual_estimate = min(max(residual_estimate, 0.0), 1.0)

        return {
            "backend": primary,
            "confidence": round(confidence, 3),
            "fallback": fallback,
            "residual_estimate": round(residual_estimate, 3),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _config_to_solver(
        cfg: dict,
        backend: str,
    ) -> tuple[RCWASolver, torch.Tensor, float, float]:
        """Convert a test config dict into (solver, geometry, wavelength, thickness)."""
        wavelength_nm = 532.0
        period_nm = (400.0, 400.0)
        n_orders = cfg.get("n_orders", 5)
        n_grid = 2 * n_orders + 1 + 10  # slightly more grid points than Fourier terms
        d_over_lambda = cfg.get("d_over_lambda", 0.1)
        loss_tangent = cfg.get("loss_tangent", 0.0)
        thickness = cfg.get("thickness_nm", d_over_lambda * wavelength_nm)

        solver = _make_solver(
            backend=backend,
            fourier_orders=n_orders,
            wavelength_nm=wavelength_nm,
            period_nm=period_nm,
        )
        geometry = _make_geometry(
            n_layers=3,
            n_grid=n_grid,
            loss_tangent=loss_tangent,
        )
        return solver, geometry, wavelength_nm, thickness

    @staticmethod
    def _describe_regime(
        backend: str,
        fwd_errors: list[float],
        grad_cosines: list[float],
        configs: Sequence[dict],
    ) -> str:
        """Describe the valid operating regime based on per-config diagnostics."""
        # Find which configs had acceptable accuracy
        tol = 0.15
        good_configs = [c for c, e in zip(configs, fwd_errors) if e < tol]

        if not good_configs:
            return f"{backend}: no valid regime found (all errors >= {tol})"

        d_vals = [c.get("d_over_lambda", 0.1) for c in good_configs]
        loss_vals = [c.get("loss_tangent", 0.0) for c in good_configs]

        min_d = min(d_vals)
        max_d = max(d_vals)
        max_loss = max(loss_vals)

        return (
            f"{backend}: valid for d/lambda in [{min_d:.2f}, {max_d:.2f}], "
            f"loss_tangent <= {max_loss:.2f}"
        )


# ---------------------------------------------------------------------------
# BackendBenchmarkTable
# ---------------------------------------------------------------------------

class BackendBenchmarkTable:
    """Generate benchmark table for RCWA backends.

    Measures forward time, backward time, memory, and accuracy
    for each backend on a grid of configurations.
    """

    def __init__(self, solver_factory=None):
        """Parameters
        ----------
        solver_factory : callable, optional
            If provided, called as ``solver_factory(backend, **kwargs)`` to
            create solver instances.  Defaults to :func:`_make_solver`.
        """
        self.solver_factory = solver_factory or _make_solver
        self._results: list[dict] = []

    def run(
        self,
        configs: Sequence[dict],
        device: str = "cpu",
    ) -> list[dict]:
        """Run benchmarks across configurations and backends.

        Parameters
        ----------
        configs : list of dict
            Each dict may contain: n_orders, d_over_lambda, loss_tangent,
            thickness_nm, wavelength_nm, period_x, period_y, n_layers, n_grid.
        device : str
            ``"cpu"`` or ``"cuda"``.

        Returns
        -------
        list of dict
            Each dict: backend, config_idx, forward_time_ms, backward_time_ms,
            peak_memory_mb, forward_error, gradient_cosine.
        """
        self._results = []
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        for ci, cfg in enumerate(configs):
            n_orders = cfg.get("n_orders", 5)
            wl = cfg.get("wavelength_nm", 532.0)
            period = (cfg.get("period_x", 400.0), cfg.get("period_y", 400.0))
            d_over_lambda = cfg.get("d_over_lambda", 0.1)
            thickness = cfg.get("thickness_nm", d_over_lambda * wl)
            loss_tangent = cfg.get("loss_tangent", 0.0)
            n_grid = cfg.get("n_grid", 2 * n_orders + 11)
            n_layers = cfg.get("n_layers", 3)

            # Reference solver
            ref_solver = self.solver_factory("eig", fourier_orders=n_orders,
                                             wavelength_nm=wl, period_nm=period)
            geo_template = _make_geometry(n_layers=n_layers, n_grid=n_grid,
                                          loss_tangent=loss_tangent)

            # Reference forward + backward
            geo_ref = geo_template.detach().clone().requires_grad_(True)
            t0 = time.perf_counter()
            out_ref = ref_solver.forward(geo_ref, wavelengths=[wl],
                                         source={"thickness_nm": thickness})
            t_fwd_ref = time.perf_counter() - t0
            loss_ref = out_ref.field[:, ref_solver.fourier_orders].sum()
            t0 = time.perf_counter()
            loss_ref.backward()
            t_bwd_ref = time.perf_counter() - t0
            ref_eff = out_ref.field.detach().sort().values
            ref_grad = geo_ref.grad.detach().clone() if geo_ref.grad is not None else None

            for backend in _ALL_BACKENDS:
                solver = self.solver_factory(backend, fourier_orders=n_orders,
                                             wavelength_nm=wl, period_nm=period)
                geo = geo_template.detach().clone().requires_grad_(True)

                # Measure memory before
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    mem_before = torch.cuda.memory_allocated()
                else:
                    mem_before = 0

                # Forward
                gc.collect()
                t0 = time.perf_counter()
                out = solver.forward(geo, wavelengths=[wl],
                                     source={"thickness_nm": thickness})
                fwd_time = (time.perf_counter() - t0) * 1000  # ms

                # Backward
                loss = out.field[:, solver.fourier_orders].sum()
                t0 = time.perf_counter()
                loss.backward()
                bwd_time = (time.perf_counter() - t0) * 1000  # ms

                # Peak memory
                if device == "cuda" and torch.cuda.is_available():
                    peak_mem = (torch.cuda.max_memory_allocated() - mem_before) / 1e6  # MB
                else:
                    # Estimate from tensor sizes (rough)
                    peak_mem = geo.element_size() * geo.numel() * 4 / 1e6  # forward+backward copies

                # Forward error
                test_eff = out.field.detach().sort().values
                denom = ref_eff.norm().item()
                fwd_err = (ref_eff - test_eff).norm().item() / max(denom, 1e-30)

                # Gradient cosine
                grad_test = geo.grad.detach() if geo.grad is not None else None
                if ref_grad is not None and grad_test is not None:
                    grad_cos = _cosine_similarity(ref_grad, grad_test)
                else:
                    grad_cos = float("nan")

                self._results.append({
                    "backend": backend,
                    "config_idx": ci,
                    "forward_time_ms": round(fwd_time, 3),
                    "backward_time_ms": round(bwd_time, 3),
                    "peak_memory_mb": round(peak_mem, 4),
                    "forward_error": round(fwd_err, 6),
                    "gradient_cosine": round(grad_cos, 6),
                })

        return self._results

    def to_markdown_table(self) -> str:
        """Format results as markdown table."""
        if not self._results:
            return "| backend | config | fwd_ms | bwd_ms | mem_MB | fwd_err | grad_cos |\n|---|---|---|---|---|---|---|\n"

        header = "| backend | config | fwd_ms | bwd_ms | mem_MB | fwd_err | grad_cos |"
        sep = "|---|---|---|---|---|---|---|"
        rows = [header, sep]

        for r in self._results:
            rows.append(
                f"| {r['backend']} | {r['config_idx']} | "
                f"{r['forward_time_ms']:.1f} | {r['backward_time_ms']:.1f} | "
                f"{r['peak_memory_mb']:.2f} | {r['forward_error']:.4f} | "
                f"{r['gradient_cosine']:.4f} |"
            )
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Operating regime table
# ---------------------------------------------------------------------------

def generate_operating_regime_table() -> list[dict]:
    """Generate the canonical operating regime table for all backends.

    Returns
    -------
    list of dict
        Each dict: backend, d_over_lambda_range, max_orders, loss_range,
        accuracy_estimate, failure_boundary.
    """
    return [
        {
            "backend": "eig",
            "d_over_lambda_range": (0.0, 10.0),
            "max_orders": 20,
            "loss_range": (0.0, 1.0),
            "accuracy_estimate": 1.0,
            "failure_boundary": "gradient instability at eigenvalue degeneracies",
        },
        {
            "backend": "eig_expm",
            "d_over_lambda_range": (0.0, 10.0),
            "max_orders": 20,
            "loss_range": (0.0, 1.0),
            "accuracy_estimate": 0.999,
            "failure_boundary": "gradient instability at eigenvalue degeneracies (same as eig)",
        },
        {
            "backend": "matrix_sqrt",
            "d_over_lambda_range": (0.0, 10.0),
            "max_orders": 20,
            "loss_range": (0.0, 1.0),
            "accuracy_estimate": 0.95,
            "failure_boundary": "near-singular P matrix (min sv < degen_tol)",
        },
        {
            "backend": "rdit",
            "d_over_lambda_range": (0.0, 0.5),
            "max_orders": 10,
            "loss_range": (0.0, 0.3),
            "accuracy_estimate": 0.9,
            "failure_boundary": "d/lambda > 0.5 (Taylor expansion diverges)",
        },
    ]
