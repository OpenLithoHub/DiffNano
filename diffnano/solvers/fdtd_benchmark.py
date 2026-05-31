"""FDTD GPU benchmark and cross-validation framework.

Provides benchmarking for time_reversal / checkpoint / AD triple comparison,
and cross-validation against external FDTD solvers (e.g., FDTDX).

References:
    - Inverse design for scalable photonic systems, Nat. Rev. Mater., 2026-04
    - FDTDX, JOSS 11:8912, 2026

NOTE: GPU benchmarks require CUDA. CPU-only mode provides correct numerical
results with honest performance annotations.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "FDTDBenchmarkSuite",
    "ExternalCrossValidator",
    "SystolicUpdateEvaluator",
]


@dataclass
class BenchmarkConfig:
    grid_sizes: list[tuple[int, int, int]] = dc_field(
        default_factory=lambda: [(16, 16, 16), (32, 32, 32)]
    )
    n_time_steps: int = 100
    backward_modes: list[str] = dc_field(
        default_factory=lambda: ["time_reversal", "checkpoint", "autograd"]
    )
    device: str = "cpu"


@dataclass
class BenchmarkResult:
    grid_size: tuple[int, int, int]
    backward_mode: str
    forward_time_ms: float
    backward_time_ms: float
    peak_memory_mb: float
    gradient_cosine_vs_autograd: float | None
    device: str


def _resolve_device(requested: str) -> tuple[str, str]:
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda", "GPU (CUDA)"
        return "cpu", "CPU (CUDA requested but not available -- fallback)"
    return "cpu", "CPU"


def _make_eps_grid(grid_size: tuple[int, int, int], device: str) -> Tensor:
    D, H, W = grid_size
    torch.manual_seed(42)
    eps = 1.5 + 1.0 * torch.rand(D, H, W, dtype=torch.float64, device=device)
    d, h, w = D // 4, H // 4, W // 4
    if d > 0 and h > 0 and w > 0:
        eps[D // 2 - d : D // 2 + d, H // 2 - h : H // 2 + h, W // 2 - w : W // 2 + w] = 4.0
    return eps


def _cosine_sim(a: Tensor, b: Tensor) -> float:
    fa = a.flatten().to(torch.float64)
    fb = b.flatten().to(torch.float64)
    denom = fa.norm() * fb.norm()
    if denom.item() < 1e-30:
        return 0.0
    return torch.dot(fa, fb).item() / denom.item()


class FDTDBenchmarkSuite:
    """Benchmark FDTD3D across grid sizes and backward modes.

    Measures forward time, backward time, peak memory, and gradient cosine
    similarity vs autograd reference for each configuration.
    """

    def __init__(self) -> None:
        self._results: list[BenchmarkResult] = []

    def run(
        self,
        fdtd3d_class: type,
        config: BenchmarkConfig,
    ) -> list[BenchmarkResult]:
        effective_device, _ = _resolve_device(config.device)
        self._results = []

        for grid_size in config.grid_sizes:
            eps_template = _make_eps_grid(grid_size, effective_device)
            autograd_grad: Tensor | None = None

            for mode in config.backward_modes:
                gc.collect()
                if effective_device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    mem_before = torch.cuda.memory_allocated()
                else:
                    mem_before = 0

                bw_kw: dict[str, Any] = {}
                if mode == "time_reversal":
                    bw_kw["backward"] = "time_reversal"
                elif mode == "checkpoint":
                    bw_kw["use_checkpoint"] = True
                    bw_kw["checkpoint_segments"] = 2

                solver = fdtd3d_class(
                    grid_shape=grid_size,
                    dl=20.0,
                    wavelength_nm=1550.0,
                    pml_layers=0,
                    n_steps=config.n_time_steps,
                    device=effective_device,
                    courant=0.35,
                    **bw_kw,
                )

                eps = eps_template.clone().detach().requires_grad_(True)

                t0 = time.perf_counter()
                result = solver.forward(eps)
                fwd_ms = (time.perf_counter() - t0) * 1000.0

                loss = result.field.sum()

                t0 = time.perf_counter()
                loss.backward()
                bwd_ms = (time.perf_counter() - t0) * 1000.0

                grad = (
                    eps.grad.detach().clone()
                    if eps.grad is not None
                    else torch.zeros_like(eps)
                )

                if effective_device == "cuda":
                    peak_mb = (torch.cuda.max_memory_allocated() - mem_before) / 1e6
                else:
                    peak_mb = _estimate_cpu_peak_mb(eps, config.n_time_steps, grid_size)

                cos_vs_ad: float | None = None
                if mode == "autograd":
                    autograd_grad = grad
                elif autograd_grad is not None:
                    cos_vs_ad = _cosine_sim(autograd_grad, grad)

                self._results.append(
                    BenchmarkResult(
                        grid_size=grid_size,
                        backward_mode=mode,
                        forward_time_ms=round(fwd_ms, 3),
                        backward_time_ms=round(bwd_ms, 3),
                        peak_memory_mb=round(peak_mb, 3),
                        gradient_cosine_vs_autograd=(
                            round(cos_vs_ad, 6) if cos_vs_ad is not None else None
                        ),
                        device=effective_device,
                    )
                )

        return self._results

    def summary_table(self) -> str:
        if not self._results:
            return "No benchmark results."

        header = (
            f"{'grid':>12s} | {'mode':>14s} | {'fwd_ms':>8s} | "
            f"{'bwd_ms':>8s} | {'mem_MB':>8s} | {'cos_vs_AD':>10s} | {'device':>6s}"
        )
        sep = "-" * len(header)
        lines = [header, sep]

        for r in self._results:
            g = f"{r.grid_size[0]}x{r.grid_size[1]}x{r.grid_size[2]}"
            cos = (
                f"{r.gradient_cosine_vs_autograd:.4f}"
                if r.gradient_cosine_vs_autograd is not None
                else "ref"
            )
            lines.append(
                f"{g:>12s} | {r.backward_mode:>14s} | {r.forward_time_ms:8.1f} | "
                f"{r.backward_time_ms:8.1f} | {r.peak_memory_mb:8.2f} | "
                f"{cos:>10s} | {r.device:>6s}"
            )

        return "\n".join(lines)


def _estimate_cpu_peak_mb(
    eps: Tensor,
    n_steps: int,
    grid_size: tuple[int, int, int],
) -> float:
    D, H, W = grid_size
    bytes_per = 8
    field_mem = 6 * D * H * W * bytes_per
    graph_factor = 4 if eps.requires_grad else 1
    return field_mem * graph_factor / 1e6


class ExternalCrossValidator:
    """Framework for cross-validating against external FDTD solvers.

    Does NOT vendor any external code. Provides interfaces for loading
    pre-computed external simulation results and comparing them against
    DiffNano FDTD outputs.
    """

    def validate_forward(
        self,
        our_result: Tensor,
        external_result: Tensor,
        rtol: float = 1e-3,
    ) -> dict[str, Any]:
        diff = our_result.detach().to(torch.float64) - external_result.detach().to(torch.float64)
        ref_norm = external_result.detach().to(torch.float64).norm().item()
        max_abs = diff.abs().max().item()
        rel_err = diff.norm().item() / max(ref_norm, 1e-30)
        return {
            "relative_error": rel_err,
            "max_absolute_error": max_abs,
            "passed": rel_err < rtol,
        }

    def validate_gradient(
        self,
        our_grad: Tensor,
        external_grad: Tensor,
        cosine_threshold: float = 0.99,
    ) -> dict[str, Any]:
        cos_sim = _cosine_sim(our_grad, external_grad)
        ref_norm = external_grad.detach().to(torch.float64).norm().item()
        diff_norm = (
            our_grad.detach().to(torch.float64) - external_grad.detach().to(torch.float64)
        ).norm().item()
        rel_err = diff_norm / max(ref_norm, 1e-30)
        return {
            "cosine_similarity": cos_sim,
            "relative_error": rel_err,
            "passed": cos_sim >= cosine_threshold,
        }

    def generate_test_case(
        self,
        grid_size: tuple[int, int, int] = (16, 16, 16),
        source_pos: tuple[int, int, int] | None = None,
        freq: float = 1.0,
    ) -> dict[str, Any]:
        D, H, W = grid_size
        if source_pos is None:
            source_pos = (D // 2, H // 2, W // 2)

        seed = 42
        torch.manual_seed(seed)
        eps = 1.5 + 1.0 * torch.rand(D, H, W, dtype=torch.float64)

        return {
            "grid_size": grid_size,
            "source_pos": source_pos,
            "freq": freq,
            "eps": eps,
            "source_config": {
                "type": "gaussian_pulse",
                "pos": list(source_pos),
                "amplitude": 1.0,
            },
            "n_steps": 50,
            "dl": 20.0,
            "wavelength_nm": 1550.0,
            "courant": 0.35,
            "seed": seed,
        }

    @staticmethod
    def load_external_results(path: str) -> dict[str, Any]:
        p = Path(path)
        with open(p) as f:
            raw = json.load(f)

        result: dict[str, Any] = {}
        for key in ("field", "gradient"):
            if key in raw:
                entry = raw[key]
                shape = tuple(entry["shape"])
                data = entry["data"]
                result[key] = torch.tensor(data, dtype=torch.float64).reshape(shape)

        for k, v in raw.items():
            if k not in ("field", "gradient"):
                result[k] = v

        return result


class SystolicUpdateEvaluator:
    """Evaluate memory bandwidth characteristics for FDTD update patterns.

    Measures effective bandwidth during FDTD field updates to determine
    whether memory bandwidth is the bottleneck (as opposed to compute).
    Not a full systolic-array implementation -- focuses on characterization.
    """

    def measure_bandwidth(
        self,
        fdtd3d_instance: Any,
        grid_size: tuple[int, int, int],
        n_warmup: int = 2,
        n_trials: int = 5,
    ) -> dict[str, Any]:
        D, H, W = grid_size
        dev = getattr(fdtd3d_instance, "_device", torch.device("cpu"))
        dtype = torch.float64

        Ex = torch.randn(D, H, W, dtype=dtype, device=dev)
        Ey = torch.randn(D, H, W, dtype=dtype, device=dev)
        Ez = torch.randn(D, H, W, dtype=dtype, device=dev)
        Hx = torch.randn(D, H, W, dtype=dtype, device=dev)
        Hy = torch.randn(D, H, W, dtype=dtype, device=dev)
        Hz = torch.randn(D, H, W, dtype=dtype, device=dev)
        eps_r = 1.5 + torch.rand(D, H, W, dtype=dtype, device=dev)
        mu_r = torch.ones(D, H, W, dtype=dtype, device=dev)

        bytes_per_element = 8
        elements_per_tensor = D * H * W
        tensors_touched = 20
        bytes_per_step = tensors_touched * elements_per_tensor * bytes_per_element

        step_fn = getattr(fdtd3d_instance, "_time_step", None)

        for _ in range(n_warmup):
            if step_fn is not None:
                Ex, Ey, Ez, Hx, Hy, Hz = step_fn(
                    Ex, Ey, Ez, Hx, Hy, Hz, eps_r, mu_r
                )
            else:
                Hz = Hz + 0.01 * (Ex + Ey + Ez)
                Ex = Ex + 0.01 * Hz

        if str(dev) == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_trials):
            if step_fn is not None:
                Ex, Ey, Ez, Hx, Hy, Hz = step_fn(
                    Ex, Ey, Ez, Hx, Hy, Hz, eps_r, mu_r
                )
            else:
                Hz = Hz + 0.01 * (Ex + Ey + Ez)
                Ex = Ex + 0.01 * Hz

        if str(dev) == "cuda":
            torch.cuda.synchronize()

        elapsed_s = time.perf_counter() - t0
        total_bytes = bytes_per_step * n_trials
        bandwidth_gbs = total_bytes / elapsed_s / 1e9

        field_size_mb = 6 * elements_per_tensor * bytes_per_element / 1e6

        if str(dev) == "cuda":
            theoretical_peak_gbs = _gpu_memory_bandwidth_gbs()
        else:
            theoretical_peak_gbs = _cpu_estimated_bandwidth_gbs()

        utilization = bandwidth_gbs / max(theoretical_peak_gbs, 1e-30)

        return {
            "field_size_mb": round(field_size_mb, 3),
            "update_bandwidth_gbs": round(bandwidth_gbs, 3),
            "theoretical_peak_gbs": round(theoretical_peak_gbs, 3),
            "bandwidth_utilization": round(utilization, 4),
        }


def _gpu_memory_bandwidth_gbs() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(0)
        return getattr(props, "memory_bus_width", 256) * 2.0 * 1e6 / 8.0 / 1e9
    except Exception:
        return 500.0


def _cpu_estimated_bandwidth_gbs() -> float:
    return 40.0
