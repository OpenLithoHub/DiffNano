"""Benchmark comparing eig vs matrix_exp RCWA propagation backends.

Measures:
  (a) Forward accuracy: max absolute difference between backends
  (b) Backward gradient NaN rate: how often gradients contain NaN/Inf
  (c) Wall time: forward + backward time per configuration

Usage:
    PYTHONPATH=. python3 scripts/benchmark_rcwa_backends.py
"""

import sys
import time

import torch

sys.path.insert(0, ".")
from diffnano.solvers import RCWASolver


def _make_grating(
    n_layers: int,
    n_grid: int,
    eps_mean: float,
    eps_imag: float = 0.0,
    seed: int = 42,
) -> torch.Tensor:
    """Create a permittivity profile with sinusoidal modulation."""
    torch.manual_seed(seed)
    x = torch.linspace(0, 4 * 3.14159, n_grid, dtype=torch.float64)
    modulation = 0.5 * torch.sin(x)
    eps_real = (eps_mean + modulation).unsqueeze(0).expand(n_layers, -1).clone()
    eps = torch.complex(eps_real, torch.full_like(eps_real, eps_imag))
    return eps.detach().requires_grad_(True)


def _run_single(
    solver: RCWASolver,
    eps: torch.Tensor,
    wavelengths: list[float],
) -> dict:
    """Run forward + backward, return metrics."""
    eps = eps.clone().detach().requires_grad_(True)
    wavelengths_t = torch.tensor(wavelengths, dtype=torch.float64)

    t0 = time.perf_counter()
    result = solver.forward(eps, wavelengths=wavelengths_t)
    loss = result.field[:, solver.fourier_orders].sum()
    loss.backward()
    elapsed = time.perf_counter() - t0

    grad = eps.grad
    grad_finite = torch.isfinite(grad).all().item() if grad is not None else False
    nan_count = 0
    if grad is not None and not grad_finite:
        nan_count = (~torch.isfinite(grad)).sum().item()

    return {
        "field": result.field.detach(),
        "grad_finite": grad_finite,
        "nan_count": nan_count,
        "grad_max": grad.abs().max().item() if (grad is not None and grad_finite) else float("nan"),
        "time_s": elapsed,
    }


def main():
    seeds = [42, 123, 456, 789, 2024]
    configs = [
        {
            "name": "lossless",
            "eps_mean": 2.25,
            "eps_imag": 0.0,
            "wavelengths": [532.0],
        },
        {
            "name": "lossy (small)",
            "eps_mean": 2.25,
            "eps_imag": 0.3,
            "wavelengths": [532.0],
        },
        {
            "name": "lossy (large)",
            "eps_mean": 2.25,
            "eps_imag": 1.0,
            "wavelengths": [532.0],
        },
        {
            "name": "metal-like",
            "eps_mean": -10.0,
            "eps_imag": 1.0,
            "wavelengths": [532.0],
        },
        {
            "name": "multi-wavelength",
            "eps_mean": 2.25,
            "eps_imag": 0.5,
            "wavelengths": [500.0, 532.0, 600.0],
        },
    ]

    solver_kwargs = dict(
        fourier_orders=3,
        wavelength_nm=532.0,
        period_nm=(400.0, 400.0),
        device="cpu",
    )
    solver_eig = RCWASolver(**solver_kwargs, solver_backend="eig")
    solver_me = RCWASolver(**solver_kwargs, solver_backend="matrix_exp")

    print("=" * 80)
    print("RCWA Backend Comparison: eig vs matrix_exp")
    print("=" * 80)
    print(f"{'Config':<22} {'Backend':<14} {'Fwd Diff':<12} {'Grad OK':<10} {'NaN':<6} {'Grad Max':<12} {'Time (s)':<10}")
    print("-" * 80)

    for cfg in configs:
        for seed in seeds:
            label = f"{cfg['name']} (s={seed})"
            eps = _make_grating(3, 50, cfg["eps_mean"], cfg["eps_imag"], seed=seed)

            r_eig = _run_single(solver_eig, eps, cfg["wavelengths"])
            r_me = _run_single(solver_me, eps, cfg["wavelengths"])

            fwd_diff = (r_eig["field"] - r_me["field"]).abs().max().item()

            print(
                f"{label:<22} {'eig':<14} {'---':<12} "
                f"{'OK' if r_eig['grad_finite'] else 'NaN':<10} "
                f"{r_eig['nan_count']:<6} "
                f"{r_eig['grad_max']:<12.4e} "
                f"{r_eig['time_s']:<10.4f}"
            )
            print(
                f"{'':<22} {'matrix_exp':<14} {fwd_diff:<12.2e} "
                f"{'OK' if r_me['grad_finite'] else 'NaN':<10} "
                f"{r_me['nan_count']:<6} "
                f"{r_me['grad_max']:<12.4e} "
                f"{r_me['time_s']:<10.4f}"
            )

    print("-" * 80)
    print("\nSummary:")
    print("  - Forward diff: should be < 1e-10 (both backends compute the same physics)")
    print("  - Grad OK: gradient should be finite for all non-degenerate inputs")
    print("  - NaN: count of NaN/Inf entries in gradient (should be 0 for varying profiles)")
    print("  - Time: wall time for forward + backward (matrix_exp may be slower due to")
    print("    matrix_exp overhead but has more stable backward in degenerate cases)")


if __name__ == "__main__":
    main()
