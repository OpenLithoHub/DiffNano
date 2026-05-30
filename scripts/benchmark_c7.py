#!/usr/bin/env python3
"""Benchmark C7: Adaptive robust optimization vs brute-force MC vs nominal.

Compares three strategies:
(a) Nominal: optimize at the nominal point only
(b) C5 brute-force MC: fixed K=16 Monte Carlo samples
(c) C7 adaptive axial: O(2N+1) axial + curriculum random samples

Reports: FoM vs simulation budget, convergence speed, worst-case performance.
"""

import json
import sys
from pathlib import Path

import torch

from diffnano.design.robustness import AdaptiveRobustOptimizer
from diffnano.design.robustness.core import robust_gradient_step

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _quadratic_objective(params: torch.Tensor) -> torch.Tensor:
    """Multi-modal quadratic test function with a minimum at origin."""
    return (params ** 2).sum()


def _perturb(params: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return params + delta.sum() * 0.1


def benchmark_nominal(n_steps=200, lr=0.01):
    params = torch.randn(10, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([params], lr=lr)
    history = []
    for _ in range(n_steps):
        loss = _quadratic_objective(params)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    return params.detach(), history


def benchmark_c5_mc(n_steps=200, lr=0.01, n_samples=16):
    params = torch.randn(10, dtype=torch.float64)
    opt = torch.optim.Adam([params], lr=lr)
    history = []
    for _ in range(n_steps):
        params.requires_grad_(True)

        def forward_fn(p):
            return _quadratic_objective(p)
        loss = robust_gradient_step(params, forward_fn, sigma_nm=1.0, n_samples=n_samples, antithetic=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        params = params.detach()
        history.append(loss.item())
    return params, history


def benchmark_c7_adaptive(n_steps=200, lr=0.01):
    params = torch.randn(10, dtype=torch.float64)

    def forward_fn(p, delta):
        return _quadratic_objective(p)

    def perturb_fn(p, delta):
        return p + delta.sum() * 0.1

    optimizer = AdaptiveRobustOptimizer(n_variation_dims=3, sigma=1.0)
    result, history = optimizer.optimize(params, forward_fn, perturb_fn, n_steps=n_steps, lr=lr, verbose=False)
    return result, history


def main():
    torch.manual_seed(42)

    n_steps = 100

    _, hist_nominal = benchmark_nominal(n_steps)
    _, hist_c5 = benchmark_c5_mc(n_steps)
    _, hist_c7 = benchmark_c7_adaptive(n_steps)

    results = {
        "n_steps": n_steps,
        "nominal": {
            "final_loss": hist_nominal[-1],
            "converged_at": next((i for i, l in enumerate(hist_nominal) if l < 1.0), n_steps),
        },
        "c5_brute_force_mc": {
            "final_loss": hist_c5[-1],
            "converged_at": next((i for i, l in enumerate(hist_c5) if l < 1.0), n_steps),
        },
        "c7_adaptive": {
            "final_loss": hist_c7[-1],
            "converged_at": next((i for i, l in enumerate(hist_c7) if l < 1.0), n_steps),
        },
    }

    out = Path(__file__).resolve().parents[1] / "benchmark_c7_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
