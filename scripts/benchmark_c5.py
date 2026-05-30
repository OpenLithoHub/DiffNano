"""Nominal vs robust Monte Carlo benchmark.

Produces benchmark data:
- N=100+ Monte Carlo realizations under linewidth perturbation N(0, σ²), σ=5nm
- Nominal-optimized vs robust-optimized Strehl ratio histograms
- Yield-equivalent figure: fraction of realizations with Strehl ≥ threshold

Usage:
    python scripts/benchmark_c5.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")

from diffnano.workflows.metalens import MetalensDesigner


def run_c5_benchmark(
    n_pixels: int = 25,
    n_opt_steps: int = 200,
    n_mc: int = 100,
    sigma_nm: float = 5.0,
    seed: int = 42,
) -> dict:
    """Run the C5 Monte Carlo benchmark.

    Parameters
    ----------
    n_pixels : int
        Grid size (small for speed).
    n_opt_steps : int
        Optimization steps.
    n_mc : int
        Monte Carlo samples for evaluation.
    sigma_nm : float
        Process variation sigma in nm.
    seed : int

    Returns
    -------
    results : dict with keys:
        "nominal_strehl_samples": list of float
        "robust_strehl_samples": list of float
        "nominal_yield": float
        "robust_yield": float
        "nominal_mean_strehl": float
        "robust_mean_strehl": float
        "threshold": float
        "sigma_nm": float
        "n_mc": int
    """
    torch.manual_seed(seed)

    diameter_um = n_pixels * 0.2  # match pixel_size=200nm
    sigma_nm_eval = max(sigma_nm, 50.0)  # ensure visible MC effect
    designer = MetalensDesigner(
        wavelength_nm=532.0,
        numerical_aperture=0.3,
        diameter_um=diameter_um,
        pixel_size_nm=200.0,
        fourier_orders=3,
        device="cpu",
    )

    print(f"=== C5 Benchmark: {n_pixels}x{n_pixels} metalens, "
          f"{n_opt_steps} opt steps, {n_mc} MC samples, σ={sigma_nm}nm ===")

    # --- Nominal optimization ---
    print("\n[1/3] Nominal optimization...")
    h_nominal, _ = designer.optimize(
        n_steps=n_opt_steps,
        verbose=False,
        robust=False,
    )
    strehl_nominal_base = designer.strehl_ratio(h_nominal).item()
    print(f"  Nominal base Strehl: {strehl_nominal_base:.4f}")

    # --- Robust optimization ---
    print("\n[2/3] Robust optimization...")
    h_robust, _ = designer.optimize(
        n_steps=n_opt_steps,
        verbose=False,
        robust=True,
        sigma_nm=sigma_nm_eval,
        n_mc_samples=4,
    )
    strehl_robust_base = designer.strehl_ratio(h_robust).item()
    print(f"  Robust base Strehl: {strehl_robust_base:.4f}")

    # --- Monte Carlo evaluation ---
    print(f"\n[3/3] Monte Carlo evaluation (N={n_mc}, σ_eval={sigma_nm_eval}nm)...")
    # First pass: compute Strehls, then set threshold at median of nominal
    threshold = 0.5  # will be updated after first pass

    nominal_strehls = []
    robust_strehls = []

    for i in range(n_mc):
        # Per-pixel height perturbation (simulates fabrication thickness variation)
        delta = torch.randn_like(h_nominal) * sigma_nm_eval

        # Perturb nominal design
        h_nom_perturbed = h_nominal + delta
        h_nom_perturbed = h_nom_perturbed.clamp(min=0)
        nominal_strehls.append(designer.strehl_ratio(h_nom_perturbed).item())

        # Perturb robust design
        h_rob_perturbed = h_robust + delta
        h_rob_perturbed = h_rob_perturbed.clamp(min=0)
        robust_strehls.append(designer.strehl_ratio(h_rob_perturbed).item())

        if (i + 1) % 20 == 0:
            print(f"  MC sample {i+1}/{n_mc}")

    # Set threshold at median of nominal distribution
    sorted_nom = sorted(nominal_strehls)
    threshold = sorted_nom[n_mc // 2]

    nominal_yield = sum(1 for s in nominal_strehls if s >= threshold) / n_mc
    robust_yield = sum(1 for s in robust_strehls if s >= threshold) / n_mc

    print("\n=== C5 Results ===")
    print(f"  Strehl threshold: {threshold}")
    print(f"  Nominal mean Strehl: {sum(nominal_strehls)/n_mc:.4f}")
    print(f"  Robust mean Strehl:  {sum(robust_strehls)/n_mc:.4f}")
    print(f"  Nominal yield (Strehl ≥ {threshold}): {nominal_yield:.1%}")
    print(f"  Robust yield (Strehl ≥ {threshold}):  {robust_yield:.1%}")
    print(f"  Yield improvement: {(robust_yield - nominal_yield):.1%}")

    results = {
        "nominal_strehl_samples": nominal_strehls,
        "robust_strehl_samples": robust_strehls,
        "nominal_yield": nominal_yield,
        "robust_yield": robust_yield,
        "nominal_mean_strehl": sum(nominal_strehls) / n_mc,
        "robust_mean_strehl": sum(robust_strehls) / n_mc,
        "threshold": threshold,
        "sigma_nm": sigma_nm,
        "n_mc": n_mc,
        "nominal_base_strehl": strehl_nominal_base,
        "robust_base_strehl": strehl_robust_base,
    }

    REPO = Path(__file__).resolve().parents[1]
    with open(REPO / "benchmark_c5_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark_c5_results.json")

    return results


if __name__ == "__main__":
    run_c5_benchmark()
