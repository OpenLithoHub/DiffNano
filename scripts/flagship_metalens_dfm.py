#!/usr/bin/env python
"""Flagship metalens + lithography DFM co-optimization demo (multi-seed).

Runs coupled (co-design: optical + litho + fab) and decoupled (optical-only
baseline) optimization across 10 seeds (42..51), then reports aggregated
results with Wilcoxon signed-rank significance tests.

Usage:
    python scripts/flagship_metalens_dfm.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffnano.workflows.dfm_metalens import DFMMetalensDesigner

N_STEPS = 150
GRID_SIZE = 20
PIXEL_SIZE_NM = 100.0
DEFAULT_SEEDS = list(range(42, 52))
RESULTS_PATH = Path(__file__).resolve().parent.parent / "flagship_metalens_results.json"


def _make_designer(device: str = "cpu") -> DFMMetalensDesigner:
    return DFMMetalensDesigner(
        wavelength_nm=940.0,
        numerical_aperture=0.3,
        diameter_um=GRID_SIZE * PIXEL_SIZE_NM / 1000,
        pixel_size_nm=PIXEL_SIZE_NM,
        n_material=2.0,
        n_ambient=1.0,
        fourier_orders=3,
        litho_wavelength_nm=193.0,
        litho_na=1.35,
        device=device,
    )


def _evaluate(designer: DFMMetalensDesigner, density: torch.Tensor, beta: float = 10.0):
    mask = designer.density_param(density, beta=beta)
    litho_result = designer.litho_model.forward(mask)
    optical_loss = designer._optical_loss(litho_result["printed_contour"]).item()
    litho_epe = litho_result["epe"].item()
    from diffnano.design.constraints_shared import combined_fabrication_penalty

    fab_penalty = combined_fabrication_penalty(mask).item()
    return optical_loss, litho_epe, fab_penalty


def _run_single_seed(seed: int, n_steps: int, device: str) -> dict:
    """Run coupled + decoupled for a single seed. Returns per-seed results."""
    torch.manual_seed(seed)

    designer = _make_designer(device=device)

    # --- Coupled (co-design) ---
    t0 = time.perf_counter()
    density_coupled, coupled_history, coupled_breakdown = designer.optimize(
        n_steps=n_steps,
        lr=1e-2,
        lambda_optical=1.0,
        lambda_litho=0.1,
        lambda_fab=0.01,
        verbose=False,
    )
    coupled_time = time.perf_counter() - t0
    opt_c, litho_c, fab_c = _evaluate(designer, density_coupled)

    # --- Decoupled baseline ---
    t0 = time.perf_counter()
    density_decoupled, decoupled_history = designer.decoupled_baseline(
        n_steps=n_steps,
        lr=1e-2,
        lambda_optical=1.0,
        lambda_fab=0.01,
        verbose=False,
    )
    decoupled_time = time.perf_counter() - t0
    opt_d, litho_d, fab_d = _evaluate(designer, density_decoupled)

    print(
        f"  seed={seed:>2d} | coupled: opt={opt_c:.6f} litho={litho_c:.6f} fab={fab_c:.6f} "
        f"t={coupled_time:.1f}s | decoupled: opt={opt_d:.6f} litho={litho_d:.6f} fab={fab_d:.6f} "
        f"t={decoupled_time:.1f}s"
    )

    return {
        "seed": seed,
        "coupled": {
            "optical_loss": opt_c,
            "litho_epe": litho_c,
            "fab_penalty": fab_c,
            "wall_time_s": round(coupled_time, 2),
            "loss_history": coupled_history,
        },
        "decoupled": {
            "optical_loss": opt_d,
            "litho_epe": litho_d,
            "fab_penalty": fab_d,
            "wall_time_s": round(decoupled_time, 2),
            "loss_history": decoupled_history,
        },
    }


def run(seeds: list[int] | None = None, device: str = "cpu") -> dict:
    if seeds is None:
        seeds = DEFAULT_SEEDS

    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, pixel: {PIXEL_SIZE_NM} nm, steps: {N_STEPS}, "
          f"seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds), device: {device}")
    print()

    per_seed = []
    for seed in seeds:
        per_seed.append(_run_single_seed(seed, N_STEPS, device))

    # --- Aggregate ---
    metrics = ["optical_loss", "litho_epe", "fab_penalty", "wall_time_s"]

    coupled_vals = {m: [s["coupled"][m] for s in per_seed] for m in metrics}
    decoupled_vals = {m: [s["decoupled"][m] for s in per_seed] for m in metrics}

    def _mean_std(vals):
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    coupled_stats = {m: _mean_std(coupled_vals[m]) for m in metrics}
    decoupled_stats = {m: _mean_std(decoupled_vals[m]) for m in metrics}

    # --- Wilcoxon signed-rank test (paired: coupled vs decoupled per seed) ---
    wilcoxon_results = {}
    for m in metrics:
        c = np.array(coupled_vals[m])
        d = np.array(decoupled_vals[m])
        diff = c - d
        if np.all(diff == 0):
            wilcoxon_results[m] = {"statistic": None, "p_value": None, "significant_005": False}
        else:
            stat, pval = wilcoxon(c, d, alternative="two-sided")
            wilcoxon_results[m] = {
                "statistic": float(stat),
                "p_value": float(pval),
                "significant_005": bool(pval < 0.05),
            }

    # --- Summary table ---
    print()
    print("=" * 90)
    print(f"  {'Metric':<22s} {'Coupled (mean±std)':>22s} {'Decoupled (mean±std)':>22s} {'Wilcoxon p':>12s}")
    print("-" * 90)
    for m in metrics:
        cm, cs = coupled_stats[m]
        dm, ds = decoupled_stats[m]
        pval = wilcoxon_results[m]["p_value"]
        p_str = f"{pval:.4f}" if pval is not None else "N/A"
        print(f"  {m:<22s} {cm:>8.6f}±{cs:<8.6f}  {dm:>8.6f}±{ds:<8.6f}  {p_str:>12s}")
    print("=" * 90)

    sig_metrics = [m for m in metrics if wilcoxon_results[m]["significant_005"]]
    if sig_metrics:
        print(f"\n  Significant differences (p<0.05): {', '.join(sig_metrics)}")
    else:
        print("\n  No metrics reach p<0.05 significance with this seed count.")

    results = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "per_seed": per_seed,
        "aggregated": {
            "coupled": {
                m: {"mean": coupled_stats[m][0], "std": coupled_stats[m][1]}
                for m in metrics
            },
            "decoupled": {
                m: {"mean": decoupled_stats[m][0], "std": decoupled_stats[m][1]}
                for m in metrics
            },
        },
        "wilcoxon": wilcoxon_results,
        "config": {
            "n_steps": N_STEPS,
            "grid_size": GRID_SIZE,
            "pixel_size_nm": PIXEL_SIZE_NM,
            "device": device,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed flagship metalens DFM benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to use (default: 42..51)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (default: cpu)")
    args = parser.parse_args()
    run(seeds=args.seeds, device=args.device)
