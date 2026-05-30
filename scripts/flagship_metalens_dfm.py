#!/usr/bin/env python
"""Flagship metalens + lithography DFM co-optimization demo.

Runs coupled (co-design: optical + litho + fab) and decoupled (optical-only
baseline) optimization on a 20x20 metalens grid, then compares results.

The coupled approach optimizes the *printed* mask quality jointly with optical
performance. The decoupled baseline ignores lithography during optimization,
then evaluates litho quality post-hoc on the result.

Usage:
    python scripts/flagship_metalens_dfm.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffnano.workflows.dfm_metalens import DFMMetalensDesigner

N_STEPS = 150
GRID_SIZE = 20
PIXEL_SIZE_NM = 100.0
SEED = 42
RESULTS_PATH = Path(__file__).resolve().parent.parent / "flagship_metalens_results.json"


def _make_designer() -> DFMMetalensDesigner:
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
        device="cpu",
    )


def _evaluate(designer: DFMMetalensDesigner, density: torch.Tensor, beta: float = 10.0):
    mask = designer.density_param(density, beta=beta)
    litho_result = designer.litho_model.forward(mask)
    optical_loss = designer._optical_loss(litho_result["printed_contour"]).item()
    litho_epe = litho_result["epe"].item()
    from diffnano.design.constraints_shared import combined_fabrication_penalty

    fab_penalty = combined_fabrication_penalty(mask).item()
    return optical_loss, litho_epe, fab_penalty


def run() -> dict:
    torch.manual_seed(SEED)

    designer = _make_designer()
    print(f"Grid: {designer.grid_shape}, pixel: {PIXEL_SIZE_NM} nm, steps: {N_STEPS}")
    print()

    # --- Coupled (co-design) ---
    print("[1/2] Coupled co-design (optical + litho + fab)...")
    t0 = time.perf_counter()
    density_coupled, coupled_history, coupled_breakdown = designer.optimize(
        n_steps=N_STEPS,
        lr=1e-2,
        lambda_optical=1.0,
        lambda_litho=0.1,
        lambda_fab=0.01,
        verbose=False,
    )
    coupled_time = time.perf_counter() - t0
    opt_c, litho_c, fab_c = _evaluate(designer, density_coupled)
    print(f"  Done in {coupled_time:.1f}s | optical={opt_c:.6f}  litho_epe={litho_c:.6f}  fab={fab_c:.6f}")

    # --- Decoupled baseline ---
    print("[2/2] Decoupled baseline (optical only, litho evaluated post-hoc)...")
    t0 = time.perf_counter()
    density_decoupled, decoupled_history = designer.decoupled_baseline(
        n_steps=N_STEPS,
        lr=1e-2,
        lambda_optical=1.0,
        lambda_fab=0.01,
        verbose=False,
    )
    decoupled_time = time.perf_counter() - t0
    opt_d, litho_d, fab_d = _evaluate(designer, density_decoupled)
    print(f"  Done in {decoupled_time:.1f}s | optical={opt_d:.6f}  litho_epe={litho_d:.6f}  fab={fab_d:.6f}")

    # --- Summary table ---
    print()
    print("=" * 70)
    print(f"  {'Metric':<22s} {'Coupled':>14s} {'Decoupled':>14s} {'Delta':>14s}")
    print("-" * 70)
    for label, vc, vd in [
        ("Optical loss", opt_c, opt_d),
        ("Litho EPE", litho_c, litho_d),
        ("Fabrication penalty", fab_c, fab_d),
    ]:
        delta = vc - vd
        sign = "+" if delta > 0 else ""
        print(f"  {label:<22s} {vc:14.6f} {vd:14.6f} {sign}{delta:13.6f}")
    print("-" * 70)
    print(f"  {'Wall time (s)':<22s} {coupled_time:14.1f} {decoupled_time:14.1f}")
    print("=" * 70)

    if litho_c < litho_d:
        pct = (litho_d - litho_c) / litho_d * 100
        print(f"\n  Coupled approach reduces litho EPE by {pct:.1f}% (fabrication-aware advantage).")
    else:
        print("\n  Litho EPE comparable or slightly higher in coupled (grid too small for clear separation).")

    results = {
        "coupled": {
            "loss_history": coupled_history,
            "optical_loss": opt_c,
            "litho_epe": litho_c,
            "fab_penalty": fab_c,
            "wall_time_s": round(coupled_time, 2),
        },
        "decoupled": {
            "loss_history": decoupled_history,
            "optical_loss": opt_d,
            "litho_epe": litho_d,
            "fab_penalty": fab_d,
            "wall_time_s": round(decoupled_time, 2),
        },
        "config": {
            "n_steps": N_STEPS,
            "grid_size": GRID_SIZE,
            "pixel_size_nm": PIXEL_SIZE_NM,
            "seed": SEED,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    run()
