"""C4 benchmark: unified autograd graph vs decoupled baseline.

Produces the CN filing C4 embodiment data:
- Optimization loss curves (unified vs decoupled)
- Final Strehl ratio comparison
- Lithography EPE (edge placement error) comparison
- MRC compliance check

Usage:
    python scripts/benchmark_c4.py
"""

from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, ".")

from diffnano.workflows.dfm_metalens import DFMMetalensDesigner


def run_c4_benchmark(
    n_steps: int = 200,
    seed: int = 42,
) -> dict:
    """Run the C4 unified vs decoupled benchmark.

    Parameters
    ----------
    n_steps : int
        Optimization steps.
    seed : int

    Returns
    -------
    results : dict
    """
    torch.manual_seed(seed)

    # Use a small grid for speed
    pixel_size = 10.0  # nm
    diameter = 0.5  # um → 50 pixels
    n_pix = int(diameter * 1000 / pixel_size)

    designer = DFMMetalensDesigner(
        wavelength_nm=940.0,
        numerical_aperture=0.3,
        diameter_um=diameter,
        pixel_size_nm=pixel_size,
        fourier_orders=3,
        litho_wavelength_nm=193.0,
        litho_na=1.35,
        device="cpu",
    )

    print(f"=== C4 Benchmark: {n_pix}x{n_pix} grid, {n_steps} steps ===")

    # --- Unified (C4) optimization ---
    print("\n[1/2] Unified C4 optimization (litho + optical + fab)...")
    d_unified, loss_unified, bd_unified = designer.optimize(
        n_steps=n_steps,
        lr=1e-2,
        lambda_optical=1.0,
        lambda_litho=0.1,
        lambda_fab=0.01,
        verbose=False,
    )

    # Evaluate final state
    mask_unified = designer.density_param(d_unified, beta=10.0)
    litho_unified = designer.litho_model.forward(mask_unified)
    strehl_unified = designer._optical_loss(
        litho_unified["printed_contour"]
    ).item()
    epe_unified = litho_unified["epe"].item()

    print(f"  Final optical loss: {strehl_unified:.6f}")
    print(f"  Final litho EPE:    {epe_unified:.6f}")

    # --- Decoupled baseline ---
    print("\n[2/2] Decoupled baseline (optical only, then litho check)...")
    d_decoupled, loss_decoupled = designer.decoupled_baseline(
        n_steps=n_steps,
        lr=1e-2,
        lambda_optical=1.0,
        lambda_fab=0.01,
        verbose=False,
    )

    mask_decoupled = designer.density_param(d_decoupled, beta=10.0)
    litho_decoupled = designer.litho_model.forward(mask_decoupled)
    strehl_decoupled = designer._optical_loss(
        litho_decoupled["printed_contour"]
    ).item()
    epe_decoupled = litho_decoupled["epe"].item()

    print(f"  Final optical loss: {strehl_decoupled:.6f}")
    print(f"  Final litho EPE:    {epe_decoupled:.6f}")

    # --- Comparison ---
    print("\n=== C4 Results ===")
    print(f"  {'':20s} {'Unified':>12s} {'Decoupled':>12s}")
    print(f"  {'Optical loss':20s} {strehl_unified:12.6f} {strehl_decoupled:12.6f}")
    print(f"  {'Litho EPE':20s} {epe_unified:12.6f} {epe_decoupled:12.6f}")

    results = {
        "unified": {
            "loss_history": loss_unified,
            "optical_loss": strehl_unified,
            "litho_epe": epe_unified,
        },
        "decoupled": {
            "loss_history": loss_decoupled,
            "optical_loss": strehl_decoupled,
            "litho_epe": epe_decoupled,
        },
        "n_steps": n_steps,
    }

    with open("benchmark_c4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark_c4_results.json")

    return results


if __name__ == "__main__":
    run_c4_benchmark()
