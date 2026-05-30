#!/usr/bin/env python3
"""Generate benchmark charts for README.

Outputs:
  docs/images/benchmark_strehl_histogram.svg  — C5 Monte Carlo Strehl distribution
  docs/images/benchmark_convergence.svg        — C4 unified vs decoupled loss curves
  docs/images/benchmark_tool_comparison.svg    — Feature comparison across tools
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── Style: neutral-gray palette for dark/light mode ──────────────────
NEUTRAL = "#888888"
STRONG = "#444444"
LIGHT_GRID = "#cccccc"
BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
PURPLE = "#8172B2"

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, color=STRONG, fontsize=11, pad=8)
    ax.set_xlabel(xlabel, color=NEUTRAL, fontsize=9)
    ax.set_ylabel(ylabel, color=NEUTRAL, fontsize=9)
    ax.tick_params(colors=NEUTRAL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(LIGHT_GRID)
    ax.grid(True, color=LIGHT_GRID, linewidth=0.4, alpha=0.6)


# ── Chart 1: C5 Strehl Ratio Histogram ──────────────────────────────
def chart_strehl_histogram():
    with open(REPO / "benchmark_c5_results.json") as f:
        data = json.load(f)

    nom = data["nominal_strehl_samples"]
    rob = data["robust_strehl_samples"]
    threshold = data["threshold"]

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    bins = np.linspace(0.53, 0.63, 25)
    ax.hist(nom, bins=bins, alpha=0.6, color=BLUE, label="Nominal", edgecolor="none")
    ax.hist(rob, bins=bins, alpha=0.6, color=ORANGE, label="Robust", edgecolor="none")
    ax.axvline(threshold, color=RED, linestyle="--", linewidth=1.2, label=f"Threshold ({threshold:.3f})")

    _style_ax(
        ax,
        title="C5 Benchmark: Strehl Ratio under Process Variation (σ = 5 nm)",
        xlabel="Strehl Ratio",
        ylabel="Monte Carlo Samples (N = 100)",
    )
    ax.legend(
        fontsize=8,
        loc="upper left",
        framealpha=0.7,
        edgecolor=LIGHT_GRID,
        labelcolor=NEUTRAL,
    )
    ax.text(
        0.97,
        0.95,
        f"Nominal yield: {data['nominal_yield']:.0%}\nRobust yield:   {data['robust_yield']:.0%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=NEUTRAL,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="none", edgecolor=LIGHT_GRID),
    )

    fig.tight_layout()
    path = OUT / "benchmark_strehl_histogram.svg"
    fig.savefig(path, format="svg", transparent=True)
    plt.close(fig)
    print(f"  → {path}")


# ── Chart 2: C4 Convergence Curves ──────────────────────────────────
def chart_convergence():
    with open(REPO / "benchmark_c4_results.json") as f:
        data = json.load(f)

    unified = data["unified"]["loss_history"]
    decoupled = data["decoupled"]["loss_history"]

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.plot(unified, color=GREEN, linewidth=1.5, label="Unified autograd")
    ax.plot(decoupled, color=RED, linewidth=1.5, label="Decoupled baseline")

    _style_ax(
        ax,
        title="C4 Benchmark: Optimization Convergence (200 steps)",
        xlabel="Step",
        ylabel="Loss",
    )
    ax.legend(
        fontsize=8,
        loc="upper right",
        framealpha=0.7,
        edgecolor=LIGHT_GRID,
        labelcolor=NEUTRAL,
    )
    ax.set_yscale("log")

    fig.tight_layout()
    path = OUT / "benchmark_convergence.svg"
    fig.savefig(path, format="svg", transparent=True)
    plt.close(fig)
    print(f"  → {path}")


# ── Chart 3: Tool Feature Comparison ────────────────────────────────
def chart_tool_comparison():
    tools = [
        "DiffNano",
        "Tidy3D\nv2.10.1",
        "MEEP\nv1.32.0",
        "TorchRDIT\n(2024)",
        "Ceviche\n(unmaintained)",
        "FDTDX\n(2026)",
    ]
    # 5-point scale: GPU, Autograd, Solver Diversity, Fabrication-aware, Open Source
    # Honest assessment — DiffNano is not the best at everything
    scores = {
        "GPU Acceleration":   [3, 5, 1, 4, 1, 5],
        "Autograd / AD":      [4, 4, 3, 5, 3, 5],
        "Solver Diversity":   [4, 2, 2, 1, 2, 1],
        "Fabrication-aware":  [4, 1, 1, 1, 1, 1],
        "Open & Free":        [5, 2, 5, 5, 5, 5],
    }
    categories = list(scores.keys())
    n_cats = len(categories)
    n_tools = len(tools)
    x = np.arange(n_cats)
    width = 0.12

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    colors = [GREEN, BLUE, ORANGE, PURPLE, RED, "#8C8C8C"]
    for i, (tool, color) in enumerate(zip(tools, colors)):
        vals = [scores[c][i] for c in categories]
        offset = (i - n_tools / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=tool.replace("\n", " "), color=color, alpha=0.8, edgecolor="none")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(v),
                        ha="center", va="bottom", fontsize=6, color=NEUTRAL)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 6)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color=NEUTRAL)

    _style_ax(
        ax,
        title="Feature Comparison (1–5 scale, author's subjective assessment)",
        ylabel="Score",
    )
    ax.legend(
        fontsize=6.5,
        ncol=3,
        loc="upper right",
        framealpha=0.7,
        edgecolor=LIGHT_GRID,
        labelcolor=NEUTRAL,
    )

    fig.tight_layout()
    path = OUT / "benchmark_tool_comparison.svg"
    fig.savefig(path, format="svg", transparent=True)
    plt.close(fig)
    print(f"  → {path}")


if __name__ == "__main__":
    print("Generating benchmark charts...")
    chart_strehl_histogram()
    chart_convergence()
    chart_tool_comparison()
    print("Done.")
