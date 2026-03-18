"""Plot Figure 6: Steering DelFlank on Ambiguous Sequences.

Panel (a): Per-layer normalized steering alpha scores over move numbers.
Panel (b): Board probe accuracy at layer 5 under baseline vs steered conditions.

Reads cached data from:
  - data/analysis_cache/steering_delflank.json (Panel A)
  - data/analysis_cache/probe_effect_delflank.json (Panel B)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from metaothello.analysis_utils import CACHE_DIR, load_json_cache
from metaothello.plotting import save_figure

logger = logging.getLogger(__name__)

FIGURE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "figures" / "steering_delflank"
RUN_NAME = "classic_delflank"
N_LAYERS = 8


def main(probe_layer_idx: int = 5) -> None:
    # Load cached data
    steering_data = load_json_cache(CACHE_DIR / "steering_delflank.json")
    probe_data = load_json_cache(CACHE_DIR / "probe_effect_delflank.json")

    if RUN_NAME not in steering_data:
        raise RuntimeError("steering_delflank.json missing. Run compute_steering_delflank.py first.")

    probe_key = f"{RUN_NAME}_L{probe_layer_idx}"
    if probe_key not in probe_data:
        raise RuntimeError(
            f"probe_effect_delflank.json missing key '{probe_key}'. "
            f"Run compute_probe_effect_delflank.py --probe_layer {probe_layer_idx} first."
        )

    steering = steering_data[RUN_NAME]
    probe = probe_data[probe_key]
    probe_label = f"L{probe_layer_idx + 1}"  # 1-indexed for display

    # Custom rcParams matching dev repo exactly
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'lines.linewidth': 1.0,
        'axes.linewidth': 1.0,
        'figure.dpi': 300,
    })

    fig, axes = plt.subplots(
        2, 1, figsize=(3.25, 4.2), gridspec_kw={'height_ratios': [1.4, 1]}
    )

    colors = cm.plasma(np.linspace(0, 0.85, N_LAYERS))
    text_halo = [pe.withStroke(linewidth=2.5, foreground="white")]

    # =========================================================================
    # Panel (a): Steering alpha score line plot
    # =========================================================================
    ax1 = axes[0]

    label_x_pos = {0: 38, 1: 4, 2: 8, 3: 13, 4: 18, 5: 23, 6: 28, 7: 33}

    moves = sorted(int(k) for k in steering.keys())

    for layer_idx in range(N_LAYERS):
        xs, ys = [], []
        for move in moves:
            val = steering[str(move)]["means"][layer_idx]
            if val is not None:
                xs.append(move)
                ys.append(val)

        if not xs:
            continue

        color = colors[layer_idx]

        # Highlight layers 2, 3, 5 (0-indexed: 1, 2, 4)
        if layer_idx in [1, 2, 4]:
            zorder, alpha, linewidth = 10, 1.0, 1.5
        else:
            zorder, alpha, linewidth = 1, 0.5, 1.0

        ax1.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)

        # Inline label
        target_x = label_x_pos.get(layer_idx, 25)
        closest_idx = min(range(len(xs)), key=lambda i: abs(xs[i] - target_x))
        x_val, y_val = xs[closest_idx], ys[closest_idx]

        ax1.text(
            x_val, y_val, str(layer_idx + 1),
            color=color, fontsize=7, fontweight='bold',
            ha='center', va='center',
            path_effects=text_halo, zorder=20,
        )

    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax1.set_xlabel('Move Number', labelpad=2)
    ax1.set_ylabel(r'$\Delta\alpha / \Delta\alpha_{\mathrm{max}}$', labelpad=2)
    ax1.text(-0.15, 1.1, '(a)', transform=ax1.transAxes, fontsize=9,
             fontweight='bold', va='bottom', ha='left')
    ax1.set_xlim(1, 40)
    ax1.set_ylim(-0.1, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([1, 10, 20, 30, 40])

    # =========================================================================
    # Panel (b): Board probe accuracy bar chart
    # =========================================================================
    ax2 = axes[1]

    baseline_acc = probe["baseline"]["mean"]
    baseline_std = probe["baseline"]["std"]
    baseline_n = probe["baseline"]["n"]

    steer_1 = probe["steered"]["1"]
    steer_2 = probe["steered"]["2"]

    conditions = ['Baseline', 'Layer 2', 'Layer 3']
    bar_colors = ['#94A3B8', colors[1], colors[2]]
    accuracies = [baseline_acc, steer_1["mean"], steer_2["mean"]]

    # Standard error of the mean for 95% CI
    sems = [
        baseline_std / np.sqrt(baseline_n),
        steer_1["std"] / np.sqrt(steer_1["n"]),
        steer_2["std"] / np.sqrt(steer_2["n"]),
    ]

    bars = ax2.bar(
        conditions, accuracies, color=bar_colors,
        edgecolor='black', linewidth=0.8, width=0.6,
    )
    ax2.errorbar(
        conditions, accuracies, yerr=[s * 1.96 for s in sems],
        fmt='none', color='black', capsize=3, capthick=1.0, linewidth=1.0,
    )

    # Value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.annotate(
            f'{acc:.1%}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha='center', va='bottom', fontsize=7, fontweight='bold',
        )

    # Delta annotations
    delta_1 = accuracies[1] - accuracies[0]
    delta_2 = accuracies[2] - accuracies[0]
    ax2.annotate(
        f'+{delta_1 * 100:.1f}%',
        xy=(1, accuracies[1] + 0.05),
        ha='center', fontsize=7, color=colors[1], fontweight='bold',
    )
    ax2.annotate(
        f'+{delta_2 * 100:.1f}%',
        xy=(2, accuracies[2] + 0.05),
        ha='center', fontsize=7, color=colors[2], fontweight='bold',
    )

    ax2.set_ylabel(f'Board Probe Acc. ({probe_label})', labelpad=2)
    ax2.text(-0.15, 1.1, '(b)', transform=ax2.transAxes, fontsize=9,
             fontweight='bold', va='bottom', ha='left')
    ax2.set_ylim(0.7, max(accuracies) + 0.1)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=baseline_acc, color='gray', linestyle=':', alpha=0.5,
                zorder=0, linewidth=0.8)

    plt.tight_layout(h_pad=1.5)

    suffix = "" if probe_layer_idx == 5 else f"_{probe_label}"
    save_figure(fig, FIGURE_DIR / f"steering_delflank_combined{suffix}")
    logger.info("Figure saved to %s", FIGURE_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Figure 6: DelFlank steering.")
    parser.add_argument("--probe_layer", type=int, default=5,
                        help="0-indexed probe layer for Panel B (default: 5).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main(probe_layer_idx=args.probe_layer)
