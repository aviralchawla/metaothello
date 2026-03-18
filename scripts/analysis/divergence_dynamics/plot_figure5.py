"""Plot Figure 5: NoMidFlip divergence dynamics (three-panel combined figure).

Panel (a): NoMidFlip probe accuracy on differing tiles vs move number.
Panel (b): Game probe fidelity vs move number, with entropy inset.
Panel (c): Causal steering delta-alpha vs move number.

Reads cached results from data/analysis_cache/.
Outputs to figures/divergence_dynamics/figure5_combined.{pdf,png}.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import numpy as np

from metaothello.analysis_utils import CACHE_DIR, load_json_cache
from metaothello.plotting import (
    ICML_FULL_WIDTH,
    save_figure,
)

logger = logging.getLogger(__name__)

FIGURE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "figures" / "divergence_dynamics"
N_LAYERS = 8
RUN_NAME = "classic_nomidflip"

# Layer colors: plasma colormap, stop at 0.8 to avoid light yellow
LAYER_CMAP = plt.cm.plasma
LAYER_COLORS = [LAYER_CMAP(x) for x in np.linspace(0, 0.8, N_LAYERS)]

# Text halo effect for layer labels
HALO = [pe.withStroke(linewidth=2.5, foreground="white")]


def _setup_style() -> None:
    """Configure matplotlib rcParams to match the dev repo's plotting style."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 1.5,
            "figure.dpi": 300,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_panel_a(ax: plt.Axes, cache: dict) -> None:
    """Panel (a): NoMidFlip probe accuracy on tiles differing between games."""
    data = cache.get(RUN_NAME, {})
    if not data:
        logger.warning("No Panel A data found.")
        return

    moves = sorted(int(m) for m in data.keys())

    # Stagger label x-positions for layers 1-8
    label_x_positions = [10, 15, 20, 25, 30, 35, 40, 45]

    for layer_idx in range(N_LAYERS):
        ys = []
        xs = []
        for m in moves:
            entry = data[str(m)]
            val = entry["means"][layer_idx]
            if val is not None:
                xs.append(m)
                ys.append(val)

        ax.plot(xs, ys, color=LAYER_COLORS[layer_idx], linewidth=1.0)

        # Inline label at staggered x position
        if xs and ys:
            xs_arr = np.array(xs)
            ys_arr = np.array(ys)
            target_x = label_x_positions[layer_idx]
            idx = np.argmin(np.abs(xs_arr - target_x))
            ax.text(
                xs_arr[idx], ys_arr[idx], str(layer_idx + 1),
                color=LAYER_COLORS[layer_idx], fontsize=7, fontweight="bold",
                ha="center", va="center", path_effects=HALO,
            )

    ax.set_xlabel("Move Number", fontsize=8)
    ax.set_ylabel("NoMidFlip Probe Accuracy\n(Differing Tiles)", fontsize=8)
    ax.axhline(y=1 / 3, color="gray", linestyle=":", alpha=0.5, linewidth=1.0, label="Chance")
    ax.set_xlim(5, 52)
    ax.set_ylim(0.33, 1.02)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.3)


def plot_panel_b(ax: plt.Axes, cache: dict) -> None:
    """Panel (b): Game probe fidelity with entropy inset."""
    data = cache.get(RUN_NAME, {})
    if not data:
        logger.warning("No Panel B data found.")
        return

    fidelity = data.get("fidelity", {})
    entropy = data.get("entropy", {})
    baseline = data.get("baseline", {})

    moves = sorted(int(m) for m in fidelity.keys())

    # Store line data for label placement
    layer_data = {}

    # Main plot: per-layer fidelity
    for layer_idx in range(N_LAYERS):
        ys = []
        xs = []
        for m in moves:
            entry = fidelity[str(m)]
            val = entry["means"][layer_idx]
            if val is not None:
                xs.append(m)
                ys.append(val)

        ax.plot(xs, ys, color=LAYER_COLORS[layer_idx], linewidth=1.0, zorder=10)
        layer_data[layer_idx] = (np.array(xs), np.array(ys))

    # Declining layers (1-4): labels on their descent where they separate
    declining_labels = {
        0: 50,   # Layer 1 - far right, lowest of declining group
        1: 44,   # Layer 2
        2: 38,   # Layer 3
        3: 32,   # Layer 4
    }

    for layer_idx, x_pos in declining_labels.items():
        if layer_idx in layer_data:
            xs_arr, ys_arr = layer_data[layer_idx]
            idx = np.argmin(np.abs(xs_arr - x_pos))
            if idx < len(xs_arr):
                ax.text(
                    xs_arr[idx], ys_arr[idx], str(layer_idx + 1),
                    color=LAYER_COLORS[layer_idx], fontsize=7, fontweight="bold",
                    ha="center", va="center", path_effects=HALO, zorder=11,
                )

    # High-staying layers (5-8): inline at staggered x positions
    high_layer_x_positions = {
        4: 6,    # Layer 5 - leftmost
        5: 14,   # Layer 6
        6: 22,   # Layer 7
        7: 30,   # Layer 8
    }

    for layer_idx, x_pos in high_layer_x_positions.items():
        if layer_idx in layer_data:
            xs_arr, ys_arr = layer_data[layer_idx]
            idx = np.argmin(np.abs(xs_arr - x_pos))
            if idx < len(xs_arr):
                ax.text(
                    xs_arr[idx], ys_arr[idx], str(layer_idx + 1),
                    color=LAYER_COLORS[layer_idx], fontsize=7, fontweight="bold",
                    ha="center", va="center", path_effects=HALO, zorder=11,
                )

    # Baseline as filled region from 0
    if baseline:
        bx = sorted(int(m) for m in baseline.keys())
        by = [baseline[str(m)]["mean"] for m in bx]
        ax.fill_between(bx, 0, by, color="#cccccc", alpha=0.4, zorder=1, label="Baseline")
        ax.text(
            30, 0.49, "Baseline",
            color="gray", fontsize=8, fontweight="bold",
            ha="center", va="bottom", zorder=2,
        )

    ax.set_xlabel("Move Number", fontsize=8)
    ax.set_ylabel("Fidelity", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_ylim(0.48, 1.01)
    ax.set_xlim(0, 62)
    ax.grid(True, alpha=0.3)

    # Entropy inset
    if entropy:
        ax_inset = ax.inset_axes([0.52, 0.385, 0.45, 0.35])
        ex = sorted(int(m) for m in entropy.keys())
        ey = [entropy[str(m)]["mean"] for m in ex]
        ax_inset.plot(ex, ey, color="#7B68EE", linewidth=1.0)
        ax_inset.set_xlabel("Move", fontsize=6, labelpad=0)
        ax_inset.set_ylabel("H(game)", fontsize=6, labelpad=0)
        ax_inset.set_ylim(-0.05, 1.05)
        ax_inset.set_xlim(0, 60)
        ax_inset.tick_params(axis="both", which="major", labelsize=5, pad=1, length=2, width=0.5)
        ax_inset.set_xticks([0, 30, 60])
        ax_inset.set_yticks([0, 0.5, 1])
        ax_inset.grid(True, alpha=0.3)
        ax_inset.set_facecolor("white")
        for spine in ax_inset.spines.values():
            spine.set_edgecolor("gray")
            spine.set_linewidth(0.6)


def plot_panel_c(ax: plt.Axes, cache: dict) -> None:
    """Panel (c): Causal steering normalized delta-alpha."""
    data = cache.get(RUN_NAME, {})
    if not data:
        logger.warning("No Panel C data found.")
        return

    moves = sorted(int(m) for m in data.keys())

    # Store data for label placement
    layer_data = {}

    for layer_idx in range(N_LAYERS):
        ys = []
        xs = []
        for m in moves:
            entry = data[str(m)]
            val = entry["means"][layer_idx]
            if val is not None:
                xs.append(m)
                ys.append(val)

        alpha_val = (layer_idx / N_LAYERS) + 0.25
        ax.plot(
            xs, ys,
            color=LAYER_COLORS[layer_idx],
            linewidth=1.0,
            alpha=min(alpha_val, 1.0),
        )
        layer_data[layer_idx] = (np.array(xs), np.array(ys))

    # Inline labels for layers that are distinguishable (4, 5, 6)
    inline_labels = {
        4: 10,   # Layer 5 - near peak (highest)
        5: 25,   # Layer 6 - moderate
        3: 15,   # Layer 4 - moderate
    }

    for layer_idx, x_pos in inline_labels.items():
        if layer_idx in layer_data:
            xs_arr, ys_arr = layer_data[layer_idx]
            idx = np.argmin(np.abs(xs_arr - x_pos))
            if idx < len(ys_arr):
                ax.text(
                    x_pos, ys_arr[idx], str(layer_idx + 1),
                    color=LAYER_COLORS[layer_idx], fontsize=7, fontweight="bold",
                    ha="center", va="center", path_effects=HALO,
                )

    # Bunched layers (1, 2, 3, 7, 8): staggered x-positions with exact y-values
    bunched_layer_x_positions = {
        0: 10,   # Layer 1
        1: 18,   # Layer 2
        2: 26,   # Layer 3
        6: 34,   # Layer 7
        7: 42,   # Layer 8
    }

    for layer_idx, x_pos in bunched_layer_x_positions.items():
        if layer_idx in layer_data:
            xs_arr, ys_arr = layer_data[layer_idx]
            idx = np.argmin(np.abs(xs_arr - x_pos))
            if idx < len(ys_arr):
                ax.text(
                    x_pos, ys_arr[idx], str(layer_idx + 1),
                    color=LAYER_COLORS[layer_idx], fontsize=6.5, fontweight="bold",
                    ha="center", va="center", path_effects=HALO,
                )

    ax.set_xlabel("Move Number", fontsize=8)
    ax.set_ylabel(r"$\Delta\alpha / \Delta\alpha_{\max}$", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_ylim(-0.12, 1.02)
    ax.set_xlim(5, 50)
    ax.grid(True, alpha=0.3)



def main() -> None:
    _setup_style()

    # Load all three caches
    cache_a = load_json_cache(CACHE_DIR / "probe_accuracy_differing.json")
    cache_b = load_json_cache(CACHE_DIR / "game_probe_fidelity.json")
    cache_c = load_json_cache(CACHE_DIR / "steering_nomidflip.json")

    if not any([cache_a, cache_b, cache_c]):
        logger.error("No cached data found. Run compute scripts first.")
        return

    fig = plt.figure(figsize=(ICML_FULL_WIDTH, 2.5))

    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[1, 1, 1],
        wspace=0.35,
        left=0.06,
        right=0.99,
        top=0.90,
        bottom=0.28,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    plot_panel_a(ax_a, cache_a)
    plot_panel_b(ax_b, cache_b)
    plot_panel_c(ax_c, cache_c)

    # Panel labels - positioned above the plots
    label_y = 1.1
    ax_a.text(-0.15, label_y, "(a)", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold", va="bottom", ha="left")
    ax_b.text(-0.15, label_y, "(b)", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold", va="bottom", ha="left")
    ax_c.text(-0.15, label_y, "(c)", transform=ax_c.transAxes,
              fontsize=10, fontweight="bold", va="bottom", ha="left")

    fig.patch.set_facecolor('white')
    save_figure(fig, FIGURE_DIR / "figure5_combined")
    logger.info("Figure saved to %s", FIGURE_DIR)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
