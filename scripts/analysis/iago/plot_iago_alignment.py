"""Plot Classic-to-Iago activation alignment via orthogonal Procrustes (Figure 4).

Plots the mean Iago alpha score for each intervention layer over move positions,
alongside the Classic baseline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metaothello.analysis_utils import BLOCK_SIZE, CACHE_DIR, load_json_cache
from metaothello.plotting import (
    ICML_HALF_WIDTH,
    apply_axis_labels,
    save_figure,
    setup_icml_style,
    trim_axes,
)

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "figures"
CACHE_FILE = CACHE_DIR / "iago_alignment.json"

MOVE_POSITIONS = np.arange(1, BLOCK_SIZE + 1)

# Layer 1 (earliest) = near-black; Layer 8 (latest) = warm peach.
_LAYER_COLORS: dict[int, str] = {
    1: "#000003",
    2: "#1e1049",
    3: "#55137d",
    4: "#892881",
    5: "#c03a75",
    6: "#ee5d5d",
    7: "#fd9969",
    8: "#fdd89a",
}


def plot_iago_alignment(cache: dict) -> None:
    """Create and save the Iago Procrustes alignment figure."""
    setup_icml_style(fig_width=ICML_HALF_WIDTH)

    run_data = cache.get("classic_iago")
    if run_data is None:
        logger.error("No 'classic_iago' data found in cache.")
        return

    fig, ax = plt.subplots(figsize=(ICML_HALF_WIDTH, ICML_HALF_WIDTH * 0.75))

    # Baseline
    base_data = run_data["baseline"]
    base_means = np.array(base_data["means"])
    base_stds = np.array(base_data["stds"])

    # Paper specifies "Classic baseline" text near the line
    ax.plot(
        MOVE_POSITIONS,
        base_means,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Classic baseline",
    )
    ax.fill_between(
        MOVE_POSITIONS,
        base_means - base_stds,
        base_means + base_stds,
        color="black",
        alpha=0.1,
        linewidth=0,
    )

    # Adding label next to the line (approximate placement)
    # The plot in Figure 4 has the label on the line
    ax.text(
        40,
        base_means[39] + 0.005,
        "Classic baseline",
        color="black",
        fontsize=7,
        ha="center",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
    )

    layers_data = run_data.get("layers", {})
    for layer in range(1, 9):
        ld = layers_data.get(str(layer))
        if not ld:
            continue

        means = np.array(ld["means"])
        stds = np.array(ld["stds"])

        ax.plot(
            MOVE_POSITIONS, means, color=_LAYER_COLORS[layer], linewidth=1.2, label=f"Layer {layer}"
        )
        ax.fill_between(
            MOVE_POSITIONS,
            means - stds,
            means + stds,
            color=_LAYER_COLORS[layer],
            alpha=0.1,
            linewidth=0,
        )

    ax.set_ylim(0.90, 1.00)
    ax.set_xlim(0, 50)

    apply_axis_labels(ax, xlabel="Move Number", ylabel="Mean Iago Alpha Score")
    trim_axes(ax)

    # Recreate the legend structure from the paper
    # "Intervention layer" title, then 2 columns of layers
    handles, labels = ax.get_legend_handles_labels()
    # Filter out baseline from handles for the main legend
    layer_handles = [h for h, lbl in zip(handles, labels, strict=False) if lbl.startswith("Layer")]
    layer_labels = [lbl for h, lbl in zip(handles, labels, strict=False) if lbl.startswith("Layer")]

    leg = ax.legend(
        layer_handles,
        layer_labels,
        title="Intervention layer",
        loc="lower right",
        ncol=2,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.5,
    )
    leg._legend_box.align = "left"

    save_figure(fig, FIGURES_DIR / "iago" / "iago_alignment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not CACHE_FILE.exists():
        logger.error(f"No cache found at {CACHE_FILE}. Run compute script first.")
        raise SystemExit(1)

    cache = load_json_cache(CACHE_FILE)
    plot_iago_alignment(cache)
