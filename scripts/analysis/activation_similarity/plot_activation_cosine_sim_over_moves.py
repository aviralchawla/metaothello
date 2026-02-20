"""Plot residual-stream activation cosine similarity over move positions.

Layout: 1x3 row of line plots, one panel per mixed-game model
(classic_nomidflip, classic_delflank, classic_iago).

Each panel shows cosine similarity between the mean resid_post activations of
the two component game types at each move position (1-59), one line per layer.
Layer colors are sampled from the magma colormap (near-black = layer 1,
warm peach = layer 8).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from metaothello.analysis_utils import BLOCK_SIZE, CACHE_DIR, load_json_cache
from metaothello.plotting import (
    ICML_FULL_WIDTH,
    RUN_LABELS,
    save_figure,
    setup_icml_style,
)

logger = logging.getLogger(__name__)
FIGURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "figures"
CACHE_FILE = CACHE_DIR / "activation_cosine_sim.json"

MOVE_POSITIONS = np.arange(1, BLOCK_SIZE + 1)
N_LAYERS = 8

MIXED_RUN_NAMES = ["classic_nomidflip", "classic_delflank", "classic_iago"]

# Magma colormap sampled at 8 evenly spaced positions (t=0.0-0.92).
# Layer 1 (earliest) = near-black; Layer 8 (latest) = warm peach.
_LAYER_COLORS: dict[int, str] = {
    1: "#000003",  # magma t=0.000
    2: "#1e1049",  # magma t=0.131
    3: "#55137d",  # magma t=0.263
    4: "#892881",  # magma t=0.394
    5: "#c03a75",  # magma t=0.526
    6: "#ee5d5d",  # magma t=0.657
    7: "#fd9969",  # magma t=0.789
    8: "#fdd89a",  # magma t=0.920
}


def plot_activation_cosine_sim_over_moves(cache: dict) -> None:
    """Create and save the activation cosine similarity figure.

    Args:
        cache: Loaded JSON cache from ``activation_cosine_sim.json``.
    """
    setup_icml_style()

    fig, axes = plt.subplots(
        1,
        len(MIXED_RUN_NAMES),
        figsize=(ICML_FULL_WIDTH, ICML_FULL_WIDTH * 0.35),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.10, bottom=0.17, top=0.82, wspace=0.08)

    for col_idx, run_name in enumerate(MIXED_RUN_NAMES):
        ax = axes[col_idx]
        entry = cache.get(run_name)

        ax.set_title(RUN_LABELS[run_name].replace("\n", " "))

        if col_idx == 0:
            ax.set_ylabel("Cosine Similarity")

        if entry is None:
            ax.set_visible(False)
            logger.warning("No cache entry for %s — panel left blank.", run_name)
            continue

        for layer in range(1, N_LAYERS + 1):
            sims = entry.get("resid_post", {}).get(str(layer))
            if sims is None:
                continue
            ax.plot(MOVE_POSITIONS, sims, color=_LAYER_COLORS[layer], linewidth=0.9)

    # Shared axis labels.
    fig.text(0.5, 0.04, "Move Number", ha="center")

    # Shared legend above figure.
    legend_handles = [
        Line2D([0], [0], color=_LAYER_COLORS[layer], linewidth=1.2, label=f"Layer {layer}")
        for layer in range(1, N_LAYERS + 1)
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        ncol=N_LAYERS,
        bbox_to_anchor=(0.5, 1.0),
    )

    save_figure(
        fig,
        FIGURES_DIR / "activation_similarity" / "activation_cosine_sim_over_moves",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot activation cosine similarity over move positions from cached results.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not CACHE_FILE.exists():
        logger.error(
            "No cache found at %s. Run compute_activation_cosine_sim.py first.", CACHE_FILE
        )
        raise SystemExit(1)

    cache = load_json_cache(CACHE_FILE)
    plot_activation_cosine_sim_over_moves(cache)
