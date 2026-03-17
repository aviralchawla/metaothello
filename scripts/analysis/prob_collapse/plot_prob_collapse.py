"""Plot Figure 10: Probability collapse after disambiguation (NoMidFlip).

Line plot showing per-layer normalized delta-P(Classic) when a classic-only
move is played at the end of an ambiguous sequence, as a function of
the divergence move number.

Reads data/analysis_cache/prob_collapse.json.
Outputs to figures/prob_collapse/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from metaothello.analysis_utils import CACHE_DIR, load_json_cache
from metaothello.plotting import save_figure

logger = logging.getLogger(__name__)

CACHE_FILE: Final[Path] = CACHE_DIR / "prob_collapse.json"
FIGURE_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "figures" / "prob_collapse"
N_LAYERS: Final[int] = 8
RUN_NAME: Final[str] = "classic_nomidflip"

# Text halo for inline layer labels
HALO = [pe.withStroke(linewidth=2.5, foreground="white")]


def _setup_style() -> None:
    """Configure matplotlib to match the published figure style."""
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


def plot_prob_collapse(cache: dict) -> None:
    """Create and save the probability collapse figure."""
    data = cache.get(RUN_NAME)
    if not data:
        logger.error("No data for %s in cache.", RUN_NAME)
        return

    # Extract move numbers (skip 'params' key)
    move_keys = sorted(int(k) for k in data if k != "params")

    _setup_style()

    colors = plt.cm.plasma(np.linspace(0, 0.8, N_LAYERS))

    fig, ax = plt.subplots(figsize=(6, 4))

    for layer_idx in range(N_LAYERS):
        xs = []
        ys = []
        for move in move_keys:
            entry = data[str(move)]
            val = entry["means"][layer_idx]
            if val is not None:
                xs.append(move)
                ys.append(val)

        if not xs:
            continue

        color = colors[layer_idx]
        alpha = layer_idx / N_LAYERS + 0.1

        ax.plot(xs, ys, linestyle="-", color=color, alpha=alpha)

        # Inline layer labels every 6 moves, starting at index 4
        start_idx = 4
        stride = 6
        for x_idx in range(start_idx, len(xs), stride):
            ax.text(
                xs[x_idx],
                ys[x_idx],
                str(layer_idx + 1),
                color=color,
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                path_effects=HALO,
            )

    ax.set_title(r"$\Delta$ P(Classic) after intersection->classic move")
    ax.set_xlabel("Move Number")
    ax.set_ylabel(r"$\Delta P(Classic; t) / (1 - P(Classic; t - 1))$")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, FIGURE_DIR / "prob_collapse_NoMidFlip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot probability collapse figure (Figure 10).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    cache = load_json_cache(CACHE_FILE)
    if not cache:
        logger.error("Cache not found: %s. Run compute script first.", CACHE_FILE)
    else:
        plot_prob_collapse(cache)
