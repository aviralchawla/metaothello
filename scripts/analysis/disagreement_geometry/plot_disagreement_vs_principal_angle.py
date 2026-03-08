"""Plot Figure 9 right panel: P(Disagreement) vs First Principal Angle.

Scatter plot showing the correlation between per-tile disagreement
probability and the first principal angle between mine/yours probe
subspaces for Classic vs NoMidFlip, with one color per layer.

Reads data/analysis_cache/disagreement_geometry.json.
Outputs to figures/disagreement_geometry/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from metaothello.analysis_utils import CACHE_DIR, load_json_cache
from metaothello.plotting import save_figure

logger = logging.getLogger(__name__)

CACHE_FILE: Final[Path] = CACHE_DIR / "disagreement_geometry.json"
FIGURE_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "figures" / "disagreement_geometry"
N_LAYERS: Final[int] = 8
RUN_NAME: Final[str] = "classic_nomidflip"


def _setup_style() -> None:
    """Configure matplotlib to match the published figure style."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
            "figure.dpi": 300,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_disagreement_vs_principal_angle(cache: dict) -> None:
    """Create and save the scatter plot."""
    data = cache.get(RUN_NAME)
    if not data:
        logger.error("No data for %s in cache.", RUN_NAME)
        return

    disagreement_prob = np.array(data["disagreement_prob"])  # (64,)
    first_angles = np.array(data["principal_angles_first"])  # (8, 64)
    r_squared = data["r_squared_principal_angle"]  # list of 8 floats

    _setup_style()
    fig, ax = plt.subplots(figsize=(5, 4))

    cmap = plt.cm.plasma

    for layer in range(N_LAYERS):
        color = cmap(layer / (N_LAYERS - 1))
        ax.scatter(
            disagreement_prob,
            first_angles[layer],
            c=[color],
            s=30,
            alpha=0.7,
            label=f"L{layer + 1} (R\u00b2={r_squared[layer]:.2f})",
            edgecolors="white",
            linewidths=0.3,
        )

    ax.set_xlabel("P(Disagreement)")
    ax.set_ylabel("First Principal Angle (\u00b0)")
    ax.tick_params(axis="both", labelsize=11)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=8,
        ncol=4,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.8,
        markerscale=0.9,
    )

    ax.set_xlim(-0.005, disagreement_prob.max() * 1.1)
    ax.set_ylim(0, first_angles.max() * 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(
        fig,
        FIGURE_DIR / "disagreement_vs_principal_angle_combined_Classic_NoMidFlip",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot P(Disagreement) vs First Principal Angle (Figure 9 right)."
    )
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
        plot_disagreement_vs_principal_angle(cache)
