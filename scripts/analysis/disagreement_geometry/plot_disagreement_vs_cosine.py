"""Plot Figure 9 left panel: P(Disagreement) vs Cosine Similarity.

Scatter plot showing the correlation between per-tile disagreement
probability and the cosine similarity of board probe weights for
Classic vs NoMidFlip, with one color per layer.

Reads data/analysis_cache/disagreement_geometry.json.
Outputs to figures/disagreement_geometry/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

from metaothello.analysis_utils import CACHE_DIR, load_json_cache

logger = logging.getLogger(__name__)

CACHE_FILE: Final[Path] = CACHE_DIR / "disagreement_geometry.json"
FIGURE_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "figures" / "disagreement_geometry"
N_LAYERS: Final[int] = 8
RUN_NAME: Final[str] = "classic_nomidflip"


def plot_disagreement_vs_cosine(cache: dict) -> None:
    """Create and save the scatter plot."""
    data = cache.get(RUN_NAME)
    if not data:
        logger.error("No data for %s in cache.", RUN_NAME)
        return

    disagreement_prob = np.array(data["disagreement_prob"])  # (64,)
    cosine_sims = np.array(data["cosine_sims"])  # (8, 64, 3)
    r_squared = data["r_squared_cosine"]  # list of 8 floats

    # Average cosine across 3 states for each (layer, tile)
    avg_cosines = cosine_sims.mean(axis=2)  # (8, 64)

    fig, ax = plt.subplots(figsize=(5, 4))

    cmap = plt.cm.plasma

    for layer in range(N_LAYERS):
        color = cmap(layer / (N_LAYERS - 1))
        ax.scatter(
            disagreement_prob,
            avg_cosines[layer],
            c=[color],
            s=30,
            alpha=0.7,
            label=f"L{layer + 1} (R\u00b2={r_squared[layer]:.2f})",
            edgecolors="white",
            linewidths=0.3,
        )

    ax.set_xlabel("P(Disagreement)", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
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
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out = FIGURE_DIR / "disagreement_vs_cosine_combined_Classic_NoMidFlip"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    logger.info("Saved figure: %s (.pdf + .png)", out)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot P(Disagreement) vs Cosine Similarity (Figure 9 left)."
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
        plot_disagreement_vs_cosine(cache)
