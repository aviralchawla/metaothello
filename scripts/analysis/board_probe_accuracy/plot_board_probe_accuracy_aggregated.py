"""Plot aggregated board probe accuracy (mean over positions) vs. layer.

Layout: 2x4 grid of panels.

- Row 0: single-game models (classic, nomidflip, delflank, iago).
- Row 1: mixed-game models (classic_nomidflip, classic_delflank, classic_iago)
  + one slot used for the shared legend.

Each panel shows x = layer (1-8), y = mean board-square accuracy aggregated
over all 59 move positions.  One line per game variant the model was probed on
(single-game panels have one line; mixed panels have two).  Line color follows
``GAME_COLORS``.  Shaded bands show 95% CI.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metaothello.analysis_utils import CACHE_DIR
from metaothello.plotting import (
    GAME_COLORS,
    GAME_LABELS,
    ICML_FULL_WIDTH,
    RUN_LABELS,
    save_figure,
    setup_icml_style,
)

logger = logging.getLogger(__name__)
FIGURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "figures"
CACHE_FILE = CACHE_DIR / "board_probe_accuracy.pkl"

_LAYERS = list(range(1, 9))
_GAME_ORDER = ["classic", "nomidflip", "delflank", "iago"]
_RUN_GRID: list[list[str | None]] = [
    ["classic", "nomidflip", "delflank", "iago"],
    ["classic_nomidflip", "classic_delflank", "classic_iago", None],
]


def _agg_layer_stats(layer_data: dict) -> tuple[float, float]:
    """Return (mean, 95%-CI half-width) aggregated over the 59 move positions.

    Args:
        layer_data: Dict with keys ``means`` and ``std_errs``, each a 59-element
            list of per-position values.

    Returns:
        Tuple of (aggregated mean, 95% CI half-width).
    """
    means = np.array(layer_data["means"])
    std_errs = np.array(layer_data["std_errs"])
    agg_mean = float(np.mean(means))
    agg_se = float(np.sqrt(np.sum(std_errs**2)) / len(std_errs))
    return agg_mean, agg_se * 1.96


def plot_board_probe_accuracy_aggregated(cache: dict) -> None:
    """Create and save the aggregated board probe accuracy figure.

    Args:
        cache: Loaded pickle cache from ``board_probe_accuracy.pkl``.
    """
    setup_icml_style()

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(ICML_FULL_WIDTH, ICML_FULL_WIDTH * 0.55),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.09, bottom=0.14, hspace=0.38, wspace=0.08)

    collected_handles: dict[str, plt.Artist] = {}

    for row_idx, row_runs in enumerate(_RUN_GRID):
        for col_idx, run_name in enumerate(row_runs):
            ax = axes[row_idx, col_idx]
            if run_name is None:
                ax.axis("off")
                continue

            # All cache entries belonging to this run.
            run_entries = {k: v for k, v in cache.items() if v.get("run_name") == run_name}

            for entry in run_entries.values():
                game_alias = entry["game_alias"]
                color = GAME_COLORS[game_alias]
                label = GAME_LABELS[game_alias]

                layer_means: list[float] = []
                layer_cis: list[float] = []
                for layer in _LAYERS:
                    ldata = entry["layers"].get(layer)
                    if ldata is None:
                        continue
                    agg_mean, ci = _agg_layer_stats(ldata)
                    layer_means.append(agg_mean)
                    layer_cis.append(ci)

                if not layer_means:
                    continue

                x = list(range(1, len(layer_means) + 1))
                y = np.array(layer_means)
                ci = np.array(layer_cis)

                (line,) = ax.plot(x, y, color=color, label=label, marker="o", markersize=2.5)
                ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.2)

                if label not in collected_handles:
                    collected_handles[label] = line

            ax.set_title(RUN_LABELS[run_name])
            ax.set_xticks(_LAYERS)

    # Shared axis labels.
    fig.text(0.5, 0.03, "Layer", ha="center")
    fig.text(0.01, 0.5, "Board Accuracy", va="center", rotation="vertical")

    # Legend in the unused 4th slot of row 1.
    ax_legend = axes[1, 3]
    ordered_handles = [
        collected_handles[GAME_LABELS[g]]
        for g in _GAME_ORDER
        if GAME_LABELS[g] in collected_handles
    ]
    ordered_labels = [GAME_LABELS[g] for g in _GAME_ORDER if GAME_LABELS[g] in collected_handles]
    ax_legend.legend(ordered_handles, ordered_labels, frameon=False, loc="center")

    save_figure(
        fig,
        FIGURES_DIR / "board_probe_accuracy" / "board_probe_accuracy_aggregated",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot aggregated board probe accuracy (mean over positions) vs. layer.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not CACHE_FILE.exists():
        logger.error("No cache found at %s. Run compute_board_probe_accuracy.py first.", CACHE_FILE)
        raise SystemExit(1)

    with CACHE_FILE.open("rb") as f:
        cache = pickle.load(f)

    plot_board_probe_accuracy_aggregated(cache)
