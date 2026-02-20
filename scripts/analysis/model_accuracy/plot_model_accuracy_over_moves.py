"""Plot model accuracy over move positions as line plots with shaded error bands."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from metaothello.analysis_utils import CACHE_DIR, Metric, load_json_cache
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
CACHE_FILE = CACHE_DIR / "model_accuracy.json"

MOVE_POSITIONS = np.arange(1, 60)  # move indices 1..59

# Layout: row 0 = single-game models, row 1 = mixed-game models (+ legend slot)
_RUN_GRID: list[list[str | None]] = [
    ["classic", "nomidflip", "delflank", "iago"],
    ["classic_nomidflip", "classic_delflank", "classic_iago", None],
]


def plot_model_accuracy_over_moves(cache: dict, metric: Metric) -> None:
    """Create and save the accuracy-over-moves figure."""
    setup_icml_style()

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(ICML_FULL_WIDTH, ICML_FULL_WIDTH * 0.55),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.08, bottom=0.12, hspace=0.35, wspace=0.08)

    collected_handles: dict[str, Line2D] = {}

    for row_idx, row_runs in enumerate(_RUN_GRID):
        for col_idx, run_name in enumerate(row_runs):
            ax = axes[row_idx, col_idx]
            if run_name is None:
                # Use this slot for the shared legend
                ax.axis("off")
                continue

            cache_key = f"{run_name}__{metric.value}"
            entry = cache.get(cache_key, {})
            for game_alias, gdata in entry.get("games", {}).items():
                means_arr = np.array(gdata["means"])
                std_errs_arr = np.array(gdata["std_errs"])
                ci = std_errs_arr * 1.96
                color = GAME_COLORS[game_alias]
                label = GAME_LABELS[game_alias]
                (line,) = ax.plot(MOVE_POSITIONS, means_arr, color=color, label=label)
                ax.fill_between(
                    MOVE_POSITIONS,
                    means_arr - ci,
                    means_arr + ci,
                    color=color,
                    alpha=0.2,
                )
                if label not in collected_handles:
                    collected_handles[label] = line

            ax.set_title(RUN_LABELS[run_name])

    # Shared axis labels
    metric_ylabel = {
        Metric.TOP1: "Top-1 Accuracy",
        Metric.CORRECT_PROB: "Valid-Move Probability",
        Metric.ALPHA: "Alpha Score",
    }[metric]
    fig.text(0.5, 0.02, "Move Number", ha="center")
    fig.text(0.01, 0.5, metric_ylabel, va="center", rotation="vertical")

    # Legend in the unused 8th panel slot
    ax_legend = axes[1, 3]
    ordered_games = ["classic", "nomidflip", "delflank", "iago"]
    legend_handles = [
        collected_handles[GAME_LABELS[g]]
        for g in ordered_games
        if GAME_LABELS[g] in collected_handles
    ]
    legend_labels = [GAME_LABELS[g] for g in ordered_games if GAME_LABELS[g] in collected_handles]
    ax_legend.legend(legend_handles, legend_labels, frameon=False, loc="center")

    save_figure(
        fig,
        FIGURES_DIR / "model_accuracy" / f"model_accuracy_over_moves_{metric.value}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot model accuracy over move positions from cached results.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=Metric.TOP1.value,
        choices=[m.value for m in Metric],
        help="Metric to plot (default: top1).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    cache = load_json_cache(CACHE_FILE)
    if not cache:
        logger.error("No cache found at %s. Run compute_model_accuracy.py first.", CACHE_FILE)
        raise SystemExit(1)

    plot_model_accuracy_over_moves(cache, Metric(args.metric))
