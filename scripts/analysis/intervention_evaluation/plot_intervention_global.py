"""Plot grouped bar chart of global intervention errors: Null vs Correct vs Cross."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from metaothello.analysis_utils import CACHE_DIR, load_json_cache
from metaothello.plotting import (
    GAME_LABELS,
    ICML_HALF_WIDTH,
    save_figure,
    setup_icml_style,
)

logger = logging.getLogger(__name__)
FIGURES_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "figures" / "intervention_evaluation"
)

# Figure 2-style colors
_COLOR_NULL: Final[str] = "gray"
_COLOR_CORRECT: Final[str] = "#2863c2"  # Smart Blue (GAME_COLORS["classic"])
_COLOR_CROSS: Final[str] = "#f28e2b"  # Warm orange


def extract_global_stats(
    results: dict[str, list[float]],
) -> tuple[float, float, float, float, int]:
    """Extract global pooled statistics from per-tile-state results."""
    all_null, all_linear = [], []
    for vals in results.values():
        if len(vals) < 5:
            continue
        null_m, _, lin_m, _, n = vals
        if n > 0 and not np.isnan(null_m) and not np.isnan(lin_m):
            all_null.extend([null_m] * int(n))
            all_linear.extend([lin_m] * int(n))

    if not all_null:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0

    n_total = len(all_null)
    null_ci = 1.96 * float(np.std(all_null, ddof=1)) / np.sqrt(n_total) if n_total > 1 else 0.0
    lin_ci = 1.96 * float(np.std(all_linear, ddof=1)) / np.sqrt(n_total) if n_total > 1 else 0.0

    return float(np.mean(all_null)), null_ci, float(np.mean(all_linear)), lin_ci, n_total


def plot_global_comparison(cache: dict) -> None:
    """Create and save the figure."""
    setup_icml_style(fig_width=ICML_HALF_WIDTH)

    models = ["classic_nomidflip", "classic_delflank"]
    # Only process models that exist in cache
    models = [m for m in models if m in cache]
    if not models:
        logger.error("No valid models found in cache.")
        return

    title_dict = {
        "classic_nomidflip": "Classic vs NoMidFlip",
        "classic_delflank": "Classic vs DelFlank",
    }

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(ICML_HALF_WIDTH, ICML_HALF_WIDTH * 0.7),
        sharey=True,
        constrained_layout=True,
    )
    if len(models) == 1:
        axes = [axes]

    bar_width = 0.25

    for ax, model in zip(axes, models, strict=False):
        model_results = cache[model]
        g1, g2 = model.split("_")

        labels, null_m, null_s, correct_m, correct_s, cross_m, cross_s = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for game in [g1, g2]:
            correct_key, cross_key = f"{game}_correct", f"{game}_cross"
            if correct_key in model_results and cross_key in model_results:
                c_stats = extract_global_stats(model_results[correct_key])
                x_stats = extract_global_stats(model_results[cross_key])

                labels.append(GAME_LABELS.get(game, game.capitalize()))
                null_m.append(c_stats[0])
                null_s.append(c_stats[1])
                correct_m.append(c_stats[2])
                correct_s.append(c_stats[3])
                cross_m.append(x_stats[2])
                cross_s.append(x_stats[3])

        x = np.arange(len(labels))

        # Colors adhering to conventions
        ax.bar(
            x - bar_width,
            null_m,
            bar_width,
            yerr=null_s,
            capsize=2,
            color=_COLOR_NULL,
            alpha=0.7,
            linewidth=0,
        )
        ax.bar(
            x,
            correct_m,
            bar_width,
            yerr=correct_s,
            capsize=2,
            color=_COLOR_CORRECT,
            alpha=0.8,
            linewidth=0,
        )
        ax.bar(
            x + bar_width,
            cross_m,
            bar_width,
            yerr=cross_s,
            capsize=2,
            color=_COLOR_CROSS,
            alpha=0.55,
            linewidth=0,
        )

        ax.set_xlabel("Game Variant")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title_dict.get(model, model))

        if model == models[0]:
            ax.set_ylabel("Prediction Error")

        # Ensure annotations don't bleed off the top
        ax.margins(y=0.25)

        # Annotations (positioned just above the mean bar)
        offset = 0.1
        for i in range(len(labels)):
            if not np.isnan(null_m[i]):
                ax.text(
                    x[i] - bar_width,
                    null_m[i] + offset,
                    f"{null_m[i]:.1f}",
                    ha="center",
                    va="bottom",
                )
            if not np.isnan(correct_m[i]):
                ax.text(
                    x[i],
                    correct_m[i] + offset,
                    f"{correct_m[i]:.1f}",
                    ha="center",
                    va="bottom",
                )
            if not np.isnan(cross_m[i]):
                ax.text(
                    x[i] + bar_width,
                    cross_m[i] + offset,
                    f"{cross_m[i]:.1f}",
                    ha="center",
                    va="bottom",
                )

    legend_handles = [
        Patch(color=_COLOR_NULL, alpha=0.7, label="Null"),
        Patch(color=_COLOR_CORRECT, alpha=0.8, label="Correct"),
        Patch(color=_COLOR_CROSS, alpha=0.55, label="Cross"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.15),
    )

    save_figure(fig, FIGURES_DIR / "intervention_global_comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot global intervention comparison.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    cache_file = CACHE_DIR / "intervention_eval.json"
    cache = load_json_cache(cache_file)
    if not cache:
        logger.error("Cache file not found or empty: %s", cache_file)
    else:
        plot_global_comparison(cache)
