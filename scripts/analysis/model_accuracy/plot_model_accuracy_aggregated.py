"""Plot aggregated model accuracy (mean over all move positions) as a grouped bar chart."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

_SINGLE_RUNS = ["classic", "nomidflip", "delflank", "iago"]
_MIXED_RUNS = ["classic_nomidflip", "classic_delflank", "classic_iago"]
_GAME_ORDER = ["classic", "nomidflip", "delflank", "iago"]


def _agg_stats(gdata: dict) -> tuple[float, float]:
    """Return (mean, 95%-CI half-width) aggregated over 59 move positions."""
    means = np.array(gdata["means"])
    std_errs = np.array(gdata["std_errs"])
    agg_mean = float(np.mean(means))
    agg_se = float(np.sqrt(np.sum(std_errs**2)) / len(std_errs))
    return agg_mean, agg_se * 1.96


def _draw_bars(
    ax: plt.Axes,
    run_names: list[str],
    cache: dict,
    metric: Metric,
    bar_width: float,
) -> dict[str, plt.Artist]:
    """Draw bars on ax, return {game_label: patch} for legend construction."""
    x = np.arange(len(run_names))
    handle_map: dict[str, plt.Artist] = {}
    for run_idx, run_name in enumerate(run_names):
        cache_key = f"{run_name}__{metric.value}"
        game_list = list(cache.get(cache_key, {}).get("games", {}).items())
        n = len(game_list)
        if n == 0:
            continue
        offsets = np.linspace(-(n - 1) * bar_width / 2, (n - 1) * bar_width / 2, n)
        for bar_idx, (game_alias, gdata) in enumerate(game_list):
            agg_mean, ci = _agg_stats(gdata)
            label = GAME_LABELS[game_alias]
            bar = ax.bar(
                x[run_idx] + offsets[bar_idx],
                agg_mean,
                width=bar_width,
                color=GAME_COLORS[game_alias],
                yerr=ci,
                capsize=2,
                label=label,
            )
            if label not in handle_map:
                handle_map[label] = bar
    ax.set_xticks(x)
    ax.set_xticklabels([RUN_LABELS[r] for r in run_names])
    return handle_map


def plot_model_accuracy_aggregated(cache: dict, metric: Metric) -> None:
    """Create and save the aggregated accuracy bar chart."""
    setup_icml_style()

    fig, (ax_single, ax_mixed) = plt.subplots(
        1,
        2,
        figsize=(ICML_FULL_WIDTH, ICML_FULL_WIDTH * 0.38),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [4, 3]},
    )

    handle_map: dict[str, plt.Artist] = {}
    handle_map.update(_draw_bars(ax_single, _SINGLE_RUNS, cache, metric, bar_width=0.5))
    handle_map.update(_draw_bars(ax_mixed, _MIXED_RUNS, cache, metric, bar_width=0.3))

    ax_single.set_title("Single-Game")
    ax_mixed.set_title("Mixed-Game")

    metric_ylabel = "Top-1 Accuracy" if metric == Metric.TOP1 else "Valid-Move Probability"
    ax_single.set_ylabel(metric_ylabel)
    fig.supxlabel("Model")

    # Shared legend above both panels, ordered canonically
    ordered_handles = [
        handle_map[GAME_LABELS[g]] for g in _GAME_ORDER if GAME_LABELS[g] in handle_map
    ]
    ordered_labels = [GAME_LABELS[g] for g in _GAME_ORDER if GAME_LABELS[g] in handle_map]
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        ncol=len(ordered_labels),
        bbox_to_anchor=(0.5, 1.06),
        frameon=False,
    )

    save_figure(
        fig,
        FIGURES_DIR / "model_accuracy" / f"model_accuracy_aggregated_{metric.value}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot aggregated model accuracy from cached results.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=Metric.TOP1.value,
        choices=[m.value for m in Metric if m != Metric.ALPHA],
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

    plot_model_accuracy_aggregated(cache, Metric(args.metric))
