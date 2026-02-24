"""Plot aggregated model accuracy (mean over all move positions) as a horizontal bar chart.

Single panel with two row groups: single-game models (top) and mixed-game models
(bottom), separated by a dotted rule and annotated with bracket labels in the margin.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

from metaothello.analysis_utils import CACHE_DIR, Metric, load_json_cache
from metaothello.plotting import (
    GAME_COLORS,
    GAME_LABELS,
    ICML_HALF_WIDTH,
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

_BAR_H_SINGLE: float = 0.45
_BAR_H_MIXED: float = 0.28
_SECTION_GAP: float = 0.45  # vertical gap between mixed and single groups
_ROW_SPACING: float = 0.7  # spacing between rows within each group


def _agg_stats(gdata: dict) -> tuple[float, float]:
    """Return (mean, 95%-CI half-width) aggregated over 59 move positions."""
    means = np.array(gdata["means"])
    std_errs = np.array(gdata["std_errs"])
    agg_mean = float(np.mean(means))
    agg_se = float(np.sqrt(np.sum(std_errs**2)) / len(std_errs))
    return agg_mean, agg_se * 1.96


def _draw_bracket(
    ax: plt.Axes,
    y_lo: float,
    y_hi: float,
    label: str,
    x_line: float = -0.30,
    color: str = "#999999",
) -> None:
    """Draw a bracket + rotated label in the left margin using a blended transform.

    Args:
        ax: Target axes.
        y_lo: Bottom of the bracket span in data coordinates.
        y_hi: Top of the bracket span in data coordinates.
        label: Text to place at the midpoint of the bracket.
        x_line: X position of the vertical bracket line in axes-fraction coordinates.
        color: Line and text color.
    """
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    y_mid = (y_lo + y_hi) / 2
    tick_len = 0.018  # horizontal tick length in axes fraction

    # Vertical spine
    ax.plot(
        [x_line, x_line],
        [y_lo, y_hi],
        transform=trans,
        color=color,
        lw=0.8,
        clip_on=False,
        solid_capstyle="butt",
    )
    # Top and bottom ticks
    for y_tick in (y_lo, y_hi):
        ax.plot(
            [x_line, x_line + tick_len],
            [y_tick, y_tick],
            transform=trans,
            color=color,
            lw=0.8,
            clip_on=False,
        )
    # Label rotated 90°, to the left of the spine
    ax.text(
        x_line - 0.01,
        y_mid,
        label,
        transform=trans,
        ha="right",
        va="center",
        color=color,
        rotation=90,
        clip_on=False,
    )


def plot_model_accuracy_aggregated(cache: dict, metric: Metric) -> None:
    """Create and save the aggregated accuracy horizontal bar chart."""
    setup_icml_style()

    fig, ax = plt.subplots(
        figsize=(ICML_HALF_WIDTH, ICML_HALF_WIDTH * 1.3),
        constrained_layout=True,
    )

    # -----------------------------------------------------------------------
    # Y positions: mixed runs at the bottom, single runs above with a gap.
    # Each group ordered bottom-to-top to match the _RUNS list top-to-bottom.
    # -----------------------------------------------------------------------
    y_mixed: dict[str, float] = {
        r: float(i) * _ROW_SPACING for i, r in enumerate(reversed(_MIXED_RUNS))
    }
    mixed_top = max(y_mixed.values()) + _BAR_H_MIXED / 2
    single_start = mixed_top + _SECTION_GAP + _BAR_H_SINGLE / 2
    y_single: dict[str, float] = {
        r: single_start + float(i) * _ROW_SPACING for i, r in enumerate(reversed(_SINGLE_RUNS))
    }

    legend_handles: dict[str, plt.Artist] = {}

    # --- Mixed-game bars (2 per run, one per game variant) ---
    for run_name in _MIXED_RUNS:
        cache_key = f"{run_name}__{metric.value}"
        game_list = list(cache.get(cache_key, {}).get("games", {}).items())
        n = len(game_list)
        if n == 0:
            continue
        y_center = y_mixed[run_name]
        offsets = np.linspace(-(n - 1) * _BAR_H_MIXED / 2, (n - 1) * _BAR_H_MIXED / 2, n)
        for idx, (game_alias, gdata) in enumerate(game_list):
            agg_mean, ci = _agg_stats(gdata)
            label = GAME_LABELS[game_alias]
            ax.barh(
                y_center + offsets[idx],
                agg_mean,
                height=_BAR_H_MIXED,
                color=GAME_COLORS[game_alias],
                xerr=ci,
                capsize=2,
            )
            if label not in legend_handles:
                legend_handles[label] = Patch(color=GAME_COLORS[game_alias])

    # --- Single-game bars (1 per run) ---
    for run_name in _SINGLE_RUNS:
        cache_key = f"{run_name}__{metric.value}"
        game_list = list(cache.get(cache_key, {}).get("games", {}).items())
        if not game_list:
            continue
        game_alias, gdata = game_list[0]
        agg_mean, ci = _agg_stats(gdata)
        label = GAME_LABELS[game_alias]
        ax.barh(
            y_single[run_name],
            agg_mean,
            height=_BAR_H_SINGLE,
            color=GAME_COLORS[game_alias],
            xerr=ci,
            capsize=2,
        )
        if label not in legend_handles:
            legend_handles[label] = Patch(color=GAME_COLORS[game_alias])

    # --- Y-axis tick labels ---
    all_y = list(y_single.values()) + list(y_mixed.values())
    all_labels = [RUN_LABELS[r] for r in list(y_single.keys()) + list(y_mixed.keys())]
    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labels)

    # --- Section bracket annotations in the left margin ---
    single_y_lo = min(y_single.values()) - _BAR_H_SINGLE / 2
    single_y_hi = max(y_single.values()) + _BAR_H_SINGLE / 2
    mixed_y_lo = min(y_mixed.values()) - _BAR_H_MIXED / 2
    mixed_y_hi = max(y_mixed.values()) + _BAR_H_MIXED / 2
    _draw_bracket(ax, single_y_lo, single_y_hi, "Single-Game")
    _draw_bracket(ax, mixed_y_lo, mixed_y_hi, "Mixed-Game")

    # --- Dotted separator between sections ---
    sep_y = (mixed_y_hi + single_y_lo) / 2
    ax.axhline(sep_y, color="#cccccc", lw=0.6, linestyle=":", zorder=1)

    # --- Axis limits and labels ---
    ax.set_xlim(left=0.9, right=1.0)
    ax.set_ylim(mixed_y_lo - 0.15, single_y_hi + 0.15)
    metric_ylabel = {
        Metric.TOP1: "Top-1 Accuracy",
        Metric.CORRECT_PROB: "Valid-Move Probability",
        Metric.ALPHA: "Alpha Score",
    }[metric]
    ax.set_xlabel(metric_ylabel)

    # --- Legend above the figure ---
    ordered_labels = [GAME_LABELS[g] for g in _GAME_ORDER if GAME_LABELS[g] in legend_handles]
    ordered_handles = [legend_handles[lbl] for lbl in ordered_labels]
    if ordered_handles:
        ax.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            ncol=len(ordered_handles),
            bbox_to_anchor=(0.5, -0.14),
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

    plot_model_accuracy_aggregated(cache, Metric(args.metric))
