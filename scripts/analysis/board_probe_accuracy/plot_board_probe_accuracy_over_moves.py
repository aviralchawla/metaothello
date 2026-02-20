"""Plot board probe accuracy over move positions as line plots, one line per layer.

Layout: 3x4 grid of panels, one per (model, game) combination.

- Row 0: single-game models (classic, nomidflip, delflank, iago).
- Row 1: classic_nomidflip x classic, classic_nomidflip x nomidflip,
         classic_delflank x classic, classic_delflank x delflank.
- Row 2: classic_iago x classic, classic_iago x iago, [legend], [hidden].

Each panel shows x = move number (1-59), y = mean board-square accuracy.
Eight lines (one per layer) are drawn with colors sampled from the viridis
colormap (lighter = earlier layers, darker = later layers).  Shaded bands
show 95% CI.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from metaothello.analysis_utils import CACHE_DIR
from metaothello.plotting import (
    GAME_LABELS,
    ICML_FULL_WIDTH,
    RUN_LABELS,
    save_figure,
    setup_icml_style,
)

logger = logging.getLogger(__name__)
FIGURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "figures"
CACHE_FILE = CACHE_DIR / "board_probe_accuracy.pkl"

MOVE_POSITIONS = np.arange(1, 60)  # move indices 1..59
N_LAYERS = 8

# Magma colormap sampled at 8 evenly spaced positions (t=0.0-0.92, trimmed to avoid near-white).
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

# Panel grid: each cell is (run_name, game_alias) or None (legend/hidden).
_PANEL_GRID: list[list[tuple[str, str] | None]] = [
    [
        ("classic", "classic"),
        ("nomidflip", "nomidflip"),
        ("delflank", "delflank"),
        ("iago", "iago"),
    ],
    [
        ("classic_nomidflip", "classic"),
        ("classic_nomidflip", "nomidflip"),
        ("classic_delflank", "classic"),
        ("classic_delflank", "delflank"),
    ],
    [
        ("classic_iago", "classic"),
        ("classic_iago", "iago"),
        None,  # legend slot
        None,  # hidden
    ],
]


def _panel_title(run_name: str, game_alias: str) -> str:
    """Return a compact panel title for the given (run, game) combination.

    Args:
        run_name: Model run name.
        game_alias: Game alias the probe was trained for.

    Returns:
        Title string.
    """
    run_label = RUN_LABELS[run_name].replace("\n", " ")
    game_label = GAME_LABELS[game_alias]
    if run_name == game_alias:
        # Single-game panel: run and game are the same, no need to repeat.
        return run_label
    return f"{run_label}\n{game_label} probe"


def plot_board_probe_accuracy_over_moves(cache: dict) -> None:
    """Create and save the board probe accuracy over moves figure.

    Args:
        cache: Loaded pickle cache from ``board_probe_accuracy.pkl``.
    """
    setup_icml_style()

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(ICML_FULL_WIDTH, ICML_FULL_WIDTH * 0.8),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.08, bottom=0.09, top=0.95, hspace=0.45, wspace=0.08)

    for row_idx, row_panels in enumerate(_PANEL_GRID):
        for col_idx, panel in enumerate(row_panels):
            ax = axes[row_idx, col_idx]

            if panel is None:
                ax.set_visible(False)
                continue

            run_name, game_alias = panel
            cache_key = f"{run_name}__{game_alias}"
            entry = cache.get(cache_key)

            if entry is None:
                ax.set_visible(False)
                logger.warning("No cache entry for %s — panel left blank.", cache_key)
                continue

            for layer in range(1, N_LAYERS + 1):
                ldata = entry["layers"].get(layer)
                if ldata is None:
                    continue
                means = np.array(ldata["means"])
                std_errs = np.array(ldata["std_errs"])
                ci = std_errs * 1.96
                color = _LAYER_COLORS[layer]

                ax.plot(MOVE_POSITIONS, means, color=color, linewidth=0.9)
                ax.fill_between(
                    MOVE_POSITIONS,
                    means - ci,
                    means + ci,
                    color=color,
                    alpha=0.2,
                )

            ax.set_title(_panel_title(run_name, game_alias))

    # Shared axis labels.
    fig.text(0.5, 0.02, "Move Number", ha="center")
    fig.text(0.01, 0.5, "Board Accuracy", va="center", rotation="vertical")

    # Legend centered across the two unused slots (2, 2) and (2, 3).
    pos_l = axes[2, 2].get_position()
    pos_r = axes[2, 3].get_position()
    ax_legend = fig.add_axes([pos_l.x0, pos_l.y0, pos_r.x1 - pos_l.x0, pos_l.height])
    ax_legend.axis("off")
    legend_handles = [
        Line2D([0], [0], color=_LAYER_COLORS[layer], linewidth=1.2, label=f"Layer {layer}")
        for layer in range(1, N_LAYERS + 1)
    ]
    ax_legend.legend(
        handles=legend_handles,
        frameon=False,
        loc="center",
        ncol=2,
    )

    save_figure(
        fig,
        FIGURES_DIR / "board_probe_accuracy" / "board_probe_accuracy_over_moves",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot board probe accuracy over move positions from cached results.",
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

    plot_board_probe_accuracy_over_moves(cache)
