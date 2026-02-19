"""Shared plotting utilities for ICML-compliant figures.

Provides a consistent visual style, color palette, and save helpers for all
MetaOthello analysis scripts and notebooks. Call :func:`setup_icml_style` once
at the top of any script before creating figures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ICML column widths (inches)
# ---------------------------------------------------------------------------
ICML_FULL_WIDTH: float = 6.75
ICML_HALF_WIDTH: float = 3.25

# ---------------------------------------------------------------------------
# Game color palette (Coolors: https://coolors.co/ff595e-ffca3a-8ac926-1982c4-6a4c93)
# ---------------------------------------------------------------------------
GAME_COLORS: dict[str, str] = {
    "classic": "#1982c4",  # Steel Blue
    "nomidflip": "#ffca3a",  # Golden Pollen
    "delflank": "#8ac926",  # Yellow Green
    "iago": "#6a4c93",  # Dusty Grape
}

# ---------------------------------------------------------------------------
# Human-readable display labels
# ---------------------------------------------------------------------------
GAME_LABELS: dict[str, str] = {
    "classic": "Classic",
    "nomidflip": "NoMidFlip",
    "delflank": "DelFlank",
    "iago": "Iago",
}

RUN_LABELS: dict[str, str] = {
    "classic": "Classic",
    "nomidflip": "NoMidFlip",
    "delflank": "DelFlank",
    "iago": "Iago",
    "classic_nomidflip": "Classic+\nNoMidFlip",
    "classic_delflank": "Classic+\nDelFlank",
    "classic_iago": "Classic+\nIago",
}


def setup_icml_style(
    fig_width: float = ICML_FULL_WIDTH,
    dpi: int = 300,
) -> None:
    """Configure matplotlib rcParams for ICML-compliant figures.

    Sets font sizes, line widths, tick sizes, DPI, and activates the seaborn
    ``ticks`` style with trimmed spines.  Call once per script before creating
    any figures.

    Args:
        fig_width: Target figure width in inches.
        dpi: Dots-per-inch for raster output.
    """
    sns.set_style("ticks")
    matplotlib.rcParams.update(
        {
            "figure.dpi": dpi,
            "figure.figsize": (fig_width, fig_width * 0.55),
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 8,
            "lines.linewidth": 1.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
        }
    )


def save_figure(fig: Figure, path: Path, *, tight: bool = True) -> None:
    """Save a figure as both PDF and PNG.

    Args:
        fig: Matplotlib Figure to save.
        path: Output path **without** extension.  Both ``path.pdf`` and
            ``path.png`` are written.
        tight: If True, use ``bbox_inches="tight"`` when saving.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(path.with_suffix(".pdf"), bbox_inches=bbox)
    fig.savefig(
        path.with_suffix(".png"),
        bbox_inches=bbox,
        dpi=matplotlib.rcParams.get("figure.dpi", 300),
    )
    logger.info("Saved figure: %s (.pdf + .png)", path)
    plt.close(fig)


def trim_axes(axes: Axes | list[Axes]) -> None:
    """Trim spines on one or more axes via :func:`seaborn.despine`.

    Args:
        axes: A single Axes or a list of Axes to trim.
    """
    ax_list = [axes] if isinstance(axes, Axes) else list(axes)
    for ax in ax_list:
        sns.despine(ax=ax)


def apply_axis_labels(
    ax: Axes,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
) -> None:
    """Set axis labels and optional title at ICML-compatible font sizes.

    Args:
        ax: The axes to label.
        xlabel: X-axis label string.
        ylabel: Y-axis label string.
        title: Axes title string (empty string omits title).
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
