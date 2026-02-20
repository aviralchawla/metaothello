"""Reproduce Figure 11: tile-level probe-weight cosine similarity heatmaps."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from matplotlib.figure import SubFigure

from metaothello.plotting import ICML_FULL_WIDTH, save_figure, setup_icml_style

logger = logging.getLogger(__name__)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DATA_DIR: Final[Path] = REPO_ROOT / "data"
OUTPUT_STEM: Final[Path] = (
    REPO_ROOT / "figures" / "probe_weight_similarity" / "tile_state_similarity_figure11"
)

_LAYERS: Final[list[int]] = list(range(1, 9))
_STATE_NAMES: Final[list[str]] = ["Opponent", "Empty", "Mine"]
_N_TILES: Final[int] = 64
_N_STATES: Final[int] = 3
_BOARD_DIM: Final[int] = 8
_VMIN: Final[float] = 0.0
_VMAX: Final[float] = 1.0


def _probe_path(run_name: str, game_alias: str, layer: int) -> Path:
    """Return the board-probe checkpoint path for one (run, game, layer)."""
    return DATA_DIR / run_name / "board_probes" / f"{game_alias}_board_L{layer}.ckpt"


def _load_probe_weights(probe_path: Path) -> torch.Tensor:
    """Load `proj.weight` from a LinearProbe checkpoint.

    Args:
        probe_path: Path to a probe checkpoint.

    Returns:
        Weight matrix of shape (192, d_model).
    """
    if not probe_path.exists():
        msg = f"Probe checkpoint not found: {probe_path}"
        raise FileNotFoundError(msg)

    state_dict = torch.load(probe_path, map_location="cpu")
    weight = state_dict.get("proj.weight")
    if weight is None:
        msg = f"Checkpoint missing 'proj.weight': {probe_path}"
        raise KeyError(msg)

    if weight.ndim != 2 or weight.shape[0] != _N_TILES * _N_STATES:
        msg = (
            "Unexpected probe weight shape at "
            f"{probe_path}: expected (192, d_model), got {tuple(weight.shape)}"
        )
        raise ValueError(msg)

    return weight.float()


def _compute_similarity_maps(run_name: str, variant_alias: str) -> dict[int, np.ndarray]:
    """Compute tile-level cosine similarity maps for all layers.

    Args:
        run_name: Mixed run containing both Classic and variant probes.
        variant_alias: Variant probe alias (`nomidflip` or `delflank`).

    Returns:
        Dict keyed by layer (1..8). Each value has shape (3, 8, 8) for
        Opponent / Empty / Mine tile-state maps.
    """
    per_layer: dict[int, np.ndarray] = {}

    for layer in _LAYERS:
        classic_w = _load_probe_weights(_probe_path(run_name, "classic", layer))
        variant_w = _load_probe_weights(_probe_path(run_name, variant_alias, layer))

        sims = F.cosine_similarity(classic_w, variant_w, dim=1).cpu().numpy()  # (192,)
        tile_state = sims.reshape(_N_TILES, _N_STATES)  # (64, 3)
        state_maps = np.stack(
            [
                tile_state[:, state_idx].reshape(_BOARD_DIM, _BOARD_DIM)
                for state_idx in range(_N_STATES)
            ],
            axis=0,
        )
        per_layer[layer] = state_maps

    return per_layer


def _annotate_cells(ax: plt.Axes, values: np.ndarray) -> None:
    """Write per-cell cosine values on top of a heatmap."""
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = float(values[row, col])
            text_color = "black" if value >= 0.55 else "white"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=2.7,
                color=text_color,
            )


def _plot_group(
    subfig: SubFigure,
    layer_maps: dict[int, np.ndarray],
    pair_title: str,
) -> None:
    """Render one side of Figure 11 (Classic vs one rule variant)."""
    gs = subfig.add_gridspec(
        nrows=len(_LAYERS),
        ncols=4,
        width_ratios=[1.0, 1.0, 1.0, 0.07],
        wspace=0.24,
        hspace=0.26,
    )
    color_ref = None

    for row_idx, layer in enumerate(_LAYERS):
        maps = layer_maps[layer]
        for state_idx, state_name in enumerate(_STATE_NAMES):
            ax = subfig.add_subplot(gs[row_idx, state_idx])
            heat = maps[state_idx]
            color_ref = ax.imshow(
                heat,
                cmap="viridis",
                vmin=_VMIN,
                vmax=_VMAX,
                interpolation="nearest",
            )
            _annotate_cells(ax, heat)

            ticks = np.arange(_BOARD_DIM)
            labels = [str(i) for i in range(1, _BOARD_DIM + 1)]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.tick_params(axis="both", labelsize=4, length=1.0, pad=0.2)

            if row_idx == 0:
                ax.set_title(state_name, pad=1.8)
            if state_idx == 0:
                ax.set_ylabel(f"L{layer}", rotation=0, labelpad=10, va="center")
            else:
                ax.set_ylabel("")
            if row_idx == len(_LAYERS) - 1:
                ax.set_xlabel("column", labelpad=1.0)
            else:
                ax.set_xlabel("")

    if color_ref is None:
        msg = "No heatmaps were plotted."
        raise RuntimeError(msg)

    cbar_ax = subfig.add_subplot(gs[:, 3])
    cbar = subfig.colorbar(color_ref, cax=cbar_ax)
    cbar.set_label("Cosine Similarity")
    cbar_ax.tick_params(labelsize=6, length=2.0)
    subfig.suptitle(pair_title, y=0.995)


def plot_figure_11() -> None:
    """Create and save the Figure 11 reproduction."""
    setup_icml_style()

    left_maps = _compute_similarity_maps("classic_nomidflip", "nomidflip")
    right_maps = _compute_similarity_maps("classic_delflank", "delflank")

    fig = plt.figure(figsize=(ICML_FULL_WIDTH, ICML_FULL_WIDTH * 1.34))
    left_subfig, right_subfig = fig.subfigures(1, 2, wspace=0.08)
    _plot_group(left_subfig, left_maps, "Classic vs. NoMidFlip")
    _plot_group(right_subfig, right_maps, "Classic vs. DelFlank")

    save_figure(fig, OUTPUT_STEM, tight=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot tile-level probe-weight similarity heatmaps (Figure 11)."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    plot_figure_11()
