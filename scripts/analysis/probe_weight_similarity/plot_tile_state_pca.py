"""2D PCA of mixed probe weights colored by tile-state cosine similarity."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from metaothello.plotting import GAME_LABELS, ICML_FULL_WIDTH, save_figure, setup_icml_style

logger = logging.getLogger(__name__)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DATA_DIR: Final[Path] = REPO_ROOT / "data"
OUTPUT_STEM: Final[Path] = REPO_ROOT / "figures" / "probe_weight_similarity" / "tile_state_pca"

_LAYERS: Final[list[int]] = list(range(1, 9))
_STATE_NAMES: Final[list[str]] = ["Opponent", "Empty", "Mine"]
_MARKERS: Final[dict[str, str]] = {"Opponent": "^", "Empty": "s", "Mine": "o"}
_N_TILES: Final[int] = 64
_N_STATES: Final[int] = 3
_VMIN: Final[float] = 0.0
_VMAX: Final[float] = 1.0
_PT_SIZE: Final[float] = 8.0


def _probe_path(run_name: str, game_alias: str, layer: int) -> Path:
    """Return the board-probe checkpoint path for one (run, game, layer)."""
    return DATA_DIR / run_name / "board_probes" / f"{game_alias}_board_L{layer}.ckpt"


def _load_probe_weights(probe_path: Path) -> torch.Tensor:
    """Load proj.weight from a LinearProbe checkpoint.

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
        msg = f"Unexpected probe weight shape: {tuple(weight.shape)}"
        raise ValueError(msg)

    return weight.float()


def _compute_pca_data(
    run_name: str,
    variant_alias: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average probe weights across layers, then project to 2D via PCA.

    Args:
        run_name: Mixed run name (e.g., ``'classic_nomidflip'``).
        variant_alias: Variant probe alias (``'nomidflip'`` or ``'delflank'``).

    Returns:
        pca_coords: ``(384, 2)`` PCA scores; rows 0..191 = classic, 192..383 = variant.
        avg_sims: ``(192,)`` per-(tile, state) cosine similarity averaged across layers.
        state_indices: ``(192,)`` state index per row (0=Opponent, 1=Empty, 2=Mine).
    """
    classic_weights: list[np.ndarray] = []
    variant_weights: list[np.ndarray] = []
    layer_sims: list[np.ndarray] = []

    for layer in _LAYERS:
        classic_w = _load_probe_weights(_probe_path(run_name, "classic", layer))
        variant_w = _load_probe_weights(_probe_path(run_name, variant_alias, layer))
        classic_weights.append(classic_w.numpy())
        variant_weights.append(variant_w.numpy())
        sims = F.cosine_similarity(classic_w, variant_w, dim=1).cpu().numpy()  # (192,)
        layer_sims.append(sims)

    avg_classic = np.mean(classic_weights, axis=0)  # (192, d_model)
    avg_variant = np.mean(variant_weights, axis=0)  # (192, d_model)
    avg_sims = np.mean(layer_sims, axis=0)  # (192,)

    all_weights = np.vstack([avg_classic, avg_variant])  # (384, d_model)
    centered = all_weights - all_weights.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pca_coords: np.ndarray = centered @ vt[:2].T  # (384, 2)

    # Row index i has state = i % 3 (probe weight rows ordered as tile*3 + state)
    state_indices = np.arange(_N_TILES * _N_STATES) % _N_STATES  # (192,)
    return pca_coords, avg_sims, state_indices


def _plot_pca_panel(
    ax: plt.Axes,
    run_name: str,
    variant_alias: str,
    pair_title: str,
) -> ScalarMappable:
    """Render one PCA scatter panel for a (classic, variant) probe pair.

    Args:
        ax: Target axes.
        run_name: Mixed run name.
        variant_alias: Variant probe alias.
        pair_title: Panel title string.

    Returns:
        ScalarMappable suitable for a shared colorbar.
    """
    pca_coords, avg_sims, state_indices = _compute_pca_data(run_name, variant_alias)
    norm = Normalize(vmin=_VMIN, vmax=_VMAX)
    cmap = plt.get_cmap("viridis")

    for state_idx, state_name in enumerate(_STATE_NAMES):
        mask = state_indices == state_idx
        tile_idxs = np.where(mask)[0]  # positions in 0..191
        variant_idxs = tile_idxs + _N_TILES * _N_STATES  # positions in 192..383
        sim_vals = avg_sims[mask]
        marker = _MARKERS[state_name]

        # Classic probes: filled markers colored by cosine similarity
        ax.scatter(
            pca_coords[tile_idxs, 0],
            pca_coords[tile_idxs, 1],
            c=sim_vals,
            cmap=cmap,
            norm=norm,
            marker=marker,
            s=_PT_SIZE,
            linewidths=0.3,
            edgecolors="black",
            zorder=3,
        )
        # Variant probes: hollow markers, edge colored by cosine similarity
        ax.scatter(
            pca_coords[variant_idxs, 0],
            pca_coords[variant_idxs, 1],
            facecolors="none",
            edgecolors=cmap(norm(sim_vals)),
            marker=marker,
            s=_PT_SIZE,
            linewidths=0.6,
            zorder=2,
        )

    ax.set_title(pair_title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


def plot_tile_state_pca() -> None:
    """Create and save the tile-state probe PCA figure."""
    setup_icml_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(ICML_FULL_WIDTH, ICML_FULL_WIDTH * 0.5),
        constrained_layout=True,
    )

    sm = _plot_pca_panel(
        axes[0],
        "classic_nomidflip",
        "nomidflip",
        f"Classic vs. {GAME_LABELS['nomidflip']}",
    )
    _plot_pca_panel(
        axes[1],
        "classic_delflank",
        "delflank",
        f"Classic vs. {GAME_LABELS['delflank']}",
    )

    # Shared colorbar on the right
    cbar = fig.colorbar(sm, ax=axes, shrink=0.85, aspect=25, pad=0.02)
    cbar.set_label("Cosine Similarity (avg. across layers)")

    # Legend: shapes encode state; filled = classic probe, hollow = variant probe
    _gray = "#666666"
    legend_handles: list[Line2D] = [
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            markerfacecolor=_gray,
            markeredgecolor=_gray,
            markersize=5,
            label="Opponent",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor=_gray,
            markeredgecolor=_gray,
            markersize=5,
            label="Empty",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=_gray,
            markeredgecolor=_gray,
            markersize=5,
            label="Mine",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=_gray,
            markeredgecolor="black",
            markersize=5,
            label="Classic (filled)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor=_gray,
            markersize=5,
            label="Variant (hollow)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.45, -0.06),
    )

    save_figure(fig, OUTPUT_STEM, tight=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot 2D PCA of mixed probe weights colored by cosine similarity."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    plot_tile_state_pca()
