"""Per-layer cosine similarity scatter between mixed probe weights, by tile state."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from metaothello.plotting import GAME_LABELS, ICML_HALF_WIDTH, save_figure, setup_icml_style

logger = logging.getLogger(__name__)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DATA_DIR: Final[Path] = REPO_ROOT / "data"
OUTPUT_STEM: Final[Path] = REPO_ROOT / "figures" / "probe_weight_similarity" / "cosine_sim_by_layer"

_LAYERS: Final[list[int]] = list(range(1, 9))
_STATE_NAMES: Final[list[str]] = ["Opponent", "Empty", "Mine"]
_MARKERS: Final[dict[str, str]] = {"Opponent": "^", "Empty": "s", "Mine": "o"}
_STATE_COLORS: Final[dict[str, str]] = {
    "Opponent": "#d62728",  # vivid red
    "Empty": "#9467bd",  # purple
    "Mine": "#f28e2b",  # orange
}
_N_TILES: Final[int] = 64
_N_STATES: Final[int] = 3


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


_STATE_X_OFFSETS: Final[dict[str, float]] = {"Opponent": -0.25, "Empty": 0.0, "Mine": 0.25}


def _compute_layer_sims(run_name: str, variant_alias: str) -> np.ndarray:
    """Load per-layer, per-tile cosine similarities.

    Args:
        run_name: Mixed run name (e.g., ``'classic_nomidflip'``).
        variant_alias: Variant probe alias (``'nomidflip'`` or ``'delflank'``).

    Returns:
        Array of shape ``(8, 64, 3)`` - layers x tiles x states.
    """
    all_sims = np.zeros((len(_LAYERS), _N_TILES, _N_STATES))

    for row_idx, layer in enumerate(_LAYERS):
        classic_w = _load_probe_weights(_probe_path(run_name, "classic", layer))
        variant_w = _load_probe_weights(_probe_path(run_name, variant_alias, layer))
        sims = F.cosine_similarity(classic_w, variant_w, dim=1).cpu().numpy()  # (192,)
        all_sims[row_idx] = sims.reshape(_N_TILES, _N_STATES)  # (64, 3)

    return all_sims


def _plot_layer_panel(
    ax: plt.Axes,
    run_name: str,
    variant_alias: str,
    pair_title: str,
    *,
    show_ylabel: bool,
) -> None:
    """Render one cosine-similarity-by-layer scatter panel.

    Args:
        ax: Target axes.
        run_name: Mixed run name.
        variant_alias: Variant probe alias.
        pair_title: Panel title string.
        show_ylabel: Whether to draw the y-axis label.
    """
    sims = _compute_layer_sims(run_name, variant_alias)  # (8, 64, 3)

    for state_idx, state_name in enumerate(_STATE_NAMES):
        color = _STATE_COLORS[state_name]
        marker = _MARKERS[state_name]
        x_offset = _STATE_X_OFFSETS[state_name]

        for layer_idx, layer in enumerate(_LAYERS):
            vals = sims[layer_idx, :, state_idx]  # (64,)
            x = np.full(len(vals), layer + x_offset)
            ax.scatter(
                x,
                vals,
                color=color,
                marker=marker,
                s=10,
                alpha=0.5,
                linewidths=0.4,
                # edgecolors="black",
                label=state_name if layer_idx == 0 else None,
            )

    ax.set_title(pair_title)
    ax.set_xlabel("Layer")
    if show_ylabel:
        ax.set_ylabel("Cosine Similarity")
    ax.set_xticks(_LAYERS)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0.0, 1.05)


def plot_cosine_sim_by_layer() -> None:
    """Create and save the per-layer cosine similarity figure."""
    setup_icml_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(ICML_HALF_WIDTH, ICML_HALF_WIDTH * 0.9),
        sharey=True,
        constrained_layout=True,
    )

    _plot_layer_panel(
        axes[0],
        "classic_nomidflip",
        "nomidflip",
        f"Classic vs. {GAME_LABELS['nomidflip']}",
        show_ylabel=False,
    )
    _plot_layer_panel(
        axes[1],
        "classic_delflank",
        "delflank",
        f"Classic vs. {GAME_LABELS['delflank']}",
        show_ylabel=False,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.06),
    )

    save_figure(fig, OUTPUT_STEM, tight=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot per-layer cosine similarity between mixed probe weights by state."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    plot_cosine_sim_by_layer()
