"""Mean cosine similarity between board-probe weights by layer, with Procrustes alignment.

Reproduces Figure 2 from the paper: three-panel bar chart comparing raw and
Procrustes-aligned cosine similarity between Classic and variant probe weights
across layers, with row-shuffled random baseline controls.

Procrustes convention (matches paper): for each layer, find R ∈ O(d) via SVD of
(W_Variant)^T W_Classic = U S V^T, so R = U V^T.  Apply to the variant matrix:
W_Variant_aligned = W_Variant @ R ≈ W_Classic.  Report cos-sim between
W_Classic[i] and W_Variant_aligned[i].

Random baseline: shuffle the 192 rows of W_Variant (keeping trained vector
magnitudes and directions), apply the same Procrustes pipeline, and average over
_N_RAND permutations.  This measures expected alignment between randomly
mismatched tile-state probe directions.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from metaothello.plotting import GAME_LABELS, ICML_FULL_WIDTH, save_figure, setup_icml_style

logger = logging.getLogger(__name__)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DATA_DIR: Final[Path] = REPO_ROOT / "data"
OUTPUT_STEM: Final[Path] = (
    REPO_ROOT / "figures" / "probe_weight_similarity" / "cosine_sim_aggregated"
)

_LAYERS: Final[list[int]] = list(range(1, 9))
_N_TILES: Final[int] = 64
_N_STATES: Final[int] = 3
_N_PROBES: Final[int] = _N_TILES * _N_STATES  # 192

# Figure 2 colors: blue for raw, orange for Procrustes-aligned
_COLOR_RAW: Final[str] = "#2863c2"  # Smart Blue (GAME_COLORS["classic"])
_COLOR_PROC: Final[str] = "#f28e2b"  # Warm orange

_BAR_WIDTH: Final[float] = 0.38
_BAR_OFFSET: Final[float] = 0.20  # centers the two bars symmetrically at each tick
_CI_Z: Final[float] = 1.96  # z-score for 95% CI
_N_RAND: Final[int] = 100  # permutation trials for random baseline

# Panel definitions: (run_name, variant_alias, title)
_PAIRS: Final[list[tuple[str, str, str]]] = [
    ("classic_delflank", "delflank", f"Classic vs. {GAME_LABELS['delflank']}"),
    ("classic_iago", "iago", f"Classic vs. {GAME_LABELS['iago']}"),
    ("classic_nomidflip", "nomidflip", f"Classic vs. {GAME_LABELS['nomidflip']}"),
]


def _probe_path(run_name: str, game_alias: str, layer: int) -> Path:
    """Return the board-probe checkpoint path for one (run, game, layer)."""
    return DATA_DIR / run_name / "board_probes" / f"{game_alias}_board_L{layer}.ckpt"


def _load_probe_weights(probe_path: Path) -> np.ndarray:
    """Load proj.weight from a LinearProbe checkpoint as a float32 numpy array.

    Args:
        probe_path: Path to a probe checkpoint.

    Returns:
        Weight matrix of shape (192, d_model), dtype float32.
    """
    if not probe_path.exists():
        msg = f"Probe checkpoint not found: {probe_path}"
        raise FileNotFoundError(msg)

    state_dict = torch.load(probe_path, map_location="cpu")
    weight = state_dict.get("proj.weight")
    if weight is None:
        msg = f"Checkpoint missing 'proj.weight': {probe_path}"
        raise KeyError(msg)

    if weight.ndim != 2 or weight.shape[0] != _N_PROBES:
        msg = f"Unexpected probe weight shape: {tuple(weight.shape)}"
        raise ValueError(msg)

    return weight.float().numpy()


def _procrustes_align(classic_w: np.ndarray, variant_w: np.ndarray) -> np.ndarray:
    """Rotate variant probe weights to align with classic.

    Finds orthogonal R minimizing ||variant_w @ R - classic_w||_F via SVD of
    (variant_w)^T @ classic_w = U S V^T, giving R = U V^T.

    Args:
        classic_w: Classic probe weights, shape (192, d_model).
        variant_w: Variant probe weights to rotate, shape (192, d_model).

    Returns:
        Aligned variant weights of the same shape, (192, d_model).
    """
    cross_cov = variant_w.T @ classic_w  # (d_model, d_model)
    u_mat, _, v_t = np.linalg.svd(cross_cov, full_matrices=True)
    rotation = u_mat @ v_t  # orthogonal, minimises ||variant_w @ R - classic_w||_F
    return variant_w @ rotation


def _layer_cosine_sims(
    classic_w: np.ndarray,
    variant_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute raw and Procrustes-aligned cosine similarities for one layer.

    Args:
        classic_w: Classic probe weights, shape (192, d_model).
        variant_w: Variant probe weights, shape (192, d_model).

    Returns:
        raw_sims: (192,) cosine similarities before alignment.
        proc_sims: (192,) cosine similarities after alignment.
    """
    classic_t = torch.from_numpy(classic_w)
    variant_t = torch.from_numpy(variant_w)
    raw_sims: np.ndarray = F.cosine_similarity(classic_t, variant_t, dim=1).numpy()

    variant_aligned = _procrustes_align(classic_w, variant_w)
    variant_aligned_t = torch.from_numpy(variant_aligned.astype(np.float32))
    proc_sims: np.ndarray = F.cosine_similarity(classic_t, variant_aligned_t, dim=1).numpy()

    return raw_sims, proc_sims


def _layer_random_baseline(
    classic_w: np.ndarray,
    variant_w: np.ndarray,
    n_rand: int = _N_RAND,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate random baseline by row-shuffling variant probe weights.

    Replaces the correct pairing of tile-state probe directions with random
    pairings (keeping the trained weight vectors intact), measuring alignment
    of randomly mismatched probes — the same null hypothesis as the paper.

    Args:
        classic_w: Classic probe weights, shape (192, d_model).
        variant_w: Variant probe weights, shape (192, d_model).
        n_rand: Number of random permutations to average over.
        seed: RNG seed for reproducibility.

    Returns:
        raw_baseline: Expected raw cosine similarity under random pairing.
        proc_baseline: Expected Procrustes-aligned cosine similarity under random pairing.
    """
    rng = np.random.default_rng(seed)
    classic_t = torch.from_numpy(classic_w)

    raw_acc = 0.0
    proc_acc = 0.0
    for _ in range(n_rand):
        perm = rng.permutation(len(variant_w))
        shuffled = variant_w[perm].astype(np.float32)

        shuffled_t = torch.from_numpy(shuffled)
        raw_acc += float(F.cosine_similarity(classic_t, shuffled_t, dim=1).mean().item())

        shuffled_aligned = _procrustes_align(classic_w, shuffled).astype(np.float32)
        shuffled_aligned_t = torch.from_numpy(shuffled_aligned)
        proc_acc += float(F.cosine_similarity(classic_t, shuffled_aligned_t, dim=1).mean().item())

    return raw_acc / n_rand, proc_acc / n_rand


def _layer_gaussian_random_baseline(
    classic_w: np.ndarray,
    variant_w: np.ndarray,
    n_rand: int = _N_RAND,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate random baseline using truly random Gaussian probe weights.

    Generates pairs of i.i.d. N(0, 1) matrices of the same shape as the real
    probe weights and computes raw and Procrustes-aligned cosine similarities.
    Unlike the shuffled baseline, weight vectors are fully random rather than
    drawn from the trained probe distribution.

    Args:
        classic_w: Classic probe weights — used only to determine shape (192, d_model).
        variant_w: Unused; kept for a uniform call signature with the shuffled baseline.
        n_rand: Number of random matrix pairs to average over.
        seed: RNG seed for reproducibility.

    Returns:
        raw_baseline: Expected raw cosine similarity for random Gaussian probes.
        proc_baseline: Expected Procrustes-aligned cosine similarity for random Gaussian probes.
    """
    rng = np.random.default_rng(seed)
    n, d = classic_w.shape

    raw_acc = 0.0
    proc_acc = 0.0
    for _ in range(n_rand):
        a_mat = rng.standard_normal((n, d)).astype(np.float32)
        b_mat = rng.standard_normal((n, d)).astype(np.float32)

        a_t = torch.from_numpy(a_mat)
        b_t = torch.from_numpy(b_mat)
        raw_acc += float(F.cosine_similarity(a_t, b_t, dim=1).mean().item())

        b_aligned = _procrustes_align(a_mat, b_mat).astype(np.float32)
        b_aligned_t = torch.from_numpy(b_aligned)
        proc_acc += float(F.cosine_similarity(a_t, b_aligned_t, dim=1).mean().item())

    return raw_acc / n_rand, proc_acc / n_rand


def _compute_panel_data(
    run_name: str,
    variant_alias: str,
    random_mode: str = "shuffled",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load probes and compute all quantities needed for one panel.

    Args:
        run_name: Mixed run name (e.g., ``'classic_nomidflip'``).
        variant_alias: Variant probe alias (e.g., ``'nomidflip'``).
        random_mode: ``'shuffled'`` (default) uses row-permuted trained weights;
            ``'gaussian'`` draws fully random N(0,1) matrices.

    Returns:
        raw_means, raw_ci, proc_means, proc_ci: Shape (8,) each — per-layer
            mean cosine similarity and 95% CI half-width.
        rand_raw: Shape (8,) — per-layer raw random baseline.
        rand_proc: Shape (8,) — per-layer Procrustes random baseline.
    """
    _baseline_fn = (
        _layer_random_baseline if random_mode == "shuffled" else _layer_gaussian_random_baseline
    )
    n_layers = len(_LAYERS)
    raw_means = np.zeros(n_layers)
    raw_ci = np.zeros(n_layers)
    proc_means = np.zeros(n_layers)
    proc_ci = np.zeros(n_layers)
    rand_raw = np.zeros(n_layers)
    rand_proc = np.zeros(n_layers)

    for i, layer in enumerate(_LAYERS):
        classic_w = _load_probe_weights(_probe_path(run_name, "classic", layer))
        variant_w = _load_probe_weights(_probe_path(run_name, variant_alias, layer))

        raw_sims, proc_sims = _layer_cosine_sims(classic_w, variant_w)
        raw_means[i] = raw_sims.mean()
        raw_ci[i] = _CI_Z * raw_sims.std() / np.sqrt(len(raw_sims))
        proc_means[i] = proc_sims.mean()
        proc_ci[i] = _CI_Z * proc_sims.std() / np.sqrt(len(proc_sims))

        rand_raw[i], rand_proc[i] = _baseline_fn(classic_w, variant_w)

        logger.debug(
            "Layer %d: raw=%.3f±%.3f  proc=%.3f±%.3f  " "rand_raw=%.3f  rand_proc=%.3f",
            layer,
            raw_means[i],
            raw_ci[i],
            proc_means[i],
            proc_ci[i],
            rand_raw[i],
            rand_proc[i],
        )

    return raw_means, raw_ci, proc_means, proc_ci, rand_raw, rand_proc


def _plot_panel(
    ax: plt.Axes,
    run_name: str,
    variant_alias: str,
    pair_title: str,
    *,
    show_ylabel: bool,
    random_mode: str = "shuffled",
) -> None:
    """Render one cosine-similarity bar chart panel.

    Args:
        ax: Target axes.
        run_name: Mixed run name.
        variant_alias: Variant probe alias.
        pair_title: Panel title.
        show_ylabel: Whether to draw the y-axis label.
        random_mode: Passed to ``_compute_panel_data``; ``'shuffled'`` or ``'gaussian'``.
    """
    raw_means, raw_ci, proc_means, proc_ci, rand_raw, rand_proc = _compute_panel_data(
        run_name, variant_alias, random_mode
    )
    x = np.arange(1, len(_LAYERS) + 1)

    ax.bar(
        x - _BAR_OFFSET,
        raw_means,
        width=_BAR_WIDTH,
        yerr=raw_ci,
        color=_COLOR_RAW,
        capsize=2,
        error_kw={"linewidth": 0.8},
    )
    ax.bar(
        x + _BAR_OFFSET,
        proc_means,
        width=_BAR_WIDTH,
        yerr=proc_ci,
        color=_COLOR_PROC,
        alpha=0.55,
        capsize=2,
        error_kw={"linewidth": 0.8},
    )

    # Per-layer random baseline markers
    ax.plot(
        x,
        rand_raw,
        marker="o",
        linestyle="",
        color="black",
        markersize=3.5,
        zorder=5,
    )
    ax.plot(
        x,
        rand_proc,
        marker="s",
        linestyle="",
        color="black",
        markersize=3.5,
        zorder=5,
    )

    ax.set_title(pair_title)
    ax.set_xlabel("Layer")
    if show_ylabel:
        ax.set_ylabel("Mean Cosine Similarity")
    ax.set_xticks(x)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0.0, 1.05)


def plot_cosine_sim_aggregated(random_mode: str = "shuffled") -> None:
    """Create and save Figure 2: mean probe cosine similarity by layer.

    Args:
        random_mode: ``'shuffled'`` (default) uses row-permuted trained weights as
            the random baseline; ``'gaussian'`` draws fully random N(0,1) matrices.
    """
    setup_icml_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(ICML_FULL_WIDTH, ICML_FULL_WIDTH * 0.33),
        sharey=True,
        constrained_layout=True,
    )

    for i, (ax, (run_name, variant_alias, title)) in enumerate(zip(axes, _PAIRS, strict=False)):
        logger.info("Computing panel: %s", title)
        _plot_panel(
            ax,
            run_name,
            variant_alias,
            title,
            show_ylabel=(i == 0),
            random_mode=random_mode,
        )

    legend_handles = [
        Patch(color=_COLOR_RAW, label="Raw"),
        Patch(color=_COLOR_PROC, alpha=0.55, label="Procrustes-aligned"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="black",
            markersize=4,
            label="Random baseline (raw)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color="black",
            markersize=4,
            label="Random baseline (Procrustes)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.12),
    )

    save_figure(fig, OUTPUT_STEM, tight=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot mean probe-weight cosine similarity by layer (Figure 2)."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--random-baseline",
        choices=["shuffled", "gaussian"],
        default="shuffled",
        help=(
            "Method for computing the random baseline. "
            "'shuffled' (default): row-permute the trained variant probe weights, "
            "preserving the weight distribution but randomising tile-state pairing. "
            "'gaussian': draw fully random N(0,1) matrices of the same shape."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    plot_cosine_sim_aggregated(random_mode=args.random_baseline)
