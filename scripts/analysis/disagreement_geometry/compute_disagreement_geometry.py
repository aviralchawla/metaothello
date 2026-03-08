"""Compute per-tile disagreement probability and probe weight geometry.

For bicompatible Classic-NoMidFlip sequences, computes:
1. Per-tile P(disagreement) — probability that a tile has opposite colors
   in the two games at a given position.
2. Per-layer cosine similarity between board probe weights (64 tiles x 3 states).
3. Per-layer first principal angle between mine/yours 2D subspaces.

Caches results to data/analysis_cache/disagreement_geometry.json.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from scipy import stats
from tqdm import tqdm

from metaothello.analysis_utils import (
    CACHE_DIR,
    generate_bicompatible_sequences,
    load_json_cache,
    save_json_cache,
)
from metaothello.games import GAME_REGISTRY

logger = logging.getLogger(__name__)

CACHE_FILE: Final[Path] = CACHE_DIR / "disagreement_geometry.json"
DATA_DIR: Final[Path] = CACHE_DIR.parent
N_LAYERS: Final[int] = 8
N_TILES: Final[int] = 64
N_STATES: Final[int] = 3  # opponent, empty, mine
RUN_NAME: Final[str] = "classic_nomidflip"


def _probe_path(run_name: str, game_alias: str, layer: int) -> Path:
    """Return the board-probe checkpoint path for one (run, game, layer)."""
    return DATA_DIR / run_name / "board_probes" / f"{game_alias}_board_L{layer}.ckpt"


def _load_probe_weights(probe_path: Path) -> np.ndarray:
    """Load proj.weight from a board probe checkpoint.

    Returns:
        Weight matrix of shape (192, d_model) as float32 numpy array.
    """
    if not probe_path.exists():
        msg = (
            f"Probe checkpoint not found: {probe_path}\n"
            "Download board probes first:\n"
            "  python scripts/download_probes.py board"
            " --run_name classic_nomidflip"
        )
        raise FileNotFoundError(msg)

    state_dict = torch.load(probe_path, map_location="cpu")
    weight = state_dict["proj.weight"]
    if weight.ndim != 2 or weight.shape[0] != N_TILES * N_STATES:
        msg = f"Unexpected probe weight shape: {tuple(weight.shape)}"
        raise ValueError(msg)

    return weight.float().numpy()


def compute_disagreement_probability(
    sequences: list[list[str]],
    game_classes: list[type],
) -> np.ndarray:
    """Compute per-tile probability of board disagreement.

    For each sequence and each intermediate position, replays both games
    and checks whether each tile has opposite colors (BLACK in one, WHITE
    in the other).  Returns the fraction of occupied-tile observations
    where disagreement occurs.

    Args:
        sequences: List of bicompatible move sequences.
        game_classes: Two game classes to compare.

    Returns:
        Array of shape (64,) with per-tile disagreement probabilities.
    """
    disagreement_counts = np.zeros(N_TILES)
    total_counts = np.zeros(N_TILES)

    for seq in tqdm(sequences, desc="Computing disagreement", leave=False):
        g1 = game_classes[0]()
        g2 = game_classes[1]()

        for move in seq:
            g1.play_move(move)
            g2.play_move(move)

            board1 = g1.board.flatten()  # (64,)
            board2 = g2.board.flatten()  # (64,)

            # Disagreement: one is BLACK(+1), the other is WHITE(-1)
            disagreement = (board1 * board2) == -1
            # Only count tiles occupied in at least one game
            occupied = (board1 != 0) | (board2 != 0)

            disagreement_counts += disagreement
            total_counts += occupied

    # Avoid division by zero for tiles never occupied
    mask = total_counts > 0
    result = np.zeros(N_TILES)
    result[mask] = disagreement_counts[mask] / total_counts[mask]
    return result


def compute_cosine_by_tile(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """Compute per-tile, per-state cosine similarity.

    Args:
        w1: Probe weights (192, d_model).
        w2: Probe weights (192, d_model).

    Returns:
        Array of shape (64, 3) with cosine similarities.
    """
    w1_t = torch.from_numpy(w1)
    w2_t = torch.from_numpy(w2)
    sims = F.cosine_similarity(w1_t, w2_t, dim=1).numpy()  # (192,)
    return sims.reshape(N_TILES, N_STATES)


def compute_principal_angle_2d(w1: np.ndarray, w2: np.ndarray, tile_idx: int) -> float:
    """Compute the first principal angle between mine/yours 2D subspaces.

    For tile ``tile_idx``, extracts the opponent (index 0) and mine
    (index 2) weight vectors from each probe and computes the first
    principal angle between the two 2D subspaces.

    Args:
        w1: Probe weights (192, d_model).
        w2: Probe weights (192, d_model).
        tile_idx: Tile index (0-63).

    Returns:
        First principal angle in degrees (0 = aligned, 90 = orthogonal).
    """
    start = tile_idx * N_STATES
    # Indices 0=opponent, 2=mine (skip 1=empty)
    indices = [start + 0, start + 2]

    v1 = w1[indices, :]  # (2, d_model)
    v2 = w2[indices, :]  # (2, d_model)

    # QR decomposition for orthonormal bases
    q1, r1 = np.linalg.qr(v1.T)  # q1: (d_model, 2)
    q2, r2 = np.linalg.qr(v2.T)  # q2: (d_model, 2)

    # Check rank
    min_rank = min(
        np.linalg.matrix_rank(r1, tol=1e-10),
        np.linalg.matrix_rank(r2, tol=1e-10),
    )
    if min_rank == 0:
        return 90.0

    # Principal angles via SVD of Q1^T @ Q2
    m = q1.T @ q2  # (2, 2)
    _, sigmas, _ = np.linalg.svd(m)
    sigmas = np.clip(sigmas, 0.0, 1.0)
    angles_deg = np.degrees(np.arccos(sigmas))

    return float(angles_deg[0])  # first (smallest) principal angle


def main(
    n_seqs_per_length: int,
    min_length: int,
    max_length: int,
    force: bool,
) -> None:
    """Run the full compute pipeline."""
    results = {} if force else load_json_cache(CACHE_FILE)

    if not force and RUN_NAME in results:
        logger.info("Results cached. Use --force to recompute.")
        return

    game_aliases = RUN_NAME.split("_")  # ["classic", "nomidflip"]
    game_classes = [GAME_REGISTRY[a] for a in game_aliases]

    # --- Phase 1: Generate bicompatible sequences and compute disagreement ---
    logger.info(
        "Generating bicompatible sequences (lengths %d-%d, %d per length)...",
        min_length,
        max_length,
        n_seqs_per_length,
    )
    all_sequences: list[list[str]] = []
    for seq_length in tqdm(range(min_length, max_length + 1), desc="Lengths"):
        seqs = generate_bicompatible_sequences(seq_length, game_classes, n_seqs_per_length)
        all_sequences.extend(seqs)

    logger.info("Total sequences: %d", len(all_sequences))

    disagreement_prob = compute_disagreement_probability(all_sequences, game_classes)
    logger.info(
        "Disagreement range: [%.4f, %.4f]",
        disagreement_prob.min(),
        disagreement_prob.max(),
    )

    # --- Phase 2: Load board probes and compute geometry ---
    layers = list(range(1, N_LAYERS + 1))
    all_cosines = np.zeros((N_LAYERS, N_TILES, N_STATES))
    all_first_angles = np.zeros((N_LAYERS, N_TILES))

    for layer_idx, layer in enumerate(layers):
        w_classic = _load_probe_weights(_probe_path(RUN_NAME, "classic", layer))
        w_variant = _load_probe_weights(_probe_path(RUN_NAME, "nomidflip", layer))

        all_cosines[layer_idx] = compute_cosine_by_tile(w_classic, w_variant)

        for tile in range(N_TILES):
            all_first_angles[layer_idx, tile] = compute_principal_angle_2d(
                w_classic, w_variant, tile
            )

    logger.info("Cosine sims computed. Shape: %s", all_cosines.shape)
    logger.info("Principal angles computed. Shape: %s", all_first_angles.shape)

    # --- Phase 3: Compute R-squared values ---
    avg_cosines = all_cosines.mean(axis=2)  # (N_LAYERS, N_TILES)
    r_sq_cosine = []
    r_sq_angle = []

    for layer_idx in range(N_LAYERS):
        r_cos, _ = stats.pearsonr(disagreement_prob, avg_cosines[layer_idx])
        r_sq_cosine.append(float(r_cos**2))

        r_ang, _ = stats.pearsonr(disagreement_prob, all_first_angles[layer_idx])
        r_sq_angle.append(float(r_ang**2))

    # --- Save ---
    results[RUN_NAME] = {
        "disagreement_prob": disagreement_prob.tolist(),
        "cosine_sims": all_cosines.tolist(),
        "principal_angles_first": all_first_angles.tolist(),
        "r_squared_cosine": r_sq_cosine,
        "r_squared_principal_angle": r_sq_angle,
        "params": {
            "n_sequences_per_length": n_seqs_per_length,
            "min_length": min_length,
            "max_length": max_length,
            "total_sequences": len(all_sequences),
        },
    }
    save_json_cache(results, CACHE_FILE)
    logger.info("Cached to %s", CACHE_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-tile disagreement probability and probe weight geometry."
    )
    parser.add_argument(
        "--n_seqs",
        type=int,
        default=1000,
        help="Bicompatible sequences per length (default: 1000).",
    )
    parser.add_argument("--min_length", type=int, default=6)
    parser.add_argument("--max_length", type=int, default=40)
    parser.add_argument("--force", action="store_true", help="Recompute even if cached.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main(args.n_seqs, args.min_length, args.max_length, args.force)
