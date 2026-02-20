"""Evaluate board probe accuracy across all MetaOthello runs and cache results.

For each of the 7 trained models this script:

1. Generates fresh random games on the fly for each game the model was trained on
   (no zarr dependency).
2. Runs batched forward passes through the GPT model, capturing residual-stream
   activations at each of the 8 layers.
3. Evaluates the linear probe for every (run_name, game_alias, layer) combination.
4. Saves per-move-position results to
   ``data/analysis_cache/board_probe_accuracy.pkl``.

Results are stored as per-position arrays of length 59, so downstream scripts can
produce both aggregated and per-move figures from the same cache without re-running
inference.

On re-run, cached entries (keyed ``{run_name}__{game_alias}``) are reused.
Pass ``--force`` to recompute everything from scratch.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from metaothello.analysis_utils import (
    ALL_RUN_NAMES,
    BLOCK_SIZE,
    CACHE_DIR,
    VOCAB_SIZE,
    get_device,
    get_game_aliases,
)
from metaothello.constants import BOARD_DIM, MAX_STEPS
from metaothello.games import GAME_REGISTRY
from metaothello.mingpt.board_probe import LinearProbe
from metaothello.mingpt.tokenizer import Tokenizer
from metaothello.mingpt.utils import get_last_ckpt, load_model_from_ckpt

logger = logging.getLogger(__name__)

CACHE_FILE = CACHE_DIR / "board_probe_accuracy.pkl"
N_LAYERS = 8
_MAX_RETRIES = 1000


# ---------------------------------------------------------------------------
# Cache I/O (pickle — board states and per-layer arrays suit binary format)
# ---------------------------------------------------------------------------


def load_pickle_cache(cache_file: Path) -> dict[str, Any]:
    """Load a pickle cache file, returning an empty dict if it doesn't exist.

    Args:
        cache_file: Path to the pickle file.

    Returns:
        Loaded dict, or empty dict if the file is missing.
    """
    if cache_file.exists():
        with cache_file.open("rb") as f:
            return pickle.load(f)
    return {}


def save_pickle_cache(results: dict[str, Any], cache_file: Path) -> None:
    """Write results to a pickle cache file, creating parent directories.

    Args:
        results: Results dict to serialise.
        cache_file: Destination path.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("wb") as f:
        pickle.dump(results, f)
    logger.info("Results cached to %s", cache_file)


# ---------------------------------------------------------------------------
# Game generation with board states
# ---------------------------------------------------------------------------


def gen_games_with_boards(
    game_alias: str,
    num_games: int,
    tokenizer: Tokenizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate fresh random games and record per-step board states.

    Generates ``num_games`` complete games (exactly ``MAX_STEPS`` moves each).
    Token sequences are encoded with the tokenizer (Iago sequences use the
    shuffled token space, consistent with model training).  Board states are
    always in the physical (absolute) representation.

    Args:
        game_alias: Key into ``GAME_REGISTRY`` (e.g. ``"classic"``).
        num_games: Number of games to generate.
        tokenizer: Tokenizer instance for encoding move histories.

    Returns:
        Tuple of:

        - **seqs**: ``int32`` array of shape ``(num_games, MAX_STEPS)`` with
          token IDs.
        - **board_states**: ``float32`` array of shape
          ``(num_games, MAX_STEPS, BOARD_DIM, BOARD_DIM)`` with the board
          snapshot after each move (absolute representation: BLACK=-1,
          WHITE=1, EMPTY=0).

    Raises:
        RuntimeError: If a valid game cannot be generated within retries.
    """
    game_class = GAME_REGISTRY[game_alias]
    seqs: list[list[int]] = []
    board_states: list[np.ndarray] = []

    for _ in tqdm(range(num_games), desc=f"Generating {game_alias} games", leave=False):
        for _attempt in range(_MAX_RETRIES):
            g = game_class()  # type: ignore[reportCallIssue]
            g.generate_random_game()
            history = g.get_history()
            if len(history) == MAX_STEPS:
                seqs.append(tokenizer.encode(history))
                boards = np.array(g.get_board_history(), dtype=np.float32)  # (MAX_STEPS, 8, 8)
                board_states.append(boards)
                break
        else:
            msg = (
                f"Could not generate a {MAX_STEPS}-step game for "
                f"'{game_alias}' after {_MAX_RETRIES} retries."
            )
            raise RuntimeError(msg)

    return np.array(seqs, dtype=np.int32), np.array(board_states, dtype=np.float32)


# ---------------------------------------------------------------------------
# Turn masking
# ---------------------------------------------------------------------------


def apply_turn_mask(board_states: np.ndarray) -> np.ndarray:
    """Convert absolute board representation to current-player-relative.

    Multiplies board states at even timesteps by -1 and odd timesteps by +1,
    matching the representation the probes were trained on (notebook convention).

    Args:
        board_states: Float array of shape ``(N, T, BOARD_DIM, BOARD_DIM)``
            with values in ``{-1, 0, 1}``.

    Returns:
        Masked array of the same shape with values in ``{-1, 0, 1}``.
    """
    t = board_states.shape[1]
    turn_mask = np.ones(t, dtype=np.float32)
    turn_mask[::2] = -1.0
    return board_states * turn_mask.reshape(1, -1, 1, 1)


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------


def _resid_post_filter(name: str) -> bool:
    """Return True for TransformerLens hook names that capture resid_post.

    Args:
        name: Hook point name from TransformerLens.

    Returns:
        True if the hook captures residual stream post-MLP activations.
    """
    return "hook_resid_post" in name


def extract_layer_acts(
    model: HookedTransformer,
    seqs: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    """Extract residual-stream activations at each layer for all sequences.

    Uses TransformerLens ``run_with_cache`` with a ``resid_post`` filter to
    capture the residual stream after every block in a single forward pass.
    The model input is ``seqs[:, :-1]`` — the standard autoregressive shift
    of length T=59.

    Args:
        model: HookedTransformer in eval mode on ``device``.
        seqs: Token array of shape ``(N, MAX_STEPS)``.
        batch_size: Sequences per forward pass.
        device: Torch device the model lives on.

    Returns:
        List of ``N_LAYERS`` arrays each of shape ``(N, T, d_model)``.
        ``layer_acts[l]`` is the residual stream after layer ``l+1``
        (0-indexed list, 1-indexed layers).
    """
    n = len(seqs)
    device_type = device.type
    all_resids: list[np.ndarray] = []

    with torch.inference_mode():
        for start in tqdm(range(0, n, batch_size), desc="Extracting activations", leave=False):
            end = min(start + batch_size, n)
            x = torch.tensor(seqs[start:end, :-1], dtype=torch.long, device=device)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device_type == "cuda"
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                _, cache = model.run_with_cache(x, names_filter=_resid_post_filter)

            # stack_activation("resid_post") → (n_layers, B, T, d_model)
            # permute → (B, T, d_model, n_layers)
            stack = cache.stack_activation("resid_post").permute(1, 2, 3, 0)
            all_resids.append(stack.float().cpu().numpy())

    stacked = np.concatenate(all_resids, axis=0)  # (N, T, d_model, n_layers)
    n_layers = stacked.shape[-1]
    return [stacked[:, :, :, layer_idx] for layer_idx in range(n_layers)]


# ---------------------------------------------------------------------------
# Probe loading and accuracy computation
# ---------------------------------------------------------------------------


def load_probe(probe_path: Path, device: torch.device) -> LinearProbe:
    """Load a LinearProbe from a state-dict checkpoint file.

    Args:
        probe_path: Path to the ``.ckpt`` file containing the state dict.
        device: Device to load the probe onto.

    Returns:
        LinearProbe in eval mode on ``device``.
    """
    probe = LinearProbe(device=device)
    state_dict = torch.load(probe_path, map_location=device)
    probe.load_state_dict(state_dict)
    probe.eval()
    return probe


def compute_probe_accuracy(
    probe: LinearProbe,
    acts: np.ndarray,
    board_states_rel: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-position board accuracy for one probe on one set of activations.

    Feeds ``(N * T)`` activation vectors through the probe in batches and
    compares argmax predictions to the ground-truth board state, computing
    mean accuracy over the 64 board squares per game-position pair.

    Args:
        probe: LinearProbe in eval mode on ``device``.
        acts: Residual-stream activations of shape ``(N, T, d_model)``.
        board_states_rel: Relative board states of shape
            ``(N, T, BOARD_DIM, BOARD_DIM)`` with values in ``{-1, 0, 1}``.
        batch_size: Examples per forward pass through the probe.
        device: Device the probe lives on.

    Returns:
        Tuple of ``(means, std_errs)`` each of shape ``(T,)`` giving the
        per-move-position mean and standard error of board-square accuracy
        across all ``N`` games.
    """
    n, t, d = acts.shape
    num_squares = BOARD_DIM * BOARD_DIM

    acts_flat = acts.reshape(n * t, d)
    boards_flat = board_states_rel.reshape(n * t, num_squares).astype(np.int32)

    all_correct: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, n * t, batch_size):
            end = min(start + batch_size, n * t)
            x = torch.tensor(acts_flat[start:end], dtype=torch.float32, device=device)
            y_true = torch.tensor(boards_flat[start:end], dtype=torch.long, device=device)

            logits, _ = probe(x)  # (B, 64, 3)
            y_pred = logits.argmax(dim=-1) - 1  # (B, 64) in {-1, 0, 1}
            correct = (y_pred == y_true).float().mean(dim=-1)  # (B,) mean over squares
            all_correct.append(correct.cpu().numpy())

    per_game_pos = np.concatenate(all_correct).reshape(n, t)  # (N, T)
    means = per_game_pos.mean(axis=0)  # (T,)
    std_errs = per_game_pos.std(axis=0, ddof=1) / np.sqrt(n)  # (T,)
    return means, std_errs


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_all(num_games: int, batch_size: int, force: bool) -> None:
    """Run probe accuracy evaluation for all 7 run_names and cache results.

    For each run, for each game alias it was trained on, loads the model,
    extracts per-layer activations from freshly generated games, then evaluates
    each layer's probe.  Results are saved incrementally after each
    (run, game) pair so partial progress survives interruption.

    Args:
        num_games: Fresh games to generate per game variant.
        batch_size: Batch size for both GPT forward passes and probe evaluation.
        force: If True, recompute all entries even if cached.
    """
    results: dict[str, Any] = {} if force else load_pickle_cache(CACHE_FILE)
    tokenizer = Tokenizer()
    device = get_device()

    for run_name in ALL_RUN_NAMES:
        game_aliases = get_game_aliases(run_name)
        ckpt_dir = CACHE_DIR.parent / run_name / "ckpts"
        probe_dir = CACHE_DIR.parent / run_name / "board_probes"

        last_ckpt, last_epoch = get_last_ckpt(ckpt_dir)
        if last_ckpt is None:
            logger.warning("No checkpoint for %s — skipping.", run_name)
            continue

        for game_alias in game_aliases:
            cache_key = f"{run_name}__{game_alias}"
            if cache_key in results:
                logger.info("Skipping %s (cached). Use --force to recompute.", cache_key)
                continue

            # Verify all probe files exist before loading the model.
            probe_paths = {
                layer: probe_dir / f"{game_alias}_board_L{layer}.ckpt"
                for layer in range(1, N_LAYERS + 1)
            }
            missing = [str(p) for p in probe_paths.values() if not p.exists()]
            if missing:
                logger.warning(
                    "Missing probe files for %s/%s: %s — skipping.",
                    run_name,
                    game_alias,
                    missing,
                )
                continue

            logger.info("Evaluating %s on %s ...", run_name, game_alias)

            # Generate fresh games and board states (no zarr).
            logger.info("  Generating %d %s games ...", num_games, game_alias)
            seqs, board_states = gen_games_with_boards(game_alias, num_games, tokenizer)
            board_states_rel = apply_turn_mask(board_states)  # (N, MAX_STEPS, 8, 8)

            # Load model and capture per-layer activations.
            model = load_model_from_ckpt(last_ckpt, VOCAB_SIZE, BLOCK_SIZE, as_tlens=True)
            model = model.to(device)
            model.eval()
            logger.info("  Loaded %s epoch %d on %s", run_name, last_epoch, device)

            layer_acts = extract_layer_acts(model, seqs, batch_size, device)
            # (list of N_LAYERS arrays, each (N, 59, 512))

            # Free model memory before probe evaluation.
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # board states for the 59 model-input positions (matching activations).
            boards_for_probe = board_states_rel[:, :BLOCK_SIZE, :, :]  # (N, 59, 8, 8)

            layer_results: dict[int, dict[str, list[float]]] = {}
            for layer in range(1, N_LAYERS + 1):
                probe = load_probe(probe_paths[layer], device)
                acts = layer_acts[layer - 1]  # (N, 59, 512)

                means, std_errs = compute_probe_accuracy(
                    probe, acts, boards_for_probe, batch_size, device
                )
                layer_results[layer] = {
                    "means": means.tolist(),
                    "std_errs": std_errs.tolist(),
                }
                logger.info("    Layer %d: mean=%.4f", layer, float(np.mean(means)))

            results[cache_key] = {
                "run_name": run_name,
                "game_alias": game_alias,
                "epoch": last_epoch,
                "num_games": num_games,
                "layers": layer_results,
            }
            save_pickle_cache(results, CACHE_FILE)

    logger.info("Done: board probe accuracy evaluation.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate board probe accuracy and cache per-move-position results.",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=1000,
        help="Number of eval games per game variant (default: 1000).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for GPT forward passes and probe evaluation (default: 256).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached results and recompute everything.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    evaluate_all(args.num_games, args.batch_size, args.force)
