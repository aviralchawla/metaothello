"""Compute activation cosine similarity for mixed-game models.

For each mixed model (classic_nomidflip, classic_delflank, classic_iago):

1. Generates fresh random games for each component game variant.
2. Extracts mean residual-stream (resid_post) activations per layer and move
   position via batched TransformerLens forward passes.
3. Computes cosine similarity between the two game types' mean activation
   vectors at each layer and move position.
4. Caches per-layer per-move-position results to
   ``data/analysis_cache/activation_cosine_sim.json``.

On re-run, cached entries (keyed by run_name) are reused.
Pass ``--force`` to recompute everything from scratch.
"""

from __future__ import annotations

import argparse
import contextlib
import logging

import numpy as np
import torch
from scipy.spatial.distance import cosine
from tqdm import tqdm

from metaothello.analysis_utils import (
    BLOCK_SIZE,
    CACHE_DIR,
    VOCAB_SIZE,
    gen_games,
    get_device,
    get_game_aliases,
    load_json_cache,
    save_json_cache,
)
from metaothello.mingpt.tokenizer import Tokenizer
from metaothello.mingpt.utils import get_last_ckpt, load_model_from_ckpt

logger = logging.getLogger(__name__)

CACHE_FILE = CACHE_DIR / "activation_cosine_sim.json"
MIXED_RUN_NAMES = ["classic_nomidflip", "classic_delflank", "classic_iago"]
N_LAYERS = 8


def _acts_filter(name: str) -> bool:
    return "hook_resid_post" in name


def compute_mean_acts(
    model: object,
    seqs: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Compute mean resid_post activations per layer and move position.

    Processes sequences in batches using TransformerLens ``run_with_cache``,
    accumulating a running sum and dividing at the end to produce mean vectors
    without storing all per-sequence activations.

    Args:
        model: HookedTransformer in eval mode on ``device``.
        seqs: Token array of shape ``(N, MAX_STEPS)``.
        batch_size: Sequences per forward pass.
        device: Torch device the model lives on.

    Returns:
        Mean activations of shape ``(T, d_model, N_LAYERS)`` where T = BLOCK_SIZE = 59.
    """
    n = len(seqs)
    sum_resid: np.ndarray | None = None
    count = 0

    with torch.inference_mode():
        for start in tqdm(range(0, n, batch_size), desc="Extracting activations", leave=False):
            end = min(start + batch_size, n)
            x = torch.tensor(seqs[start:end, :-1], dtype=torch.long, device=device)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                _, cache = model.run_with_cache(x, names_filter=_acts_filter)

            # resid_post: (N_LAYERS, B, T, d_model) -> (B, T, d_model, N_LAYERS)
            resid_batch = (
                cache.stack_activation("resid_post").permute(1, 2, 3, 0).float().cpu().numpy()
            )

            b = resid_batch.shape[0]
            if sum_resid is None:
                sum_resid = np.zeros(resid_batch.shape[1:], dtype=np.float64)

            sum_resid += resid_batch.sum(axis=0)  # (T, d_model, N_LAYERS)
            count += b

    assert sum_resid is not None
    return sum_resid / count  # (T, d_model, N_LAYERS)


def compute_cosine_sims(
    mean_a: np.ndarray,
    mean_b: np.ndarray,
) -> dict[str, list[float]]:
    """Compute per-layer per-position cosine similarities between two mean activation arrays.

    Args:
        mean_a: Mean activations of shape ``(T, d_model, N_LAYERS)`` for game A.
        mean_b: Mean activations of shape ``(T, d_model, N_LAYERS)`` for game B.

    Returns:
        Dict mapping layer string (``"1"`` - ``"8"``) to list of T cosine
        similarity values.
    """
    t_len, _, n_layers = mean_a.shape
    sims: dict[str, list[float]] = {}
    for layer_idx in range(n_layers):
        layer_sims: list[float] = []
        for t in range(t_len):
            a = mean_a[t, :, layer_idx]
            b = mean_b[t, :, layer_idx]
            layer_sims.append(float(1.0 - cosine(a, b)))
        sims[str(layer_idx + 1)] = layer_sims
    return sims


def evaluate_all(num_games: int, batch_size: int, force: bool) -> None:
    """Run activation cosine similarity evaluation for all mixed models.

    Args:
        num_games: Fresh games to generate per game variant.
        batch_size: Sequences per forward pass through the model.
        force: If True, recompute all entries even if cached.
    """
    results: dict = {} if force else load_json_cache(CACHE_FILE)
    tokenizer = Tokenizer()
    device = get_device()

    for run_name in MIXED_RUN_NAMES:
        if run_name in results and not force:
            logger.info("Skipping %s (cached). Use --force to recompute.", run_name)
            continue

        game_aliases = get_game_aliases(run_name)
        ckpt_dir = CACHE_DIR.parent / run_name / "ckpts"
        last_ckpt, last_epoch = get_last_ckpt(ckpt_dir)
        if last_ckpt is None:
            logger.warning("No checkpoint for %s — skipping.", run_name)
            continue

        logger.info("Processing %s (epoch %d) ...", run_name, last_epoch)
        model = load_model_from_ckpt(last_ckpt, VOCAB_SIZE, BLOCK_SIZE, as_tlens=True)
        model = model.to(device)
        model.eval()

        mean_acts: dict[str, np.ndarray] = {}
        for alias in game_aliases:
            logger.info("  Generating %d %s games ...", num_games, alias)
            seqs, _ = gen_games(alias, num_games, tokenizer)
            logger.info("  Extracting activations for %s ...", alias)
            mean_acts[alias] = compute_mean_acts(model, seqs, batch_size, device)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        game_a, game_b = game_aliases
        results[run_name] = {
            "game_a": game_a,
            "game_b": game_b,
            "num_seqs": num_games,
            "epoch": last_epoch,
            "resid_post": compute_cosine_sims(mean_acts[game_a], mean_acts[game_b]),
        }
        save_json_cache(results, CACHE_FILE)
        logger.info("  Saved %s.", run_name)

    logger.info("Done: activation cosine similarity evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute activation cosine similarity for mixed-game models.",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=2000,
        help="Games per variant for activation extraction (default: 2000).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Forward-pass batch size (default: 64).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached results and recompute everything.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    evaluate_all(args.num_games, args.batch_size, args.force)
