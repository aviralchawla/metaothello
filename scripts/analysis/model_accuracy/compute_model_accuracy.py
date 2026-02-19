"""Evaluate model accuracy across all MetaOthello runs and cache results.

For each of the 7 trained models this script:
1. Generates fresh random games on the fly for each game the model was trained on.
2. Runs batched inference to compute the selected metric.
3. Saves per-move-position results to ``data/analysis_cache/model_accuracy.json``.

Results are stored as per-position arrays (length 59), not scalars, so that
downstream scripts can produce both aggregated and per-move figures from the
same cache without re-running inference.

On re-run, cached entries are reused automatically.  Pass ``--force`` to
recompute everything from scratch.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from metaothello.analysis_utils import (
    ALL_RUN_NAMES,
    BLOCK_SIZE,
    CACHE_DIR,
    VOCAB_SIZE,
    Metric,
    gen_games,
    get_device,
    get_game_aliases,
    load_json_cache,
    save_json_cache,
)
from metaothello.mingpt.tokenizer import Tokenizer
from metaothello.mingpt.utils import get_last_ckpt, load_model_from_ckpt

logger = logging.getLogger(__name__)

CACHE_FILE = CACHE_DIR / "model_accuracy.json"


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metric(
    logits: torch.Tensor,
    metric: Metric,
    valid_mask: torch.Tensor | None = None,
) -> np.ndarray:
    """Compute per-position metric values for a single batch.

    Args:
        logits: Raw model output, shape ``(B, T, vocab_size)``.
        metric: Which metric to compute.
        valid_mask: Float tensor of shape ``(B, T, vocab_size)`` with 1.0 at
            valid-move token positions and 0.0 elsewhere.  Required for
            both ``Metric.TOP1`` and ``Metric.CORRECT_PROB``.

    Returns:
        Float numpy array of shape ``(B, T)`` with per-game per-position
        metric values.

    Raises:
        NotImplementedError: If ``metric == Metric.ALPHA``.
    """
    if metric == Metric.ALPHA:
        raise NotImplementedError("Alpha-score is not yet implemented.")

    if valid_mask is None:
        raise ValueError("valid_mask is required for all implemented metrics")

    if metric == Metric.TOP1:
        preds = logits.argmax(dim=-1)  # (B, T)
        values = valid_mask.gather(dim=-1, index=preds.unsqueeze(-1)).squeeze(-1)
    else:  # CORRECT_PROB
        probs = F.softmax(logits, dim=-1)  # (B, T, vocab_size)
        values = (probs * valid_mask).sum(dim=-1)  # (B, T)

    return values.cpu().numpy()  # (B, T)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    model: Any,
    seqs: np.ndarray,
    metric: Metric,
    batch_size: int,
    device: torch.device,
    valid_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run batched inference and compute the selected metric.

    Args:
        model: A GPT model in eval mode on ``device``.
        seqs: Token array of shape ``(N, MAX_STEPS)``.
        metric: Metric to compute.
        batch_size: Sequences per forward pass.
        device: Torch device the model lives on.
        valid_masks: Boolean array of shape ``(N, MAX_STEPS, vocab_size)``
            as returned by ``gen_games``.

    Returns:
        Tuple of ``(means, std_errs)`` each of shape ``(T,)`` = ``(59,)``,
        giving the per-move-position mean and standard error across all games.
    """
    n = len(seqs)
    all_values: list[np.ndarray] = []
    device_type = device.type

    with torch.inference_mode():
        for start in tqdm(range(0, n, batch_size), desc="Inference", leave=False):
            end = min(start + batch_size, n)
            batch_np = seqs[start:end]

            x = torch.tensor(batch_np[:, :-1], dtype=torch.long, device=device)

            vm = valid_masks[start:end, 1:]  # (batch, T, vocab_size)
            valid_mask_tensor = torch.tensor(vm, dtype=torch.float32, device=device)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device_type == "cuda"
                else contextlib.nullcontext()
            )

            with autocast_ctx:
                logits, _ = model(x)

            vals = compute_metric(logits.float(), metric, valid_mask_tensor)
            all_values.append(vals)

    stacked = np.concatenate(all_values, axis=0)  # (N, T)
    means = stacked.mean(axis=0)  # (T,)
    std_errs = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])  # (T,)
    return means, std_errs


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_all(
    metric: Metric,
    num_games: int,
    batch_size: int,
    force: bool,
) -> None:
    """Run evaluation for all 7 run_names and cache results.

    Args:
        metric: Metric to evaluate.
        num_games: Fresh games to generate per game variant.
        batch_size: Inference batch size.
        force: If True, recompute all entries even if cached.
    """
    results = {} if force else load_json_cache(CACHE_FILE)
    tokenizer = Tokenizer()
    device = get_device()

    for run_name in ALL_RUN_NAMES:
        cache_key = f"{run_name}__{metric.value}"
        if cache_key in results:
            logger.info("Skipping %s (cached). Use --force to recompute.", run_name)
            continue

        logger.info("Evaluating run: %s", run_name)
        ckpt_dir = CACHE_DIR.parent / run_name / "ckpts"
        last_ckpt, last_epoch = get_last_ckpt(ckpt_dir)
        if last_ckpt is None:
            logger.warning("No checkpoint for %s — skipping.", run_name)
            continue

        model = load_model_from_ckpt(last_ckpt, VOCAB_SIZE, BLOCK_SIZE, as_tlens=False)
        model = model.to(device)
        model.eval()
        logger.info("Loaded %s epoch %d on %s", run_name, last_epoch, device)

        game_aliases = get_game_aliases(run_name)
        run_results: dict[str, dict[str, Any]] = {}

        for game_alias in game_aliases:
            logger.info("  %s → %s: generating %d games...", run_name, game_alias, num_games)
            seqs, valid_masks = gen_games(game_alias, num_games, tokenizer)
            means, std_errs = run_inference(model, seqs, metric, batch_size, device, valid_masks)
            run_results[game_alias] = {
                "means": means.tolist(),
                "std_errs": std_errs.tolist(),
                "num_games": num_games,
                "epoch": last_epoch,
            }
            logger.info(
                "    %s on %s: mean=%.4f (avg over %d positions)",
                run_name,
                game_alias,
                float(means.mean()),
                len(means),
            )

        results[cache_key] = {
            "run_name": run_name,
            "metric": metric.value,
            "epoch": last_epoch,
            "games": run_results,
        }
        save_json_cache(results, CACHE_FILE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model accuracy and cache per-move-position results.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=Metric.TOP1.value,
        choices=[m.value for m in Metric if m != Metric.ALPHA],
        help="Evaluation metric (default: top1).",
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
        help="Inference batch size (default: 256).",
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

    evaluate_all(Metric(args.metric), args.num_games, args.batch_size, args.force)
