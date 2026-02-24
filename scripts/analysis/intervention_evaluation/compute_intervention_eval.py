"""Evaluate linear probe interventions for each tile-state combination.

For mixed-game models (classic_nomidflip, classic_delflank), this script measures
the effect of intervening on the residual stream using probe weights. It tests
both the "correct" probe (trained on the game being played) and the "cross" probe
(trained on the other game in the mix).
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from metaothello.analysis_utils import (
    BLOCK_SIZE,
    CACHE_DIR,
    VOCAB_SIZE,
    get_device,
    load_json_cache,
    save_json_cache,
)
from metaothello.constants import BOARD_DIM, WHITE
from metaothello.games import GAME_REGISTRY
from metaothello.mingpt.tokenizer import Tokenizer
from metaothello.mingpt.utils import get_last_ckpt, load_model_from_ckpt

logger = logging.getLogger(__name__)

CACHE_FILE = CACHE_DIR / "intervention_eval.json"
N_LAYERS = 8

# Intervention constants
ALPHA_FLIP = 5.0
ALPHA_ERASE = 2.0
ALPHA_PLACE = 2.0


def tile_to_algebraic(r: int, c: int) -> str:
    """Convert (row, col) to algebraic notation like 'e4'."""
    col_letter = chr(ord("a") + c)
    row_number = r + 1
    return f"{col_letter}{row_number}"


def calculate_errors(
    logits: torch.Tensor,
    target_valid_moves: list[str],
    tokenizer: Tokenizer,
) -> float:
    """Calculate prediction errors based on Top-K predictions.

    Error = False Positives + False Negatives
    Based on Top-K predictions where K = len(valid_moves).
    """
    valid_tokens = set(tokenizer.encode(target_valid_moves))
    k = len(valid_tokens)

    if k == 0:
        return 0.0

    probs = F.softmax(logits[0, -1, :], dim=-1)
    top_k_indices = torch.topk(probs, k).indices.cpu().numpy()
    pred_tokens = set(top_k_indices)

    fp = len(pred_tokens - valid_tokens)
    fn = len(valid_tokens - pred_tokens)

    return float(fp + fn)


def create_additive_hook(diff_vector: torch.Tensor, alpha: float) -> Any:
    """Create hook function to inject probe vector into activations."""
    diff = diff_vector.detach()
    diff = diff / (diff.norm() + 1e-8)

    def _hook(act: torch.Tensor, hook: Any) -> torch.Tensor:
        # act is (batch, seq_len, d_model)
        act = act.clone()
        act[:, -1, :] = act[:, -1, :] + alpha * diff.to(act.device)
        return act

    return _hook


def get_board_to_mineyours(seq: list[str], game_class: type) -> tuple[np.ndarray, Any]:
    """Reconstruct board and convert to Mine(1)/Yours(-1)/Empty(0) perspective."""
    game = game_class()
    game.recover_from_history(seq)
    board = game.board.copy()
    if game.next_color == WHITE:
        board *= -1
    return board, game


def jaccard_index(l1: list[str | None], l2: list[str | None]) -> float:
    """Calculate Jaccard similarity between two lists."""
    s1, s2 = set(l1), set(l2)
    if not s1 and not s2:
        return 1.0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    if union == 0:
        return 1.0
    return float(intersection) / union


def get_target_valid_moves(game: Any, r: int, c: int, target_state: int) -> list[str | None]:
    """Get valid moves if board state was changed at (r,c) to target_state."""
    game_modified = deepcopy(game)
    if game.next_color == WHITE:
        game_modified.board[r, c] = -target_state if target_state != 0 else 0
    else:
        game_modified.board[r, c] = target_state

    game_modified.valid_moves = None
    return game_modified.get_all_valid_moves()


def load_probes(run_name: str, game_alias: str, device: torch.device) -> torch.Tensor:
    """Load probes for all layers and reshape to (8, 8, 3, d_model)."""
    probe_dir = CACHE_DIR.parent / run_name / "board_probes"
    probes = []
    for layer in range(1, N_LAYERS + 1):
        probe_path = probe_dir / f"{game_alias}_board_L{layer}.ckpt"
        state_dict = torch.load(probe_path, map_location=device)
        weight = state_dict["proj.weight"].detach()
        probes.append(weight.reshape(BOARD_DIM, BOARD_DIM, 3, -1))
    return torch.stack(probes)  # (N_LAYERS, 8, 8, 3, d_model)


def evaluate_interventions(
    model: Any,
    tokenizer: Tokenizer,
    game_class: type,
    probes: torch.Tensor,
    device: torch.device,
    samples_per_combo: int = 50,
) -> dict[str, list[float]]:
    """Evaluate interventions for all tile-state combinations.

    Returns dict mapping 'tile_state' -> [null_m, null_s, lin_m, lin_s, n].
    """
    total_combos = 64 * 3
    experiments: dict[tuple[int, int, int], list[tuple[list[str], int, int, list[str]]]] = (
        defaultdict(list)
    )
    target_states = [-1, 0, 1]

    # Pre-generate sequences and select experiments
    # Keep generating games and extracting truncations until we have enough samples
    pbar_sel = tqdm(
        total=total_combos * samples_per_combo,
        desc="Selecting experiments",
        leave=False,
    )

    while sum(len(v) for v in experiments.values()) < total_combos * samples_per_combo:
        g = game_class()
        g.generate_random_game()
        hist = g.get_history()

        # Don't try all truncations to keep it random, but try a few
        max_t = len(hist)
        if max_t <= 5:
            continue

        for _ in range(3):
            t = np.random.randint(5, max_t)
            seq = hist[:t]
            board_my, game = get_board_to_mineyours(seq, game_class)
            old_valid_moves = game.get_all_valid_moves()

            for r in range(BOARD_DIM):
                for c in range(BOARD_DIM):
                    curr_state = int(board_my[r, c])
                    for ts in target_states:
                        if curr_state == ts:
                            continue

                        key = (r, c, ts)
                        if len(experiments[key]) >= samples_per_combo:
                            continue

                        target_valid = get_target_valid_moves(game, r, c, ts)

                        jaccard = jaccard_index(old_valid_moves, target_valid)
                        if jaccard > 1.0:  # jaccard_threshold=1.0 by default from original
                            continue

                        experiments[key].append((seq, curr_state, ts, target_valid))
                        pbar_sel.update(1)

                        # Stop searching this sequence if we found one
                        break

    pbar_sel.close()

    results_dict = {}
    pbar_eval = tqdm(total=total_combos * samples_per_combo, desc="Evaluating", leave=False)

    for r in range(BOARD_DIM):
        for c in range(BOARD_DIM):
            for ts in target_states:
                key = (r, c, ts)
                exps = experiments[key]
                if not exps:
                    continue

                null_errs = []
                lin_errs = []

                val_idx = ts + 1

                for seq, curr_state, target_state, target_valid in exps:
                    if target_state == 0:
                        alpha = ALPHA_ERASE
                    elif curr_state == 0:
                        alpha = ALPHA_PLACE
                    else:
                        alpha = ALPHA_FLIP

                    input_tokens = torch.tensor(tokenizer.encode(seq), device=device).unsqueeze(0)

                    # Null
                    with torch.inference_mode():
                        clean_logits = model(input_tokens)
                    null_err = calculate_errors(clean_logits, target_valid, tokenizer)
                    null_errs.append(null_err)

                    # Linear intervention
                    hooks = []
                    for layer in range(N_LAYERS):
                        vec = probes[layer, r, c, val_idx]
                        hooks.append(
                            (
                                f"blocks.{layer}.hook_resid_post",
                                create_additive_hook(vec, alpha),
                            )
                        )

                    with torch.inference_mode():
                        patched_logits = model.run_with_hooks(input_tokens, fwd_hooks=hooks)
                    lin_err = calculate_errors(patched_logits, target_valid, tokenizer)
                    lin_errs.append(lin_err)

                    pbar_eval.update(1)

                tile_id = tile_to_algebraic(r, c)
                state_name = {-1: "yours", 0: "empty", 1: "mine"}[ts]
                k = f"{tile_id}_{state_name}"

                n = len(null_errs)
                if n > 0:
                    results_dict[k] = [
                        float(np.mean(null_errs)),
                        float(np.std(null_errs, ddof=1) if n > 1 else 0.0),
                        float(np.mean(lin_errs)),
                        float(np.std(lin_errs, ddof=1) if n > 1 else 0.0),
                        n,
                    ]

    pbar_eval.close()
    return results_dict


def main(samples_per_combo: int, force: bool) -> None:
    """Run the intervention evaluation for all mixed-game models.

    Args:
        samples_per_combo: Number of samples to collect for each (tile, state) pair.
        force: Whether to recompute results even if they exist in the cache.
    """
    results: dict[str, Any] = {} if force else load_json_cache(CACHE_FILE)
    tokenizer = Tokenizer()
    device = get_device()

    models_to_eval = ["classic_nomidflip", "classic_delflank"]

    for run_name in models_to_eval:
        if run_name not in results:
            results[run_name] = {}

        g1, g2 = run_name.split("_")

        # Check if all needed entries exist
        needed = [f"{g1}_correct", f"{g2}_correct", f"{g1}_cross", f"{g2}_cross"]
        if not force and all(k in results[run_name] for k in needed):
            logger.info("Skipping %s (cached). Use --force to recompute.", run_name)
            continue

        ckpt_dir = CACHE_DIR.parent / run_name / "ckpts"
        last_ckpt, _ = get_last_ckpt(ckpt_dir)
        if last_ckpt is None:
            logger.warning("No checkpoint for %s — skipping.", run_name)
            continue

        logger.info("Evaluating %s...", run_name)
        model = load_model_from_ckpt(last_ckpt, VOCAB_SIZE, BLOCK_SIZE, as_tlens=True)
        model = model.to(device)
        model.eval()

        probes_g1 = load_probes(run_name, g1, device)
        probes_g2 = load_probes(run_name, g2, device)

        for game_alias, game_probes, is_correct in [
            (g1, probes_g1, True),
            (g2, probes_g2, True),
            (g1, probes_g2, False),
            (g2, probes_g1, False),
        ]:
            key = f"{game_alias}_{'correct' if is_correct else 'cross'}"
            if not force and key in results[run_name]:
                continue

            logger.info("  Running %s", key)
            res = evaluate_interventions(
                model=model,
                tokenizer=tokenizer,
                game_class=GAME_REGISTRY[game_alias],
                probes=game_probes,
                device=device,
                samples_per_combo=samples_per_combo,
            )
            results[run_name][key] = res
            save_json_cache(results, CACHE_FILE)

    logger.info("Done: intervention evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate linear probe interventions.")
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Samples per tile-state combination.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute all cached results.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main(args.samples, args.force)
