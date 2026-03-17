"""Compute normalized P(Classic) change after disambiguation moves.

For sequences valid in both Classic and NoMidFlip that diverge at each
move number, plays a classic-only move and measures how the game identity
probe's P(Classic) changes.  The change is normalized by the remaining
room: delta-P / (1 - P_pre).

Requires:
- Mixed classic_nomidflip model checkpoint
- Per-layer game identity probes

Caches results to data/analysis_cache/prob_collapse.json.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from metaothello.analysis_utils import (
    BLOCK_SIZE,
    CACHE_DIR,
    VOCAB_SIZE,
    generate_diverging_sequences,
    get_device,
    load_json_cache,
    save_json_cache,
)
from metaothello.games import GAME_REGISTRY
from metaothello.mingpt.tokenizer import Tokenizer
from metaothello.mingpt.utils import get_last_ckpt, load_model_from_ckpt

logger = logging.getLogger(__name__)

CACHE_FILE: Final[Path] = CACHE_DIR / "prob_collapse.json"
DATA_DIR: Final[Path] = CACHE_DIR.parent
N_LAYERS: Final[int] = 8
RUN_NAME: Final[str] = "classic_nomidflip"


def load_game_probes(
    run_name: str, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load per-layer game identity probe weights and biases.

    Returns list of (weight, bias) tuples (one per layer).
    weight: (2, d_model), bias: (2,).
    """
    probe_dir = DATA_DIR / run_name / "game_probes"
    if not probe_dir.exists():
        msg = (
            f"Game probe directory not found: {probe_dir}\n"
            "Download game probes first:\n"
            "  python scripts/download_probes.py game"
            " --run_name classic_nomidflip\n"
            "Or train from scratch:\n"
            "  python scripts/training/train_game_probe.py"
            " --run_name classic_nomidflip"
        )
        raise FileNotFoundError(msg)

    probes = []
    for layer in range(1, N_LAYERS + 1):
        path = probe_dir / f"game_L{layer}.ckpt"
        if not path.exists():
            msg = f"Game probe not found: {path}\nDownload or train game probes (see above)."
            raise FileNotFoundError(msg)
        state_dict = torch.load(path, map_location=device)
        weight = state_dict["proj.weight"].detach()  # (2, d_model)
        if "proj.bias" in state_dict:
            bias = state_dict["proj.bias"].detach()  # (2,)
        else:
            bias = torch.zeros(2, device=device)
        probes.append((weight, bias))
    return probes


def find_classic_only_move(
    seq: list[str],
    classic_class: type,
    variant_class: type,
) -> str | None:
    """Find a move valid in Classic but not in the variant.

    Replays the sequence in both games and returns a random move from the
    set difference (Classic valid - variant valid).  Returns None if no
    such move exists.
    """
    g_classic = classic_class()
    g_variant = variant_class()
    for m in seq:
        g_classic.play_move(m)
        g_variant.play_move(m)

    valid_classic = {m for m in g_classic.get_all_valid_moves() if m is not None}
    valid_variant = {m for m in g_variant.get_all_valid_moves() if m is not None}
    only_classic = valid_classic - valid_variant

    if not only_classic:
        return None
    return random.choice(list(only_classic))


def main(num_sequences: int, force: bool) -> None:
    """Run the probability collapse experiment."""
    results = {} if force else load_json_cache(CACHE_FILE)

    if not force and RUN_NAME in results:
        logger.info("Results cached. Use --force to recompute.")
        return

    device = get_device()
    tokenizer = Tokenizer()
    classic_class = GAME_REGISTRY["classic"]
    variant_class = GAME_REGISTRY["nomidflip"]
    game_classes = [classic_class, variant_class]

    # Load model
    ckpt_dir = DATA_DIR / RUN_NAME / "ckpts"
    if not ckpt_dir.exists() or not any(ckpt_dir.glob("*.ckpt")):
        msg = (
            f"No model checkpoints found in {ckpt_dir}\n"
            "Download the model first:\n"
            "  python scripts/download_models.py models"
            " --run_name classic_nomidflip"
        )
        raise FileNotFoundError(msg)
    last_ckpt, _ = get_last_ckpt(ckpt_dir)
    model = load_model_from_ckpt(last_ckpt, VOCAB_SIZE, BLOCK_SIZE, as_tlens=True)
    model = model.to(device)
    model.eval()

    # Load game identity probes
    game_probes = load_game_probes(RUN_NAME, device)
    hook_names = [f"blocks.{i}.hook_resid_post" for i in range(N_LAYERS)]

    move_points = list(range(6, 50))

    # Accumulators: move -> layer -> [normalized_delta values]
    delta_by_move: dict[int, list[list[float]]] = defaultdict(lambda: [[] for _ in range(N_LAYERS)])

    for move in tqdm(move_points, desc="Move points"):
        try:
            seqs = generate_diverging_sequences(move, game_classes, num_sequences, tokenizer)
        except RuntimeError:
            logger.warning("Could not generate enough sequences at t=%d", move)
            continue

        for seq in seqs:
            classic_only_move = find_classic_only_move(seq, classic_class, variant_class)
            if classic_only_move is None:
                continue

            # Encode pre-move and post-move sequences
            pre_tokens = tokenizer.encode(seq)
            post_tokens = tokenizer.encode([*seq, classic_only_move])

            # Truncate to model context window
            pre_tokens = pre_tokens[:BLOCK_SIZE]
            post_tokens = post_tokens[:BLOCK_SIZE]

            pre_t = torch.tensor([pre_tokens], dtype=torch.long, device=device)
            post_t = torch.tensor([post_tokens], dtype=torch.long, device=device)

            with torch.inference_mode():
                _, cache_pre = model.run_with_cache(
                    pre_t,
                    names_filter=lambda n: n in hook_names,
                )
                _, cache_post = model.run_with_cache(
                    post_t,
                    names_filter=lambda n: n in hook_names,
                )

            for layer_idx in range(N_LAYERS):
                w, b = game_probes[layer_idx]

                # Evaluate at last token position
                act_pre = cache_pre[hook_names[layer_idx]][0, -1, :]
                act_post = cache_post[hook_names[layer_idx]][0, -1, :]

                logits_pre = act_pre @ w.T + b
                logits_post = act_post @ w.T + b

                p_pre = F.softmax(logits_pre, dim=-1)[0].item()
                p_post = F.softmax(logits_post, dim=-1)[0].item()

                delta = p_post - p_pre
                # Normalize by remaining room
                denom = 1.0 - p_pre
                normalized = delta / denom if abs(denom) > 1e-8 else 0.0

                delta_by_move[move][layer_idx].append(normalized)

    # Aggregate: mean and std per (move, layer)
    result_data: dict[str, dict] = {}
    for move in move_points:
        if move not in delta_by_move:
            continue
        means = []
        stds = []
        for layer_idx in range(N_LAYERS):
            vals = delta_by_move[move][layer_idx]
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
            else:
                means.append(None)
                stds.append(None)
        result_data[str(move)] = {"means": means, "stds": stds}

    result_data["params"] = {
        "num_sequences": num_sequences,
        "move_range": [6, 49],
    }

    results[RUN_NAME] = result_data
    save_json_cache(results, CACHE_FILE)
    logger.info("Cached to %s", CACHE_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute normalized P(Classic) change after disambiguation."
    )
    parser.add_argument(
        "--num_seqs",
        type=int,
        default=10000,
        help="Diverging sequences per move (default: 10000). Use 200 for fast test.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if cached.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main(args.num_seqs, args.force)
