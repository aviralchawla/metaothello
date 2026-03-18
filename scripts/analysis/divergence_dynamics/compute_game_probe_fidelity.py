"""Compute game identity probe fidelity vs Bayesian ground truth.

For sequences from the mixed Classic-NoMidFlip training distribution,
evaluates how well per-layer game identity probes recover P(Classic | s_{<t})
compared to the Bayesian ground truth. Also evaluates the analytic baseline.

Fidelity = 1 - |P_probe(Classic) - P_GT(Classic)|

Caches results to data/analysis_cache/game_probe_fidelity.json.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from metaothello.analysis_utils import (
    BLOCK_SIZE,
    CACHE_DIR,
    VOCAB_SIZE,
    calculate_ground_truth,
    get_device,
    load_json_cache,
    save_json_cache,
)
from metaothello.constants import MAX_STEPS

# Analytic probe feature dimension: 60 moves * 66 vocab tokens
_ANALYTIC_INPUT_DIM = MAX_STEPS * VOCAB_SIZE
from metaothello.games import GAME_REGISTRY
from metaothello.mingpt.tokenizer import Tokenizer
from metaothello.mingpt.utils import get_last_ckpt, load_model_from_ckpt

logger = logging.getLogger(__name__)

CACHE_FILE = CACHE_DIR / "game_probe_fidelity.json"
N_LAYERS = 8
RUN_NAME = "classic_nomidflip"


def load_game_probes(run_name: str, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load per-layer game identity probe weights and biases.

    Returns list of (weight, bias) tuples (one per layer).
    weight: (2, d_model), bias: (2,).
    The probe outputs [P(game_0), P(game_1)] after softmax.
    """
    probe_dir = CACHE_DIR.parent / run_name / "game_probes"
    probes = []
    for layer in range(1, N_LAYERS + 1):
        path = probe_dir / f"game_L{layer}.ckpt"
        state_dict = torch.load(path, map_location=device)
        weight = state_dict["proj.weight"].detach()  # (2, d_model)
        if "proj.bias" in state_dict:
            bias = state_dict["proj.bias"].detach()  # (2,)
        else:
            bias = torch.zeros(2, device=device)
        probes.append((weight, bias))
    return probes


def load_analytic_probe(run_name: str, device: torch.device):
    """Load the analytic baseline probe weights."""
    path = CACHE_DIR.parent / run_name / "analytic_probe.ckpt"
    if not path.exists():
        logger.warning("No analytic probe found at %s", path)
        return None
    state_dict = torch.load(path, map_location=device)
    return state_dict["proj.weight"].detach(), state_dict["proj.bias"].detach()


def binary_entropy(p: float) -> float:
    """H(p) = -p*log2(p) - (1-p)*log2(1-p), clamped for stability."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def main(num_games: int, force: bool) -> None:
    results = {} if force else load_json_cache(CACHE_FILE)

    if not force and RUN_NAME in results:
        logger.info("Results cached. Use --force to recompute.")
        return

    device = get_device()
    tokenizer = Tokenizer()
    game_aliases = RUN_NAME.split("_")
    game_classes = [GAME_REGISTRY[a] for a in game_aliases]

    # Load model
    ckpt_dir = CACHE_DIR.parent / RUN_NAME / "ckpts"
    last_ckpt, _ = get_last_ckpt(ckpt_dir)
    model = load_model_from_ckpt(last_ckpt, VOCAB_SIZE, BLOCK_SIZE, as_tlens=True)
    model = model.to(device)
    model.eval()

    # Load probes
    game_probes = load_game_probes(RUN_NAME, device)
    analytic_weights = load_analytic_probe(RUN_NAME, device)

    hook_names = [f"blocks.{i}.hook_resid_post" for i in range(N_LAYERS)]

    # Generate sequences: half from each game
    all_seqs = []
    for game_alias in game_aliases:
        gc = GAME_REGISTRY[game_alias]
        for _ in tqdm(range(num_games // 2), desc=f"Generating {game_alias}", leave=False):
            g = gc()
            g.generate_random_game()
            hist = g.get_history()
            if len(hist) >= 10:
                all_seqs.append(hist)

    logger.info("Collected %d sequences.", len(all_seqs))

    # Per-move fidelity accumulators: move -> layer -> [fidelity values]
    fidelity_by_move = defaultdict(lambda: [[] for _ in range(N_LAYERS)])
    baseline_by_move = defaultdict(list)
    entropy_by_move = defaultdict(list)

    for seq in tqdm(all_seqs, desc="Evaluating fidelity"):
        # Ground truth P(game | s_{1..t})
        p_g = calculate_ground_truth(seq, game_classes, tokenizer, skip_p=True)
        # p_g shape: (T, 2), p_g[t, 0] = P(classic | moves 1..t)
        # At model position t (having seen moves 0..t), ground truth is p_g[t+1]

        # Get model activations (truncate to model context window)
        encoded = tokenizer.encode(seq)[:BLOCK_SIZE]
        tokens = torch.tensor([encoded], dtype=torch.long, device=device)
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n in hook_names
            )

        seq_len = min(len(seq) - 1, BLOCK_SIZE - 1)

        for t in range(seq_len):
            p_classic_gt = float(p_g[t + 1, 0])  # posterior after seeing moves 0..t

            # Game entropy
            entropy_by_move[t].append(binary_entropy(p_classic_gt))

            # Per-layer probe fidelity
            for layer_idx in range(N_LAYERS):
                act = cache[hook_names[layer_idx]][0, t, :]  # (d_model,)
                w, b = game_probes[layer_idx]
                logits = act @ w.T + b  # (2,)
                p_probe = F.softmax(logits, dim=-1)[0].item()  # P(classic)
                fidelity = 1.0 - abs(p_probe - p_classic_gt)
                fidelity_by_move[t][layer_idx].append(fidelity)

            # Analytic baseline fidelity
            if analytic_weights is not None:
                # Inline one-hot encoding of move prefix
                feat = torch.zeros(_ANALYTIC_INPUT_DIM, device=device)
                for step in range(min(t + 1, len(seq))):  # encode moves 0..t
                    tid = tokenizer.stoi[seq[step]]
                    feat[step * VOCAB_SIZE + tid] = 1.0
                w, b = analytic_weights
                logits = feat @ w.T + b  # (2,)
                p_baseline = F.softmax(logits, dim=-1)[0].item()
                baseline_fidelity = 1.0 - abs(p_baseline - p_classic_gt)
                baseline_by_move[t].append(baseline_fidelity)

    # Aggregate
    move_numbers = sorted(fidelity_by_move.keys())
    result_data = {
        "move_numbers": move_numbers,
        "fidelity": {},
        "entropy": {},
        "baseline": {},
    }

    for t in move_numbers:
        layer_means = []
        layer_stds = []
        for layer_idx in range(N_LAYERS):
            vals = fidelity_by_move[t][layer_idx]
            if vals:
                layer_means.append(float(np.mean(vals)))
                layer_stds.append(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
            else:
                layer_means.append(None)
                layer_stds.append(None)
        result_data["fidelity"][str(t)] = {"means": layer_means, "stds": layer_stds}

        evals = entropy_by_move[t]
        result_data["entropy"][str(t)] = {
            "mean": float(np.mean(evals)),
            "std": float(np.std(evals, ddof=1) if len(evals) > 1 else 0.0),
        }

        bvals = baseline_by_move.get(t, [])
        if bvals:
            result_data["baseline"][str(t)] = {
                "mean": float(np.mean(bvals)),
                "std": float(np.std(bvals, ddof=1) if len(bvals) > 1 else 0.0),
            }

    results[RUN_NAME] = result_data
    save_json_cache(results, CACHE_FILE)
    logger.info("Done: game probe fidelity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute game probe fidelity.")
    parser.add_argument("--num_games", type=int, default=10000,
                        help="Total games (split between game types).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main(args.num_games, args.force)
