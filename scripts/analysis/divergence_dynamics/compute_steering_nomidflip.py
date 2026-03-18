"""Compute causal steering effects using game identity probe weight vectors.

For sequences valid in both Classic and NoMidFlip, injects the steering vector
Delta_mu = mu_NoMid - mu_Classic (derived from game identity probe weights)
into the residual stream at each layer. Measures the normalized increase in
alpha-score relative to NoMidFlip-valid moves.

Caches results to data/analysis_cache/steering_nomidflip.json.
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
    alpha_score,
    calculate_ground_truth,
    generate_diverging_sequences,
    get_device,
    load_json_cache,
    save_json_cache,
)
from metaothello.games import GAME_REGISTRY
from metaothello.mingpt.tokenizer import Tokenizer
from metaothello.mingpt.utils import get_last_ckpt, load_model_from_ckpt

logger = logging.getLogger(__name__)

CACHE_FILE = CACHE_DIR / "steering_nomidflip.json"
N_LAYERS = 8
RUN_NAME = "classic_nomidflip"
STEERING_SCALE = 6.8  # lambda in the paper


def load_steering_vectors(run_name: str, device: torch.device) -> list[torch.Tensor]:
    """Load per-layer steering vectors from game probe weights.

    Steering vector = W[1] - W[0] where W[i] is the weight row for game i.
    W[0] = Classic weight, W[1] = NoMidFlip weight.
    Delta_mu = mu_NoMid - mu_Classic.

    Returns list of (d_model,) tensors, one per layer.
    """
    probe_dir = CACHE_DIR.parent / run_name / "game_probes"
    vectors = []
    for layer in range(1, N_LAYERS + 1):
        path = probe_dir / f"game_L{layer}.ckpt"
        state_dict = torch.load(path, map_location=device)
        weight = state_dict["proj.weight"].detach()  # (2, d_model)
        # weight[0] = Classic direction, weight[1] = NoMidFlip direction
        delta_mu = weight[1] - weight[0]  # mu_NoMid - mu_Classic
        vectors.append(delta_mu)
    return vectors


def create_steering_hook(
    steering_vec: torch.Tensor, scale: float
) -> callable:
    """Create hook that adds scaled steering vector to last-position activations."""
    vec = steering_vec.detach()

    def _hook(act: torch.Tensor, hook) -> torch.Tensor:
        act = act.clone()
        act[:, -1, :] = act[:, -1, :] + scale * vec.to(act.device)
        return act

    return _hook


def main(num_seqs_per_point: int, force: bool) -> None:
    results = {} if force else load_json_cache(CACHE_FILE)

    if not force and RUN_NAME in results:
        logger.info("Results cached. Use --force to recompute.")
        return

    device = get_device()
    tokenizer = Tokenizer()
    game_classes = [GAME_REGISTRY["nomidflip"], GAME_REGISTRY["classic"]]

    # Load model
    ckpt_dir = CACHE_DIR.parent / RUN_NAME / "ckpts"
    last_ckpt, _ = get_last_ckpt(ckpt_dir)
    model = load_model_from_ckpt(last_ckpt, VOCAB_SIZE, BLOCK_SIZE, as_tlens=True)
    model = model.to(device)
    model.eval()

    # Load steering vectors
    steering_vecs = load_steering_vectors(RUN_NAME, device)

    # For each move number, generate diverging sequences and measure steering effect
    move_points = list(range(6, 50, 1))

    # Results: move -> layer -> [delta_alpha values]
    delta_alpha_by_move = defaultdict(lambda: [[] for _ in range(N_LAYERS)])

    for div_point in tqdm(move_points, desc="Move points"):
        try:
            seqs = generate_diverging_sequences(
                div_point, game_classes, num_seqs_per_point, tokenizer
            )
        except RuntimeError:
            logger.warning("Could not generate enough sequences at t=%d", div_point)
            continue

        for seq in seqs:
            encoded = tokenizer.encode(seq)[:BLOCK_SIZE]
            tokens = torch.tensor([encoded], dtype=torch.long, device=device)

            # Ground truth distribution for NoMidFlip at this position
            p_next_gt, p_g = calculate_ground_truth(
                seq, game_classes, tokenizer, skip_p=False
            )

            # Baseline (no intervention) alpha toward NoMidFlip
            with torch.inference_mode():
                base_logits = model(tokens)

            base_probs = F.softmax(base_logits[0, -1, :], dim=-1).cpu().numpy()

            # NoMidFlip ground truth at last position: uniform over NoMidFlip-valid moves
            nmf_game = GAME_REGISTRY["nomidflip"]()
            for move in seq:
                if move not in nmf_game.get_all_valid_moves():
                    break
                nmf_game.play_move(move)
            nmf_valid = nmf_game.get_all_valid_moves()
            if not nmf_valid:
                continue

            nmf_gt = np.zeros(VOCAB_SIZE)
            nmf_tokens = tokenizer.encode(nmf_valid)
            nmf_gt[nmf_tokens] = 1.0 / len(nmf_valid)

            try:
                base_alpha = alpha_score(base_probs, nmf_gt)
            except (ValueError, ZeroDivisionError):
                continue

            # Per-layer steering
            for layer_idx in range(N_LAYERS):
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
                hook_fn = create_steering_hook(steering_vecs[layer_idx], STEERING_SCALE)

                with torch.inference_mode():
                    steered_logits = model.run_with_hooks(
                        tokens, fwd_hooks=[(hook_name, hook_fn)]
                    )

                steered_probs = F.softmax(steered_logits[0, -1, :], dim=-1).cpu().numpy()

                try:
                    steered_alpha = alpha_score(steered_probs, nmf_gt)
                except (ValueError, ZeroDivisionError):
                    continue

                # Normalized delta alpha: (steered - base) / (1 - base)
                if base_alpha < 1.0:
                    norm_delta = (steered_alpha - base_alpha) / (1.0 - base_alpha)
                else:
                    norm_delta = 0.0

                delta_alpha_by_move[div_point][layer_idx].append(norm_delta)

    # Aggregate
    result_data = {}
    for move in move_points:
        if move not in delta_alpha_by_move:
            continue
        layer_means = []
        layer_stds = []
        for layer_idx in range(N_LAYERS):
            vals = delta_alpha_by_move[move][layer_idx]
            if vals:
                layer_means.append(float(np.mean(vals)))
                layer_stds.append(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
            else:
                layer_means.append(None)
                layer_stds.append(None)
        result_data[str(move)] = {"means": layer_means, "stds": layer_stds}

    results[RUN_NAME] = result_data
    save_json_cache(results, CACHE_FILE)
    logger.info("Done: steering NoMidFlip.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute steering effects for NoMidFlip.")
    parser.add_argument("--num_seqs", type=int, default=1000,
                        help="Sequences per divergence point.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main(args.num_seqs, args.force)
