"""Compute board probe accuracy under steering at early layers.

For sequences valid in both Classic and DelFlank, measures how steering at
layers 1 and 2 (0-indexed) affects the DelFlank board probe accuracy at
layer 5 (0-indexed). This demonstrates that amplifying the game identity
signal causally improves the model's internal world model.

Caches results to data/analysis_cache/probe_effect_delflank.json.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from metaothello.analysis_utils import (
    BLOCK_SIZE,
    CACHE_DIR,
    VOCAB_SIZE,
    generate_diverging_sequences,
    get_board_states,
    get_device,
    load_json_cache,
    save_json_cache,
)
from metaothello.constants import BOARD_DIM
from metaothello.games import GAME_REGISTRY
from metaothello.mingpt.tokenizer import Tokenizer
from metaothello.mingpt.utils import get_last_ckpt, load_model_from_ckpt

logger = logging.getLogger(__name__)

N_LAYERS = 8
RUN_NAME = "classic_delflank"
STEERING_SCALE = 7.5
STEERING_LAYERS = [1, 2]  # 0-indexed layers for steering


def load_steering_vectors(run_name: str, device: torch.device) -> dict[int, torch.Tensor]:
    """Load steering vectors for specified layers."""
    probe_dir = CACHE_DIR.parent / run_name / "game_probes"
    vectors = {}
    for layer_idx in STEERING_LAYERS:
        file_layer = layer_idx + 1  # 1-indexed filename
        path = probe_dir / f"game_L{file_layer}.ckpt"
        state_dict = torch.load(path, map_location=device)
        weight = state_dict["proj.weight"].detach()  # (2, d_model)
        delta_mu = weight[1] - weight[0]  # mu_DelFlank - mu_Classic
        vectors[layer_idx] = delta_mu
    return vectors


def load_board_probe(
    run_name: str, probe_layer_idx: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load DelFlank board probe weights for the given probe layer (0-indexed).

    Returns (weight, bias) where weight is (8, 8, 3, d_model) and bias is (8, 8, 3).
    """
    probe_file_layer = probe_layer_idx + 1  # 1-indexed for filename
    probe_path = (
        CACHE_DIR.parent / run_name / "board_probes"
        / f"delflank_board_L{probe_file_layer}.ckpt"
    )
    state_dict = torch.load(probe_path, map_location=device)
    probe_w = state_dict["proj.weight"].detach().reshape(BOARD_DIM, BOARD_DIM, 3, -1)
    if "proj.bias" in state_dict:
        probe_b = state_dict["proj.bias"].detach().reshape(BOARD_DIM, BOARD_DIM, 3)
    else:
        probe_b = torch.zeros(BOARD_DIM, BOARD_DIM, 3, device=device)
    return probe_w, probe_b


def create_steering_hook(steering_vec: torch.Tensor, scale: float) -> callable:
    """Create hook that adds scaled steering vector to last-position activations."""
    vec = steering_vec.detach()

    def _hook(act: torch.Tensor, hook) -> torch.Tensor:
        act = act.clone()
        act[:, -1, :] = act[:, -1, :] + scale * vec.to(act.device)
        return act

    return _hook


def compute_board_accuracy(
    act: torch.Tensor,
    probe_w: torch.Tensor,
    probe_b: torch.Tensor,
    gt_classes: np.ndarray,
) -> float:
    """Compute board probe accuracy from activations.

    Args:
        act: Activation tensor of shape (d_model,).
        probe_w: Probe weights (8, 8, 3, d_model).
        probe_b: Probe bias (8, 8, 3).
        gt_classes: Ground truth classes (64,) with values in {0, 1, 2}.

    Returns:
        Fraction of correctly predicted tiles.
    """
    logits = torch.einsum("ijcd,d->ijc", probe_w, act) + probe_b  # (8, 8, 3)
    preds = logits.argmax(dim=-1).flatten().cpu().numpy()  # (64,)
    return float((preds == gt_classes).mean())


def main(num_seqs_per_point: int, probe_layer_idx: int, force: bool) -> None:
    cache_file = CACHE_DIR / "probe_effect_delflank.json"
    cache_key = f"{RUN_NAME}_L{probe_layer_idx}"
    results = {} if force else load_json_cache(cache_file)

    if not force and cache_key in results:
        logger.info("Results cached. Use --force to recompute.")
        return

    device = get_device()
    tokenizer = Tokenizer()
    game_classes = [GAME_REGISTRY["delflank"], GAME_REGISTRY["classic"]]

    # Load model
    ckpt_dir = CACHE_DIR.parent / RUN_NAME / "ckpts"
    last_ckpt, _ = get_last_ckpt(ckpt_dir)
    model = load_model_from_ckpt(last_ckpt, VOCAB_SIZE, BLOCK_SIZE, as_tlens=True)
    model = model.to(device)
    model.eval()

    # Load steering vectors and board probe
    steering_vecs = load_steering_vectors(RUN_NAME, device)
    probe_w, probe_b = load_board_probe(RUN_NAME, probe_layer_idx, device)

    hook_name_probe = f"blocks.{probe_layer_idx}.hook_resid_post"
    move_points = list(range(1, 21))  # Moves 1-20

    # Collect per-sequence accuracies
    baseline_accs = []
    steered_accs = {layer: [] for layer in STEERING_LAYERS}

    for div_point in tqdm(move_points, desc="Move points"):
        try:
            seqs = generate_diverging_sequences(
                div_point, game_classes, num_seqs_per_point, tokenizer
            )
        except RuntimeError:
            logger.warning("Could not generate enough sequences at t=%d", div_point)
            continue

        for seq in seqs:
            # Get ground truth board state under DelFlank
            board_gt = get_board_states(seq, game_classes[0])  # (8, 8) values in {-1, 0, 1}
            gt_classes = (board_gt + 1).astype(int).flatten()  # {0, 1, 2} -> (64,)

            tokens = torch.tensor(
                [tokenizer.encode(seq)[:BLOCK_SIZE]], dtype=torch.long, device=device
            )

            # Baseline: no steering
            with torch.inference_mode():
                _, cache = model.run_with_cache(
                    tokens, names_filter=lambda n: n == hook_name_probe
                )
            act = cache[hook_name_probe][0, -1, :]
            acc = compute_board_accuracy(act, probe_w, probe_b, gt_classes)
            baseline_accs.append(acc)

            # Steered conditions
            for steer_layer in STEERING_LAYERS:
                hook_name_steer = f"blocks.{steer_layer}.hook_resid_post"
                hook_fn = create_steering_hook(
                    steering_vecs[steer_layer], STEERING_SCALE
                )

                model.add_hook(hook_name_steer, hook_fn)
                with torch.inference_mode():
                    _, cache = model.run_with_cache(
                        tokens,
                        names_filter=lambda n: n == hook_name_probe,
                    )
                model.reset_hooks()
                act = cache[hook_name_probe][0, -1, :]
                acc = compute_board_accuracy(act, probe_w, probe_b, gt_classes)
                steered_accs[steer_layer].append(acc)

    # Aggregate
    result_data = {
        "probe_layer": probe_layer_idx,
        "steering_scale": STEERING_SCALE,
        "num_moves": len(move_points),
        "num_seqs_per_move": num_seqs_per_point,
        "baseline": {
            "mean": float(np.mean(baseline_accs)),
            "std": float(np.std(baseline_accs, ddof=1)),
            "n": len(baseline_accs),
        },
        "steered": {},
    }
    for layer in STEERING_LAYERS:
        vals = steered_accs[layer]
        result_data["steered"][str(layer)] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "n": len(vals),
        }

    results[cache_key] = result_data
    save_json_cache(results, cache_file)
    logger.info("Done: probe effect DelFlank.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute board probe accuracy under steering for DelFlank."
    )
    parser.add_argument("--num_seqs", type=int, default=200,
                        help="Sequences per divergence point.")
    parser.add_argument("--probe_layer", type=int, default=5,
                        help="0-indexed layer for board probe evaluation (default: 5).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main(args.num_seqs, args.probe_layer, args.force)
