"""Compute NoMidFlip board probe accuracy on tiles that differ between games.

For sequences valid in both Classic and NoMidFlip, identifies tiles where the
two games produce different board states, then evaluates the NoMidFlip board
probe's accuracy on only those differing tiles, per layer and move number.

Caches results to data/analysis_cache/probe_accuracy_differing.json.
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

CACHE_FILE = CACHE_DIR / "probe_accuracy_differing.json"
N_LAYERS = 8
RUN_NAME = "classic_nomidflip"


def load_board_probes(
    run_name: str, game_alias: str, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load board probes for all layers. Returns list of (weight, bias) tuples.

    weight: (8, 8, 3, d_model), bias: (8, 8, 3).
    """
    probe_dir = CACHE_DIR.parent / run_name / "board_probes"
    probes = []
    for layer in range(1, N_LAYERS + 1):
        path = probe_dir / f"{game_alias}_board_L{layer}.ckpt"
        state_dict = torch.load(path, map_location=device)
        weight = state_dict["proj.weight"].detach()
        weight = weight.reshape(BOARD_DIM, BOARD_DIM, 3, -1)
        if "proj.bias" in state_dict:
            bias = state_dict["proj.bias"].detach().reshape(BOARD_DIM, BOARD_DIM, 3)
        else:
            bias = torch.zeros(BOARD_DIM, BOARD_DIM, 3, device=device)
        probes.append((weight, bias))
    return probes


def probe_accuracy_on_tiles(
    activations: torch.Tensor,
    probes: list[tuple[torch.Tensor, torch.Tensor]],
    board_gt: np.ndarray,
    differing_mask: np.ndarray,
) -> list[float]:
    """Compute soft probe accuracy on masked tiles for each layer.

    Soft accuracy = mean probability mass on the correct class over differing tiles.

    Args:
        activations: Shape (N_LAYERS, d_model) -- activations at last position.
        probes: List of N_LAYERS (weight, bias) tuples; weight (8,8,3,d_model), bias (8,8,3).
        board_gt: Ground truth board (8, 8) with values in {-1, 0, 1}.
        differing_mask: Boolean (8, 8) indicating which tiles differ between games.

    Returns:
        List of per-layer soft accuracy floats on differing tiles.
    """
    # Convert board_gt to class indices: -1->0 (yours), 0->1 (empty), 1->2 (mine)
    gt_classes = (board_gt + 1).astype(int)  # {0, 1, 2}
    gt_classes_t = torch.tensor(gt_classes, dtype=torch.long)
    differing_mask_t = torch.tensor(differing_mask, dtype=torch.bool)

    accuracies = []
    for layer_idx in range(N_LAYERS):
        probe_w, probe_b = probes[layer_idx]  # (8,8,3,d_model), (8,8,3)
        act = activations[layer_idx]  # (d_model,)

        # Compute logits for all tiles: (8, 8, 3)
        logits = torch.einsum("ijcd,d->ijc", probe_w, act) + probe_b
        probs = torch.softmax(logits, dim=-1)  # (8, 8, 3)

        # Soft accuracy on differing tiles only
        if differing_mask.sum() > 0:
            correct_probs = probs[differing_mask_t, gt_classes_t[differing_mask_t]]
            correct = correct_probs.mean().item()
        else:
            correct = float("nan")
        accuracies.append(float(correct))

    return accuracies


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

    # Load NoMidFlip board probes
    probes = load_board_probes(RUN_NAME, "nomidflip", device)

    hook_names = [f"blocks.{i}.hook_resid_post" for i in range(N_LAYERS)]

    # For each divergence point (move number), compute accuracy
    move_points = list(range(6, 51, 1))
    accuracy_by_move = defaultdict(lambda: [[] for _ in range(N_LAYERS)])

    for div_point in tqdm(move_points, desc="Divergence points"):
        try:
            seqs = generate_diverging_sequences(
                div_point, game_classes, num_seqs_per_point, tokenizer
            )
        except RuntimeError:
            logger.warning("Could not generate enough sequences at t=%d", div_point)
            continue

        for seq in seqs:
            # Get board states under both games
            board_nomidflip = get_board_states(seq, game_classes[0])
            board_classic = get_board_states(seq, game_classes[1])
            differing = board_classic != board_nomidflip

            if not differing.any():
                continue

            # Get model activations at last position
            tokens = torch.tensor(
                [tokenizer.encode(seq)], dtype=torch.long, device=device
            )
            with torch.inference_mode():
                _, cache = model.run_with_cache(
                    tokens, names_filter=lambda n: n in hook_names
                )

            # Stack layer activations at last position: (N_LAYERS, d_model)
            acts = torch.stack([cache[name][0, -1, :] for name in hook_names])

            accs = probe_accuracy_on_tiles(acts, probes, board_nomidflip, differing)
            for layer_idx, acc in enumerate(accs):
                if not np.isnan(acc):
                    accuracy_by_move[div_point][layer_idx].append(acc)

    # Aggregate: mean accuracy per (move, layer)
    result_data = {}
    for move in move_points:
        if move not in accuracy_by_move:
            continue
        layer_means = []
        layer_stds = []
        for layer_idx in range(N_LAYERS):
            vals = accuracy_by_move[move][layer_idx]
            if vals:
                layer_means.append(float(np.mean(vals)))
                layer_stds.append(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
            else:
                layer_means.append(None)
                layer_stds.append(None)
        result_data[str(move)] = {"means": layer_means, "stds": layer_stds}

    results[RUN_NAME] = result_data
    save_json_cache(results, CACHE_FILE)
    logger.info("Done: probe accuracy on differing tiles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute board probe accuracy on tiles differing between games."
    )
    parser.add_argument("--num_seqs", type=int, default=500,
                        help="Sequences per divergence point.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main(args.num_seqs, args.force)
