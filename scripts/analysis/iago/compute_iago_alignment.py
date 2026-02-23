"""Compute Classic-to-Iago activation alignment via orthogonal Procrustes.

1. Estimates a global transformation Omega aligning Classic to Iago resid_post.
2. Applies Omega at each layer independently on Classic sequences.
3. Computes the Iago alpha score of the model's predictions.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm

from metaothello.analysis_utils import (
    BLOCK_SIZE,
    CACHE_DIR,
    VOCAB_SIZE,
    alpha_score,
    calculate_ground_truth,
    get_device,
    load_json_cache,
    save_json_cache,
)
from metaothello.games import GAME_REGISTRY
from metaothello.mingpt.tokenizer import Tokenizer
from metaothello.mingpt.utils import get_last_ckpt, load_model_from_ckpt

logger = logging.getLogger(__name__)

CACHE_FILE = CACHE_DIR / "iago_alignment.json"
RUN_NAME = "classic_iago"


def _acts_filter(name: str) -> bool:
    return "hook_resid_post" in name


def get_paired_sequences(num_games: int, tokenizer: Tokenizer) -> tuple[np.ndarray, np.ndarray]:
    """Generate paired Classic and Iago sequences from the same physical games."""
    classic_cls = GAME_REGISTRY["classic"]
    iago_cls = GAME_REGISTRY["iago"]
    iago_game = iago_cls()

    seqs_c = []
    seqs_i = []

    for _ in tqdm(range(num_games), desc="Generating paired games", leave=False):
        for _attempt in range(1000):
            c_game = classic_cls()
            c_game.generate_random_game()
            history = c_game.get_history()
            if len(history) == BLOCK_SIZE + 1:
                seqs_c.append(tokenizer.encode(history))
                seqs_i.append(tokenizer.encode([iago_game.mapping[m] for m in history]))
                break
        else:
            raise RuntimeError("Could not generate full game.")

    return np.array(seqs_c, dtype=np.int32), np.array(seqs_i, dtype=np.int32)


def extract_all_activations(
    model: object,
    seqs: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract all resid_post activations: shape (N, T, N_LAYERS, d_model)."""
    n = len(seqs)
    all_acts = []

    with torch.inference_mode():
        for start in tqdm(range(0, n, batch_size), desc="Extracting acts", leave=False):
            end = min(start + batch_size, n)
            x = torch.tensor(seqs[start:end, :-1], dtype=torch.long, device=device)
            _, cache = model.run_with_cache(x, names_filter=_acts_filter)

            # resid_post: (N_LAYERS, B, T, d_model) -> (B, T, N_LAYERS, d_model)
            acts = cache.stack_activation("resid_post").permute(1, 2, 0, 3).cpu().numpy()
            all_acts.append(acts)

    return np.concatenate(all_acts, axis=0)


def compute_omega(acts_c: np.ndarray, acts_i: np.ndarray) -> np.ndarray:
    """Compute orthogonal Procrustes from Classic to Iago."""
    flat_c = acts_c.reshape(-1, acts_c.shape[-1])
    flat_i = acts_i.reshape(-1, acts_i.shape[-1])
    # Returns R such that ||flat_c @ R - flat_i||_F is minimized
    omega, _ = orthogonal_procrustes(flat_c, flat_i)
    return omega


def run_intervened_inference(
    model: object,
    seqs_c: np.ndarray,
    seqs_i: np.ndarray,
    omega: np.ndarray,
    layer_idx: int,
    batch_size: int,
    device: torch.device,
    tokenizer: Tokenizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference with Omega intervention at layer_idx and compute Iago alpha score."""
    n, t_plus_1 = seqs_c.shape
    t = t_plus_1 - 1

    omega_t = torch.tensor(omega, dtype=torch.float32, device=device)

    def hook_fn(acts: torch.Tensor, hook: object) -> torch.Tensor:
        return acts @ omega_t

    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    all_q = np.empty((n, t, tokenizer.vocab_size), dtype=np.float32)

    with torch.inference_mode():
        for start in tqdm(
            range(0, n, batch_size), desc=f"Intervention L{layer_idx + 1}", leave=False
        ):
            end = min(start + batch_size, n)
            x = torch.tensor(seqs_c[start:end, :-1], dtype=torch.long, device=device)

            logits = model.run_with_hooks(x, return_type="logits", fwd_hooks=[(hook_name, hook_fn)])
            all_q[start:end] = F.softmax(logits.float(), dim=-1).cpu().numpy()

    # Phase 2: compute alpha scores against Iago ground truth
    all_scores = []
    iago_cls = GAME_REGISTRY["iago"]

    for idx in tqdm(range(n), desc="Scoring", leave=False):
        names_i = tokenizer.decode(seqs_i[idx].tolist())
        try:
            p_gt, _ = calculate_ground_truth(names_i, [iago_cls], tokenizer, skip_p=False)
        except ValueError:
            all_scores.append([float("nan")] * t)
            continue

        q = all_q[idx]
        row = []
        for pos in range(len(p_gt)):
            try:
                row.append(alpha_score(p_gt[pos], q[pos]))
            except ValueError:
                row.append(float("nan"))

        row.extend([float("nan")] * (t - len(row)))
        all_scores.append(row)

    stacked = np.array(all_scores)
    n_valid = np.sum(~np.isnan(stacked), axis=0)
    means = np.nanmean(stacked, axis=0)
    std_errs = np.where(
        n_valid > 1,
        np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1)),
        np.nan,
    )
    return means, std_errs


def run_baseline_inference(
    model: object,
    seqs_c: np.ndarray,
    seqs_i: np.ndarray,
    batch_size: int,
    device: torch.device,
    tokenizer: Tokenizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Run normal inference on Classic sequences and compute Classic alpha score.

    The paper plots 'Classic baseline', which is just normal alpha score of the model
    on Classic sequences.
    """
    n, t_plus_1 = seqs_c.shape
    t = t_plus_1 - 1

    all_q = np.empty((n, t, tokenizer.vocab_size), dtype=np.float32)
    with torch.inference_mode():
        for start in tqdm(range(0, n, batch_size), desc="Baseline inference", leave=False):
            end = min(start + batch_size, n)
            x = torch.tensor(seqs_c[start:end, :-1], dtype=torch.long, device=device)
            logits = model(x)
            all_q[start:end] = F.softmax(logits.float(), dim=-1).cpu().numpy()

    all_scores = []
    classic_cls = GAME_REGISTRY["classic"]

    for idx in tqdm(range(n), desc="Baseline scoring", leave=False):
        names_c = tokenizer.decode(seqs_c[idx].tolist())
        try:
            p_gt, _ = calculate_ground_truth(names_c, [classic_cls], tokenizer, skip_p=False)
        except ValueError:
            all_scores.append([float("nan")] * t)
            continue

        q = all_q[idx]
        row = []
        for pos in range(len(p_gt)):
            try:
                row.append(alpha_score(p_gt[pos], q[pos]))
            except ValueError:
                row.append(float("nan"))

        row.extend([float("nan")] * (t - len(row)))
        all_scores.append(row)

    stacked = np.array(all_scores)
    n_valid = np.sum(~np.isnan(stacked), axis=0)
    means = np.nanmean(stacked, axis=0)
    std_errs = np.where(
        n_valid > 1,
        np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1)),
        np.nan,
    )
    return means, std_errs


def main(num_train: int, num_test: int, batch_size: int, force: bool) -> None:
    """Execute the full Procrustes alignment and intervention pipeline."""
    results = {} if force else load_json_cache(CACHE_FILE)
    if not force and "classic_iago" in results:
        logger.info("Skipping computation (cached). Use --force to recompute.")
        return

    tokenizer = Tokenizer()
    device = get_device()

    ckpt_dir = CACHE_DIR.parent / RUN_NAME / "ckpts"
    last_ckpt, last_epoch = get_last_ckpt(ckpt_dir)
    if last_ckpt is None:
        logger.error(f"No checkpoint found for {RUN_NAME}")
        return

    model = load_model_from_ckpt(last_ckpt, VOCAB_SIZE, BLOCK_SIZE, as_tlens=True)
    model = model.to(device)
    model.eval()

    # 1. Estimate Omega
    logger.info("Generating %d train games for Procrustes...", num_train)
    train_c, train_i = get_paired_sequences(num_train, tokenizer)
    logger.info("Extracting activations...")
    acts_c = extract_all_activations(model, train_c, batch_size, device)
    acts_i = extract_all_activations(model, train_i, batch_size, device)

    logger.info("Computing Omega...")
    omega = compute_omega(acts_c, acts_i)

    # Free memory
    del acts_c, acts_i

    # 2. Test Interventions
    logger.info("Generating %d test games for intervention...", num_test)
    test_c, test_i = get_paired_sequences(num_test, tokenizer)

    res_layers = {}
    for layer in range(8):
        logger.info("Testing intervention at Layer %d", layer + 1)
        means, stds = run_intervened_inference(
            model, test_c, test_i, omega, layer, batch_size, device, tokenizer
        )
        res_layers[str(layer + 1)] = {"means": means.tolist(), "stds": stds.tolist()}

    logger.info("Computing Classic baseline...")
    base_means, base_stds = run_baseline_inference(
        model, test_c, test_i, batch_size, device, tokenizer
    )

    results["classic_iago"] = {
        "num_train": num_train,
        "num_test": num_test,
        "epoch": last_epoch,
        "layers": res_layers,
        "baseline": {"means": base_means.tolist(), "stds": base_stds.tolist()},
    }

    save_json_cache(results, CACHE_FILE)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, default=1000)
    parser.add_argument("--num_test", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main(args.num_train, args.num_test, args.batch_size, args.force)
