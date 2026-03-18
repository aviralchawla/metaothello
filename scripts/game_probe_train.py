"""Train a linear game identity probe on model activations.

For a given mixed-game model (identified by ``--model_name``) and transformer
layer (``--layer``), this script:

1. Generates random games for each component game variant.
2. Runs the model to extract residual-stream activations on the fly.
3. Assigns binary labels (0 for the first game, 1 for the second).
4. Trains a linear probe (``nn.Linear(d_model, 2)``) to predict game identity.
5. Saves the trained probe to
   ``data/{model_name}/game_probes/game_L{layer}.ckpt``.

The probe outputs logits ``[score_game0, score_game1]``; apply softmax
to get ``P(game_i | activation)``.

Usage::

    python scripts/game_probe_train.py --model_name classic_nomidflip --layer 5
    # or via Makefile:
    make train-game-probe MODEL_NAME=classic_nomidflip LAYER=5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import TensorDataset
from tqdm import tqdm

from metaothello.analysis_utils import (
    BLOCK_SIZE,
    VOCAB_SIZE,
    get_device,
)
from metaothello.games import GAME_REGISTRY
from metaothello.mingpt.board_probe import ProbeTrainer
from metaothello.mingpt.tokenizer import Tokenizer
from metaothello.mingpt.utils import get_last_ckpt, load_model_from_ckpt, set_seed

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
N_LAYERS = 8


class GameIdentityProbe(nn.Module):
    """Linear probe for binary game identity classification.

    Maps residual-stream activations to a 2-class logit vector.
    Compatible with :class:`~metaothello.mingpt.board_probe.ProbeTrainer`.

    The saved state dict contains ``proj.weight`` of shape ``(2, d_model)``
    and ``proj.bias`` of shape ``(2,)``, matching the format expected by
    downstream analysis scripts.
    """

    def __init__(
        self,
        device: torch.device | str,
        input_dim: int = 512,
    ) -> None:
        """Initialize the game identity probe.

        Args:
            device: Target device for the model.
            input_dim: Dimensionality of the input activations (d_model).
        """
        super().__init__()
        self.proj = nn.Linear(input_dim, 2)
        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logits and optionally the cross-entropy loss.

        Args:
            x: Input activations of shape ``(batch_size, input_dim)``.
            y: Game identity labels of shape ``(batch_size,)`` with values
                in ``{0, 1}``. When None, loss is not computed.

        Returns:
            Tuple of ``(logits, loss)``. Logits have shape
            ``(batch_size, 2)``. Loss is None when y is None.
        """
        logits = self.proj(x)  # (batch, 2)
        if y is None:
            return logits, None
        loss = F.cross_entropy(logits, y.long())
        return logits, loss


def generate_games(
    game_alias: str,
    num_games: int,
) -> list[list[str]]:
    """Generate random complete games for a given variant.

    Args:
        game_alias: Key into ``GAME_REGISTRY``.
        num_games: Number of games to generate.

    Returns:
        List of move-name sequences.
    """
    game_class = GAME_REGISTRY[game_alias]
    seqs: list[list[str]] = []
    max_retries = 1000
    for _ in tqdm(range(num_games), desc=f"Generating {game_alias}", leave=False):
        for _attempt in range(max_retries):
            g = game_class()
            g.generate_random_game()
            history = g.get_history()
            if len(history) >= 10:
                seqs.append(history)
                break
    return seqs


def extract_activations(
    model: torch.nn.Module,
    seqs: list[list[str]],
    tokenizer: Tokenizer,
    layer: int,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Run model inference and extract residual-stream activations for one layer.

    Args:
        model: TransformerLens HookedTransformer model.
        seqs: List of move-name sequences.
        tokenizer: Tokenizer for encoding sequences.
        layer: 1-indexed layer to extract (1-8).
        device: Torch device for inference.
        batch_size: Batch size for inference.

    Returns:
        Tensor of shape ``(total_positions, d_model)`` with activations
        from every valid position across all sequences.
    """
    hook_name = f"blocks.{layer - 1}.hook_resid_post"
    all_acts = []

    for i in tqdm(range(0, len(seqs), batch_size), desc="Extracting activations", leave=False):
        batch_seqs = seqs[i : i + batch_size]

        # Encode and pad to same length
        encoded = [tokenizer.encode(s)[:BLOCK_SIZE] for s in batch_seqs]
        max_len = max(len(e) for e in encoded)
        padded = [e + [0] * (max_len - len(e)) for e in encoded]
        tokens = torch.tensor(padded, dtype=torch.long, device=device)

        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)

        acts = cache[hook_name]  # (batch, seq_len, d_model)

        # Extract valid (non-padded) positions
        for j, enc in enumerate(encoded):
            seq_len = min(len(enc), BLOCK_SIZE)
            all_acts.append(acts[j, :seq_len, :].cpu())

    return torch.cat(all_acts, dim=0)


def split_train_test(
    x: torch.Tensor,
    y: torch.Tensor,
    test_frac: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shuffle and split data into train and test sets.

    Args:
        x: Input tensor of shape ``(N, d_model)``.
        y: Label tensor of shape ``(N,)``.
        test_frac: Fraction of data to reserve for testing.

    Returns:
        Tuple of ``(x_train, y_train, x_test, y_test)``.
    """
    n = x.shape[0]
    perm = torch.randperm(n)
    x, y = x[perm], y[perm]
    split = int(n * (1 - test_frac))
    return x[:split], y[:split], x[split:], y[split:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a linear game identity probe on model activations.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help=(
            "Mixed-game model run name matching a directory under data/ "
            "(e.g., classic_nomidflip, classic_delflank)"
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Transformer layer to train the probe on (1-indexed, 1-8)",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=2000,
        help="Number of games per variant to generate (default: 2000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for model inference (default: 64)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    set_seed(42)

    # Determine game aliases from the model name
    game_aliases = args.model_name.split("_")
    for alias in game_aliases:
        if alias not in GAME_REGISTRY:
            logger.error("Unknown game alias '%s' in model name '%s'.", alias, args.model_name)
            sys.exit(1)
    if len(game_aliases) < 2:
        logger.error("Game identity probes require a mixed-game model (e.g., classic_nomidflip).")
        sys.exit(1)

    logger.info("Model: %s, Layer: %d", args.model_name, args.layer)
    logger.info("Games: %s, %d games each", game_aliases, args.num_games)

    # Load model
    device = get_device()
    ckpt_dir = DATA / args.model_name / "ckpts"
    last_ckpt, last_epoch = get_last_ckpt(ckpt_dir)
    if last_ckpt is None:
        logger.error("No checkpoint found in %s. Train a model first.", ckpt_dir)
        sys.exit(1)
    logger.info("Loading model from epoch %d", last_epoch)
    model = load_model_from_ckpt(last_ckpt, VOCAB_SIZE, BLOCK_SIZE, as_tlens=True)
    model = model.to(device)
    model.eval()

    tokenizer = Tokenizer()

    # Generate games and extract activations for each variant
    all_x = []
    all_y = []
    for game_idx, alias in enumerate(game_aliases):
        seqs = generate_games(alias, args.num_games)
        logger.info("Generated %d %s games", len(seqs), alias)

        x = extract_activations(model, seqs, tokenizer, args.layer, device, args.batch_size)
        y = torch.full((x.shape[0],), game_idx, dtype=torch.long)
        all_x.append(x)
        all_y.append(y)
        logger.info("  %s: %d activation samples", alias, x.shape[0])

    # Free model memory before training probe
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    x_all = torch.cat(all_x, dim=0)
    y_all = torch.cat(all_y, dim=0)
    logger.info("Total samples: %d", x_all.shape[0])

    x_train, y_train, x_test, y_test = split_train_test(x_all, y_all, test_frac=0.2)
    logger.info("Train size: %d, Test size: %d", x_train.shape[0], x_test.shape[0])

    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)

    # Setup probe and trainer
    probe_device: torch.device | str = (
        torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    )
    probe = GameIdentityProbe(device=probe_device)

    tr_config = {
        "max_epochs": 10,
        "batch_size": 1024,
        "lr": 0.0003,
        "wd": 0.01,
        "betas": [0.9, 0.95],
        "grad_norm_clip": 1.0,
        "num_workers": 0,
    }

    try:
        trainer = ProbeTrainer(probe, train_data, test_data, tr_config)
        logger.info("Training on device: %s", trainer.device)
        trainer.train()
    except Exception:
        logger.exception("Training failed.")
        sys.exit(1)

    # Save probe — unwrap DataParallel if necessary
    model_to_save = probe.module if hasattr(probe, "module") else probe
    probe_save_path = DATA / args.model_name / "game_probes" / f"game_L{args.layer}.ckpt"
    probe_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_to_save.state_dict(), probe_save_path)
    logger.info("Probe saved to %s", probe_save_path)
