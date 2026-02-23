"""Train a linear board probe on cached residual-stream activations.

For a given model (identified by ``--model_name``), game variant
(``--game``), and transformer layer (``--layer``), this script:

1. Loads cached activations and board states from a Zarr store.
2. Applies turn-based masking to board states (current-player-relative).
3. Trains a ``LinearProbe`` using ``ProbeTrainer``.
4. Saves the trained probe to
   ``data/{model_name}/board_probes/{game}_board_L{layer}.ckpt``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import TensorDataset

from metaothello.mingpt.board_probe import LinearProbe, ProbeTrainer
from metaothello.mingpt.utils import get_last_ckpt, set_seed

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"


def apply_turn_mask(board_states: np.ndarray) -> np.ndarray:
    """Convert absolute board representation to current-player-relative.

    Multiplies board states at even timesteps by -1 and odd timesteps by +1,
    matching the representation used by ``compute_board_probe_accuracy.py``.

    Args:
        board_states: Array of shape ``(N, T, BOARD_DIM, BOARD_DIM)``
            with values in ``{-1, 0, 1}``.

    Returns:
        Masked array of the same shape.
    """
    t = board_states.shape[1]
    turn_mask = np.ones(t, dtype=np.float32)
    turn_mask[::2] = -1.0
    return board_states * turn_mask.reshape(1, -1, 1, 1)


def get_embeddings_and_boards(
    ds: xr.Dataset,
    model_name: str,
    epoch: int,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract activation embeddings and turn-masked board states from a Zarr dataset.

    Args:
        ds: xarray Dataset opened from a Zarr store containing cached
            activations (``{model_name}_epoch{epoch}_resid_post``) and
            board states (``board_state``).
        model_name: Model run name used to locate the activation array.
        epoch: Checkpoint epoch number embedded in the activation array name.
        layer: 1-indexed transformer layer to extract.

    Returns:
        Tuple of ``(x, y)`` where ``x`` has shape ``(N*T, d_model)`` and
        ``y`` has shape ``(N*T, 64)``.
    """
    resid_key = f"{model_name}_epoch{epoch}_resid_post"
    # resid_post shape: (N, MAX_STEPS, d_model, n_layers) — drop last position, select layer
    embeddings = ds[resid_key][:, :-1, :, layer - 1].values
    # board_state shape: (N, MAX_STEPS, 8, 8) — drop last position
    board_states = ds["board_state"][:, :-1, :, :].values
    board_states = apply_turn_mask(board_states)

    x = torch.tensor(embeddings, dtype=torch.float32).reshape(-1, embeddings.shape[-1])
    y = torch.tensor(board_states, dtype=torch.int).reshape(-1, 64)

    return x, y


def split_train_test(
    x: torch.Tensor,
    y: torch.Tensor,
    test_frac: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split data into train and test sets.

    Args:
        x: Input tensor of shape ``(N, d_model)``.
        y: Target tensor of shape ``(N, 64)``.
        test_frac: Fraction of data to reserve for testing.

    Returns:
        Tuple of ``(x_train, y_train, x_test, y_test)``.
    """
    n = x.shape[0]
    split = int(n * (1 - test_frac))

    return x[:split], y[:split], x[split:], y[split:]


def find_data_entry(config: dict, game: str) -> dict:
    """Find the data entry for a specific game in the config.

    Args:
        config: Parsed config dict with a ``"data"`` list.
        game: Game alias to look up (e.g. ``"classic"``).

    Returns:
        The matching data entry dict.

    Raises:
        ValueError: If no entry matches the requested game.
    """
    for entry in config["data"]:
        if entry["game"] == game:
            return entry
    available = [e["game"] for e in config["data"]]
    msg = f"Game '{game}' not found in config. Available: {available}"
    raise ValueError(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a linear board probe on cached model activations.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model run name matching a directory under data/ (e.g., classic, classic_nomidflip)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Transformer layer to train the probe on (1-indexed, 1-8)",
    )
    parser.add_argument(
        "--game",
        type=str,
        required=True,
        help="Game variant whose data to probe on (e.g., classic, iago)",
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

    # Load config
    config_path = DATA / args.model_name / "board_probe_train_config.json"
    with open(config_path) as f:
        config = json.load(f)
    logger.info("Training probe with config: %s", config)
    logger.info("Model: %s, Game: %s, Layer: %d", args.model_name, args.game, args.layer)

    # Find data entry for the requested game
    data_entry = find_data_entry(config, args.game)
    data_path = REPO_ROOT / data_entry["path"]

    # Determine epoch from the model's last checkpoint
    ckpt_dir = DATA / args.model_name / "ckpts"
    _last_ckpt, last_epoch = get_last_ckpt(ckpt_dir)
    if last_epoch == 0:
        logger.error("No checkpoint found in %s. Train a model first.", ckpt_dir)
        sys.exit(1)
    logger.info("Using activations from epoch %d", last_epoch)

    # Load dataset
    ds = xr.open_zarr(data_path)
    x, y = get_embeddings_and_boards(ds, args.model_name, last_epoch, args.layer)
    x_train, y_train, x_test, y_test = split_train_test(x, y, test_frac=0.2)
    logger.info("Train size: %d, Test size: %d", x_train.shape[0], x_test.shape[0])

    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)

    # Setup probe and trainer
    probe = LinearProbe(device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    tr_config = config["training"]

    try:
        trainer = ProbeTrainer(probe, train_data, test_data, tr_config)
        logger.info("Training on device: %s", trainer.device)
        trainer.train()
    except Exception:
        logger.exception("Training failed.")
        sys.exit(1)

    # Save probe
    probe_save_path = (
        DATA / args.model_name / "board_probes" / f"{args.game}_board_L{args.layer}.ckpt"
    )
    probe_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), probe_save_path)
    logger.info("Probe saved to %s", probe_save_path)
