"""Train a linear board probe on cached residual-stream activations.

For a given model (identified by ``--model_name``), game variant
(``--game``), and transformer layer (``--layer``), this script:

1. Streams cached activations and board states from a Zarr store in chunks
   to avoid loading the full dataset into memory.
2. Applies turn-based masking to board states (current-player-relative).
3. Trains a ``LinearProbe`` via a streaming training loop.
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
import zarr
from tqdm import tqdm

from metaothello.mingpt.board_probe import LinearProbe
from metaothello.mingpt.utils import get_last_ckpt, set_seed

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"

# Number of games to load per zarr chunk. Balances memory usage against I/O
# overhead — each chunk reads all layers then discards unneeded ones.
_CHUNK_GAMES = 1024


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


def load_chunk(
    resid_arr: zarr.Array,
    board_arr: zarr.Array,
    layer_idx: int,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a chunk of games from zarr, extracting one layer and applying turn mask.

    Args:
        resid_arr: Zarr array of shape ``(N, MAX_STEPS, d_model, n_layers)``.
        board_arr: Zarr array of shape ``(N, MAX_STEPS, BOARD_DIM, BOARD_DIM)``.
        layer_idx: 0-indexed layer to extract.
        start: First game index (inclusive).
        end: Last game index (exclusive).

    Returns:
        Tuple of ``(x, y)`` where ``x`` has shape ``(chunk*T, d_model)``
        and ``y`` has shape ``(chunk*T, 64)``.
    """
    # Read only the chunk from zarr — drops last position (no model output there)
    resid = resid_arr[start:end, :-1, :, layer_idx]  # (chunk, 59, d_model)
    boards = board_arr[start:end, :-1, :, :]  # (chunk, 59, 8, 8)
    boards = apply_turn_mask(boards)

    x = torch.tensor(resid.reshape(-1, resid.shape[-1]), dtype=torch.float32)
    y = torch.tensor(boards.reshape(-1, 64), dtype=torch.int)
    return x, y


def run_epoch(
    probe: torch.nn.Module,
    resid_arr: zarr.Array,
    board_arr: zarr.Array,
    layer_idx: int,
    game_start: int,
    game_end: int,
    batch_size: int,
    device: torch.device | int,
    optimizer: torch.optim.Optimizer | None = None,
    grad_norm_clip: float = 1.0,
    epoch_num: int = 0,
    split: str = "train",
) -> float:
    """Run one epoch by streaming through zarr in game-level chunks.

    Args:
        probe: LinearProbe model (possibly wrapped in DataParallel).
        resid_arr: Zarr array for residual-stream activations.
        board_arr: Zarr array for board states.
        layer_idx: 0-indexed layer to extract.
        game_start: First game index for this split (inclusive).
        game_end: Last game index for this split (exclusive).
        batch_size: Mini-batch size for gradient updates.
        device: Device the model lives on.
        optimizer: Optimizer for training; None for eval.
        grad_norm_clip: Max gradient norm for clipping.
        epoch_num: Current epoch number for logging.
        split: ``"train"`` or ``"test"`` for logging.

    Returns:
        Mean loss over all batches in the epoch.
    """
    is_train = optimizer is not None
    probe.train(is_train)
    all_losses: list[float] = []

    num_chunks = (game_end - game_start + _CHUNK_GAMES - 1) // _CHUNK_GAMES
    pbar = tqdm(range(num_chunks), desc=f"epoch {epoch_num + 1} {split}", leave=False)

    for chunk_idx in pbar:
        cs = game_start + chunk_idx * _CHUNK_GAMES
        ce = min(cs + _CHUNK_GAMES, game_end)
        x, y_target = load_chunk(resid_arr, board_arr, layer_idx, cs, ce)

        # Shuffle within chunk for training
        if is_train:
            perm = torch.randperm(x.shape[0])
            x, y_target = x[perm], y_target[perm]

        # Mini-batch loop
        for i in range(0, x.shape[0], batch_size):
            xb = x[i : i + batch_size].to(device)
            yb = y_target[i : i + batch_size].to(device)

            with torch.set_grad_enabled(is_train):
                _logits, loss = probe(xb, yb)
                loss = loss.mean()
                all_losses.append(loss.item())

            if is_train:
                probe.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), grad_norm_clip)
                optimizer.step()

        pbar.set_description(f"epoch {epoch_num + 1} {split} loss {np.mean(all_losses):.4f}")

    return float(np.mean(all_losses))


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

    # Open zarr arrays directly (lazy — no data loaded yet)
    store = zarr.open(str(data_path), mode="r")
    resid_key = f"{args.model_name}_epoch{last_epoch}_resid_post"
    resid_arr = store[resid_key]
    board_arr = store["board_state"]
    num_games = resid_arr.shape[0]

    # Train/test split at the game level
    test_frac = 0.2
    train_end = int(num_games * (1 - test_frac))
    logger.info(
        "Dataset: %d games (%d train, %d test)",
        num_games,
        train_end,
        num_games - train_end,
    )

    # Setup probe and optimizer
    if torch.cuda.is_available():
        device: torch.device | int = torch.cuda.current_device()
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    probe = LinearProbe(device=device)
    if torch.cuda.is_available():
        probe = torch.nn.DataParallel(probe).to(device)

    tr = config["training"]
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=tr["lr"],
        weight_decay=tr["wd"],
        betas=tuple(tr.get("betas", (0.9, 0.95))),
    )
    layer_idx = args.layer - 1  # convert 1-indexed CLI arg to 0-indexed

    try:
        best_test_loss = float("inf")
        for epoch in range(tr["max_epochs"]):
            train_loss = run_epoch(
                probe,
                resid_arr,
                board_arr,
                layer_idx,
                game_start=0,
                game_end=train_end,
                batch_size=tr["batch_size"],
                device=device,
                optimizer=optimizer,
                grad_norm_clip=tr["grad_norm_clip"],
                epoch_num=epoch,
                split="train",
            )
            test_loss = run_epoch(
                probe,
                resid_arr,
                board_arr,
                layer_idx,
                game_start=train_end,
                game_end=num_games,
                batch_size=tr["batch_size"],
                device=device,
                optimizer=None,
                epoch_num=epoch,
                split="test",
            )
            logger.info(
                "Epoch %d — train loss: %.4f, test loss: %.4f",
                epoch + 1,
                train_loss,
                test_loss,
            )
            if test_loss < best_test_loss:
                best_test_loss = test_loss

        logger.info("Training complete. Best test loss: %.4f", best_test_loss)
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
