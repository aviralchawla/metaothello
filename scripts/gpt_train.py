from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from metaothello.mingpt.dataset import SequenceDataset
from metaothello.mingpt.tokenizer import Tokenizer
from metaothello.mingpt.trainer import Trainer, TrainerConfig
from metaothello.mingpt.utils import (
    load_fresh_model,
    load_model_from_ckpt,
    set_seed,
    shuffle_data,
    split_train_test,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"


def get_dataset(data_path: Path) -> np.ndarray:
    """Open a Zarr game dataset and return the pre-tokenised sequence array.

    Args:
        data_path: Path to the .zarr store produced by generate_data.py.

    Returns:
        Numpy array of shape (num_games, MAX_STEPS) with dtype int32.
    """
    ds = xr.open_zarr(data_path)
    return ds["seqs"].values


def get_last_ckpt(ckpt_dir: Path) -> tuple[Path | None, int]:
    """Return the most recent checkpoint file and its epoch number.

    Args:
        ckpt_dir: Directory containing .ckpt files named epoch_{n}.ckpt.

    Returns:
        Tuple of (checkpoint_path, epoch_number). Returns (None, 0) if the
        directory does not exist or contains no checkpoints.
    """
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True)
        return None, 0
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return None, 0
    ckpts = sorted(ckpts, key=lambda x: int(x.stem.split("_")[-1]))
    last_ckpt = ckpts[-1]
    last_epoch = int(last_ckpt.stem.split("_")[-1])
    return last_ckpt, last_epoch


def load_data(
    config: dict,
    test_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and split game sequence data from Zarr stores.

    Args:
        config: Training configuration dict with a 'data' list, each entry
            containing a 'path' key relative to the repository root.
        test_frac: Fraction of each dataset to reserve for the test split.

    Returns:
        Tuple of (train_data, test_data) as numpy arrays of shape
        (n_games, MAX_STEPS).
    """
    if len(config["data"]) == 1:
        ds = get_dataset(REPO_ROOT / config["data"][0]["path"])
        return split_train_test(ds, test_frac=test_frac)

    train = []
    test = []
    for dataset in config["data"]:
        ds = get_dataset(REPO_ROOT / dataset["path"])
        train_ds, test_ds = split_train_test(ds, test_frac=test_frac)
        train.append(train_ds)
        test.append(test_ds)

    return np.concatenate(train, axis=0), np.concatenate(test, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT model on Othello game sequences.")
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Training run name matching a directory under data/ (e.g., classic)",
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
    with open(DATA / args.run_name / "train_config.json") as f:
        config = json.load(f)
    logger.info("Training with config: %s", config)

    # Load dataset
    train, test = load_data(config, test_frac=0.2)
    logger.info("Train size: %d, Test size: %d", len(train), len(test))

    train_ds = SequenceDataset(shuffle_data(train), Tokenizer(), tokenize=False)
    test_ds = SequenceDataset(shuffle_data(test), Tokenizer(), tokenize=False)

    # Check checkpoints
    ckpt_dir = DATA / "ckpts" / config["run_name"]
    last_ckpt, last_epoch = get_last_ckpt(ckpt_dir)
    logger.info("Last checkpoint: %s, last epoch: %d", last_ckpt, last_epoch)

    # Load model
    if last_ckpt is None:
        model = load_fresh_model(train_ds.vocab_size, train_ds.block_size)
    else:
        model = load_model_from_ckpt(last_ckpt, train_ds.vocab_size, train_ds.block_size)

    # Training config
    config_training = config["training"]
    tconf = TrainerConfig(
        max_epochs=config_training["max_epochs"],
        batch_size=config_training["batch_size"],
        learning_rate=config_training["learning_rate"],
        lr_decay=config_training["lr_decay"],
        betas=tuple(config_training["betas"]),
        grad_norm_clip=config_training["grad_norm_clip"],
        weight_decay=config_training["weight_decay"],
        warmup_tokens=len(train_ds) * train_ds.block_size * 5 if last_epoch == 0 else 0,
        final_tokens=config_training["max_epochs"] * len(train_ds) * (train_ds.block_size - 1),
        num_workers=0,
        ckpt_path=ckpt_dir,
    )

    try:
        trainer = Trainer(model, train_ds, test_ds, tconf)
        logger.info("Training on device: %s", trainer.device)
        trainer.train(last_epoch)
    except Exception:
        logger.exception("Training failed.")
        sys.exit(1)
