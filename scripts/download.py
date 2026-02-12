"""Download pretrained models and pregenerated datasets from HuggingFace Hub.

HuggingFace repo: aviralchawla/metaothello
Files are saved locally preserving the repo's directory structure, so the
paths you use on HF directly determine where they land locally. Adjust the
pattern-builder functions below (_model_patterns, _data_patterns) to match
your actual repo layout.

Usage (via Makefile — preferred):
    make download-all
    make download-models
    make download-model  GAME=classic
    make download-data
    make download-data-game  GAME=classic
    make download-data-split GAME=classic SPLIT=train

Usage (direct):
    python scripts/download.py all
    python scripts/download.py models [--game GAME]
    python scripts/download.py data   [--game GAME] [--split SPLIT]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_REPO = "aviralchawla/metaothello"
HF_REPO_TYPE = "model"  # Change to "dataset" if the HF repo is a dataset repo

GAMES = ["classic", "nomidflip", "delflank", "iago"]
SPLITS = ["train", "val"]

# Local root directories (relative to repo root, i.e. where you run make from)
DATA_DIR = Path("data/games")
MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# Path pattern builders — EDIT THESE to match your HF repo layout.
#
# snapshot_download preserves repo structure under local_dir, so if your
# repo has  models/classic/run1.ckpt  and local_dir=MODELS_DIR, it lands at
# models/classic/run1.ckpt locally.
# ---------------------------------------------------------------------------


def _model_patterns(game: str | None) -> list[str]:
    """Glob patterns selecting .ckpt files, optionally scoped to one game."""
    if game is not None:
        return [f"models/{game}/*.ckpt"]  # TODO: adjust to your repo layout
    return ["models/**/*.ckpt"]


def _data_patterns(game: str | None, split: str | None) -> list[str]:
    """Glob patterns selecting Zarr data files, optionally scoped."""
    if game is not None and split is not None:
        return [f"data/{game}/{split}/**"]  # TODO: adjust to your repo layout
    if game is not None:
        return [f"data/{game}/**"]
    return ["data/**"]


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def _download(allow_patterns: list[str], local_dir: Path, label: str) -> None:
    """Run snapshot_download for the given patterns and log progress."""
    logger.info("Downloading %s → %s", label, local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO,
        repo_type=HF_REPO_TYPE,
        allow_patterns=allow_patterns,
        local_dir=str(local_dir),
    )
    logger.info("Done: %s", label)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_all(_args: argparse.Namespace) -> None:
    """Download all models and all data."""
    _download(_model_patterns(None), MODELS_DIR, "all models")
    _download(_data_patterns(None, None), DATA_DIR, "all data")


def cmd_models(args: argparse.Namespace) -> None:
    """Download models, optionally filtered to a single game."""
    game: str | None = args.game
    label = f"model [{game}]" if game else "all models"
    _download(_model_patterns(game), MODELS_DIR, label)


def cmd_data(args: argparse.Namespace) -> None:
    """Download data, optionally filtered by game and/or split."""
    game: str | None = args.game
    split: str | None = args.split

    if split is not None and game is None:
        logger.error("--split requires --game to be specified.")
        sys.exit(1)

    parts = [p for p in [game, split] if p is not None]
    label = f"data [{'/'.join(parts)}]" if parts else "all data"
    _download(_data_patterns(game, split), DATA_DIR, label)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download models / data from aviralchawla/metaothello on HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- all --
    sub.add_parser("all", help="Download all models and all data.")

    # -- models --
    p_models = sub.add_parser("models", help="Download GPT models.")
    p_models.add_argument(
        "--game",
        choices=GAMES,
        default=None,
        help="Restrict to a single game variant (default: all games).",
    )

    # -- data --
    p_data = sub.add_parser("data", help="Download game data.")
    p_data.add_argument(
        "--game",
        choices=GAMES,
        default=None,
        help="Restrict to a single game variant (default: all games).",
    )
    p_data.add_argument(
        "--split",
        choices=SPLITS,
        default=None,
        help="Restrict to a single split; requires --game (default: all splits).",
    )

    return parser


def main() -> None:
    """Entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    handlers: dict[str, object] = {
        "all": cmd_all,
        "models": cmd_models,
        "data": cmd_data,
    }
    handlers[args.command](args)  # type: ignore[operator]


if __name__ == "__main__":
    main()
