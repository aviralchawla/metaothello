"""Download pretrained models and pregenerated datasets from HuggingFace Hub.

Assets are hosted across two HuggingFace repositories:

- **Dataset repo** (``datasets/aviralchawla/metaothello``): Zarr training data
  stored flat at the repo root (e.g. ``train_classic_20M.zarr/``).  Downloaded
  into ``data/<game>/`` locally.

- **Model repo** (``aviralchawla/metaothello``): Checkpoints organised by
  run name (e.g. ``classic/epoch_250.ckpt``).  Downloaded into
  ``data/<run_name>/ckpts/`` locally.

Usage (via Makefile -- preferred)::

    make download-all
    make download-models
    make download-model  RUN_NAME=classic
    make download-data
    make download-data-game GAME=classic

Usage (direct)::

    python scripts/download.py all
    python scripts/download.py models [--run_name RUN_NAME]
    python scripts/download.py data   [--game GAME]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_DATA_REPO = "aviralchawla/metaothello"
HF_MODEL_REPO = "aviralchawla/metaothello"

GAMES = ["classic", "nomidflip", "delflank", "iago"]
RUN_NAMES = [
    "classic",
    "nomidflip",
    "delflank",
    "iago",
    "classic_nomidflip",
    "classic_delflank",
    "classic_iago",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_data(game: str | None = None) -> None:
    """Download Zarr training data from the HF dataset repo.

    The dataset repo stores zarr directories flat at the root
    (e.g. ``train_classic_20M.zarr/``).  This function downloads them
    into ``data/<game>/`` to match the local layout expected by
    ``gpt_train.py``.

    Args:
        game: Restrict to a single game variant.  Downloads all four
            single-game datasets when *None*.
    """
    games = [game] if game else GAMES

    for g in games:
        pattern = f"train_{g}_*M.zarr/**"
        logger.info("Downloading data for '%s' ...", g)

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                allow_patterns=[pattern],
                local_dir=tmp,
            )

            # Move each zarr directory into data/<game>/
            tmp_path = Path(tmp)
            for zarr_dir in sorted(tmp_path.glob(f"train_{g}_*M.zarr")):
                dest = DATA_DIR / g / zarr_dir.name
                if dest.exists():
                    logger.info("  Skipping %s (already exists)", dest.relative_to(REPO_ROOT))
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(zarr_dir), str(dest))
                logger.info("  Saved %s", dest.relative_to(REPO_ROOT))

    logger.info("Done: data download.")


def download_models(run_name: str | None = None) -> None:
    """Download model checkpoints from the HF model repo.

    The model repo stores checkpoints as
    ``<run_name>/epoch_<N>.ckpt``.  This function downloads them into
    ``data/<run_name>/ckpts/`` to match the local layout used by
    ``gpt_train.py``.

    Args:
        run_name: Restrict to a single run.  Downloads all seven runs
            when *None*.
    """
    runs = [run_name] if run_name else RUN_NAMES

    for run in runs:
        pattern = f"{run}/epoch_*.ckpt"
        logger.info("Downloading model checkpoints for '%s' ...", run)

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download(
                repo_id=HF_MODEL_REPO,
                repo_type="model",
                allow_patterns=[pattern],
                local_dir=tmp,
            )

            # Move ckpts from <tmp>/<run>/ into data/<run>/ckpts/
            src_dir = Path(tmp) / run
            if not src_dir.exists():
                logger.warning("  No checkpoints found for '%s' on HuggingFace.", run)
                continue

            dest_dir = DATA_DIR / run / "ckpts"
            dest_dir.mkdir(parents=True, exist_ok=True)
            for ckpt in sorted(src_dir.glob("*.ckpt")):
                dest = dest_dir / ckpt.name
                if dest.exists():
                    logger.info("  Skipping %s (already exists)", dest.relative_to(REPO_ROOT))
                    continue
                shutil.move(str(ckpt), str(dest))
                logger.info("  Saved %s", dest.relative_to(REPO_ROOT))

    logger.info("Done: model download.")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_all(_args: argparse.Namespace) -> None:
    """Download all models and all data."""
    download_data()
    download_models()


def cmd_models(args: argparse.Namespace) -> None:
    """Download model checkpoints."""
    download_models(run_name=args.run_name)


def cmd_data(args: argparse.Namespace) -> None:
    """Download training data."""
    download_data(game=args.game)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Download models / data from aviralchawla/metaothello on HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- all --
    sub.add_parser("all", help="Download all models and all data.")

    # -- models --
    p_models = sub.add_parser("models", help="Download GPT model checkpoints.")
    p_models.add_argument(
        "--run_name",
        choices=RUN_NAMES,
        default=None,
        help="Restrict to a single run (default: all runs).",
    )

    # -- data --
    p_data = sub.add_parser("data", help="Download training data (Zarr).")
    p_data.add_argument(
        "--game",
        choices=GAMES,
        default=None,
        help="Restrict to a single game variant (default: all games).",
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
