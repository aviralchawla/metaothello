"""Download probe checkpoints from HuggingFace Hub.

Probes are hosted in the model repo (``aviralchawla/metaothello``):

- **Board probes**: ``board_probes/<run_name>/<game>_board_L<N>.ckpt``
  → ``data/<run_name>/board_probes/``
- **Game identity probes**: ``game_probes/<run_name>/game_L<N>.ckpt``
  → ``data/<run_name>/game_probes/``
- **Analytic probes**: ``analytic_probes/<run_name>/analytic_probe.ckpt``
  → ``data/<run_name>/analytic_probe.ckpt``

Usage (via Makefile -- preferred)::

    make download-board-probes
    make download-board-probe RUN_NAME=classic
    make download-game-probes
    make download-game-probe RUN_NAME=classic_nomidflip

Usage (direct)::

    python scripts/download_probes.py board
    python scripts/download_probes.py board --run_name classic
    python scripts/download_probes.py game
    python scripts/download_probes.py game --run_name classic_nomidflip
    python scripts/download_probes.py all
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

HF_MODEL_REPO = "aviralchawla/metaothello"

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


def _download_probe_dir(
    hf_prefix: str,
    local_subdir: str,
    run_name: str | None = None,
    *,
    run_names: list[str] | None = None,
) -> None:
    """Generic helper to download probe checkpoints from a HF subdirectory.

    Downloads from ``<hf_prefix>/<run>/`` on HuggingFace into
    ``data/<run>/<local_subdir>/`` locally.

    Args:
        hf_prefix: Directory prefix in the HF repo (e.g. ``"board_probes"``).
        local_subdir: Subdirectory name under ``data/<run>/`` (e.g. ``"board_probes"``).
        run_name: Restrict to a single run.  Downloads all runs when *None*.
        run_names: Override the list of runs to iterate over.
    """
    runs = [run_name] if run_name else (run_names or RUN_NAMES)

    for run in runs:
        pattern = f"{hf_prefix}/{run}/*.ckpt"
        logger.info("Downloading %s for '%s' ...", hf_prefix, run)

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download(
                repo_id=HF_MODEL_REPO,
                repo_type="model",
                allow_patterns=[pattern],
                local_dir=tmp,
            )

            src_dir = Path(tmp) / hf_prefix / run
            if not src_dir.exists():
                logger.warning("  No %s found for '%s' on HuggingFace.", hf_prefix, run)
                continue

            dest_dir = DATA_DIR / run / local_subdir
            dest_dir.mkdir(parents=True, exist_ok=True)
            for ckpt in sorted(src_dir.glob("*.ckpt")):
                dest = dest_dir / ckpt.name
                if dest.exists():
                    logger.info("  Skipping %s (already exists)", dest.relative_to(REPO_ROOT))
                    continue
                shutil.move(str(ckpt), str(dest))
                logger.info("  Saved %s", dest.relative_to(REPO_ROOT))

    logger.info("Done: %s download.", hf_prefix)


def download_board_probes(run_name: str | None = None) -> None:
    """Download board probe checkpoints from the HF model repo."""
    _download_probe_dir("board_probes", "board_probes", run_name)


def download_game_probes(run_name: str | None = None) -> None:
    """Download game identity probe checkpoints from the HF model repo."""
    _download_probe_dir("game_probes", "game_probes", run_name)


def download_analytic_probes(run_name: str | None = None) -> None:
    """Download analytic baseline probe checkpoints from the HF model repo.

    Analytic probes are stored as
    ``analytic_probes/<run_name>/analytic_probe.ckpt`` on HuggingFace and
    downloaded to ``data/<run_name>/analytic_probe.ckpt`` locally.
    """
    runs = [run_name] if run_name else RUN_NAMES

    for run in runs:
        pattern = f"analytic_probes/{run}/*.ckpt"
        logger.info("Downloading analytic probes for '%s' ...", run)

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download(
                repo_id=HF_MODEL_REPO,
                repo_type="model",
                allow_patterns=[pattern],
                local_dir=tmp,
            )

            src_dir = Path(tmp) / "analytic_probes" / run
            if not src_dir.exists():
                logger.warning("  No analytic probes found for '%s' on HuggingFace.", run)
                continue

            for ckpt in sorted(src_dir.glob("*.ckpt")):
                dest = DATA_DIR / run / ckpt.name
                if dest.exists():
                    logger.info("  Skipping %s (already exists)", dest.relative_to(REPO_ROOT))
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ckpt), str(dest))
                logger.info("  Saved %s", dest.relative_to(REPO_ROOT))

    logger.info("Done: analytic probe download.")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_all(args: argparse.Namespace) -> None:
    """Download all probe types."""
    download_board_probes(run_name=args.run_name)
    download_game_probes(run_name=args.run_name)
    download_analytic_probes(run_name=args.run_name)


def cmd_board(args: argparse.Namespace) -> None:
    """Download board probes."""
    download_board_probes(run_name=args.run_name)


def cmd_game(args: argparse.Namespace) -> None:
    """Download game identity probes."""
    download_game_probes(run_name=args.run_name)


def cmd_analytic(args: argparse.Namespace) -> None:
    """Download analytic baseline probes."""
    download_analytic_probes(run_name=args.run_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Download probe checkpoints from aviralchawla/metaothello on HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run_name",
        choices=RUN_NAMES,
        default=None,
        help="Restrict to a single run (default: all runs).",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("all", help="Download all probe types (board + game + analytic).")
    sub.add_parser("board", help="Download board probe checkpoints.")
    sub.add_parser("game", help="Download game identity probe checkpoints.")
    sub.add_parser("analytic", help="Download analytic baseline probe checkpoints.")

    return parser


def main() -> None:
    """Entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    handlers = {
        "all": cmd_all,
        "board": cmd_board,
        "game": cmd_game,
        "analytic": cmd_analytic,
    }

    if args.command is None:
        # Default: download board probes (backward compatible)
        download_board_probes(run_name=args.run_name)
    else:
        handlers[args.command](args)


if __name__ == "__main__":
    main()
