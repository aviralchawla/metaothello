"""Download board probe checkpoints from HuggingFace Hub.

Probes are hosted in the model repo (``aviralchawla/metaothello``) under
``board_probes/<run_name>/<game>_board_L<N>.ckpt``.  Downloaded into
``data/<run_name>/board_probes/`` locally.

Usage (via Makefile -- preferred)::

    make download-board-probes
    make download-board-probe RUN_NAME=classic

Usage (direct)::

    python scripts/download_probes.py
    python scripts/download_probes.py --run_name classic
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
# Download helper
# ---------------------------------------------------------------------------


def download_probes(run_name: str | None = None) -> None:
    """Download board probe checkpoints from the HF model repo.

    The model repo stores probes as
    ``board_probes/<run_name>/<game>_board_L<N>.ckpt``.  This function
    downloads them into ``data/<run_name>/board_probes/`` to match the
    local layout expected by probe analysis scripts.

    Args:
        run_name: Restrict to a single run.  Downloads all seven runs
            when *None*.
    """
    runs = [run_name] if run_name else RUN_NAMES

    for run in runs:
        pattern = f"board_probes/{run}/*.ckpt"
        logger.info("Downloading board probes for '%s' ...", run)

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download(
                repo_id=HF_MODEL_REPO,
                repo_type="model",
                allow_patterns=[pattern],
                local_dir=tmp,
            )

            src_dir = Path(tmp) / "board_probes" / run
            if not src_dir.exists():
                logger.warning("  No board probes found for '%s' on HuggingFace.", run)
                continue

            dest_dir = DATA_DIR / run / "board_probes"
            dest_dir.mkdir(parents=True, exist_ok=True)
            for ckpt in sorted(src_dir.glob("*.ckpt")):
                dest = dest_dir / ckpt.name
                if dest.exists():
                    logger.info("  Skipping %s (already exists)", dest.relative_to(REPO_ROOT))
                    continue
                shutil.move(str(ckpt), str(dest))
                logger.info("  Saved %s", dest.relative_to(REPO_ROOT))

    logger.info("Done: board probe download.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Download board probes from aviralchawla/metaothello on HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run_name",
        choices=RUN_NAMES,
        default=None,
        help="Restrict to a single run (default: all runs).",
    )
    return parser


def main() -> None:
    """Entry point."""
    args = _build_parser().parse_args()
    download_probes(run_name=args.run_name)


if __name__ == "__main__":
    main()
