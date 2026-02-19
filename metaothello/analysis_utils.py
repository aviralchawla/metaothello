"""Shared utilities for MetaOthello analysis scripts.

Provides constants, enums, device detection, cache I/O, game generation, and
config helpers used across all ``scripts/analysis/`` modules.  Import from here
instead of duplicating definitions in individual scripts.
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from metaothello.constants import MAX_STEPS
from metaothello.games import GAME_REGISTRY
from metaothello.mingpt.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants (derived from package location, not script location)
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
"""Absolute path to the repository root directory."""

CACHE_DIR: Path = REPO_ROOT / "data" / "analysis_cache"
"""Default directory for cached analysis results (JSON)."""

# ---------------------------------------------------------------------------
# Model / tokenizer constants
# ---------------------------------------------------------------------------

VOCAB_SIZE: int = 66
"""Tokenizer vocabulary size (64 squares + PAD + pass)."""

BLOCK_SIZE: int = MAX_STEPS - 1  # 59
"""Model context window length (T = MAX_STEPS - 1)."""

ALL_RUN_NAMES: list[str] = [
    "classic",
    "nomidflip",
    "delflank",
    "iago",
    "classic_nomidflip",
    "classic_delflank",
    "classic_iago",
]
"""All trained model run names."""


# ---------------------------------------------------------------------------
# Metric enum
# ---------------------------------------------------------------------------


class Metric(StrEnum):
    """Evaluation metric selector.

    Used by compute scripts to select which metric to evaluate.
    ``ALPHA`` is defined but not yet implemented.
    """

    TOP1 = "top1"
    CORRECT_PROB = "correct_prob"
    ALPHA = "alpha"


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    """Detect and return the best available torch device.

    Prefers CUDA, falls back to MPS (Apple Silicon), then CPU.

    Returns:
        A ``torch.device`` for the best available accelerator.
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    elif hasattr(torch, "mps") and torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def get_game_aliases(run_name: str) -> list[str]:
    """Extract game aliases from a run's training config.

    Args:
        run_name: Run name matching a directory under ``data/``.

    Returns:
        List of game alias strings (e.g. ``["classic", "nomidflip"]``).
    """
    config_path = REPO_ROOT / "data" / run_name / "train_config.json"
    with config_path.open() as f:
        config = json.load(f)
    return [entry["game"] for entry in config["data"]]


# ---------------------------------------------------------------------------
# JSON cache I/O
# ---------------------------------------------------------------------------


def load_json_cache(cache_file: Path) -> dict[str, Any]:
    """Load a JSON cache file, returning an empty dict if it doesn't exist.

    Args:
        cache_file: Path to the JSON cache file.

    Returns:
        Parsed JSON dict, or empty dict if the file is missing.
    """
    if cache_file.exists():
        with cache_file.open() as f:
            return json.load(f)
    return {}


def save_json_cache(results: dict[str, Any], cache_file: Path) -> None:
    """Write results to a JSON cache file, creating parent directories.

    Args:
        results: Results dict to serialize.
        cache_file: Path to write the JSON file to.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results cached to %s", cache_file)


# ---------------------------------------------------------------------------
# Game generation with valid-move masks
# ---------------------------------------------------------------------------

_MAX_RETRIES: int = 1000


def gen_games(
    game_alias: str,
    num_games: int,
    tokenizer: Tokenizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate fresh random games and record per-step valid-move masks.

    Generates ``num_games`` complete games (exactly ``MAX_STEPS`` moves each)
    for the given variant.  Each game is replayed step-by-step to build a
    boolean mask of which tokens are legal at each position.

    For Iago, physical valid moves are remapped through the game's syntax map
    before encoding, so the mask is in the model's shuffled-token space.

    Args:
        game_alias: Key into ``GAME_REGISTRY`` (e.g. ``"classic"``).
        num_games: Number of games to generate.
        tokenizer: Tokenizer instance for encoding move histories.

    Returns:
        Tuple of:
        - **seqs**: ``int32`` array of shape ``(num_games, MAX_STEPS)`` with
          token IDs.
        - **valid_masks**: ``bool`` array of shape
          ``(num_games, MAX_STEPS, vocab_size)`` where ``valid_masks[i, s]``
          is ``True`` at token positions that are valid moves before step
          ``s`` is played.

    Raises:
        RuntimeError: If a valid game cannot be generated within retries.
    """
    game_class = GAME_REGISTRY[game_alias]
    seqs: list[list[int]] = []
    valid_masks: list[np.ndarray] = []
    vocab_size = tokenizer.vocab_size

    for _ in tqdm(range(num_games), desc=f"Generating {game_alias} games", leave=False):
        for _attempt in range(_MAX_RETRIES):
            g = game_class()  # type: ignore[reportCallIssue]
            g.generate_random_game()
            history = g.get_history()
            if len(history) == MAX_STEPS:
                seqs.append(tokenizer.encode(history))

                g_replay = game_class()  # type: ignore[reportCallIssue]
                has_mapping = hasattr(g_replay, "mapping")
                game_masks = np.zeros((MAX_STEPS, vocab_size), dtype=bool)

                for step in range(MAX_STEPS):
                    valid_physical = g_replay.get_all_valid_moves()
                    valid_names = (
                        [g_replay.mapping[m] for m in valid_physical]
                        if has_mapping
                        else valid_physical
                    )
                    for name in valid_names:
                        game_masks[step, tokenizer.stoi[name]] = True

                    if has_mapping:
                        g_replay.play_move(g_replay.reverse_mapping[history[step]])
                    else:
                        g_replay.play_move(history[step])

                valid_masks.append(game_masks)
                break
        else:
            msg = (
                f"Could not generate a {MAX_STEPS}-step game for "
                f"'{game_alias}' after {_MAX_RETRIES} retries."
            )
            raise RuntimeError(msg)

    return np.array(seqs, dtype=np.int32), np.array(valid_masks, dtype=bool)
