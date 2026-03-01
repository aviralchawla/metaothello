"""Shared utilities for MetaOthello analysis scripts.

Provides constants, enums, device detection, cache I/O, game generation, and
config helpers used across all ``scripts/analysis/`` modules.  Import from here
instead of duplicating definitions in individual scripts.
"""

from __future__ import annotations

import json
import logging
import random
import warnings
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from metaothello.constants import MAX_STEPS
from metaothello.games import GAME_REGISTRY, Iago
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


def get_all_next_valid(game_class: type, seq: list[str]) -> list[str]:
    """Return all valid next moves (in token space) given a move sequence.

    Args:
        game_class: A game class from ``GAME_REGISTRY``.
        seq: Sequence of move token names played so far.

    Returns:
        List of valid move token names at the current position.
    """
    game = game_class()
    if game_class is Iago:
        for move in seq:
            game.play_move(game.reverse_mapping[move])
        return [game.mapping[m] for m in game.get_all_valid_moves()]
    for move in seq:
        game.play_move(move)
    return game.get_all_valid_moves()


def calculate_branching(seq: list[str | None], game_class: type) -> list[list[str | None]]:
    """Record the set of valid moves (in token space) at each step of a sequence.

    Replays ``seq`` move-by-move and collects the valid-move set before each
    move is played.  If an invalid move is encountered, the remaining entries
    are filled with empty lists and a warning is issued.

    Args:
        seq: Sequence of move token names.
        game_class: A game class from ``GAME_REGISTRY``.

    Returns:
        ``valid_sets`` where ``valid_sets[t]`` is the list of valid move token
        names available *before* move ``t`` is played.  Length equals
        ``len(seq)``.
    """
    game = game_class()
    valid_sets: list[list[str | None]] = []

    for t, c in enumerate(seq):
        if game_class is Iago:
            all_valid = [game.mapping[m] for m in game.get_all_valid_moves()]
            valid_sets.append(all_valid)
            if c not in all_valid:
                warnings.warn(
                    "Invalid Iago move encountered. Sequence truncated.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                valid_sets.extend([[]] * (len(seq) - t - 1))
                break
            game.play_move(game.reverse_mapping[c])
        else:
            all_valid = game.get_all_valid_moves()
            valid_sets.append(all_valid)
            if c not in all_valid:
                warnings.warn(
                    "Invalid move encountered. Sequence truncated.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                valid_sets.extend([[]] * (len(seq) - t - 1))
                break
            game.play_move(c)

    return valid_sets


def calculate_game_probabilities(
    valid_sets_list: list[list[list[str | None]]],
    moves_list: list[str | None],
    tokenizer: Tokenizer,
) -> np.ndarray:
    """Compute Bayesian posterior game probabilities at each move position.

    Uses a uniform prior over games and updates via Bayes' rule: each move
    multiplies the likelihood of a game by ``1 / |valid_moves|`` if the move
    was legal, or by 0 if illegal.

    Args:
        valid_sets_list: Per-game valid-move sets as returned by
            :func:`calculate_branching`, one entry per game.
        moves_list: Sequence of move token names actually played.
        tokenizer: Tokenizer used to verify moves are in vocabulary.

    Returns:
        ``p_g`` array of shape ``(T+1, n_games)`` where ``p_g[0]`` is the
        uniform prior and ``p_g[t]`` is ``P(game | moves 1..t)``.

    Raises:
        ValueError: If the sequence is impossible under all provided game rules.
    """
    n_games = len(valid_sets_list)
    p: list[list[float]] = [[1 / n_games] * n_games]  # uniform prior
    x: list[float] = [1.0] * n_games  # P(observed moves | game i)

    for t, move in enumerate(moves_list):
        assert move in tokenizer.stoi, f"Move {move!r} not in tokenizer vocabulary"
        for i in range(n_games):
            v = valid_sets_list[i][t]  # valid moves before move t
            if move not in v:
                x[i] = 0.0  # illegal under game i
            else:
                x[i] *= 1 / len(v)

        total = sum(x)
        if total == 0:
            raise ValueError("Sequence is impossible under all provided game rules.")

        p.append([xi / total for xi in x])

    return np.array(p)


def calculate_ground_truth(
    seq: list[str | None],
    game_classes: list[type],
    tokenizer: Tokenizer,
    skip_p: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the ground-truth next-move distribution and game posteriors.

    Args:
        seq: Sequence of move token names.
        game_classes: List of game classes to consider.
        tokenizer: Tokenizer for encoding move names to token IDs.
        skip_p: If ``True``, return only the game-posterior array ``p_g``
            (cheaper).  If ``False``, also compute and return the full
            next-move probability array ``p_next``.

    Returns:
        If ``skip_p`` is ``True``: ``p_g`` of shape ``(T, n_games)`` where
        ``p_g[t, i] = P(game i | moves 1..t)``.

        If ``skip_p`` is ``False``: tuple ``(p_next, p_g)`` where ``p_next``
        has shape ``(T, vocab_size)`` with the Bayesian next-move distribution
        at each step, and ``p_g`` is as above.
    """
    n_games = len(game_classes)
    valid_sets = [calculate_branching(seq, G) for G in game_classes]
    p_g = calculate_game_probabilities(valid_sets, seq, tokenizer)

    if skip_p:
        return p_g[:-1, :]

    p_next = np.zeros((len(seq) - 1, tokenizer.vocab_size))
    for t in range(len(seq) - 1):
        for i in range(n_games):
            v = valid_sets[i][t + 1]
            if len(v) == 0:
                continue
            valid_tokens = tokenizer.encode(v)
            p_next[t, valid_tokens] += p_g[t + 1, i] / len(v)

        # Normalize by the total probability mass of games that can continue
        total_valid_mass = sum(
            p_g[t + 1, i] for i in range(n_games) if len(valid_sets[i][t + 1]) > 0
        )
        if total_valid_mass > 0:
            p_next[t] /= total_valid_mass

        if not np.isclose(np.sum(p_next[t]), 1.0):
            raise ValueError(f"p_next[{t}] does not sum to 1 (got {np.sum(p_next[t]):.6f})")

    return p_next, p_g[:-1, :]


def alpha_score(
    p: np.ndarray | list,
    q: np.ndarray | list,
    u: np.ndarray | list | None = None,
) -> float:
    """Compute the alpha score between model distribution q and ground truth p.

    The alpha score is defined as ``1 - KL(p || q) / KL(p || u)``, where u is
    a reference (default: uniform) distribution.  A score of 1 means q = p
    (perfect); a score of 0 means q is no better than the reference u.

    Args:
        p: Ground-truth probability distribution. Must be non-negative and sum
            to 1.
        q: Model probability distribution. Must be non-negative and sum to 1.
        u: Reference distribution (default: uniform over ``len(p)`` outcomes).
            Must be non-negative and sum to 1.

    Returns:
        Alpha score as a float.

    Raises:
        TypeError: If ``p`` or ``q`` cannot be converted to a numpy array.
        ValueError: If shape, non-negativity, or normalization constraints fail,
            or if ``KL(p || u) == 0`` (p equals the reference distribution).
    """
    try:
        p = np.asarray(p, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError("p must be convertible to a numpy array") from e
    try:
        q = np.asarray(q, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError("q must be convertible to a numpy array") from e

    if p.shape != q.shape:
        raise ValueError(f"p and q must have the same shape, got {p.shape} vs {q.shape}")
    if not np.all(p >= 0):
        raise ValueError("p must contain only non-negative values")
    if not np.all(q >= 0):
        raise ValueError("q must contain only non-negative values")
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError(f"p must sum to 1, but got {np.sum(p)}")
    if not np.isclose(np.sum(q), 1.0):
        raise ValueError(f"q must sum to 1, but got {np.sum(q)}")

    if u is None:
        u = np.ones(len(p)) / len(p)
    else:
        try:
            u = np.asarray(u, dtype=float)
        except (TypeError, ValueError) as e:
            raise TypeError("u must be convertible to a numpy array") from e
        if not np.all(u >= 0):
            raise ValueError("u must contain only non-negative values")
        if not np.isclose(np.sum(u), 1.0):
            raise ValueError(f"u must sum to 1, but got {np.sum(u)}")

    # 0 * log(0) = 0 convention
    log_p = np.where(p > 0, np.log(p), 0.0)
    log_q = np.where(q > 0, np.log(q), np.log(1e-10))
    log_u = np.where(u > 0, np.log(u), np.log(1e-10))

    numerator = float(np.sum(p * (log_p - log_q)))  # KL(p || q)
    denominator = float(np.sum(p * (log_p - log_u)))  # KL(p || u)

    if np.isclose(denominator, 0.0):
        raise ValueError("KL(p || u) == 0: p equals the reference distribution u.")

    return 1.0 - numerator / denominator


def generate_diverging_sequences(
    divergence_point: int,
    game_classes: list[type],
    num_sequences: int,
    tokenizer: Tokenizer | None = None,
    *,
    max_attempts: int = 50_000,
) -> list[list[str]]:
    """Generate sequences valid in all games that diverge at a given move.

    Uses DFS with random move shuffling and backtracking to find sequences
    of length ``divergence_point`` that are valid in both games, and where
    the valid-move sets at the end satisfy ``|V1 - V2| > 0`` **and**
    ``|V2 - V1| > 0`` (i.e. the symmetric difference is non-empty in both
    directions).

    Each DFS attempt builds a sequence one move at a time, choosing from
    the intersection of valid moves in both games (shuffled randomly for
    diversity).  If a dead end is reached the search backtracks.  When
    the sequence reaches the target length, the divergence criterion is
    checked.

    Args:
        divergence_point: Desired sequence length at which the two games'
            valid-move sets must diverge.
        game_classes: List of exactly two game classes.
        num_sequences: Number of diverging sequences to collect.
        tokenizer: Unused (kept for backward compatibility with callers).
        max_attempts: Maximum number of full DFS restarts before giving up.
            Each restart is an independent DFS search from the empty
            sequence.

    Returns:
        List of move-name sequences (each of length ``divergence_point``).

    Raises:
        RuntimeError: If ``num_sequences`` cannot be reached within
            ``max_attempts`` DFS restarts.
    """
    assert len(game_classes) == 2, "generate_diverging_sequences only supports 2 games"

    def _dfs(game_classes: list[type], sequence: list[str], target_len: int) -> list[str] | None:
        """DFS with backtracking to find one diverging sequence."""
        g1 = game_classes[0]()
        g2 = game_classes[1]()
        for m in sequence:
            g1.play_move(m)
            g2.play_move(m)

        v1 = set(g1.get_all_valid_moves())
        v2 = set(g2.get_all_valid_moves())
        # Remove Nones
        v1 = {m for m in v1 if m is not None}
        v2 = {m for m in v2 if m is not None}
        v_intersection = v1.intersection(v2)

        if len(v1) == 0 or len(v2) == 0 or len(v_intersection) == 0:
            return None

        if len(sequence) == target_len:
            # Divergence criterion: symmetric difference non-empty in BOTH directions
            if len(v1 - v2) == 0 or len(v2 - v1) == 0:
                return None
            return sequence

        # Randomly shuffle the intersection for diversity
        v_list = list(v_intersection)
        random.shuffle(v_list)

        for m in v_list:
            result = _dfs(game_classes, sequence + [m], target_len)
            if result is not None:
                return result

        return None

    found: list[list[str]] = []
    attempts = 0

    with tqdm(total=num_sequences, desc=f"Diverging seqs (t={divergence_point})", leave=False) as pbar:
        while len(found) < num_sequences:
            if attempts >= max_attempts:
                break
            attempts += 1
            result = _dfs(game_classes, [], divergence_point)
            if result is not None:
                found.append(result)
                pbar.update(1)

    if len(found) < num_sequences:
        raise RuntimeError(
            f"Only found {len(found)}/{num_sequences} diverging sequences "
            f"at t={divergence_point} after {max_attempts} DFS attempts."
        )

    return found


def get_board_states(seq: list[str], game_class: type) -> np.ndarray:
    """Replay a move sequence and return the resulting board as mine/yours/empty.

    Converts the raw board (BLACK=-1, WHITE=1, EMPTY=0) to the perspective
    of the player to move: mine=1, yours=-1, empty=0.

    Args:
        seq: Move token names to replay.
        game_class: Game class to use for replay.

    Returns:
        Board array of shape ``(8, 8)`` with values in ``{-1, 0, 1}``.
    """
    from metaothello.constants import WHITE

    game = game_class()
    for move in seq:
        game.play_move(move)
    board = game.board.copy()
    if game.next_color == WHITE:
        board *= -1
    return board
