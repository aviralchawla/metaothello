from __future__ import annotations

import argparse
import logging
import multiprocessing
import multiprocessing.synchronize
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from tqdm import tqdm

from metaothello.constants import MAX_STEPS
from metaothello.games import ClassicOthello, DeleteFlanking, Iago, NoMiddleFlip
from metaothello.metaothello import MetaOthello
from metaothello.mingpt.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

GAME_REGISTRY = {cls.alias: cls for cls in [ClassicOthello, NoMiddleFlip, DeleteFlanking, Iago]}
# Use all available CPUs for the pool (useful when running on HPC)
NUM_CPU = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
MAX_RETRIES = 1000
tokenizer = Tokenizer()
REPO_ROOT = Path(__file__).resolve().parent.parent


def _gen_single_game(
    game_class: type[MetaOthello],
) -> tuple[list[str | None], list[np.ndarray]]:
    """Generate a single game with MAX_STEPS moves."""
    for _ in range(MAX_RETRIES):
        g = game_class()
        g.generate_random_game()
        if len(g.get_history()) == MAX_STEPS:
            return g.get_history(), g.get_board_history()

    raise RuntimeError(
        f"Failed to generate a valid {MAX_STEPS}-move game after {MAX_RETRIES} attempts "
        f"for {game_class.__name__}."
    )


def _gen_single_game_dataset(game_class: type[MetaOthello], save_board: bool) -> xr.Dataset:
    """Generate a single game and return as xarray Dataset."""
    np.random.seed(None)
    history, board_history = _gen_single_game(game_class)
    seqs = tokenizer.encode(history)
    if save_board:
        return xr.Dataset(
            data_vars={
                "seqs": (["move"], np.array(seqs, dtype=np.int32)),
                "board_state": (["move", "x", "y"], np.array(board_history)),
            }
        )
    return xr.Dataset(data_vars={"seqs": (["move"], np.array(seqs, dtype=np.int32))})


def _worker_process(
    _worker_id: int,
    game_class: type[MetaOthello],
    task_queue: multiprocessing.Queue[tuple[int, int]],
    save_path: str,
    save_board: bool,
    lock: multiprocessing.synchronize.Lock,
    file_initialized: Any,
    progress_counter: Any,
) -> None:
    """Worker process that generates chunks of games and saves them to Zarr.

    Each worker:
    1. Gets a chunk assignment from the task queue
    2. Generates chunk_size games
    3. Saves the chunk to Zarr (thread-safe via lock)
    4. Reports completed games to the main process via progress_counter
    5. Repeats until task_queue is empty
    """
    while True:
        try:
            _, num_games = task_queue.get_nowait()
        except queue.Empty:
            break

        # Generate games for this chunk
        results = []
        for _ in range(num_games):
            results.append(_gen_single_game_dataset(game_class, save_board))

        # Concatenate and save
        ds_chunk = xr.concat(results, dim="game")

        # Thread-safe file write
        with lock:
            if not file_initialized.value:
                ds_chunk.to_zarr(save_path, zarr_format=2, mode="w")
                file_initialized.value = True
            else:
                ds_chunk.to_zarr(save_path, mode="a", append_dim="game")

        # Report completed games to main process (atomic increment)
        with progress_counter.get_lock():
            progress_counter.value += num_games


def generate_with_parallel_workers(
    n: int,
    game_class: type[MetaOthello],
    num_workers: int,
    chunk_size: int,
    save_path: str,
    save_board: bool,
) -> None:
    """Generate data using parallel workers with a single global progress bar.

    Args:
        n: Total number of games to generate.
        game_class: The game class to instantiate.
        num_workers: Number of parallel worker processes.
        chunk_size: Max number of games each worker generates before saving.
        save_path: Path to save the Zarr dataset.
        save_board: Flag to save or not save the board state history.
    """
    logger.info("Starting generation: %d games with %d workers.", n, num_workers)
    logger.info("Chunk size: %d games per save operation.", chunk_size)
    logger.info("Save path: %s", save_path)

    if parent := os.path.dirname(save_path):
        os.makedirs(parent, exist_ok=True)

    # Create task queue: list of (chunk_id, num_games) tuples
    task_queue: multiprocessing.Queue[tuple[int, int]] = multiprocessing.Queue()
    remaining = n
    chunk_id = 0

    while remaining > 0:
        current_chunk = min(chunk_size, remaining)
        task_queue.put((chunk_id, current_chunk))
        remaining -= current_chunk
        chunk_id += 1

    logger.info("Created %d chunks to process.", chunk_id)

    # Shared state: lock for Zarr writes, initialization flag, and global game counter
    lock = multiprocessing.Lock()
    file_initialized = multiprocessing.Value("b", False)
    progress_counter = multiprocessing.Value("i", 0)

    overall_start = time.time()

    # Spawn worker processes
    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=_worker_process,
            args=(
                i,
                game_class,
                task_queue,
                save_path,
                save_board,
                lock,
                file_initialized,
                progress_counter,
            ),
        )
        p.start()
        processes.append(p)

    # Single progress bar driven by polling the shared counter
    with tqdm(total=n, desc=f"Generating {game_class.alias}", unit="game") as pbar:
        reported = 0
        while any(p.is_alive() for p in processes):
            current = progress_counter.value
            if current > reported:
                pbar.update(current - reported)
                reported = current
            time.sleep(0.5)
        # Drain any remaining count after all processes finish
        pbar.update(progress_counter.value - reported)

    for p in processes:
        p.join()

    total_elapsed = time.time() - overall_start
    logger.info("Done. %d games saved to %s in %.1fs.", n, save_path, total_elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random Othello game data.")
    parser.add_argument(
        "--num_games",
        type=float,
        default=1,
        help="Number of games to generate, in millions (default: 1)",
    )
    parser.add_argument(
        "--game",
        type=str,
        required=True,
        choices=GAME_REGISTRY.keys(),
        help="Game variant to generate data for",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_CPU,
        help=f"Number of parallel workers (default: {NUM_CPU})",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50_000,
        help="Number of games each worker generates before saving (default: 10000)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "board_train"],
        help="Dataset split to generate (default: train)",
    )
    parser.add_argument(
        "--save_board",
        tyep=bool,
        default=False,
        help="If enabled, the script also saves board state history to the dataset",
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

    N = int(args.num_games * 1_000_000)
    if N <= 0:
        logger.error(
            "Number of games must be positive. Got --num_games=%.4f (=%d games).", args.num_games, N
        )
        sys.exit(1)

    GameClass = GAME_REGISTRY[args.game]
    logger.info("Game variant: %s (%s)", GameClass.__name__, args.game)
    logger.info("Split: %s", args.split)
    logger.info("Total games to generate: %d (%.2fM)", N, args.num_games)

    save_path = (
        REPO_ROOT
        / "data"
        / "games"
        / args.game
        / args.split
        / f"{args.game}_{int(args.num_games)}.zarr"
    )

    try:
        generate_with_parallel_workers(
            N, GameClass, args.num_workers, args.chunk_size, str(save_path), args.save_board
        )
    except Exception:
        logger.exception("Data generation failed.")
        sys.exit(1)
