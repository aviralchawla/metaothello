from __future__ import annotations

import argparse
import contextlib
import logging
import threading
from pathlib import Path
from queue import Queue
from typing import Any

import numpy as np
import torch
import xarray as xr
import zarr
from tqdm import tqdm
from transformer_lens import HookedTransformer

from metaothello.games import ClassicOthello, DeleteFlanking, Iago, NoMiddleFlip
from metaothello.mingpt.dataset import SequenceDataset
from metaothello.mingpt.tokenizer import Tokenizer
from metaothello.mingpt.utils import load_model_from_ckpt

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
GAME_REGISTRY = {cls.alias: cls for cls in [ClassicOthello, NoMiddleFlip, DeleteFlanking, Iago]}


def get_dataset(data_path: Path) -> np.ndarray:
    """Open a Zarr game dataset and return the pre-tokenised sequence array.

    Args:
        data_path: Path to the .zarr store produced by generate_data.py.

    Returns:
        Numpy array of shape (num_games, MAX_STEPS) with dtype int32.
    """
    ds = xr.open_zarr(data_path)
    return ds["seqs"].values


def get_last_ckpt(ckpt_dir: Path) -> Path:
    """Return the most recent checkpoint path in ckpt_dir.

    Args:
        ckpt_dir: Directory containing .ckpt files named epoch_{n}.ckpt.

    Returns:
        Path to the latest checkpoint file.
    """
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda x: int(x.stem.split("_")[-1]))
    return ckpts[-1]


def writer_worker(
    q: Queue[Any],
    logits_arr: zarr.Array,
    residpost_arr: zarr.Array,
    error_bucket: list[BaseException],
) -> None:
    """Drain the queue and write batches to Zarr.

    Runs in a background thread. Any exception raised during a Zarr write is
    captured into ``error_bucket`` rather than silently discarded, so the main
    thread can detect and re-raise it after joining.

    Args:
        q: Queue of (start_idx, end_idx, logits_data, resid_data) tuples.
            A ``None`` item signals the thread to exit.
        logits_arr: Zarr array for logit outputs.
        residpost_arr: Zarr array for residual-stream outputs.
        error_bucket: List to which any unhandled exception is appended.
    """
    try:
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break
            start_idx, end_idx, logits_data, resid_data = item
            logits_arr[start_idx:end_idx] = logits_data
            residpost_arr[start_idx:end_idx] = resid_data
            q.task_done()
    except Exception as exc:
        error_bucket.append(exc)
        q.task_done()


def stream_logits_cache(
    model: HookedTransformer,
    seqs: np.ndarray,
    logits_arr: zarr.Array,
    residpost_arr: zarr.Array,
    batch_size: int = 4096,
) -> None:
    """Stream logits and residual activations from the model into Zarr arrays.

    Runs inference in batches and offloads Zarr writes to a background thread
    to overlap GPU computation with disk I/O. If the writer thread raises an
    exception (e.g. a shape mismatch on write), the batch loop terminates early
    and the exception is re-raised in the main thread after the thread is joined.

    Args:
        model: A TransformerLens HookedTransformer model in eval mode.
        seqs: Pre-tokenised integer sequences of shape (num_games, MAX_STEPS).
        logits_arr: Pre-allocated Zarr array for logit outputs.
        residpost_arr: Pre-allocated Zarr array for residual-stream outputs.
        batch_size: Number of sequences to process per forward pass.
    """
    device = next(model.parameters()).device
    device_type = device.type

    def cache_filter(name: str) -> bool:
        return "hook_resid_post" in name

    writer_errors: list[BaseException] = []
    write_queue: Queue[Any] = Queue(maxsize=2)
    writer_thread = threading.Thread(
        target=writer_worker,
        args=(write_queue, logits_arr, residpost_arr, writer_errors),
        daemon=True,
    )
    writer_thread.start()

    try:
        with torch.inference_mode():
            num_batches = (len(seqs) + batch_size - 1) // batch_size

            for batch_idx in tqdm(range(num_batches), desc="Processing Batches"):
                if writer_errors:
                    raise RuntimeError("Writer thread failed.") from writer_errors[0]

                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(seqs))

                batch_tensor = torch.tensor(
                    seqs[start_idx:end_idx],
                    dtype=torch.long,
                    device=device,
                )[:, :-1]  # Shape: [batch_size, 59]

                # autocast only runs on CUDA; CPU and MPS use a no-op context
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if device_type == "cuda"
                    else contextlib.nullcontext()
                )

                with autocast_ctx:
                    # logits: [batch, seq_len, vocab]
                    logits, cache = model.run_with_cache(
                        batch_tensor,
                        names_filter=cache_filter,
                    )

                logits_np = logits.float().cpu().numpy()
                stack_resid = cache.stack_activation("resid_post")
                stack_resid = stack_resid.permute(1, 2, 3, 0)  # [batch, seq_len, d_model, n_layers]
                resid_np = stack_resid.float().cpu().numpy()

                padded_logits = np.pad(
                    logits_np,
                    ((0, 0), (0, 1), (0, 0)),
                    mode="constant",
                    constant_values=np.nan,
                )
                padded_resids = np.pad(
                    resid_np,
                    ((0, 0), (0, 1), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=np.nan,
                )

                write_queue.put((start_idx, end_idx, padded_logits, padded_resids))
                del logits, cache, stack_resid, logits_np, resid_np

    finally:
        write_queue.put(None)
        writer_thread.join(timeout=60)
        if writer_thread.is_alive():
            logger.error("Writer thread did not exit within 60 s; Zarr output may be incomplete.")
        if writer_errors:
            raise RuntimeError("Writer thread failed.") from writer_errors[0]


def mod_dsarray(
    grp: zarr.Group,
    model_name: str,
    num_seqs: int,
    n_ctx: int,
    d_vocab: int,
    d_model: int,
    n_layers: int,
) -> tuple[zarr.Array, zarr.Array]:
    """Create or replace logits and resid_post arrays in the Zarr store.

    Args:
        grp: Open Zarr group backed by the game data store.
        model_name: Model run name used to prefix array names.
        num_seqs: Number of game sequences (first dimension of arrays).
        n_ctx: Context window size (number of positions per sequence).
        d_vocab: Vocabulary size.
        d_model: Model embedding dimension.
        n_layers: Number of transformer layers.

    Returns:
        Tuple of (logits_arr, residpost_arr) zarr arrays.
    """
    logits_arr = grp.require_array(
        f"{model_name}_logits",
        shape=(num_seqs, n_ctx + 1, d_vocab),
        dtype="float32",
        fill_value=np.nan,
        overwrite=True,  # keep it for now because we might want to overwrite with newer activations
    )
    logits_arr.attrs["_ARRAY_DIMENSIONS"] = ["game", "move", "vocab"]

    residpost_arr = grp.require_array(
        f"{model_name}_resid_post",
        shape=(num_seqs, n_ctx + 1, d_model, n_layers),
        dtype="float32",
        fill_value=np.nan,
        overwrite=True,
    )
    residpost_arr.attrs["_ARRAY_DIMENSIONS"] = ["game", "move", "d_model", "layer"]

    return logits_arr, residpost_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate model activations for Othello game sequences.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Model run name matching a directory under data/ (e.g. classic)",
    )
    parser.add_argument(
        "--game",
        type=str,
        required=True,
        choices=GAME_REGISTRY.keys(),
        help="Game variant (classic, nomidflip, delflank, iago)",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "board_train"],
        help="Dataset split to load (e.g. train)",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        required=True,
        help="Number of games in the target dataset, in millions (e.g. 20)",
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
    logger.info("Running with config: %s", args)

    data_path = REPO_ROOT / "data" / args.game / f"{args.split}_{args.game}_{args.num_games}M.zarr"
    ckpt_dir = REPO_ROOT / "data" / args.run_name / "ckpts"

    seqs = get_dataset(data_path)
    seq_ds = SequenceDataset(seqs, Tokenizer(), tokenize=False)

    last_ckpt = get_last_ckpt(ckpt_dir)
    model = load_model_from_ckpt(last_ckpt, seq_ds.vocab_size, seq_ds.block_size, as_tlens=True)
    model.eval()

    grp = zarr.open(str(data_path), mode="a")
    logits_arr, residpost_arr = mod_dsarray(
        grp,
        args.run_name,
        len(seqs),
        model.cfg.n_ctx,
        model.cfg.d_vocab,
        model.cfg.d_model,
        model.cfg.n_layers,
    )

    stream_logits_cache(model, seqs, logits_arr, residpost_arr)

    zarr.consolidate_metadata(str(data_path))
