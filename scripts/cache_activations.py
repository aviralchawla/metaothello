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

from metaothello.constants import MAX_STEPS
from metaothello.mingpt.utils import get_last_ckpt, load_model_from_ckpt

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Tokenizer.vocab_size = len([PAD] + 64 squares + [None]) = 66.
# block_size = MAX_STEPS - 1 = 59 (autoregressive context window).
_VOCAB_SIZE: int = 66
_BLOCK_SIZE: int = MAX_STEPS - 1


def _resid_post_filter(name: str) -> bool:
    """Return True for TransformerLens hook names that capture resid_post.

    Args:
        name: Hook point name from TransformerLens.

    Returns:
        True if the hook captures residual stream post-MLP activations.
    """
    return "hook_resid_post" in name


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
            try:
                logits_arr[start_idx:end_idx] = logits_data
                residpost_arr[start_idx:end_idx] = resid_data
            except Exception as exc:
                raise RuntimeError(
                    f"Zarr write failed for batch slice [{start_idx}:{end_idx}]."
                ) from exc
            q.task_done()
    except Exception as exc:
        error_bucket.append(exc)
        q.task_done()


def stream_logits_cache(
    model: HookedTransformer,
    seqs: xr.DataArray,
    num_seqs: int,
    logits_arr: zarr.Array,
    residpost_arr: zarr.Array,
    batch_size: int = 4096,
) -> None:
    """Stream logits and residual activations from the model into Zarr arrays.

    Runs inference in batches and offloads Zarr writes to a background thread
    to overlap GPU computation with disk I/O. Sequences are loaded lazily from
    the xarray DataArray one batch at a time to avoid holding the full dataset
    in memory.

    If the writer thread raises an exception (e.g. a shape mismatch on write),
    the batch loop terminates early and the exception is re-raised in the main
    thread after the thread is joined.

    Args:
        model: A TransformerLens HookedTransformer model in eval mode.
        seqs: Lazy xarray DataArray backed by a Zarr store. Shape
            (num_games, MAX_STEPS), dtype int32. Slices are materialised
            per-batch via ``.values`` to keep memory usage bounded.
        num_seqs: Total number of sequences in ``seqs``.
        logits_arr: Pre-allocated Zarr array for logit outputs.
        residpost_arr: Pre-allocated Zarr array for residual-stream outputs.
        batch_size: Number of sequences to process per forward pass.
    """
    device = next(model.parameters()).device
    device_type = device.type

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
            num_batches = (num_seqs + batch_size - 1) // batch_size

            for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
                if writer_errors:
                    raise RuntimeError("Writer thread failed.") from writer_errors[0]

                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_seqs)

                # Materialise one batch from the lazy xarray DataArray.
                # Sequential chunk-aligned access is efficient on Zarr.
                batch_np = seqs[start_idx:end_idx].values

                batch_tensor = torch.tensor(
                    batch_np,
                    dtype=torch.long,
                    device=device,
                )[:, :-1]  # Shape: (batch, MAX_STEPS-1) = (batch, 59)

                # autocast only runs on CUDA; CPU and MPS use a no-op context
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if device_type == "cuda"
                    else contextlib.nullcontext()
                )

                with autocast_ctx:
                    logits, cache = model.run_with_cache(
                        batch_tensor,
                        names_filter=_resid_post_filter,
                    )

                logits_np = logits.float().cpu().numpy()
                stack_resid = cache.stack_activation("resid_post")
                stack_resid = stack_resid.permute(1, 2, 3, 0)
                resid_np = stack_resid.float().cpu().numpy()

                # The model processes positions 0..MAX_STEPS-2 (59 tokens).
                # Output arrays span all MAX_STEPS positions (60 slots) to
                # align with the full game sequence. The final position
                # (index 59) receives NaN because no model output exists for
                # it — it is only ever an input token, never predicted.
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

    finally:
        write_queue.put(None)
        writer_thread.join(timeout=60)
        if writer_thread.is_alive():
            logger.error("Writer thread did not exit within 60 s; Zarr output may be incomplete.")
        if writer_errors:
            raise RuntimeError("Writer thread failed.") from writer_errors[0]


def create_activation_arrays(
    grp: zarr.Group,
    model_name: str,
    epoch: int,
    num_seqs: int,
    d_vocab: int,
    d_model: int,
    n_layers: int,
    overwrite: bool,
    batch_size: int,
) -> tuple[zarr.Array, zarr.Array]:
    """Create logits and resid_post arrays in the Zarr store.

    Array names follow the pattern ``{model_name}_epoch{epoch}_logits`` and
    ``{model_name}_epoch{epoch}_resid_post``, so activations from different
    checkpoints within the same run coexist without collision.

    Arrays are chunked with ``batch_size`` games per chunk along the first
    dimension so that each writer-thread write corresponds to exactly one
    Zarr chunk, avoiding partial-chunk I/O overhead.

    Args:
        grp: Open Zarr group backed by the game data store.
        model_name: Model run name used to prefix array names.
        epoch: Checkpoint epoch number, embedded in the array name.
        num_seqs: Number of game sequences (first dimension of arrays).
        d_vocab: Vocabulary size.
        d_model: Model embedding dimension.
        n_layers: Number of transformer layers.
        overwrite: If True, overwrite existing arrays. If False, raise an
            error if an array with the same name already exists.
        batch_size: Number of sequences per batch; determines chunk size
            along the first dimension.

    Returns:
        Tuple of (logits_arr, residpost_arr) zarr arrays.
    """
    prefix = f"{model_name}_epoch{epoch}"
    logits_name = f"{prefix}_logits"
    resid_name = f"{prefix}_resid_post"

    if not overwrite:
        if logits_name in grp:
            raise FileExistsError(
                f"Array '{logits_name}' already exists. Pass --force to overwrite."
            )
        if resid_name in grp:
            raise FileExistsError(
                f"Array '{resid_name}' already exists. Pass --force to overwrite."
            )

    logits_arr = grp.require_array(
        logits_name,
        shape=(num_seqs, MAX_STEPS, d_vocab),
        dtype="float32",
        fill_value=np.nan,
        overwrite=overwrite,
        chunks=(batch_size, MAX_STEPS, d_vocab),
    )
    logits_arr.attrs["_ARRAY_DIMENSIONS"] = ["game", "move", "vocab"]

    residpost_arr = grp.require_array(
        resid_name,
        shape=(num_seqs, MAX_STEPS, d_model, n_layers),
        dtype="float32",
        fill_value=np.nan,
        overwrite=overwrite,
        chunks=(batch_size, MAX_STEPS, d_model, n_layers),
    )
    residpost_arr.attrs["_ARRAY_DIMENSIONS"] = ["game", "move", "d_model", "layer"]

    return logits_arr, residpost_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache model activations for Othello game sequences into a Zarr store.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to the existing .zarr store (produced by generate_data.py).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help=(
            "Model run name. Used to locate checkpoints under "
            "data/{run_name}/ckpts/ and to prefix activation array names "
            "in the Zarr store (e.g. 'classic' at epoch 250 produces "
            "'classic_epoch250_logits' and 'classic_epoch250_resid_post')."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Number of sequences per forward pass (default: 4096).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing activation arrays. Without this flag, the "
            "script exits with an error if activation arrays for --run_name "
            "already exist in the Zarr store."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.info("Running with config: %s", args)

    ckpt_dir = REPO_ROOT / "data" / args.run_name / "ckpts"
    last_ckpt, last_epoch = get_last_ckpt(ckpt_dir)
    if last_ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint found in {ckpt_dir}. " "Train a model first with scripts/gpt_train.py."
        )

    model = load_model_from_ckpt(last_ckpt, _VOCAB_SIZE, _BLOCK_SIZE, as_tlens=True)
    model.eval()
    device = next(model.parameters()).device
    logger.info("Model loaded from %s on device: %s", last_ckpt, device)

    ds = xr.open_zarr(args.data_path)
    seqs_da = ds["seqs"]
    num_seqs = seqs_da.sizes["game"]
    logger.info("Dataset: %d sequences from %s", num_seqs, args.data_path)

    grp = zarr.open(str(args.data_path), mode="a")
    logits_arr, residpost_arr = create_activation_arrays(
        grp,
        args.run_name,
        last_epoch,
        num_seqs,
        model.cfg.d_vocab,
        model.cfg.d_model,
        model.cfg.n_layers,
        overwrite=args.force,
        batch_size=args.batch_size,
    )

    stream_logits_cache(
        model,
        seqs_da,
        num_seqs,
        logits_arr,
        residpost_arr,
        batch_size=args.batch_size,
    )

    zarr.consolidate_metadata(str(args.data_path))
    logger.info("Done. Metadata consolidated at %s.", args.data_path)
