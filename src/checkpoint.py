"""Atomic model checkpoints for replicated and self-sharded embeddings.

Two layers live here. ``save_model_checkpoint`` / ``load_model_checkpoint``
persist model weights (and, for sharded tables, their optimizer accumulators) in
a reshardable layout. On top of that, ``stage_training_checkpoint`` /
``load_training_checkpoint`` add everything a crashed run needs to continue:
the global step, the dense and replicated sparse optimizer state, and each
rank's input-scan cursor. :class:`CheckpointUploader` publishes staged steps to
a run directory (local or HDFS) with a commit marker, so an interrupted upload
can never be mistaken for a resumable step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
import errno
from hashlib import sha256
import json
import logging
import math
import os
from pathlib import Path
import queue
import shutil
import struct
import threading
import time
from typing import Any

import torch
import torch.distributed as torch_dist
from torch import Tensor, nn

from .checkpoint_store import CheckpointStore, download_tree, open_checkpoint_store
from .config import AppConfig
from .embeddings import EmbeddingShardSpec, ShardedEmbedding, sharded_embedding_modules
from .features import vocab_strategy_fingerprint
from .optim import ShardedAdagrad, ShardedRowWiseAdagrad

logger = logging.getLogger(__name__)


def _padding_idx_value(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)

SHARDED_CHECKPOINT_FORMAT = "mdl_sharded_embedding_v1"
# v2 splits one rank's tables across several bounded-size files so neither the
# CPU copy nor the staging directory ever holds a whole shard at once.
SHARDED_CHECKPOINT_CHUNK_FORMAT = "mdl_sharded_embedding_v2"
SHARDED_SPARSE_STREAM_FORMAT = "mdl_sharded_sparse_stream_v1"
_SHARDED_CHECKPOINT_READABLE = frozenset(
    {
        SHARDED_CHECKPOINT_FORMAT,
        SHARDED_CHECKPOINT_CHUNK_FORMAT,
        SHARDED_SPARSE_STREAM_FORMAT,
    }
)
TRAINING_CHECKPOINT_FORMAT = "mdl_training_checkpoint_v1"

# Layout of one committed step directory inside a run directory.
STEP_DIR_PREFIX = "step-"
_STEP_DIR_DIGITS = 9
MODEL_SUBDIR = "model"
MODEL_FILE = "model.pt"
TRAIN_STATE_FILE = "train_state.pt"
CHECKPOINT_MANIFEST = "checkpoint.json"
COMMIT_MARKER = "_COMMIT"
LATEST_POINTER = "_latest.json"

# Tables are packed into files up to this many bytes. A single table larger than
# the budget still gets its own file: splitting one table across files would put
# a row-range dimension into the reshard path for very little extra headroom.
DEFAULT_SHARD_CHUNK_BYTES = 2 * 1024 * 1024 * 1024
_SPARSE_SEGMENT_MAGIC = b"MDLSPV1\n"
_SPARSE_COPY_BLOCK_BYTES = 64 * 1024 * 1024


def _checkpoint_metadata(config: AppConfig) -> dict[str, Any]:
    return {
        "model_name": config.model.name,
        "task_names": config.task_names,
        "vocab_strategy_hash": vocab_strategy_fingerprint(config),
    }


def _validate_checkpoint_metadata(
    config: AppConfig,
    payload: dict[str, Any],
) -> None:
    if payload.get("model_name") not in {None, config.model.name}:
        raise ValueError("checkpoint model_name does not match current config")
    task_names = payload.get("task_names")
    if task_names is not None and list(task_names) != config.task_names:
        raise ValueError("checkpoint task_names do not match current config")
    if payload.get("vocab_strategy_hash") != vocab_strategy_fingerprint(config):
        raise ValueError("checkpoint vocab_strategy_hash does not match current config")


class CheckpointStagingSpaceError(RuntimeError):
    """The local staging directory could not hold this rank's checkpoint.

    Raised instead of the zip writer's ``unexpected pos`` / ``file write
    failed`` pair, which names neither the directory that filled up nor how much
    room the save actually needed.
    """


def _free_bytes(path: Path) -> int | None:
    """Bytes writable at ``path`` (or its nearest existing parent)."""

    for candidate in (path, *path.parents):
        try:
            stat = os.statvfs(candidate)
        except OSError:
            continue
        return int(stat.f_bavail) * int(stat.f_frsize)
    return None


def _format_bytes(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / (1024 ** 3):.1f}GiB"


def _is_out_of_space(error: BaseException) -> bool:
    """Whether a failed write looks like the filesystem running out of room.

    ``torch.save`` reports a short write through the zip writer, which discards
    the underlying errno, so the message has to be matched as well.
    """

    if isinstance(error, OSError) and error.errno in {errno.ENOSPC, errno.EDQUOT}:
        return True
    text = str(error).lower()
    return "no space left" in text or "file write failed" in text


def _discard_partial_file(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        logger.debug("could not remove partial checkpoint file %s", path)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
    except BaseException as error:
        # A half-written archive is pure overhead: nothing can read it, and on
        # the filesystem that just filled up it keeps the next attempt from
        # succeeding.
        _discard_partial_file(temporary)
        if _is_out_of_space(error):
            raise CheckpointStagingSpaceError(
                f"ran out of space writing {path.name} under {path.parent}: "
                f"{_format_bytes(_free_bytes(path.parent))} free. Point "
                "training.checkpoint.staging_dir at a filesystem with room for "
                "every local rank's shard."
            ) from error
        raise
    os.replace(temporary, path)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except BaseException as error:
        _discard_partial_file(temporary)
        if _is_out_of_space(error):
            raise CheckpointStagingSpaceError(
                f"ran out of space writing {path.name} under {path.parent}: "
                f"{_format_bytes(_free_bytes(path.parent))} free"
            ) from error
        raise
    os.replace(temporary, path)


def _state_to_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_state_to_cpu(item) for item in value)
    return value


def _sharded_state_keys(model: nn.Module) -> set[str]:
    keys: set[str] = set()
    for name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, ShardedEmbedding):
            keys.add(f"{name}.weight" if name else "weight")
    return keys


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _sharded_optimizer_row_bytes(
    module: ShardedEmbedding,
    sharded_optimizer: "ShardedAdagrad | ShardedRowWiseAdagrad | None",
) -> int:
    """Accumulator bytes per embedding row, ``0`` when nothing is saved.

    Row-Wise Adagrad keeps one FP32 scalar per row; plain Adagrad keeps a full
    row. Both are derived from tensors this rank already owns, so the number is
    the same on every rank.
    """

    if sharded_optimizer is None:
        return 0
    state = sharded_optimizer.state.get(module.weight)
    if not state:
        return 0
    rows = max(1, int(module.weight.size(0)))
    return max(0, _tensor_bytes(state) // rows)


def _table_plan_bytes(
    module: ShardedEmbedding,
    *,
    world_size: int,
    row_bytes: int,
) -> int:
    """Bytes one rank writes for a table, computed from global table shape.

    Chunk boundaries must be identical on every rank, so the plan is derived
    from ``num_embeddings`` (global, replicated) rather than the local row count
    (which differs by a row or two depending on the owner mapping).
    """

    per_row = module.embedding_dim * module.weight.element_size() + row_bytes
    rows = math.ceil(int(module.num_embeddings) / max(1, int(world_size)))
    return rows * per_row


def plan_shard_chunks(
    modules: Sequence[ShardedEmbedding],
    *,
    world_size: int,
    chunk_bytes: int = DEFAULT_SHARD_CHUNK_BYTES,
    sharded_optimizer: "ShardedAdagrad | ShardedRowWiseAdagrad | None" = None,
) -> list[list[ShardedEmbedding]]:
    """Pack tables into files of at most ``chunk_bytes``, in a stable order."""

    budget = max(1, int(chunk_bytes))
    chunks: list[list[ShardedEmbedding]] = []
    current: list[ShardedEmbedding] = []
    current_bytes = 0
    for module in sorted(modules, key=lambda item: item.table_name):
        size = _table_plan_bytes(
            module,
            world_size=world_size,
            row_bytes=_sharded_optimizer_row_bytes(module, sharded_optimizer),
        )
        if current and current_bytes + size > budget:
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(module)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks


@dataclass(frozen=True)
class StagingSpaceEstimate:
    """How much local scratch one rank needs to stage a checkpoint."""

    total_bytes: int
    peak_bytes: int
    chunk_count: int
    largest_chunk_bytes: int

    def describe(self) -> str:
        return (
            f"total={_format_bytes(self.total_bytes)} "
            f"peak={_format_bytes(self.peak_bytes)} "
            f"chunks={self.chunk_count}"
        )


def estimate_staging_space(
    model: nn.Module,
    *,
    rank: int = 0,
    world_size: int = 1,
    dense_optimizer: torch.optim.Optimizer | None = None,
    replicated_sparse_optimizer: torch.optim.Optimizer | None = None,
    sharded_optimizer: "ShardedAdagrad | ShardedRowWiseAdagrad | None" = None,
    chunk_bytes: int = DEFAULT_SHARD_CHUNK_BYTES,
    upload_window: int = 1,
    split_large_tables: bool = False,
) -> StagingSpaceEstimate:
    """Predict this rank's staging footprint before anything is written.

    ``peak_bytes`` assumes published files are deleted as the upload drains, so
    it is what a healthy save actually needs. ``total_bytes`` is the fallback
    when uploads cannot keep up and every file stays until the step finishes.
    """

    modules = sharded_embedding_modules(model)
    chunk_sizes: list[int] = []
    if modules:
        if split_large_tables:
            budget = max(1, int(chunk_bytes))
            for module in modules:
                size = _table_plan_bytes(
                    module,
                    world_size=world_size,
                    row_bytes=_sharded_optimizer_row_bytes(
                        module, sharded_optimizer
                    ),
                )
                while size > 0:
                    piece = min(size, budget)
                    chunk_sizes.append(piece)
                    size -= piece
        else:
            for chunk in plan_shard_chunks(
                modules,
                world_size=world_size,
                chunk_bytes=chunk_bytes,
                sharded_optimizer=sharded_optimizer,
            ):
                chunk_sizes.append(
                    sum(
                        _table_plan_bytes(
                            module,
                            world_size=world_size,
                            row_bytes=_sharded_optimizer_row_bytes(
                                module, sharded_optimizer
                            ),
                        )
                        for module in chunk
                    )
                )
        sharded_keys = _sharded_state_keys(model)
        replicated = sum(
            _tensor_bytes(value)
            for key, value in model.state_dict().items()
            if key not in sharded_keys
        )
    else:
        replicated = _tensor_bytes(model.state_dict())
    extras = 0
    if rank == 0:
        extras += replicated
        for optimizer in (dense_optimizer, replicated_sparse_optimizer):
            if optimizer is not None:
                extras += _tensor_bytes(optimizer.state)
    total = sum(chunk_sizes) + extras
    largest = max(chunk_sizes, default=0)
    # One chunk is being written while ``upload_window`` earlier ones may still
    # be waiting for the store to accept them.
    peak = min(total, largest * (max(0, int(upload_window)) + 1) + extras)
    return StagingSpaceEstimate(
        total_bytes=total,
        peak_bytes=peak,
        chunk_count=len(chunk_sizes),
        largest_chunk_bytes=largest,
    )


def check_staging_space(
    staging_dir: str | Path,
    estimate: StagingSpaceEstimate,
    *,
    local_ranks: int = 1,
    headroom: float = 1.1,
    enforce: bool = True,
) -> str:
    """Raise when ``staging_dir`` cannot hold the concurrent local ranks.

    Every rank on a node stages into the same filesystem, so the requirement is
    the per-rank figure times the local rank count, not one checkpoint.
    """

    path = Path(staging_dir)
    free = _free_bytes(path)
    ranks = max(1, int(local_ranks))
    required = int(estimate.peak_bytes * ranks * max(1.0, headroom))
    comfortable = int(estimate.total_bytes * ranks * max(1.0, headroom))
    summary = (
        f"staging_dir={path} free={_format_bytes(free)} "
        f"needed={_format_bytes(required)} "
        f"(local_ranks={ranks} {estimate.describe()})"
    )
    if free is None:
        return summary
    if free < required:
        message = (
            "training.checkpoint staging has too little room: "
            f"{summary}. Set training.checkpoint.staging_dir to a filesystem "
            "with more space (the system temp directory is often a small "
            "RAM-backed tmpfs)."
        )
        if enforce:
            raise CheckpointStagingSpaceError(message)
        logger.error("%s", message)
        return summary
    if free < comfortable:
        logger.warning(
            "checkpoint staging has room for the streamed peak but not for a "
            "whole step (%s); a slow run directory will make saves block. %s",
            _format_bytes(comfortable),
            summary,
        )
    return summary


def shard_chunk_file(index: int, rank: int, world_size: int) -> str:
    """Name of one packed group of tables belonging to one rank."""

    return f"shard-{int(index):04d}-rank-{int(rank):05d}-of-{int(world_size):05d}.pt"


def legacy_rank_file(rank: int, world_size: int) -> str:
    """Name of the single-file-per-rank layout written before chunking."""

    return f"rank-{int(rank):05d}-of-{int(world_size):05d}.pt"


def shard_file_names(
    manifest: dict[str, Any],
    *,
    rank: int,
    world_size: int,
) -> list[str]:
    """Embedding shard files one rank must read, newest and legacy layouts.

    A restart at the saved world size only touches its own files. Resharding
    reads every saved owner, because the rows this rank now owns were spread
    across all of them.
    """

    saved_world_size = int(manifest["world_size"])
    resharding = saved_world_size != world_size

    def _pick(files: Sequence[str]) -> list[str]:
        if resharding:
            return list(files)
        if not 0 <= rank < len(files):
            raise ValueError(
                f"checkpoint has {len(files)} shard files per group but this run "
                f"is rank {rank} of {world_size}"
            )
        return [files[rank]]

    chunks = manifest.get("chunks")
    if chunks:
        names: list[str] = []
        for chunk in chunks:
            names.extend(_pick(chunk["rank_files"]))
        return names
    return _pick(manifest["rank_files"])


def _table_payload(
    module: ShardedEmbedding,
    sharded_optimizer: ShardedAdagrad | ShardedRowWiseAdagrad | None,
) -> dict[str, Any]:
    optimizer_state = None
    if sharded_optimizer is not None:
        state = sharded_optimizer.state.get(module.weight)
        if state:
            optimizer_state = _state_to_cpu(state)
    return {
        "weight": module.weight.detach().cpu(),
        "num_embeddings": module.num_embeddings,
        "embedding_dim": module.embedding_dim,
        "padding_idx": module.padding_idx,
        "shard_spec": asdict(module.shard_spec),
        "optimizer_state": optimizer_state,
    }


_DTYPE_TO_NAME = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


class _HashingWriter:
    """Small file wrapper that computes the segment checksum while writing."""

    def __init__(self, raw: Any) -> None:
        self.raw = raw
        self.digest = sha256()

    def write(self, payload: bytes | bytearray | memoryview) -> int:
        self.digest.update(payload)
        return int(self.raw.write(payload))


def _write_raw_tensor(stream: _HashingWriter, value: Tensor) -> None:
    """Write a contiguous CPU tensor without dtype conversion or pickle."""

    cpu = value.detach().contiguous().cpu()
    # NumPy does not expose BF16 on every supported version.  A uint8 view
    # preserves the exact bytes and lets ``memoryview`` avoid a second bytes()
    # allocation for a multi-hundred-MiB block.
    raw = cpu.view(torch.uint8).numpy()
    stream.write(memoryview(raw))


def _global_ids_for_local_rows(
    module: ShardedEmbedding,
    local_rows: Tensor,
    *,
    rank: int,
) -> Tensor:
    rows = local_rows.to(device="cpu", dtype=torch.int64)
    spec = module.shard_spec
    if spec.strategy == "table_wise":
        if spec.table_owner != rank and rows.numel():
            raise ValueError(
                f"rank {rank} cannot save rows owned by table-wise table "
                f"{module.table_name!r} on rank {spec.table_owner}"
            )
        return rows
    residue = (rank - spec.cyclic_offset) % spec.world_size
    result = rows * spec.world_size + residue
    if result.numel() and int(result[-1]) >= int(module.num_embeddings):
        raise ValueError(
            f"local row mapping exceeds table {module.table_name!r} bounds"
        )
    return result


def _iter_full_local_rows(module: ShardedEmbedding, max_rows: int) -> Any:
    count = int(module.weight.shape[0])
    limit = max(1, int(max_rows))
    for start in range(0, count, limit):
        yield torch.arange(start, min(count, start + limit), dtype=torch.int64)


def _iter_sparse_rows(
    module: ShardedEmbedding,
    optimizer: ShardedAdagrad | ShardedRowWiseAdagrad | None,
    *,
    kind: str,
    max_rows: int,
) -> Any:
    if kind == "full":
        yield from _iter_full_local_rows(module, max_rows)
        return
    if optimizer is None or not bool(getattr(optimizer, "tracks_dirty_rows", False)):
        raise RuntimeError(
            "dirty-row sparse checkpoint requested without an optimizer dirty-row "
            "tracker"
        )
    yield from optimizer.iter_dirty_rows(module.weight, max_rows=max_rows)


def _write_selected_tensor_rows(
    stream: _HashingWriter,
    tensor: Tensor,
    local_rows: Tensor,
) -> None:
    row_bytes = max(1, int(tensor[0].numel() * tensor.element_size())) if tensor.shape[0] else 1
    block_rows = max(1, _SPARSE_COPY_BLOCK_BYTES // row_bytes)
    for start in range(0, int(local_rows.numel()), block_rows):
        count = min(block_rows, int(local_rows.numel()) - start)
        indices = local_rows.narrow(0, start, count).to(
            device=tensor.device,
            dtype=torch.long,
        )
        selected = tensor.detach().index_select(0, indices)
        _write_raw_tensor(stream, selected)
        del indices, selected


def _write_sparse_segment(
    path: Path,
    *,
    module: ShardedEmbedding,
    optimizer_state: dict[str, Any] | None,
    local_rows: Tensor,
    global_rows: Tensor,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    weight = module.weight
    accumulator = None if optimizer_state is None else optimizer_state.get("sum")
    if not isinstance(accumulator, Tensor):
        raise ValueError(
            f"sparse checkpoint requires optimizer accumulator for "
            f"{module.table_name!r}"
        )
    weight_dtype = _DTYPE_TO_NAME.get(weight.dtype)
    accumulator_dtype = _DTYPE_TO_NAME.get(accumulator.dtype)
    if weight_dtype is None or accumulator_dtype is None:
        raise TypeError(
            f"unsupported sparse checkpoint dtype for {module.table_name!r}: "
            f"weight={weight.dtype}, accumulator={accumulator.dtype}"
        )
    count = int(local_rows.numel())
    weight_shape = [count, *[int(item) for item in weight.shape[1:]]]
    accumulator_shape = [count, *[int(item) for item in accumulator.shape[1:]]]
    ids_bytes = count * 8
    weight_bytes = math.prod(weight_shape) * weight.element_size()
    accumulator_bytes = math.prod(accumulator_shape) * accumulator.element_size()
    header = {
        "format": SHARDED_SPARSE_STREAM_FORMAT,
        "table_name": module.table_name,
        "rank": int(rank),
        "world_size": int(world_size),
        "row_count": count,
        "ids_dtype": "int64",
        "weight_dtype": weight_dtype,
        "weight_shape": weight_shape,
        "accumulator_dtype": accumulator_dtype,
        "accumulator_shape": accumulator_shape,
        "ids_bytes": ids_bytes,
        "weight_bytes": weight_bytes,
        "accumulator_bytes": accumulator_bytes,
    }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(temporary, "wb") as raw:
            stream = _HashingWriter(raw)
            stream.write(_SPARSE_SEGMENT_MAGIC)
            stream.write(struct.pack("<Q", len(encoded)))
            stream.write(encoded)
            _write_raw_tensor(stream, global_rows.to(dtype=torch.int64))
            _write_selected_tensor_rows(stream, weight, local_rows)
            _write_selected_tensor_rows(stream, accumulator, local_rows)
            checksum = stream.digest.hexdigest()
        os.replace(temporary, path)
    except BaseException:
        _discard_partial_file(temporary)
        raise
    return {
        "file": path.name,
        "row_count": count,
        "first_global_id": None if count == 0 else int(global_rows[0]),
        "last_global_id": None if count == 0 else int(global_rows[-1]),
        "size_bytes": int(path.stat().st_size),
        "sha256": checksum,
    }


def _save_streaming_sharded_checkpoint(
    config: AppConfig,
    model: nn.Module,
    checkpoint_path: Path,
    *,
    rank: int,
    world_size: int,
    process_group: torch_dist.ProcessGroup | None,
    sharded_optimizer: ShardedAdagrad | ShardedRowWiseAdagrad | None,
    chunk_bytes: int,
    sparse_kind: str,
    sparse_parent_checkpoint: str | None,
    generation: int,
    publish: Callable[[str], None] | None,
) -> list[str]:
    """Write rank-owned sparse rows in bounded raw segments plus rank-0 dense."""

    if sparse_kind not in {"full", "delta"}:
        raise ValueError("sparse_kind must be full or delta")
    modules = sorted(sharded_embedding_modules(model), key=lambda item: item.table_name)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def record(name: str) -> None:
        written.append(name)
        if publish is not None:
            publish(name)

    rank_tables: dict[str, Any] = {}
    for table_index, module in enumerate(modules):
        optimizer_state = _sharded_optimizer_state(sharded_optimizer, module.weight)
        accumulator = None if optimizer_state is None else optimizer_state.get("sum")
        if not isinstance(accumulator, Tensor):
            raise ValueError(
                f"sparse checkpoint has no optimizer accumulator for "
                f"{module.table_name!r}"
            )
        bytes_per_row = (
            8
            + math.prod(module.weight.shape[1:]) * module.weight.element_size()
            + math.prod(accumulator.shape[1:]) * accumulator.element_size()
        )
        rows_per_segment = max(1, int(chunk_bytes) // max(1, bytes_per_row))
        segments: list[dict[str, Any]] = []
        row_count = 0
        for segment_index, local_rows in enumerate(
            _iter_sparse_rows(
                module,
                sharded_optimizer,
                kind=sparse_kind,
                max_rows=rows_per_segment,
            )
        ):
            local_rows = local_rows.to(device="cpu", dtype=torch.int64).contiguous()
            if local_rows.numel() == 0:
                continue
            global_rows = _global_ids_for_local_rows(module, local_rows, rank=rank)
            name = (
                f"sparse-t{table_index:04d}-s{segment_index:05d}-"
                f"rank-{rank:05d}-of-{world_size:05d}.bin"
            )
            segment = _write_sparse_segment(
                checkpoint_path / name,
                module=module,
                optimizer_state=optimizer_state,
                local_rows=local_rows,
                global_rows=global_rows,
                rank=rank,
                world_size=world_size,
            )
            segments.append(segment)
            row_count += int(local_rows.numel())
            record(name)
        step_value = None if optimizer_state is None else optimizer_state.get("step")
        rank_tables[module.table_name] = {
            "num_embeddings": int(module.num_embeddings),
            "embedding_dim": int(module.embedding_dim),
            "padding_idx": _padding_idx_value(module.padding_idx),
            "shard_spec": asdict(module.shard_spec),
            "optimizer_step": (
                None if not isinstance(step_value, Tensor) else float(step_value)
            ),
            "row_count": row_count,
            "segments": segments,
        }

    rank_manifest_name = f"sparse-manifest-rank-{rank:05d}.json"
    _atomic_json_save(
        {
            "format": SHARDED_SPARSE_STREAM_FORMAT,
            "version": 1,
            "kind": sparse_kind,
            "rank": int(rank),
            "world_size": int(world_size),
            "generation": int(generation),
            "sparse_parent_checkpoint": sparse_parent_checkpoint,
            "tables": rank_tables,
        },
        checkpoint_path / rank_manifest_name,
    )
    record(rank_manifest_name)

    dense_file = "dense.pt"
    if rank == 0:
        sharded_keys = _sharded_state_keys(model)
        dense_state = {
            key: _state_to_cpu(value)
            for key, value in model.state_dict().items()
            if key not in sharded_keys
        }
        _atomic_torch_save(
            {"model_state_dict": dense_state, **_checkpoint_metadata(config)},
            checkpoint_path / dense_file,
        )
        del dense_state
        record(dense_file)
    if world_size > 1:
        torch_dist.barrier(group=process_group)
    if rank == 0:
        _atomic_json_save(
            {
                "format": SHARDED_SPARSE_STREAM_FORMAT,
                "version": 1,
                "kind": sparse_kind,
                "generation": int(generation),
                "sparse_parent_checkpoint": sparse_parent_checkpoint,
                "world_size": int(world_size),
                "dense_file": dense_file,
                "rank_manifests": [
                    f"sparse-manifest-rank-{item:05d}.json"
                    for item in range(world_size)
                ],
                "tables": {
                    module.table_name: {
                        "num_embeddings": int(module.num_embeddings),
                        "embedding_dim": int(module.embedding_dim),
                        "padding_idx": _padding_idx_value(module.padding_idx),
                    }
                    for module in modules
                },
                "training_metadata": {
                    "sparse_optimizer": config.training.sparse_optimizer,
                },
                **_checkpoint_metadata(config),
            },
            checkpoint_path / "manifest.json",
        )
        record("manifest.json")
    if world_size > 1:
        torch_dist.barrier(group=process_group)
    return written


def save_model_checkpoint(
    config: AppConfig,
    model: nn.Module,
    path: str | Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    process_group: torch_dist.ProcessGroup | None = None,
    sharded_optimizer: ShardedAdagrad | ShardedRowWiseAdagrad | None = None,
    chunk_bytes: int = DEFAULT_SHARD_CHUNK_BYTES,
    sparse_stream: bool = False,
    sparse_kind: str = "full",
    sparse_parent_checkpoint: str | None = None,
    generation: int = 0,
    publish: Callable[[str], None] | None = None,
) -> list[str]:
    """Save one replicated file or a manifest plus this rank's shard files.

    Returns the files written, relative to ``path``. ``publish`` is called with
    each name as soon as it lands, so a caller that copies files to a run
    directory can drain (and delete) them while the next chunk is still being
    built — that is what keeps staging bounded by one chunk instead of a whole
    shard.
    """

    checkpoint_path = Path(path)
    sharded_modules = sharded_embedding_modules(model)
    written: list[str] = []

    def _record(name: str) -> None:
        written.append(name)
        if publish is not None:
            publish(name)

    if not sharded_modules:
        if rank == 0:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_torch_save(
                {
                    "model_state_dict": _state_to_cpu(model.state_dict()),
                    **_checkpoint_metadata(config),
                },
                checkpoint_path,
            )
            # A replicated model is one file next to the checkpoint, so names are
            # relative to its parent; sharded names are relative to the directory.
            _record(checkpoint_path.name)
        return written

    if sparse_stream:
        return _save_streaming_sharded_checkpoint(
            config,
            model,
            checkpoint_path,
            rank=rank,
            world_size=world_size,
            process_group=process_group,
            sharded_optimizer=sharded_optimizer,
            chunk_bytes=chunk_bytes,
            sparse_kind=sparse_kind,
            sparse_parent_checkpoint=sparse_parent_checkpoint,
            generation=generation,
            publish=publish,
        )

    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid rank/world_size for sharded checkpoint")
    if any(module.world_size != world_size for module in sharded_modules):
        raise RuntimeError("model sharding plan does not match checkpoint world size")
    if checkpoint_path.exists() and not checkpoint_path.is_dir():
        raise ValueError(
            "a sharded checkpoint path must be a directory, but a file already exists: "
            f"{checkpoint_path}"
        )
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    chunks = plan_shard_chunks(
        sharded_modules,
        world_size=world_size,
        chunk_bytes=chunk_bytes,
        sharded_optimizer=sharded_optimizer,
    )
    for index, chunk in enumerate(chunks):
        # Build, write, and drop one group at a time. Materializing every table
        # first would put a second copy of the whole shard in host memory before
        # a single byte reached the disk.
        payload = {
            "format": SHARDED_CHECKPOINT_CHUNK_FORMAT,
            "rank": rank,
            "world_size": world_size,
            "chunk": index,
            "tables": {
                module.table_name: _table_payload(module, sharded_optimizer)
                for module in chunk
            },
        }
        chunk_file = shard_chunk_file(index, rank, world_size)
        try:
            _atomic_torch_save(payload, checkpoint_path / chunk_file)
        finally:
            del payload
        _record(chunk_file)

    dense_file = "dense.pt"
    if rank == 0:
        sharded_keys = _sharded_state_keys(model)
        dense_state = {
            key: _state_to_cpu(value)
            for key, value in model.state_dict().items()
            if key not in sharded_keys
        }
        _atomic_torch_save(
            {"model_state_dict": dense_state, **_checkpoint_metadata(config)},
            checkpoint_path / dense_file,
        )
        del dense_state
        _record(dense_file)
    if world_size > 1:
        torch_dist.barrier(group=process_group)
    if rank == 0:
        table_metadata = {
            module.table_name: {
                "num_embeddings": module.num_embeddings,
                "embedding_dim": module.embedding_dim,
                "padding_idx": module.padding_idx,
            }
            for module in sharded_modules
        }
        _atomic_json_save(
            {
                "format": SHARDED_CHECKPOINT_CHUNK_FORMAT,
                "version": 2,
                "world_size": world_size,
                "dense_file": dense_file,
                "chunks": [
                    {
                        "index": index,
                        "tables": [module.table_name for module in chunk],
                        "rank_files": [
                            shard_chunk_file(index, item, world_size)
                            for item in range(world_size)
                        ],
                    }
                    for index, chunk in enumerate(chunks)
                ],
                "tables": table_metadata,
                "training_metadata": {
                    "sparse_optimizer": config.training.sparse_optimizer,
                },
                **_checkpoint_metadata(config),
            },
            checkpoint_path / "manifest.json",
        )
        _record("manifest.json")
    if world_size > 1:
        torch_dist.barrier(group=process_group)
    return written


def _sharded_optimizer_state(
    optimizer: ShardedAdagrad | ShardedRowWiseAdagrad | None,
    weight: Tensor,
) -> dict[str, Any] | None:
    if optimizer is None:
        return None
    state = optimizer.state.get(weight)
    return state if state else None


def _restore_sharded_optimizer_rows(
    state: dict[str, Any],
    saved_state: dict[str, Any] | None,
    *,
    source_rows: Tensor,
    target_rows: Tensor,
    table_name: str,
) -> None:
    """Copy the saved rows of one embedding's accumulator into local state.

    Accumulators are indexed by local row exactly like the weight, so the same
    owner mapping that reshards weights also reshards Adagrad state.
    """

    if saved_state is None:
        raise ValueError(
            f"checkpoint has no sparse optimizer state for table {table_name!r}; "
            "resume would silently restart Adagrad accumulators"
        )
    accumulator = state.get("sum")
    saved_accumulator = saved_state.get("sum")
    if not isinstance(accumulator, Tensor) or not isinstance(saved_accumulator, Tensor):
        raise ValueError(
            f"invalid sparse optimizer accumulator for table {table_name!r}"
        )
    if accumulator.dim() != saved_accumulator.dim():
        raise ValueError(
            "checkpoint sparse optimizer state does not match "
            f"training.sparse_optimizer for table {table_name!r}"
        )
    values = saved_accumulator.index_select(0, source_rows.to(saved_accumulator.device))
    accumulator.index_copy_(
        0,
        target_rows.to(accumulator.device),
        values.to(device=accumulator.device, dtype=accumulator.dtype),
    )
    saved_step = saved_state.get("step")
    step = state.get("step")
    if isinstance(saved_step, Tensor) and isinstance(step, Tensor):
        # Every rank advances the shared schedule together, so the largest seen
        # value is the run's step count even when a table saw no rows locally.
        step.fill_(max(float(step), float(saved_step)))


def _load_dense_sharded_state(
    config: AppConfig,
    model: nn.Module,
    checkpoint_path: Path,
    manifest: dict[str, Any],
    device: torch.device,
) -> None:
    dense_path = checkpoint_path / str(manifest["dense_file"])
    dense_payload = torch.load(dense_path, map_location=device)
    _validate_checkpoint_metadata(config, dense_payload)
    missing, unexpected = model.load_state_dict(
        dense_payload["model_state_dict"], strict=False
    )
    expected_missing = _sharded_state_keys(model)
    if unexpected:
        raise ValueError(
            "checkpoint contains unexpected dense model keys: "
            + ", ".join(sorted(unexpected))
        )
    if set(missing) != expected_missing:
        absent = set(missing) - expected_missing
        extra = expected_missing - set(missing)
        details = []
        if absent:
            details.append("unexpected missing=" + ",".join(sorted(absent)))
        if extra:
            details.append(
                "sharded keys present in dense file=" + ",".join(sorted(extra))
            )
        raise ValueError("invalid dense checkpoint state: " + "; ".join(details))


def _read_sparse_segment_header(path: Path) -> tuple[dict[str, Any], int]:
    with open(path, "rb") as stream:
        if stream.read(len(_SPARSE_SEGMENT_MAGIC)) != _SPARSE_SEGMENT_MAGIC:
            raise ValueError(f"invalid sparse segment magic in {path.name}")
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated sparse segment header in {path.name}")
        header_length = int(struct.unpack("<Q", raw_length)[0])
        if header_length <= 0 or header_length > 16 * 1024 * 1024:
            raise ValueError(f"invalid sparse segment header length in {path.name}")
        encoded = stream.read(header_length)
        if len(encoded) != header_length:
            raise ValueError(f"truncated sparse segment header in {path.name}")
    header = json.loads(encoded.decode("utf-8"))
    if header.get("format") != SHARDED_SPARSE_STREAM_FORMAT:
        raise ValueError(f"unsupported sparse segment format in {path.name}")
    return header, len(_SPARSE_SEGMENT_MAGIC) + 8 + header_length


def _verify_file_checksum(path: Path, expected: str | None) -> None:
    if not expected:
        return
    digest = sha256()
    with open(path, "rb") as stream:
        while True:
            payload = stream.read(_SPARSE_COPY_BLOCK_BYTES)
            if not payload:
                break
            digest.update(payload)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"sparse checkpoint checksum mismatch for {path.name}: "
            f"expected {expected}, got {actual}"
        )


def _tensor_from_raw(payload: bytes, dtype: torch.dtype, shape: list[int]) -> Tensor:
    # Clone detaches the tensor from the temporary bytearray backing store.
    value = torch.frombuffer(bytearray(payload), dtype=dtype).clone()
    return value.reshape(tuple(int(item) for item in shape))


def _load_sparse_segment(
    path: Path,
    segment_metadata: dict[str, Any],
    *,
    module: ShardedEmbedding,
    optimizer: ShardedAdagrad | ShardedRowWiseAdagrad | None,
    rank: int,
) -> int:
    _verify_file_checksum(path, segment_metadata.get("sha256"))
    header, payload_offset = _read_sparse_segment_header(path)
    if header.get("table_name") != module.table_name:
        raise ValueError(
            f"sparse segment {path.name} belongs to {header.get('table_name')!r}, "
            f"not {module.table_name!r}"
        )
    count = int(header["row_count"])
    if count != int(segment_metadata.get("row_count", count)):
        raise ValueError(f"sparse segment row count mismatch in {path.name}")
    ids_bytes = int(header["ids_bytes"])
    weight_bytes = int(header["weight_bytes"])
    accumulator_bytes = int(header["accumulator_bytes"])
    expected_size = payload_offset + ids_bytes + weight_bytes + accumulator_bytes
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"sparse segment {path.name} has size {path.stat().st_size}, "
            f"expected {expected_size}"
        )
    weight_dtype = _NAME_TO_DTYPE.get(str(header["weight_dtype"]))
    accumulator_dtype = _NAME_TO_DTYPE.get(str(header["accumulator_dtype"]))
    if weight_dtype is None or accumulator_dtype is None:
        raise ValueError(f"unsupported dtype in sparse segment {path.name}")
    weight_shape = [int(item) for item in header["weight_shape"]]
    accumulator_shape = [int(item) for item in header["accumulator_shape"]]
    if not weight_shape or weight_shape[0] != count:
        raise ValueError(f"invalid weight shape in sparse segment {path.name}")
    if not accumulator_shape or accumulator_shape[0] != count:
        raise ValueError(f"invalid accumulator shape in sparse segment {path.name}")

    with open(path, "rb") as stream:
        stream.seek(payload_offset)
        raw_ids = stream.read(ids_bytes)
    if len(raw_ids) != ids_bytes:
        raise ValueError(f"truncated row IDs in sparse segment {path.name}")
    global_ids = _tensor_from_raw(raw_ids, torch.int64, [count])
    if global_ids.numel() > 1 and not bool((global_ids[1:] > global_ids[:-1]).all()):
        raise ValueError(f"row IDs are not strictly sorted in {path.name}")
    if global_ids.numel() and (
        int(global_ids[0]) < 0 or int(global_ids[-1]) >= module.num_embeddings
    ):
        raise ValueError(f"row IDs are outside {module.table_name!r} in {path.name}")

    owned = module.shard_spec.owner(global_ids) == int(rank)
    selected_total = int(owned.sum())
    weight_row_elements = max(1, math.prod(weight_shape[1:]))
    weight_row_bytes = weight_row_elements * torch.empty((), dtype=weight_dtype).element_size()
    accumulator_row_elements = max(1, math.prod(accumulator_shape[1:]))
    accumulator_row_bytes = (
        accumulator_row_elements
        * torch.empty((), dtype=accumulator_dtype).element_size()
    )
    block_rows = max(
        1,
        _SPARSE_COPY_BLOCK_BYTES // max(weight_row_bytes, accumulator_row_bytes),
    )
    weight_offset = payload_offset + ids_bytes
    accumulator_offset = weight_offset + weight_bytes
    optimizer_state = (
        None if optimizer is None else optimizer.state.get(module.weight)
    )
    accumulator = None if not optimizer_state else optimizer_state.get("sum")
    if optimizer is not None and not isinstance(accumulator, Tensor):
        raise ValueError(
            f"optimizer has no accumulator for sparse table {module.table_name!r}"
        )
    with torch.no_grad(), open(path, "rb") as stream:
        for start in range(0, count, block_rows):
            rows = min(block_rows, count - start)
            block_owned = owned.narrow(0, start, rows)
            source_rows = torch.nonzero(block_owned, as_tuple=False).flatten()
            if source_rows.numel() == 0:
                continue
            block_global_ids = global_ids.narrow(0, start, rows)
            target_global_ids = block_global_ids.index_select(0, source_rows)
            target_rows = module.shard_spec.local_row_ids(target_global_ids)

            stream.seek(weight_offset + start * weight_row_bytes)
            raw_weight = stream.read(rows * weight_row_bytes)
            if len(raw_weight) != rows * weight_row_bytes:
                raise ValueError(f"truncated weights in sparse segment {path.name}")
            block_weight = _tensor_from_raw(
                raw_weight,
                weight_dtype,
                [rows, *weight_shape[1:]],
            ).index_select(0, source_rows)
            module.weight.index_copy_(
                0,
                target_rows.to(module.weight.device),
                block_weight.to(
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                ),
            )

            if isinstance(accumulator, Tensor):
                stream.seek(accumulator_offset + start * accumulator_row_bytes)
                raw_accumulator = stream.read(rows * accumulator_row_bytes)
                if len(raw_accumulator) != rows * accumulator_row_bytes:
                    raise ValueError(
                        f"truncated optimizer state in sparse segment {path.name}"
                    )
                block_accumulator = _tensor_from_raw(
                    raw_accumulator,
                    accumulator_dtype,
                    [rows, *accumulator_shape[1:]],
                ).index_select(0, source_rows)
                accumulator.index_copy_(
                    0,
                    target_rows.to(accumulator.device),
                    block_accumulator.to(
                        device=accumulator.device,
                        dtype=accumulator.dtype,
                    ),
                )
    return selected_total


def _load_streaming_sharded_checkpoint(
    config: AppConfig,
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    process_group: torch_dist.ProcessGroup | None,
    *,
    sharded_optimizer: ShardedAdagrad | ShardedRowWiseAdagrad | None,
    load_dense: bool,
) -> None:
    manifest = json.loads(
        (checkpoint_path / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("format") != SHARDED_SPARSE_STREAM_FORMAT:
        raise ValueError("unsupported streaming sparse checkpoint format")
    _validate_checkpoint_metadata(config, manifest)
    if load_dense:
        _load_dense_sharded_state(config, model, checkpoint_path, manifest, device)
    modules = {module.table_name: module for module in sharded_embedding_modules(model)}
    if set(modules) != set(manifest.get("tables", {})):
        raise ValueError("checkpoint embedding table set does not match current model")
    rank, world_size = (0, 1)
    if torch_dist.is_available() and torch_dist.is_initialized():
        rank = torch_dist.get_rank(process_group)
        world_size = torch_dist.get_world_size(process_group)
    saved_world_size = int(manifest["world_size"])
    saved_sparse_optimizer = manifest.get("training_metadata", {}).get(
        "sparse_optimizer"
    )
    if sharded_optimizer is not None and saved_sparse_optimizer not in {
        None,
        config.training.sparse_optimizer,
    }:
        raise ValueError(
            "checkpoint sparse optimizer does not match the current configuration"
        )
    rank_manifest_names = list(manifest.get("rank_manifests", ()))
    if saved_world_size == world_size:
        rank_manifest_names = [rank_manifest_names[rank]]
    covered = {name: 0 for name in modules}
    for rank_manifest_name in rank_manifest_names:
        rank_manifest = json.loads(
            (checkpoint_path / rank_manifest_name).read_text(encoding="utf-8")
        )
        if rank_manifest.get("format") != SHARDED_SPARSE_STREAM_FORMAT:
            raise ValueError(f"invalid sparse rank manifest {rank_manifest_name}")
        for table_name, table in rank_manifest.get("tables", {}).items():
            module = modules.get(table_name)
            if module is None:
                raise ValueError(f"checkpoint has unknown sparse table {table_name!r}")
            if (
                int(table["num_embeddings"]) != module.num_embeddings
                or int(table["embedding_dim"]) != module.embedding_dim
                or _padding_idx_value(table["padding_idx"])
                != _padding_idx_value(module.padding_idx)
            ):
                raise ValueError(
                    f"checkpoint metadata does not match embedding {table_name!r}"
                )
            for segment in table.get("segments", ()):
                covered[table_name] += _load_sparse_segment(
                    checkpoint_path / str(segment["file"]),
                    segment,
                    module=module,
                    optimizer=sharded_optimizer,
                    rank=rank,
                )
            if sharded_optimizer is not None:
                state = sharded_optimizer.state[module.weight]
                step = state.get("step")
                saved_step = table.get("optimizer_step")
                if isinstance(step, Tensor) and saved_step is not None:
                    step.fill_(max(float(step), float(saved_step)))
    if manifest.get("kind") == "full":
        incomplete = [
            name
            for name, module in modules.items()
            if covered.get(name, 0) != int(module.weight.shape[0])
        ]
        if incomplete:
            raise ValueError(
                "full sparse checkpoint did not cover every local row for: "
                + ", ".join(sorted(incomplete))
            )


def _load_sharded_checkpoint(
    config: AppConfig,
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    process_group: torch_dist.ProcessGroup | None,
    sharded_optimizer: ShardedAdagrad | ShardedRowWiseAdagrad | None = None,
) -> None:
    manifest_path = checkpoint_path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"sharded checkpoint is missing {manifest_path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") not in _SHARDED_CHECKPOINT_READABLE:
        raise ValueError("unsupported sharded checkpoint format")
    _validate_checkpoint_metadata(config, manifest)
    dense_path = checkpoint_path / str(manifest["dense_file"])
    dense_payload = torch.load(dense_path, map_location=device)
    _validate_checkpoint_metadata(config, dense_payload)
    missing, unexpected = model.load_state_dict(
        dense_payload["model_state_dict"], strict=False
    )
    expected_missing = _sharded_state_keys(model)
    if unexpected:
        raise ValueError(
            "checkpoint contains unexpected dense model keys: "
            + ", ".join(sorted(unexpected))
        )
    if set(missing) != expected_missing:
        absent = set(missing) - expected_missing
        extra = expected_missing - set(missing)
        details = []
        if absent:
            details.append("unexpected missing=" + ",".join(sorted(absent)))
        if extra:
            details.append("sharded keys present in dense file=" + ",".join(sorted(extra)))
        raise ValueError("invalid dense checkpoint state: " + "; ".join(details))

    modules = {module.table_name: module for module in sharded_embedding_modules(model)}
    manifest_tables = manifest.get("tables", {})
    if set(modules) != set(manifest_tables):
        raise ValueError("checkpoint embedding table set does not match current model")
    if sharded_optimizer is not None:
        saved_sparse_optimizer = manifest.get("training_metadata", {}).get(
            "sparse_optimizer"
        )
        if saved_sparse_optimizer not in {None, config.training.sparse_optimizer}:
            raise ValueError(
                "checkpoint training_metadata.sparse_optimizer "
                f"{saved_sparse_optimizer!r} does not match current "
                f"training.sparse_optimizer {config.training.sparse_optimizer!r}"
            )
    rank, world_size = (0, 1)
    if torch_dist.is_available() and torch_dist.is_initialized():
        rank = torch_dist.get_rank(process_group)
        world_size = torch_dist.get_world_size(process_group)
    saved_world_size = int(manifest["world_size"])
    # Same-size restarts need only this rank's files. Different-size loads stream
    # every saved owner and write directly into the new local rows; no full-table
    # reconstruction is allocated.
    shard_files = [
        checkpoint_path / name
        for name in shard_file_names(
            manifest,
            rank=rank,
            world_size=world_size,
        )
    ]
    map_location = device if saved_world_size == world_size else "cpu"
    filled = {
        name: torch.zeros(module.weight.size(0), dtype=torch.bool)
        for name, module in modules.items()
    }
    with torch.no_grad():
        for shard_file in shard_files:
            # One file at a time: a resharding load would otherwise hold every
            # saved rank's tables in host memory simultaneously.
            payload = torch.load(shard_file, map_location=map_location)
            if payload.get("format") not in _SHARDED_CHECKPOINT_READABLE:
                raise ValueError("invalid embedding rank shard format")
            saved_rank = int(payload["rank"])
            if int(payload["world_size"]) != saved_world_size:
                raise ValueError("inconsistent world size across embedding shard files")
            for table_name, table in payload["tables"].items():
                module = modules.get(table_name)
                if module is None:
                    raise ValueError(
                        f"checkpoint shard holds unknown table {table_name!r}"
                    )
                if (
                    int(table["num_embeddings"]) != module.num_embeddings
                    or int(table["embedding_dim"]) != module.embedding_dim
                    or _padding_idx_value(table["padding_idx"])
                    != _padding_idx_value(module.padding_idx)
                ):
                    raise ValueError(
                        f"checkpoint metadata does not match embedding {table_name!r}"
                    )
                saved_spec = EmbeddingShardSpec(**table["shard_spec"])
                global_ids = torch.arange(module.num_embeddings, dtype=torch.long)
                saved_owned = saved_spec.owner(global_ids) == saved_rank
                saved_global_ids = global_ids[saved_owned]
                saved_weight = table["weight"]
                if saved_weight.size(0) != saved_global_ids.numel():
                    raise ValueError(
                        f"checkpoint shard row count is invalid for {table_name!r}"
                    )
                current_owned = module.shard_spec.owner(saved_global_ids) == rank
                source_rows = torch.nonzero(current_owned, as_tuple=False).flatten()
                target_global_ids = saved_global_ids[current_owned]
                target_rows = module.shard_spec.local_row_ids(target_global_ids)
                values = saved_weight.index_select(
                    0, source_rows.to(saved_weight.device)
                ).to(
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                module.weight.index_copy_(
                    0, target_rows.to(module.weight.device), values
                )
                if sharded_optimizer is not None:
                    _restore_sharded_optimizer_rows(
                        sharded_optimizer.state[module.weight],
                        table.get("optimizer_state"),
                        source_rows=source_rows,
                        target_rows=target_rows,
                        table_name=table_name,
                    )
                filled[table_name].index_fill_(0, target_rows.cpu(), True)
            del payload
    incomplete = [name for name, mask in filled.items() if not bool(mask.all())]
    if incomplete:
        raise ValueError(
            "checkpoint did not cover all local rows for tables: "
            + ", ".join(sorted(incomplete))
        )


def load_model_checkpoint(
    config: AppConfig,
    model: nn.Module,
    path: str | Path,
    *,
    device: torch.device,
    process_group: torch_dist.ProcessGroup | None = None,
    sharded_optimizer: ShardedAdagrad | ShardedRowWiseAdagrad | None = None,
) -> None:
    """Load a legacy replicated file or a reshardable checkpoint directory."""

    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        manifest_path = checkpoint_path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format") == SHARDED_SPARSE_STREAM_FORMAT:
                _load_streaming_sharded_checkpoint(
                    config,
                    model,
                    checkpoint_path,
                    device,
                    process_group,
                    sharded_optimizer=sharded_optimizer,
                    load_dense=True,
                )
                return
        _load_sharded_checkpoint(
            config,
            model,
            checkpoint_path,
            device,
            process_group,
            sharded_optimizer=sharded_optimizer,
        )
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    _validate_checkpoint_metadata(config, checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])


# --- Run directories: step naming, discovery, and retention ---


def step_directory_name(step: int) -> str:
    """Return the zero-padded directory name for one saved step."""

    return f"{STEP_DIR_PREFIX}{int(step):0{_STEP_DIR_DIGITS}d}"


def parse_step_directory(name: str) -> int | None:
    """Return the step encoded in a directory name, or None when unrelated."""

    if not name.startswith(STEP_DIR_PREFIX):
        return None
    suffix = name[len(STEP_DIR_PREFIX) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def rank_ready_marker(rank: int) -> str:
    """Marker a rank writes once all of its files reached the run directory."""

    return f"_READY-rank-{int(rank):05d}"


def rank_progress_file(rank: int) -> str:
    """Human-readable record of where one rank's input scan stopped."""

    return f"progress-rank-{int(rank):05d}.json"


@dataclass(frozen=True)
class CheckpointDataWindow:
    """One contiguous half-open UTC hour interval consumed by training."""

    start: str
    end: str
    inputs: tuple[str, ...] = ()
    # Persist these separately from ``inputs``: manifests intentionally do not
    # repeat every HDFS URI, but resume still needs to detect a rewritten hour.
    input_manifest_digest: str | None = None
    input_count: int | None = None

    def manifest_digest(self) -> str:
        if self.input_manifest_digest is not None:
            return self.input_manifest_digest
        return sha256("\n".join(self.inputs).encode("utf-8")).hexdigest()

    def manifest_count(self) -> int:
        if self.input_count is not None:
            return int(self.input_count)
        return len(self.inputs)

    def to_payload(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "input_manifest_digest": self.manifest_digest(),
            "input_count": self.manifest_count(),
        }

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any] | None
    ) -> "CheckpointDataWindow | None":
        if not payload:
            return None
        return cls(
            start=str(payload["start"]),
            end=str(payload["end"]),
            input_manifest_digest=payload.get("input_manifest_digest"),
            input_count=(
                None
                if payload.get("input_count") is None
                else int(payload["input_count"])
            ),
        )


@dataclass(frozen=True)
class DataCursor:
    """Where one rank's Parquet scan stood when a checkpoint was taken.

    ``position`` indexes the rank's deterministic work list (whole files under
    ``reader.shard_unit: file``, row groups otherwise). ``prefix_digest`` covers
    the already-consumed part of that list, so a restart can tell "the same
    inputs plus new partitions" (resumable) from "the inputs were rewritten"
    (must rescan). ``split_key`` ties the cursor to one split and rank.

    ``position`` is where the *reader* stood, which leads the trainer by the
    prefetch and IPC queues. ``rewind`` is how many items back a restart must
    start so those read-but-untrained rows are replayed; ``emitted_rows`` and
    ``rows_trained`` are the measurements it came from.
    """

    work_unit: str
    position: int
    prefix_digest: str | None = None
    split_key: str | None = None
    rank: int = 0
    world_size: int = 1
    rewind: int = 1
    emitted_rows: int = 0
    rows_trained: int = 0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "DataCursor | None":
        if not payload:
            return None
        known = {item.name for item in dataclass_fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known})


@dataclass(frozen=True)
class CommittedCheckpoint:
    """A step directory that finished uploading and passed its commit marker."""

    step: int
    directory: str
    uri: str
    world_size: int = 0
    saved_at: float = 0.0
    lineage_id: str | None = None
    generation: int = 0
    sparse_kind: str = "full"
    parent_checkpoint: str | None = None
    sparse_parent_checkpoint: str | None = None
    data_window: CheckpointDataWindow | None = None
    window_complete: bool = False
    trained_through: str | None = None


def _step_entries(store: CheckpointStore) -> list[tuple[int, str]]:
    steps: list[tuple[int, str]] = []
    for entry in store.list_entries():
        if not entry.is_dir:
            continue
        step = parse_step_directory(entry.name)
        if step is not None:
            steps.append((step, entry.name))
    steps.sort()
    return steps


def list_committed_checkpoints(store: CheckpointStore) -> list[CommittedCheckpoint]:
    """Return every committed step in the run directory, oldest first."""

    committed: list[CommittedCheckpoint] = []
    for step, name in _step_entries(store):
        if not store.exists(name, COMMIT_MARKER):
            continue
        world_size = 0
        saved_at = 0.0
        metadata: dict[str, Any] = {}
        try:
            metadata = store.read_json(name, CHECKPOINT_MANIFEST)
            world_size = int(metadata.get("world_size", 0))
            saved_at = float(metadata.get("saved_at", 0.0))
        except Exception:  # noqa: BLE001 - a readable commit marker is enough
            pass
        committed.append(
            CommittedCheckpoint(
                step=step,
                directory=name,
                uri=store.uri(name),
                world_size=world_size,
                saved_at=saved_at,
                lineage_id=metadata.get("lineage_id"),
                generation=int(metadata.get("generation", 0)),
                sparse_kind=str(metadata.get("sparse_kind", "full")),
                parent_checkpoint=metadata.get("parent_checkpoint"),
                sparse_parent_checkpoint=metadata.get("sparse_parent_checkpoint"),
                data_window=CheckpointDataWindow.from_payload(
                    metadata.get("data_window")
                ),
                window_complete=bool(metadata.get("window_complete", False)),
                trained_through=metadata.get("trained_through"),
            )
        )
    return committed


def latest_committed_checkpoint(
    store: CheckpointStore,
) -> CommittedCheckpoint | None:
    """Return the newest resumable step, or None for an empty/fresh run."""

    committed = list_committed_checkpoints(store)
    return committed[-1] if committed else None


def resolve_resume_checkpoint(
    store: CheckpointStore,
    spec: str | None,
    *,
    lineage_id: str | None = None,
    requested_windows: Sequence[CheckpointDataWindow] = (),
) -> CommittedCheckpoint | None:
    """Interpret ``training.checkpoint.resume`` against a run directory.

    ``auto``/``latest`` pick the newest committed step, ``none`` disables
    resuming, and an integer or ``step-000012000`` name pins one step.
    """

    text = (spec or "auto").strip()
    if text.lower() in {"none", "off", "false", ""}:
        return None
    if text.lower() in {"auto", "latest", "latest_predecessor", "continue_window"}:
        committed = list_committed_checkpoints(store)
        if lineage_id is not None:
            committed = [
                item
                for item in committed
                if item.lineage_id in {None, lineage_id}
            ]
        if requested_windows:
            requested_by_bounds = {
                (item.start, item.end): item for item in requested_windows
            }
            first_start = requested_windows[0].start
            last_end = requested_windows[-1].end
            legacy = [item for item in committed if item.trained_through is None]
            aware = [item for item in committed if item.trained_through is not None]
            eligible: list[CommittedCheckpoint] = []
            for item in aware:
                if not (item.trained_through <= last_end):
                    continue
                if item.data_window is not None:
                    requested = requested_by_bounds.get(
                        (item.data_window.start, item.data_window.end)
                    )
                    # A completed checkpoint whose watermark equals the new
                    # range start is a valid predecessor even though its own
                    # preceding window is not part of the new input manifest.
                    needs_requested_window = (
                        not item.window_complete
                        or str(item.trained_through) > first_start
                    )
                    if needs_requested_window and requested is None:
                        continue
                    if requested is not None:
                        saved_digest = item.data_window.input_manifest_digest
                        if (
                            saved_digest is not None
                            and saved_digest != requested.manifest_digest()
                        ):
                            continue
                        saved_count = item.data_window.input_count
                        if (
                            saved_count is not None
                            and int(saved_count) != requested.manifest_count()
                        ):
                            continue
                elif item.trained_through < first_start:
                    # A predecessor is safe only when its watermark is exactly
                    # the requested start (which is admitted by the normal
                    # path above). Silently jumping a time gap loses data.
                    continue
                eligible.append(item)
            if eligible:
                return max(
                    eligible,
                    key=lambda item: (
                        str(item.trained_through),
                        int(item.generation),
                        int(item.step),
                    ),
                )
            if aware:
                newest = max(
                    aware,
                    key=lambda item: (
                        str(item.trained_through),
                        int(item.generation),
                        int(item.step),
                    ),
                )
                raise ValueError(
                    "no committed checkpoint in lineage "
                    f"{lineage_id!r} aligns with requested data range "
                    f"[{first_start}, {last_end}); newest checkpoint "
                    f"{newest.directory} has trained_through="
                    f"{newest.trained_through}. Use a contiguous range, an "
                    "explicitly different lineage, or resume=none to start fresh."
                )
            # Legacy checkpoints have no data-time watermark. They remain
            # usable only when the lineage has no window-aware checkpoint,
            # preserving migration from the old full format without masking a
            # known discontinuity.
            return legacy[-1] if legacy else None
        return committed[-1] if committed else None
    step = parse_step_directory(text)
    if step is None and text.isdigit():
        step = int(text)
    if step is None:
        raise ValueError(
            "training.checkpoint.resume must be auto, latest, none, a step number, "
            f"or a step directory name; received {spec!r}"
        )
    directory = step_directory_name(step)
    if not store.exists(directory, COMMIT_MARKER):
        raise FileNotFoundError(
            f"requested resume step {step} is not committed under {store.root_uri}"
        )
    for item in list_committed_checkpoints(store):
        if item.directory == directory:
            return item
    return CommittedCheckpoint(step=step, directory=directory, uri=store.uri(directory))


def prune_run_directory(store: CheckpointStore, keep_last: int) -> list[str]:
    """Delete superseded steps; return the directory names that were removed.

    Uncommitted directories older than the newest committed step are remnants of
    an interrupted upload and are removed regardless of ``keep_last``.
    """

    entries = _step_entries(store)
    if not entries:
        return []
    committed = [
        (step, name) for step, name in entries if store.exists(name, COMMIT_MARKER)
    ]
    newest_committed = committed[-1][0] if committed else -1
    doomed: list[str] = []
    if keep_last > 0 and len(committed) > keep_last:
        # Delta checkpoints are not independently resumable. Keep every sparse
        # ancestor reachable from the retained heads; a later full compaction
        # has no sparse parent, which naturally makes the old chain collectible.
        manifests: dict[str, dict[str, Any]] = {}
        for _step, name in committed:
            try:
                manifests[name] = store.read_json(name, CHECKPOINT_MANIFEST)
            except Exception:
                manifests[name] = {}
        protected = {
            name for _step, name in committed[len(committed) - keep_last :]
        }
        pending = list(protected)
        while pending:
            name = pending.pop()
            parent = manifests.get(name, {}).get("sparse_parent_checkpoint")
            if parent and parent in manifests and parent not in protected:
                protected.add(parent)
                pending.append(parent)
        doomed.extend(name for _step, name in committed if name not in protected)
    committed_names = {name for _step, name in committed}
    doomed.extend(
        name
        for step, name in entries
        if name not in committed_names and step < newest_committed
    )
    removed: list[str] = []
    for name in doomed:
        try:
            store.remove_tree(name)
            removed.append(name)
        except Exception as error:  # noqa: BLE001 - retention must never be fatal
            logger.warning("checkpoint retention could not remove %s: %s", name, error)
    return removed


# --- Saving one training step ---


@dataclass(frozen=True)
class StagedCheckpoint:
    """Files one rank wrote locally, ready to be published to the run directory."""

    step: int
    staging_dir: Path
    relative_files: tuple[str, ...]
    cleanup_staging: bool = True
    lineage_id: str | None = None
    parent_checkpoint: str | None = None
    sparse_parent_checkpoint: str | None = None
    sparse_kind: str = "full"
    generation: int = 0
    data_window: CheckpointDataWindow | None = None
    window_complete: bool = False
    trained_through: str | None = None
    attempt_id: str | None = None


def stage_training_checkpoint(
    config: AppConfig,
    model: nn.Module,
    staging_dir: str | Path,
    *,
    step: int,
    rows: int,
    rank: int = 0,
    world_size: int = 1,
    dense_optimizer: torch.optim.Optimizer | None = None,
    replicated_sparse_optimizer: torch.optim.Optimizer | None = None,
    sharded_optimizer: ShardedAdagrad | ShardedRowWiseAdagrad | None = None,
    data_cursor: DataCursor | None = None,
    process_group: torch_dist.ProcessGroup | None = None,
    elapsed_seconds: float = 0.0,
    run_name: str | None = None,
    cleanup_staging: bool = True,
    chunk_bytes: int = DEFAULT_SHARD_CHUNK_BYTES,
    sparse_stream: bool = False,
    sparse_kind: str = "full",
    parent_checkpoint: str | None = None,
    sparse_parent_checkpoint: str | None = None,
    generation: int = 0,
    lineage_id: str | None = None,
    data_window: CheckpointDataWindow | None = None,
    window_complete: bool = False,
    trained_through: str | None = None,
    attempt_id: str | None = None,
    publish: Callable[[str], bool] | None = None,
) -> StagedCheckpoint:
    """Write this rank's share of a resumable checkpoint to a local directory.

    ``publish`` is offered each file as it lands and returns whether it took
    ownership (uploaded and removed it). Files it declines stay for the caller's
    normal publish pass, so a stalled run directory degrades to the old
    "everything waits in staging" behaviour instead of blocking the step loop.
    """

    staging_path = Path(staging_dir)
    staging_path.mkdir(parents=True, exist_ok=True)
    sharded = bool(sharded_embedding_modules(model))
    model_relpath = MODEL_SUBDIR if sharded else MODEL_FILE
    model_prefix = f"{MODEL_SUBDIR}/" if sharded else ""

    relative_files: list[str] = []
    published: set[str] = set()

    def _offer(relative: str) -> None:
        relative_files.append(relative)
        if publish is not None and publish(relative):
            published.add(relative)

    model_files = save_model_checkpoint(
        config,
        model,
        staging_path / model_relpath,
        rank=rank,
        world_size=world_size,
        process_group=process_group,
        sharded_optimizer=sharded_optimizer,
        chunk_bytes=chunk_bytes,
        sparse_stream=sparse_stream,
        sparse_kind=sparse_kind,
        sparse_parent_checkpoint=sparse_parent_checkpoint,
        generation=generation,
        publish=(
            None
            if publish is None
            else (lambda name: _offer(f"{model_prefix}{name}"))
        ),
    )
    if publish is None:
        relative_files.extend(f"{model_prefix}{name}" for name in model_files)

    progress_name = rank_progress_file(rank)
    _atomic_json_save(
        {
            "format": TRAINING_CHECKPOINT_FORMAT,
            "step": int(step),
            "rows": int(rows),
            "rank": int(rank),
            "world_size": int(world_size),
            "saved_at": time.time(),
            "saved_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "data_cursor": None if data_cursor is None else data_cursor.to_payload(),
            "data_window": None if data_window is None else data_window.to_payload(),
            "window_complete": bool(window_complete),
            "trained_through": trained_through,
            "attempt_id": attempt_id,
        },
        staging_path / progress_name,
    )
    _offer(progress_name)

    if rank == 0:
        _atomic_torch_save(
            {
                "format": TRAINING_CHECKPOINT_FORMAT,
                "step": int(step),
                "world_size": int(world_size),
                "generation": int(generation),
                "dense_optimizer": (
                    None
                    if dense_optimizer is None
                    else _state_to_cpu(dense_optimizer.state_dict())
                ),
                "replicated_sparse_optimizer": (
                    None
                    if replicated_sparse_optimizer is None
                    else _state_to_cpu(replicated_sparse_optimizer.state_dict())
                ),
                **_checkpoint_metadata(config),
            },
            staging_path / TRAIN_STATE_FILE,
        )
        _offer(TRAIN_STATE_FILE)
        _atomic_json_save(
            {
                "format": TRAINING_CHECKPOINT_FORMAT,
                "version": 1,
                "step": int(step),
                "world_size": int(world_size),
                "lineage_id": lineage_id,
                "parent_checkpoint": parent_checkpoint,
                "sparse_parent_checkpoint": sparse_parent_checkpoint,
                "sparse_kind": sparse_kind,
                "generation": int(generation),
                "model_path": model_relpath,
                "sharded_embeddings": sharded,
                "run_name": run_name,
                "saved_at": time.time(),
                "saved_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "elapsed_seconds": float(elapsed_seconds),
                "data_window": (
                    None if data_window is None else data_window.to_payload()
                ),
                "window_complete": bool(window_complete),
                "trained_through": trained_through,
                "attempt_id": attempt_id,
                "training_metadata": {
                    "sparse_optimizer": config.training.sparse_optimizer,
                    "embedding_distribution": config.training.embedding_distribution,
                    "batch_size": config.training.batch_size,
                    "gradient_accumulation_steps": (
                        config.training.gradient_accumulation_steps
                    ),
                },
                **_checkpoint_metadata(config),
            },
            staging_path / CHECKPOINT_MANIFEST,
        )
        _offer(CHECKPOINT_MANIFEST)

    if published:
        logger.debug(
            "checkpoint step %d streamed %d/%d files while staging",
            int(step),
            len(published),
            len(relative_files),
        )
    return StagedCheckpoint(
        step=int(step),
        staging_dir=staging_path,
        relative_files=tuple(relative_files),
        cleanup_staging=cleanup_staging,
        lineage_id=lineage_id,
        parent_checkpoint=parent_checkpoint,
        sparse_parent_checkpoint=sparse_parent_checkpoint,
        sparse_kind=sparse_kind,
        generation=int(generation),
        data_window=data_window,
        window_complete=bool(window_complete),
        trained_through=trained_through,
        attempt_id=attempt_id,
    )


# --- Publishing staged steps to the run directory ---


@dataclass(frozen=True)
class _StagedFile:
    """One staged file handed to the uploader before its step finished."""

    step: int
    directory: str
    relative: str
    source: Path


@dataclass(frozen=True)
class _AbortedCheckpoint:
    """Cleanup marker ordered behind files from a failed staging attempt."""

    step: int
    staging_dir: Path
    cleanup_staging: bool


class CheckpointUploader:
    """Publishes staged steps to the run directory without stalling training.

    Every rank uploads its own files and then drops a ready marker. Rank 0 waits
    for the full set before writing ``_COMMIT``; readers treat a step without
    that marker as non-existent, so a crash mid-upload leaves nothing that can
    be resumed from by mistake.

    Files can also be handed over while the step is still being written (see
    :meth:`stream_publisher`). They travel on the same queue as the step that
    owns them, so the ready marker is still written after the last byte lands.
    """

    def __init__(
        self,
        store: CheckpointStore,
        *,
        rank: int = 0,
        world_size: int = 1,
        keep_last: int = 3,
        ready_timeout_sec: float = 1800.0,
        poll_interval_sec: float = 2.0,
        asynchronous: bool = True,
        max_pending: int = 1,
        stream_window: int = 2,
        stream_timeout_sec: float = 300.0,
    ) -> None:
        self._store = store
        self._rank = int(rank)
        self._world_size = max(1, int(world_size))
        self._keep_last = int(keep_last)
        self._ready_timeout_sec = float(ready_timeout_sec)
        self._poll_interval_sec = float(poll_interval_sec)
        self._asynchronous = bool(asynchronous)
        self._stream_window = max(1, int(stream_window))
        self._stream_timeout_sec = max(0.0, float(stream_timeout_sec))
        self._max_pending_steps = max(1, int(max_pending))
        # File streaming is bounded, but the step finalizer is not.  A bounded
        # mixed queue can fill with file items and then reject the finalizer;
        # the old rejection path removed the staging directory even though the
        # queued items still referenced files inside it.  Keep FIFO ordering in
        # one unbounded queue and enforce only the streamed-file window with a
        # semaphore, so the finalizer is always queued behind every file it owns.
        self._queue: queue.Queue = queue.Queue()
        self._stream_slots = threading.BoundedSemaphore(self._stream_window)
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        # Names that already reached the run directory, so the publish pass can
        # skip them. Ordering on the queue guarantees this is complete before the
        # owning step is finalized; the lock also covers synchronous uploads and
        # failure cleanup, which may run on the caller's thread.
        self._streamed: dict[int, set[str]] = {}
        self._streamed_lock = threading.Lock()
        # A queue remaining full is not evidence that HDFS is wedged: one 2GiB
        # file can take several minutes while making steady progress.  The
        # generation changes after every completed remote write chunk, allowing
        # backpressure to time out only after *no byte progress*.
        self._progress_lock = threading.Lock()
        self._progress_generation = 0
        self._uploaded_bytes: dict[int, int] = {}
        self._last_progress_log_at: dict[int, float] = {}
        # Bound whole staged checkpoints separately from streamed files.  The
        # coordinator reserves before it writes anything, avoiding a pile-up of
        # complete local checkpoints when an earlier HDFS publish is still live.
        self._pending_condition = threading.Condition()
        self._pending_steps: set[int] = set()
        self._step_outcomes: dict[int, bool] = {}
        self._closed = False
        self.published_steps: list[int] = []
        self.failed_steps: list[int] = []
        self.dropped_steps: list[int] = []
        if self._asynchronous:
            self._thread = threading.Thread(
                target=self._run,
                name=f"mdl-checkpoint-upload-rank{self._rank}",
                daemon=True,
            )
            self._thread.start()

    @property
    def store(self) -> CheckpointStore:
        return self._store

    @property
    def stream_window(self) -> int:
        """Staged files that may be awaiting upload while the next one is built."""

        return self._stream_window

    def wait_for_step_capacity(
        self,
        step: int,
        timeout_sec: float,
        *,
        heartbeat: Callable[[], None] | None = None,
    ) -> bool:
        """Wait for a whole-checkpoint slot without reserving it."""

        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        with self._pending_condition:
            while (
                int(step) not in self._pending_steps
                and len(self._pending_steps) >= self._max_pending_steps
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._pending_condition.wait(timeout=min(0.5, remaining))
                if heartbeat is not None:
                    heartbeat()
            return True

    def reserve_step(self, step: int) -> bool:
        """Reserve capacity before staging; idempotent for the same step."""

        value = int(step)
        with self._pending_condition:
            if value in self._pending_steps:
                return True
            if len(self._pending_steps) >= self._max_pending_steps:
                return False
            self._step_outcomes.pop(value, None)
            self._pending_steps.add(value)
            return True

    def _finish_step(self, step: int, *, succeeded: bool) -> None:
        with self._pending_condition:
            self._step_outcomes[int(step)] = bool(succeeded)
            self._pending_steps.discard(int(step))
            self._pending_condition.notify_all()

    def wait_for_step(
        self,
        step: int,
        timeout_sec: float,
        *,
        heartbeat: Callable[[], None] | None = None,
    ) -> bool | None:
        """Return publish outcome, or ``None`` when the deadline expires."""

        value = int(step)
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        with self._pending_condition:
            while value not in self._step_outcomes:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._pending_condition.wait(timeout=min(0.5, remaining))
                if heartbeat is not None:
                    heartbeat()
            return self._step_outcomes[value]

    def stream_publisher(
        self,
        staging_dir: Path,
        step: int,
        *,
        enabled: bool = True,
        heartbeat: Callable[[], None] | None = None,
    ) -> Callable[[str], bool]:
        """Sink that uploads staged files as they are written, then deletes them.

        Local run directories stage straight into their destination, so there is
        nothing to copy and nothing that may be deleted; those pass
        ``enabled=False`` and keep every file. Backpressure resets its deadline
        whenever an upload writes another chunk. Only a true no-progress window
        opens the per-step circuit; that file and every later file then stay for
        the normal publish pass without each waiting for another full timeout.
        """

        directory = step_directory_name(step)
        reserved = self.reserve_step(step)
        streaming_disabled = not reserved
        if not reserved:
            logger.warning(
                "checkpoint step %d cannot stream while %d prior step(s) are "
                "still pending; files will remain local unless the step is dropped",
                step,
                self._max_pending_steps,
            )

        def _publish_file(relative: str) -> bool:
            nonlocal streaming_disabled
            if not enabled:
                return False
            if streaming_disabled:
                return False
            item = _StagedFile(
                step=int(step),
                directory=directory,
                relative=relative,
                source=Path(staging_dir) / relative,
            )
            if not self._asynchronous:
                return self._upload_staged_file(item)
            generation, uploaded = self._progress_snapshot(step)
            deadline = time.monotonic() + self._stream_timeout_sec
            while True:
                wait_for = min(0.5, max(0.0, deadline - time.monotonic()))
                if self._stream_slots.acquire(timeout=wait_for):
                    # This queue is intentionally unbounded; the semaphore is
                    # the streamed-file capacity. The put therefore cannot
                    # strand an acquired slot behind queue.Full.
                    self._queue.put_nowait(item)
                    return True
                if heartbeat is not None:
                    heartbeat()
                current_generation, current_uploaded = self._progress_snapshot(step)
                if current_generation != generation:
                    generation = current_generation
                    uploaded = current_uploaded
                    deadline = time.monotonic() + self._stream_timeout_sec
                    continue
                if time.monotonic() >= deadline:
                    streaming_disabled = True
                    logger.warning(
                        "checkpoint step %d stopped streaming at %s after no "
                        "upload byte progress for %.0fs (uploaded %.2fGiB on "
                        "rank %d); this and later files stay in staging for the "
                        "ordered final publish",
                        step,
                        relative,
                        self._stream_timeout_sec,
                        uploaded / (1024**3),
                        self._rank,
                    )
                    return False

        return _publish_file

    def submit(self, staged: StagedCheckpoint) -> bool:
        """Queue the step finalizer behind all streamed files it owns.

        The finalizer must never be dropped for file-level backpressure. It is
        what retries declined/failed streams, writes the rank-ready marker, and
        eventually cleans staging. Since file capacity is enforced separately,
        appending this one control item does not make file buffering unbounded.
        """

        if not self.reserve_step(staged.step):
            # No streamed file from this step could have been accepted without
            # the same reservation, so cleanup is safe in this one drop path.
            self.dropped_steps.append(staged.step)
            logger.warning(
                "checkpoint step %d dropped before publish: %d prior step(s) "
                "are still pending for %s",
                staged.step,
                self._max_pending_steps,
                self._store.root_uri,
            )
            self._cleanup(staged)
            self._finish_step(staged.step, succeeded=False)
            return False
        if not self._asynchronous:
            self._publish(staged)
            return True
        self._queue.put_nowait(staged)
        return True

    def abort_step(
        self,
        step: int,
        staging_dir: Path,
        *,
        cleanup_staging: bool,
    ) -> None:
        """Order failed-stage cleanup after any already accepted file items."""

        aborted = _AbortedCheckpoint(
            step=int(step),
            staging_dir=Path(staging_dir),
            cleanup_staging=bool(cleanup_staging),
        )
        if not self._asynchronous:
            self._cleanup_aborted(aborted)
            return
        self._queue.put_nowait(aborted)

    def _note_upload_progress(self, step: int, byte_count: int) -> None:
        increment = max(0, int(byte_count))
        if increment <= 0:
            return
        now = time.monotonic()
        report: int | None = None
        with self._progress_lock:
            self._uploaded_bytes[step] = self._uploaded_bytes.get(step, 0) + increment
            self._progress_generation += 1
            last_report = self._last_progress_log_at.get(step)
            if self._rank == 0 and (
                last_report is None or now - last_report >= 60.0
            ):
                self._last_progress_log_at[step] = now
                report = self._uploaded_bytes[step]
        if report is not None:
            try:
                print(
                    f"Checkpoint upload | step={step} rank=0 "
                    f"transferred_gib={report / (1024**3):.2f}",
                    flush=True,
                )
            except Exception:
                # Diagnostics must never turn a successful remote write into a
                # retried upload merely because stdout is being torn down.
                pass

    def _progress_snapshot(self, step: int) -> tuple[int, int]:
        with self._progress_lock:
            return self._progress_generation, self._uploaded_bytes.get(step, 0)

    def drain(self, timeout_sec: float | None = None) -> bool:
        """Block until queued uploads finish; returns False on timeout."""

        if not self._asynchronous:
            return True
        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
        while not self._queue.empty():
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.1)
        # unfinished_tasks covers the item currently in flight.
        while self._queue.unfinished_tasks:
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.1)
        return True

    def close(self, timeout_sec: float | None = 600.0) -> None:
        if self._closed:
            return
        self._closed = True
        drained = self.drain(timeout_sec)
        self._stopping.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        if not drained:
            logger.warning(
                "checkpoint uploader for %s did not finish within %.0fs",
                self._store.root_uri,
                timeout_sec or 0.0,
            )

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if isinstance(item, _StagedFile):
                    # Count the in-flight file in the window. Combined with the
                    # next chunk being serialized, this matches the staging
                    # preflight's ``upload_window + 1`` peak calculation.
                    try:
                        self._upload_staged_file(item)
                    finally:
                        self._stream_slots.release()
                elif isinstance(item, _AbortedCheckpoint):
                    self._cleanup_aborted(item)
                else:
                    self._publish(item)
            finally:
                self._queue.task_done()

    def _upload_staged_file(self, item: _StagedFile) -> bool:
        """Copy one staged file to the run directory and free the local copy."""

        try:
            self._store.upload_file(
                item.source,
                item.directory,
                *item.relative.split("/"),
                progress=lambda count: self._note_upload_progress(item.step, count),
            )
        except Exception as error:  # noqa: BLE001 - the publish pass retries
            logger.warning(
                "checkpoint step %d could not stream %s: %s",
                item.step,
                item.relative,
                error,
            )
            return False
        with self._streamed_lock:
            self._streamed.setdefault(item.step, set()).add(item.relative)
        _discard_partial_file(item.source)
        return True

    def _publish(self, staged: StagedCheckpoint) -> None:
        directory = step_directory_name(staged.step)
        with self._streamed_lock:
            already = self._streamed.pop(staged.step, set())
        succeeded = False
        try:
            self._store.makedirs(directory)
            for relative in staged.relative_files:
                if relative in already:
                    continue
                self._store.upload_file(
                    staged.staging_dir / relative,
                    directory,
                    *relative.split("/"),
                    progress=lambda count: self._note_upload_progress(
                        staged.step, count
                    ),
                )
                if staged.cleanup_staging:
                    # Freeing each file as it lands keeps a large multi-rank step
                    # from holding its whole footprint until the very end.
                    _discard_partial_file(staged.staging_dir / relative)
            self._store.write_json(
                {
                    "rank": self._rank,
                    "step": staged.step,
                    "files": list(staged.relative_files),
                    "attempt_id": staged.attempt_id,
                    "ready_at": time.time(),
                },
                directory,
                rank_ready_marker(self._rank),
            )
            if self._rank == 0:
                if not self._wait_for_ranks(
                    directory,
                    staged.step,
                    attempt_id=staged.attempt_id,
                ):
                    self.failed_steps.append(staged.step)
                    return
                self._store.write_json(
                    {
                        "step": staged.step,
                        "attempt_id": staged.attempt_id,
                        "committed_at": time.time(),
                    },
                    directory,
                    COMMIT_MARKER,
                )
                self._store.write_json(
                    {
                        "step": staged.step,
                        "directory": directory,
                        "uri": self._store.uri(directory),
                        "world_size": self._world_size,
                        "lineage_id": staged.lineage_id,
                        "generation": staged.generation,
                        "sparse_kind": staged.sparse_kind,
                        "data_window": (
                            None
                            if staged.data_window is None
                            else staged.data_window.to_payload()
                        ),
                        "window_complete": staged.window_complete,
                        "trained_through": staged.trained_through,
                        "updated_at": time.time(),
                        "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    },
                    LATEST_POINTER,
                )
                prune_run_directory(self._store, self._keep_last)
            self.published_steps.append(staged.step)
            succeeded = True
            if self._rank == 0:
                print(
                    f"Checkpoint committed | step={staged.step} "
                    f"uri={self._store.uri(directory)}",
                    flush=True,
                )
            logger.info(
                "checkpoint step %d published to %s",
                staged.step,
                self._store.uri(directory),
            )
        except Exception as error:  # noqa: BLE001 - a failed upload must not kill training
            self.failed_steps.append(staged.step)
            logger.error(
                "checkpoint step %d failed to publish to %s: %s",
                staged.step,
                self._store.uri(directory),
                error,
            )
        finally:
            self._cleanup(staged)
            self._finish_step(staged.step, succeeded=succeeded)

    def _wait_for_ranks(
        self,
        directory: str,
        step: int,
        *,
        attempt_id: str | None,
    ) -> bool:
        if self._world_size <= 1:
            return True
        expected = {rank_ready_marker(rank) for rank in range(self._world_size)}
        deadline = time.monotonic() + self._ready_timeout_sec
        while True:
            present: set[str] = set()
            for entry in self._store.list_entries(directory):
                if entry.name not in expected:
                    continue
                if attempt_id is not None:
                    try:
                        marker = self._store.read_json(directory, entry.name)
                    except Exception:
                        continue
                    if marker.get("attempt_id") != attempt_id:
                        continue
                present.add(entry.name)
            missing = expected - present
            if not missing:
                return True
            if time.monotonic() > deadline or self._stopping.is_set():
                logger.error(
                    "checkpoint step %d not committed: %d rank file set(s) missing "
                    "after %.0fs (%s)",
                    step,
                    len(missing),
                    self._ready_timeout_sec,
                    ", ".join(sorted(missing)[:4]),
                )
                return False
            time.sleep(self._poll_interval_sec)

    def _cleanup(self, staged: StagedCheckpoint) -> None:
        with self._streamed_lock:
            self._streamed.pop(staged.step, None)
        with self._progress_lock:
            self._uploaded_bytes.pop(staged.step, None)
            self._last_progress_log_at.pop(staged.step, None)
        if not staged.cleanup_staging:
            return
        shutil.rmtree(staged.staging_dir, ignore_errors=True)

    def _cleanup_aborted(self, aborted: _AbortedCheckpoint) -> None:
        with self._streamed_lock:
            self._streamed.pop(aborted.step, None)
        with self._progress_lock:
            self._uploaded_bytes.pop(aborted.step, None)
            self._last_progress_log_at.pop(aborted.step, None)
        if aborted.cleanup_staging:
            shutil.rmtree(aborted.staging_dir, ignore_errors=True)
        self._finish_step(aborted.step, succeeded=False)


# --- Resuming ---


@dataclass(frozen=True)
class ResumedTrainingState:
    """What a restarted run recovered from a committed checkpoint."""

    step: int
    rows: int
    world_size: int
    source_uri: str
    data_cursor: DataCursor | None = None
    lineage_id: str | None = None
    generation: int = 0
    sparse_kind: str = "full"
    parent_checkpoint: str | None = None
    data_window: CheckpointDataWindow | None = None
    window_complete: bool = False
    trained_through: str | None = None


_LOCAL_SPARSE_CHAIN = ".sparse-chain.json"


def _fetch_one_checkpoint_for_rank(
    store: CheckpointStore,
    directory: str,
    local_dir: Path,
    *,
    rank: int,
    world_size: int,
    sparse_only: bool = False,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    manifest = store.read_json(directory, CHECKPOINT_MANIFEST)
    saved_world_size = int(manifest.get("world_size", 1))
    if saved_world_size != world_size or not manifest.get("sharded_embeddings", False):
        download_tree(store, local_dir, directory)
        return

    wanted = [
        CHECKPOINT_MANIFEST,
        f"{MODEL_SUBDIR}/manifest.json",
    ]
    if not sparse_only:
        wanted.extend(
            (
                TRAIN_STATE_FILE,
                rank_progress_file(rank),
                f"{MODEL_SUBDIR}/dense.pt",
            )
        )
    for relative in wanted:
        parts = relative.split("/")
        if not store.exists(directory, *parts):
            continue
        store.download_file(local_dir / Path(*parts), directory, *parts)

    model_manifest_path = local_dir / MODEL_SUBDIR / "manifest.json"
    if not model_manifest_path.exists():
        return
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if model_manifest.get("format") == SHARDED_SPARSE_STREAM_FORMAT:
        rank_manifests = list(model_manifest.get("rank_manifests", ()))
        names = [rank_manifests[rank]]
        for name in names:
            store.download_file(
                local_dir / MODEL_SUBDIR / name,
                directory,
                MODEL_SUBDIR,
                name,
            )
            rank_manifest = json.loads(
                (local_dir / MODEL_SUBDIR / name).read_text(encoding="utf-8")
            )
            for table in rank_manifest.get("tables", {}).values():
                for segment in table.get("segments", ()):
                    segment_name = str(segment["file"])
                    store.download_file(
                        local_dir / MODEL_SUBDIR / segment_name,
                        directory,
                        MODEL_SUBDIR,
                        segment_name,
                    )
        return
    if sparse_only and store.exists(directory, MODEL_SUBDIR, "dense.pt"):
        # Legacy bases cannot load sparse rows independently; preserve their
        # dense file for migration. Streaming full bases take the sparse-only
        # path above and avoid this download.
        store.download_file(
            local_dir / MODEL_SUBDIR / "dense.pt",
            directory,
            MODEL_SUBDIR,
            "dense.pt",
        )
    for name in shard_file_names(
        model_manifest,
        rank=rank,
        world_size=world_size,
    ):
        store.download_file(
            local_dir / MODEL_SUBDIR / name,
            directory,
            MODEL_SUBDIR,
            name,
        )


def fetch_checkpoint_for_rank(
    store: CheckpointStore,
    checkpoint: CommittedCheckpoint,
    destination: str | Path,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> Path:
    """Materialize the files this rank needs to resume, and return the local dir.

    Same-size restarts copy only this rank's embedding shard. A resharding
    restart needs every saved shard, so the whole step is fetched instead.
    """

    local_dir = Path(destination)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Walk only sparse dependencies. Dense parameters and control/optimizer
    # state are full in the target checkpoint, while a delta sparse snapshot
    # needs its nearest full base and intervening deltas.
    chain = [checkpoint.directory]
    seen = {checkpoint.directory}
    current = checkpoint.directory
    while True:
        manifest = store.read_json(current, CHECKPOINT_MANIFEST)
        if current == checkpoint.directory:
            expected_lineage = manifest.get("lineage_id")
        elif (
            expected_lineage is not None
            and manifest.get("lineage_id") not in {None, expected_lineage}
        ):
            raise ValueError(
                f"checkpoint sparse chain crosses lineage at {current}"
            )
        parent = manifest.get("sparse_parent_checkpoint")
        if not parent:
            if str(manifest.get("sparse_kind", "full")) == "delta":
                raise ValueError(
                    f"checkpoint {current} is a sparse delta without a full base"
                )
            break
        parent = str(parent)
        if parent in seen:
            raise ValueError(f"checkpoint sparse-parent cycle at {parent}")
        if not store.exists(parent, COMMIT_MARKER):
            raise FileNotFoundError(
                f"checkpoint {current} depends on uncommitted/missing {parent}"
            )
        seen.add(parent)
        chain.append(parent)
        current = parent
    chain.reverse()

    local_chain: list[str] = []
    for directory in chain:
        target = (
            local_dir
            if directory == checkpoint.directory
            else local_dir / "_sparse_chain" / directory
        )
        _fetch_one_checkpoint_for_rank(
            store,
            directory,
            target,
            rank=rank,
            world_size=world_size,
            sparse_only=directory != checkpoint.directory,
        )
        local_chain.append(os.path.relpath(target, local_dir))
    if len(local_chain) > 1:
        _atomic_json_save({"paths": local_chain}, local_dir / _LOCAL_SPARSE_CHAIN)
    return local_dir


def load_training_checkpoint(
    config: AppConfig,
    model: nn.Module,
    local_dir: str | Path,
    *,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    dense_optimizer: torch.optim.Optimizer | None = None,
    replicated_sparse_optimizer: torch.optim.Optimizer | None = None,
    sharded_optimizer: ShardedAdagrad | ShardedRowWiseAdagrad | None = None,
    process_group: torch_dist.ProcessGroup | None = None,
    source_uri: str = "",
) -> ResumedTrainingState:
    """Restore weights, optimizer state, step, and this rank's data cursor."""

    checkpoint_dir = Path(local_dir)
    manifest = json.loads(
        (checkpoint_dir / CHECKPOINT_MANIFEST).read_text(encoding="utf-8")
    )
    if manifest.get("format") != TRAINING_CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported training checkpoint format {manifest.get('format')!r}"
        )
    _validate_checkpoint_metadata(config, manifest)

    chain_path = checkpoint_dir / _LOCAL_SPARSE_CHAIN
    if chain_path.exists():
        chain_payload = json.loads(chain_path.read_text(encoding="utf-8"))
        chain_dirs = [checkpoint_dir / str(item) for item in chain_payload["paths"]]
        if not chain_dirs or chain_dirs[-1].resolve() != checkpoint_dir.resolve():
            raise ValueError("local sparse checkpoint chain does not end at target")
        base_manifest = json.loads(
            (chain_dirs[0] / CHECKPOINT_MANIFEST).read_text(encoding="utf-8")
        )
        base_model_path = chain_dirs[0] / str(base_manifest["model_path"])
        base_model_manifest_path = base_model_path / "manifest.json"
        base_model_manifest = (
            json.loads(base_model_manifest_path.read_text(encoding="utf-8"))
            if base_model_manifest_path.exists()
            else {}
        )
        if base_model_manifest.get("format") == SHARDED_SPARSE_STREAM_FORMAT:
            _load_streaming_sharded_checkpoint(
                config,
                model,
                base_model_path,
                device,
                process_group,
                sharded_optimizer=sharded_optimizer,
                load_dense=False,
            )
        else:
            load_model_checkpoint(
                config,
                model,
                base_model_path,
                device=device,
                process_group=process_group,
                sharded_optimizer=sharded_optimizer,
            )
        for item_dir in chain_dirs[1:]:
            item_manifest = json.loads(
                (item_dir / CHECKPOINT_MANIFEST).read_text(encoding="utf-8")
            )
            model_path = item_dir / str(item_manifest["model_path"])
            model_manifest = json.loads(
                (model_path / "manifest.json").read_text(encoding="utf-8")
            )
            if model_manifest.get("format") != SHARDED_SPARSE_STREAM_FORMAT:
                raise ValueError(
                    "a sparse delta chain contains a non-streaming child checkpoint"
                )
            _load_streaming_sharded_checkpoint(
                config,
                model,
                model_path,
                device,
                process_group,
                sharded_optimizer=sharded_optimizer,
                load_dense=False,
            )
        # Dense is a full rank-0 snapshot at every generation, so the target
        # replaces the base dense state after sparse deltas have been replayed.
        target_model_path = checkpoint_dir / str(manifest["model_path"])
        target_model_manifest = json.loads(
            (target_model_path / "manifest.json").read_text(encoding="utf-8")
        )
        _load_dense_sharded_state(
            config,
            model,
            target_model_path,
            target_model_manifest,
            device,
        )
    else:
        model_path = checkpoint_dir / str(manifest["model_path"])
        model_manifest_path = model_path / "manifest.json"
        if model_manifest_path.exists():
            model_manifest = json.loads(
                model_manifest_path.read_text(encoding="utf-8")
            )
            if (
                model_manifest.get("format") == SHARDED_SPARSE_STREAM_FORMAT
                and model_manifest.get("kind") == "delta"
            ):
                raise ValueError(
                    "cannot restore a sparse delta without its full dependency chain"
                )
        load_model_checkpoint(
            config,
            model,
            model_path,
            device=device,
            process_group=process_group,
            sharded_optimizer=sharded_optimizer,
        )

    train_state = torch.load(
        checkpoint_dir / TRAIN_STATE_FILE,
        map_location=device,
    )
    _validate_checkpoint_metadata(config, train_state)
    dense_state = train_state.get("dense_optimizer")
    if dense_optimizer is not None and dense_state is not None:
        dense_optimizer.load_state_dict(dense_state)
    replicated_state = train_state.get("replicated_sparse_optimizer")
    if replicated_sparse_optimizer is not None and replicated_state is not None:
        replicated_sparse_optimizer.load_state_dict(replicated_state)

    step = int(train_state.get("step", manifest.get("step", 0)))
    saved_world_size = int(manifest.get("world_size", 1))
    rows = 0
    cursor: DataCursor | None = None
    progress_path = checkpoint_dir / rank_progress_file(rank)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        rows = int(progress.get("rows", 0))
        cursor = DataCursor.from_payload(progress.get("data_cursor"))
    if cursor is not None and saved_world_size != world_size:
        # Shard assignment is a function of world size; a cursor recorded under a
        # different topology points at files this rank no longer owns.
        logger.warning(
            "checkpoint was saved with world_size=%d but this run uses %d; "
            "input scan restarts from the beginning of each rank's shard",
            saved_world_size,
            world_size,
        )
        cursor = None
    if bool(manifest.get("window_complete", False)):
        # A completed window resumes at the next window's first partition, not
        # at the final prefetched position of the window that just committed.
        cursor = None
    return ResumedTrainingState(
        step=step,
        rows=rows,
        world_size=saved_world_size,
        source_uri=source_uri or str(checkpoint_dir),
        data_cursor=cursor,
        lineage_id=manifest.get("lineage_id"),
        generation=int(manifest.get("generation", 0)),
        sparse_kind=str(manifest.get("sparse_kind", "full")),
        parent_checkpoint=manifest.get("parent_checkpoint"),
        data_window=CheckpointDataWindow.from_payload(manifest.get("data_window")),
        window_complete=bool(manifest.get("window_complete", False)),
        trained_through=manifest.get("trained_through"),
    )


def open_run_store(base_uri: str, run_name: str | None = None) -> CheckpointStore:
    """Return the store for one run directory, creating it when missing."""

    store = open_checkpoint_store(base_uri)
    if run_name:
        store = store.child(run_name)
    store.makedirs()
    return store
