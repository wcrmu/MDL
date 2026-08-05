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
import json
import logging
import math
import os
from pathlib import Path
import queue
import shutil
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

SHARDED_CHECKPOINT_FORMAT = "mdl_sharded_embedding_v1"
# v2 splits one rank's tables across several bounded-size files so neither the
# CPU copy nor the staging directory ever holds a whole shard at once.
SHARDED_CHECKPOINT_CHUNK_FORMAT = "mdl_sharded_embedding_v2"
_SHARDED_CHECKPOINT_READABLE = frozenset(
    {SHARDED_CHECKPOINT_FORMAT, SHARDED_CHECKPOINT_CHUNK_FORMAT}
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
) -> StagingSpaceEstimate:
    """Predict this rank's staging footprint before anything is written.

    ``peak_bytes`` assumes published files are deleted as the upload drains, so
    it is what a healthy save actually needs. ``total_bytes`` is the fallback
    when uploads cannot keep up and every file stays until the step finishes.
    """

    modules = sharded_embedding_modules(model)
    chunk_sizes: list[int] = []
    if modules:
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
                    or int(table["padding_idx"]) != module.padding_idx
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
        try:
            manifest = store.read_json(name, CHECKPOINT_MANIFEST)
            world_size = int(manifest.get("world_size", 0))
            saved_at = float(manifest.get("saved_at", 0.0))
        except Exception:  # noqa: BLE001 - a readable commit marker is enough
            pass
        committed.append(
            CommittedCheckpoint(
                step=step,
                directory=name,
                uri=store.uri(name),
                world_size=world_size,
                saved_at=saved_at,
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
) -> CommittedCheckpoint | None:
    """Interpret ``training.checkpoint.resume`` against a run directory.

    ``auto``/``latest`` pick the newest committed step, ``none`` disables
    resuming, and an integer or ``step-000012000`` name pins one step.
    """

    text = (spec or "auto").strip()
    if text.lower() in {"none", "off", "false", ""}:
        return None
    if text.lower() in {"auto", "latest"}:
        return latest_committed_checkpoint(store)
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
        doomed.extend(name for _step, name in committed[: len(committed) - keep_last])
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
                "model_path": model_relpath,
                "sharded_embeddings": sharded,
                "run_name": run_name,
                "saved_at": time.time(),
                "saved_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "elapsed_seconds": float(elapsed_seconds),
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
    )


# --- Publishing staged steps to the run directory ---


@dataclass(frozen=True)
class _StagedFile:
    """One staged file handed to the uploader before its step finished."""

    step: int
    directory: str
    relative: str
    source: Path


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
        self._stream_window = max(0, int(stream_window))
        self._stream_timeout_sec = max(0.0, float(stream_timeout_sec))
        self._queue: queue.Queue = queue.Queue(
            maxsize=max(1, int(max_pending)) + self._stream_window
        )
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        # Names that already reached the run directory, so the publish pass can
        # skip them. Ordering on the queue guarantees this is complete before the
        # owning step is finalized; the lock only covers synchronous uploads and
        # dropped steps, which run on the caller's thread.
        self._streamed: dict[int, set[str]] = {}
        self._streamed_lock = threading.Lock()
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
        ``enabled=False`` and keep every file. When the queue stays full past
        ``stream_timeout_sec`` the sink declines, which leaves the file for the
        normal publish pass rather than blocking the step loop forever.
        """

        directory = step_directory_name(step)
        with self._streamed_lock:
            # A step that failed while staging never reaches ``_publish``, which
            # is what normally clears its names. Reaching a later step means no
            # earlier one can still be finalized.
            for earlier in [key for key in self._streamed if key < int(step)]:
                del self._streamed[earlier]

        def _publish_file(relative: str) -> bool:
            if not enabled:
                return False
            item = _StagedFile(
                step=int(step),
                directory=directory,
                relative=relative,
                source=Path(staging_dir) / relative,
            )
            if not self._asynchronous:
                return self._upload_staged_file(item)
            deadline = time.monotonic() + self._stream_timeout_sec
            while True:
                try:
                    self._queue.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    if heartbeat is not None:
                        heartbeat()
                    if time.monotonic() >= deadline:
                        logger.warning(
                            "checkpoint step %d kept %s in staging: the run "
                            "directory is not draining within %.0fs",
                            step,
                            relative,
                            self._stream_timeout_sec,
                        )
                        return False

        return _publish_file

    def submit(self, staged: StagedCheckpoint) -> bool:
        """Queue a staged step. Returns False when it was dropped for backpressure."""

        if not self._asynchronous:
            self._publish(staged)
            return True
        try:
            # Streamed files share the queue, so a momentarily full queue is
            # normal; only a genuinely stuck uploader should drop a step.
            self._queue.put(staged, timeout=self._stream_timeout_sec)
        except queue.Full:
            # Dropping is safer than blocking the training loop: the next
            # cadence writes a fresh step, and the stale staging dir is removed.
            self.dropped_steps.append(staged.step)
            logger.warning(
                "checkpoint step %d dropped: a previous upload to %s is still "
                "running; increase training.checkpoint.every_steps",
                staged.step,
                self._store.root_uri,
            )
            self._cleanup(staged)
            return False
        return True

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
                    self._upload_staged_file(item)
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
        try:
            self._store.makedirs(directory)
            for relative in staged.relative_files:
                if relative in already:
                    continue
                self._store.upload_file(
                    staged.staging_dir / relative,
                    directory,
                    *relative.split("/"),
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
                    "ready_at": time.time(),
                },
                directory,
                rank_ready_marker(self._rank),
            )
            if self._rank == 0:
                if not self._wait_for_ranks(directory, staged.step):
                    self.failed_steps.append(staged.step)
                    return
                self._store.write_json(
                    {"step": staged.step, "committed_at": time.time()},
                    directory,
                    COMMIT_MARKER,
                )
                self._store.write_json(
                    {
                        "step": staged.step,
                        "directory": directory,
                        "uri": self._store.uri(directory),
                        "world_size": self._world_size,
                        "updated_at": time.time(),
                        "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    },
                    LATEST_POINTER,
                )
                prune_run_directory(self._store, self._keep_last)
            self.published_steps.append(staged.step)
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

    def _wait_for_ranks(self, directory: str, step: int) -> bool:
        if self._world_size <= 1:
            return True
        expected = {rank_ready_marker(rank) for rank in range(self._world_size)}
        deadline = time.monotonic() + self._ready_timeout_sec
        while True:
            present = {entry.name for entry in self._store.list_entries(directory)}
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
        if not staged.cleanup_staging:
            return
        shutil.rmtree(staged.staging_dir, ignore_errors=True)


# --- Resuming ---


@dataclass(frozen=True)
class ResumedTrainingState:
    """What a restarted run recovered from a committed checkpoint."""

    step: int
    rows: int
    world_size: int
    source_uri: str
    data_cursor: DataCursor | None = None


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
    manifest = store.read_json(checkpoint.directory, CHECKPOINT_MANIFEST)
    saved_world_size = int(manifest.get("world_size", 1))
    if saved_world_size != world_size or not manifest.get("sharded_embeddings", False):
        download_tree(store, local_dir, checkpoint.directory)
        return local_dir

    wanted = [
        CHECKPOINT_MANIFEST,
        TRAIN_STATE_FILE,
        rank_progress_file(rank),
        f"{MODEL_SUBDIR}/manifest.json",
        f"{MODEL_SUBDIR}/dense.pt",
    ]
    for relative in wanted:
        parts = relative.split("/")
        if not store.exists(checkpoint.directory, *parts):
            continue
        store.download_file(local_dir / Path(*parts), checkpoint.directory, *parts)

    # Which embedding files belong to this rank depends on the layout the save
    # used, so the model manifest decides rather than a name built here.
    model_manifest_path = local_dir / MODEL_SUBDIR / "manifest.json"
    if model_manifest_path.exists():
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        for name in shard_file_names(
            model_manifest,
            rank=rank,
            world_size=world_size,
        ):
            store.download_file(
                local_dir / MODEL_SUBDIR / name,
                checkpoint.directory,
                MODEL_SUBDIR,
                name,
            )
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

    load_model_checkpoint(
        config,
        model,
        checkpoint_dir / str(manifest["model_path"]),
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
    return ResumedTrainingState(
        step=step,
        rows=rows,
        world_size=saved_world_size,
        source_uri=source_uri or str(checkpoint_dir),
        data_cursor=cursor,
    )


def open_run_store(base_uri: str, run_name: str | None = None) -> CheckpointStore:
    """Return the store for one run directory, creating it when missing."""

    store = open_checkpoint_store(base_uri)
    if run_name:
        store = store.child(run_name)
    store.makedirs()
    return store
