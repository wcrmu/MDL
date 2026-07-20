from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from importlib import import_module
import inspect
import logging
import math
import os
from pathlib import Path
import queue
import sqlite3
import tempfile
import threading
from time import perf_counter
from typing import Any, Callable, Iterator

import torch
import torch.distributed as torch_dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from .config import (
    AppConfig,
    DDPConfig,
    ParquetSplitConfig,
    QuickEvalConfig,
    ReaderConfig,
)
from .checkpoint import load_model_checkpoint, save_model_checkpoint
from .dataloader import (
    FeatureBatch,
    _column_array,
    _require_pyarrow,
    _safe_table_take,
    discover_scenario_values,
    iter_flat_tables,
    move_feature_batch,
    pin_feature_batch,
    resolve_auto_scenarios,
    table_to_feature_batch,
)
from .features import load_vocab_maps
from .embeddings import (
    ShardedEmbedding,
    consume_sharded_embedding_stats,
    sharded_embedding_modules,
)
from .model import build_model
from .modules.attention import varlen_attention_available
from .modules.mlp import SparseMoEPerTokenFFN
from .optim import ShardedAdagrad, ShardedRowWiseAdagrad


logger = logging.getLogger(__name__)

_CONTROL_PROCESS_GROUP: torch_dist.ProcessGroup | None = None
# Cold scenario discovery on HDFS can exceed the default 10-minute store wait
# before the first collective. Keep process-group timeouts generous so peers
# survive a slow rank-0 metadata pass; discovery itself is also optimized.
_PROCESS_GROUP_TIMEOUT = timedelta(minutes=60)


def _varlen_attention_reasons(config: AppConfig) -> tuple[str, ...]:
    """Human-readable reasons this config needs ``torch.nn.attention.varlen``."""

    reasons: list[str] = []
    if config.model.name in {"longer", "onetrans", "mdl_onetrans"}:
        reasons.append(f"model={config.model.name}")
    longer_sequences = [
        sequence.name
        for sequence in config.sequences
        if sequence.encoder == "longer"
    ]
    if longer_sequences:
        reasons.append("LONGER sequences=" + ",".join(longer_sequences))
    return tuple(reasons)


def _requires_varlen_attention(config: AppConfig) -> bool:
    """True when strict flash would execute the packed Varlen Flash path."""

    return bool(_varlen_attention_reasons(config))


def _needs_padded_sdpa_flash(config: AppConfig) -> bool:
    """True when strict flash would also exercise ordinary padded SDPA Flash.

    LONGER / OneTrans S-streams use Varlen. Ordinary padded FlashAttention is
    only required when MDL constructs ``DomainAwareAttention`` for enabled
    task/scenario feature interactions. Plain RankMixer token mixing does not
    use padded Flash, so both capabilities are independent.
    """

    model = config.model
    if model.name not in {"mdl_rankmixer", "mdl_onetrans"}:
        return False
    task_attention_enabled = (
        model.use_task_tokens and model.use_task_feature_interaction
    )
    scenario_attention_enabled = (
        model.use_scenario_tokens and model.use_scenario_feature_interaction
    )
    return task_attention_enabled or scenario_attention_enabled


def _ordinary_sdpa_flash_available() -> bool:
    """True when this PyTorch/GPU build reports padded SDPA Flash available."""

    return bool(
        getattr(
            torch.backends.cuda,
            "is_flash_attention_available",
            lambda: False,
        )()
    )


def _attention_runtime_description(
    config: AppConfig,
    device: torch.device,
) -> str:
    """Validate requested attention backend against local capabilities.

    Entry points must call this after the device is known and before scenario
    discovery, model construction, or synthetic data generation.
    """

    requested = getattr(config.runtime, "attention_backend", "auto")
    reasons = _varlen_attention_reasons(config)
    needs_varlen = _requires_varlen_attention(config)
    needs_padded = _needs_padded_sdpa_flash(config)
    varlen_api = varlen_attention_available()
    if requested == "flash":
        if device.type != "cuda":
            raise RuntimeError(
                "runtime.attention_backend='flash' requires CUDA, but the resolved "
                f"device is {device}"
            )
        if needs_varlen and not varlen_api:
            detail = "; ".join(reasons) if reasons else "configured Varlen Flash paths"
            raise RuntimeError(
                "runtime.attention_backend='flash' requires "
                "torch.nn.attention.varlen.varlen_attn for "
                f"{detail}, but the API is unavailable in "
                f"torch={torch.__version__}. Use runtime.attention_backend='sdpa' "
                "on this platform, or install a CUDA-enabled PyTorch build that "
                "exposes torch.nn.attention.varlen.varlen_attn and validate it on "
                "the target GPU (PyTorch 2.10+ is normally required)."
            )
        if needs_padded and not _ordinary_sdpa_flash_available():
            raise RuntimeError(
                "runtime.attention_backend='flash' was requested, but this "
                "PyTorch/GPU build reports the ordinary SDPA FlashAttention "
                "backend unavailable"
            )
        if needs_varlen and needs_padded:
            implementation = "varlen_flash+padded_sdpa_flash"
        elif needs_varlen:
            implementation = "varlen_flash"
        else:
            implementation = "padded_sdpa_flash"
        return (
            f"requested=flash resolved={implementation} "
            f"flash_path_requires_varlen={needs_varlen} "
            f"flash_path_requires_padded_sdpa={needs_padded} "
            f"varlen_api_available={varlen_api} "
            f"strict=true device={device} precision={config.runtime.precision}"
        )
    return (
        f"requested={requested} resolved=padded_sdpa "
        f"kernel_policy=runtime_dispatch "
        f"flash_path_requires_varlen={needs_varlen} "
        f"flash_path_requires_padded_sdpa={needs_padded} "
        f"varlen_api_available={varlen_api} "
        f"strict=false device={device} precision={config.runtime.precision}"
    )


# Public aliases for benchmark / tuner / tests.
attention_runtime_description = _attention_runtime_description
needs_varlen_flash = _requires_varlen_attention
needs_padded_sdpa_flash = _needs_padded_sdpa_flash
ordinary_sdpa_flash_available = _ordinary_sdpa_flash_available
requires_varlen_attention = _requires_varlen_attention
varlen_attention_reasons = _varlen_attention_reasons


@dataclass(frozen=True)
class TrainResult:
    steps: int
    last_loss: float
    rows: int = 0
    elapsed_seconds: float = 0.0

    @property
    def steps_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.steps / self.elapsed_seconds

    @property
    def rows_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.rows / self.elapsed_seconds


@dataclass(frozen=True)
class TrainStepTrace:
    """Synchronized timings for one training step.

    Tracing is opt-in because collecting phase timings synchronizes the device at
    phase boundaries. Normal training therefore pays no synchronization or
    callback overhead. Performance benchmarks use these traces for stable p95
    and dataloader-wait measurements while aggregate throughput continues to be
    reported separately by :class:`TrainResult`.
    """

    step: int
    rank_active: bool
    active_ranks: int
    rows: int
    input_tokens: int
    padded_token_slots: int
    step_seconds: float
    dataloader_wait_seconds: float
    h2d_seconds: float
    forward_seconds: float
    backward_seconds: float
    sparse_sync_seconds: float
    optimizer_seconds: float
    sparse_local_rows: int
    sparse_global_rows: int
    sparse_payload_bytes: int


TrainStepObserver = Callable[[TrainStepTrace], None]


@dataclass(frozen=True)
class PredictResult:
    rows: int
    output_path: Path | None


@dataclass(frozen=True)
class EvaluateResult:
    rows: int
    group_metric_name: str | None
    metrics: dict[str, dict[str, float | int | None]]
    auc_histogram_bins: int = 65536


@dataclass(frozen=True)
class QuickEvalResult:
    rows: int
    metrics: dict[str, dict[str, float | int | None]]
    elapsed_seconds: float


ExternalTrainAdapter = Callable[..., TrainResult | dict[str, Any]]


class _NoOpGradScaler:
    def is_enabled(self) -> bool:
        return False


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool = False
    control_group: torch_dist.ProcessGroup | None = None


@dataclass(frozen=True)
class _NamedSparseParameter:
    """A row-sparse embedding parameter with its stable model name."""

    name: str
    parameter: nn.Parameter


@dataclass(frozen=True)
class _ParameterGroups:
    """Optimizer ownership plus the COO subset that bypasses DDP reduction."""

    dense_optimizer: tuple[nn.Parameter, ...]
    embedding_optimizer: tuple[nn.Parameter, ...]
    sparse_sync: tuple[_NamedSparseParameter, ...]
    sharded_optimizer: tuple[nn.Parameter, ...] = ()
    sharded_ddp_ignore: tuple[_NamedSparseParameter, ...] = ()


@dataclass(frozen=True)
class _SparseSyncStats:
    local_rows: int = 0
    global_rows: int = 0
    logical_payload_bytes: int = 0


@dataclass(frozen=True)
class _SparseTableSpec:
    ref: _NamedSparseParameter
    row_offset: int


@dataclass(frozen=True)
class _SparseGroupSpec:
    embedding_dim: int
    dtype: torch.dtype
    tables: tuple[_SparseTableSpec, ...]
    total_rows: int


class _DDPGraphAuditor:
    """Observe representative reducer participation without changing policy."""

    def __init__(
        self,
        model: nn.Module,
        *,
        ignored_parameter_ids: set[int],
        max_steps: int,
    ) -> None:
        self.parameters = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in ignored_parameter_ids
        )
        self.max_steps = max_steps
        self.patterns: list[tuple[str, ...]] = []

    def observe(self) -> None:
        if len(self.patterns) >= self.max_steps:
            return
        self.patterns.append(
            tuple(
                name
                for name, parameter in self.parameters
                if parameter.grad is None
            )
        )

    def report(self, context: DistributedContext) -> str | None:
        if not context.enabled or not self.patterns:
            return None
        gathered: list[list[tuple[str, ...]]] = [
            [] for _ in range(context.world_size)
        ]
        torch_dist.all_gather_object(gathered, self.patterns)
        all_patterns = [pattern for rank_patterns in gathered for pattern in rank_patterns]
        unused = sorted({name for pattern in all_patterns for name in pattern})
        stable = bool(all_patterns) and all(
            pattern == all_patterns[0] for pattern in all_patterns[1:]
        )
        if unused:
            recommendation = (
                "candidate_static_graph_after_extended_validation"
                if stable
                else "keep_safe_find_unused"
            )
        else:
            recommendation = (
                "candidate_find_unused_false_or_static_graph_after_extended_validation"
                if stable
                else "keep_safe_find_unused"
            )
        return (
            f"observed_rank_steps={len(all_patterns)} usage_stable={str(stable).lower()} "
            f"unused_count={len(unused)} recommendation={recommendation} "
            f"unused={','.join(unused[:20]) or '-'}"
        )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def distributed_rank() -> int:
    return _env_int("RANK", 0)


def is_main_process() -> bool:
    return distributed_rank() == 0


def _select_device(config: AppConfig, local_rank: int | None = None) -> torch.device:
    requested = config.runtime.device
    if requested.startswith("cuda") and torch.cuda.is_available():
        if local_rank is not None:
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device(requested)
    if requested != "cpu" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _setup_distributed(config: AppConfig) -> DistributedContext:
    global _CONTROL_PROCESS_GROUP
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    enabled = world_size > 1
    device = _select_device(config, local_rank if enabled else None)
    initialized_here = False

    if enabled and not torch_dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch_dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=_PROCESS_GROUP_TIMEOUT,
        )
        initialized_here = True

    control_group: torch_dist.ProcessGroup | None = None
    if enabled and device.type == "cuda":
        if _CONTROL_PROCESS_GROUP is None:
            try:
                _CONTROL_PROCESS_GROUP = torch_dist.new_group(
                    backend="gloo",
                    timeout=_PROCESS_GROUP_TIMEOUT,
                )
            except TypeError:
                # Older PyTorch builds reject timeout= on new_group.
                try:
                    _CONTROL_PROCESS_GROUP = torch_dist.new_group(backend="gloo")
                except RuntimeError as error:
                    logger.warning(
                        "Could not create a CPU control process group; active-rank "
                        "coordination will synchronize CUDA: %s",
                        error,
                    )
            except RuntimeError as error:
                logger.warning(
                    "Could not create a CPU control process group; active-rank "
                    "coordination will synchronize CUDA: %s",
                    error,
                )
        control_group = _CONTROL_PROCESS_GROUP

    return DistributedContext(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        initialized_here=initialized_here,
        control_group=control_group,
    )


def _cleanup_distributed(context: DistributedContext) -> None:
    global _CONTROL_PROCESS_GROUP
    if context.initialized_here and torch_dist.is_initialized():
        if _CONTROL_PROCESS_GROUP is not None:
            torch_dist.destroy_process_group(_CONTROL_PROCESS_GROUP)
            _CONTROL_PROCESS_GROUP = None
        torch_dist.destroy_process_group()


def _resolve_distributed_auto_scenarios(
    config: AppConfig,
    context: DistributedContext,
) -> AppConfig:
    """Discover on rank 0 and broadcast one stable raw-scene ordering."""

    scenarios = getattr(config, "scenarios", None)
    if scenarios is None or not getattr(scenarios, "auto_discover", False):
        return config
    if not context.enabled:
        resolved = resolve_auto_scenarios(config)
        logger.info("Discovered raw scene_id values: %s", resolved.scenarios.names)
        return resolved

    payload: list[dict[str, Any] | None] = [None]
    if context.rank == 0:
        try:
            payload[0] = {
                "values": list(discover_scenario_values(config)),
                "error": None,
            }
        except Exception as error:  # Broadcast failure instead of hanging peers.
            payload[0] = {"values": None, "error": str(error)}
    # Prefer the CPU control group so peers wait on gloo while rank 0 scans
    # Parquet, instead of lazily initializing NCCL before discovery finishes.
    if context.control_group is not None:
        torch_dist.broadcast_object_list(payload, src=0, group=context.control_group)
    else:
        torch_dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not isinstance(result, dict):
        raise RuntimeError("distributed scenario discovery broadcast an invalid payload")
    error = result.get("error")
    if error:
        raise RuntimeError(f"automatic scenario discovery failed: {error}")
    values = result.get("values")
    if not isinstance(values, list):
        raise RuntimeError("automatic scenario discovery did not broadcast a value list")
    resolved = resolve_auto_scenarios(config, values)
    if context.rank == 0:
        logger.info("Discovered raw scene_id values: %s", resolved.scenarios.names)
    return resolved


def _load_external_train_adapter(dotted_path: str | None) -> ExternalTrainAdapter:
    if not dotted_path:
        raise ValueError("external sparse parameter-server training requires an adapter dotted path")
    module_name, separator, attribute_name = dotted_path.partition(":")
    if not separator:
        module_name, separator, attribute_name = dotted_path.rpartition(".")
    if not module_name or not attribute_name:
        raise ValueError(
            "training.sparse_parameter_server_adapter must be 'package.module:function' "
            "or 'package.module.function'"
        )
    module = import_module(module_name)
    adapter = getattr(module, attribute_name)
    if not callable(adapter):
        raise TypeError(f"sparse parameter-server adapter {dotted_path!r} is not callable")
    return adapter


def _coerce_train_result(result: TrainResult | dict[str, Any]) -> TrainResult:
    if isinstance(result, TrainResult):
        return result
    if isinstance(result, dict):
        return TrainResult(
            steps=int(result.get("steps", 0)),
            last_loss=float(result.get("last_loss", 0.0)),
            rows=int(result.get("rows", 0)),
            elapsed_seconds=float(result.get("elapsed_seconds", 0.0)),
        )
    raise TypeError("external training adapter must return TrainResult or a dict")


def iter_candidate_tables(
    config: AppConfig,
    split_name: str,
    shard_rank: int = 0,
    shard_world_size: int = 1,
    require_labels: bool = True,
) -> Iterator[object]:
    yield from iter_flat_tables(
        config,
        split_name,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
        require_labels=require_labels,
    )


def _slice_table(table: object, batch_size: int) -> Iterator[object]:
    for offset in range(0, table.num_rows, batch_size):
        yield table.slice(offset, batch_size)


def _shuffle_table(table: object, generator: torch.Generator) -> object:
    if table.num_rows <= 1:
        return table
    permutation = torch.randperm(table.num_rows, generator=generator)
    return _safe_table_take(table, permutation)


def _request_group_tables(
    split: ParquetSplitConfig,
    table: object,
) -> Iterator[object]:
    """Yield one table per request while preserving candidate order."""

    if not split.reader.deduplicate_request_features:
        yield table
        return
    if split.request_id is None:
        raise ValueError("request-grouped batching requires request_id")

    request_ids = _column_array(table, split.request_id).to_pylist()
    positions_by_request: dict[Any, list[int]] = {}
    for row_index, request_id in enumerate(request_ids):
        if request_id is None:
            raise ValueError(
                f"request_id column {split.request_id!r} contains null at row {row_index}"
            )
        try:
            positions_by_request.setdefault(request_id, []).append(row_index)
        except TypeError as error:
            raise ValueError(
                f"request_id column {split.request_id!r} must contain hashable scalars"
            ) from error

    for positions in positions_by_request.values():
        first = positions[0]
        if positions == list(range(first, first + len(positions))):
            yield table.slice(first, len(positions))
        else:
            yield _safe_table_take(table, positions)


def _shuffle_table_groups(
    tables: list[object],
    generator: torch.Generator,
) -> list[object]:
    if len(tables) <= 1:
        return tables
    permutation = torch.randperm(len(tables), generator=generator).tolist()
    return [tables[index] for index in permutation]


def _iter_shuffled_candidate_tables(
    config: AppConfig,
    split_name: str,
    shard_rank: int,
    shard_world_size: int,
    require_labels: bool,
) -> Iterator[object]:
    """Bounded deterministic shuffle with request groups kept intact."""

    reader = _split_reader(config, split_name)
    split = config.data.train if split_name == "train" else config.data.test
    if split is None:
        raise ValueError(f"split {split_name!r} is not configured")
    candidate_source = iter_candidate_tables(
        config,
        split_name,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
        require_labels=require_labels,
    )
    source = (
        group
        for table in candidate_source
        if table.num_rows
        for group in _request_group_tables(split, table)
    )
    if reader.shuffle_buffer_rows == 0:
        yield from source
        return

    generator = torch.Generator(device="cpu")
    generator.manual_seed(reader.shuffle_seed + shard_rank)
    if not reader.deduplicate_request_features:
        pa, _pc, _ds, _pq = _require_pyarrow()
        buffered: object | None = None
        for table in source:
            combined = (
                table
                if buffered is None
                else pa.concat_tables([buffered, table])
            )
            if combined.num_rows <= reader.shuffle_buffer_rows:
                buffered = combined
                continue
            shuffled = _shuffle_table(combined, generator)
            emitted_rows = shuffled.num_rows - reader.shuffle_buffer_rows
            yield shuffled.slice(0, emitted_rows)
            buffered = shuffled.slice(emitted_rows)
        if buffered is not None and buffered.num_rows:
            yield _shuffle_table(buffered, generator)
        return

    buffered_groups: list[object] = []
    buffered_rows = 0
    for table in source:
        buffered_groups.append(table)
        buffered_rows += table.num_rows
        if buffered_rows <= reader.shuffle_buffer_rows:
            continue
        shuffled = _shuffle_table_groups(buffered_groups, generator)
        emitted: list[object] = []
        while shuffled and buffered_rows > reader.shuffle_buffer_rows:
            group = shuffled.pop(0)
            emitted.append(group)
            buffered_rows -= group.num_rows
        yield from emitted
        buffered_groups = shuffled
    yield from _shuffle_table_groups(buffered_groups, generator)


def _iter_group_preserving_batches(
    tables: Iterator[object],
    batch_size: int,
) -> Iterator[object]:
    """Pack request groups without splitting unless one group exceeds capacity."""

    pa, _pc, _ds, _pq = _require_pyarrow()
    buffered: list[object] = []
    buffered_rows = 0
    for original in tables:
        table = original
        while table.num_rows > batch_size:
            if buffered_rows:
                yield pa.concat_tables(buffered)
                buffered = []
                buffered_rows = 0
            yield table.slice(0, batch_size)
            table = table.slice(batch_size)
        if not table.num_rows:
            continue
        if buffered_rows and buffered_rows + table.num_rows > batch_size:
            yield pa.concat_tables(buffered)
            buffered = []
            buffered_rows = 0
        buffered.append(table)
        buffered_rows += table.num_rows
        if buffered_rows == batch_size:
            yield pa.concat_tables(buffered)
            buffered = []
            buffered_rows = 0
    if buffered_rows:
        yield pa.concat_tables(buffered)


def _table_sequence_lengths(config: AppConfig, sequence: Any, table: object) -> Tensor:
    pa, pc, _ds, _pq = _require_pyarrow()
    source = sequence.fields[0].source
    array = _column_array(table, source)
    if pa.types.is_dictionary(array.type):
        dictionary_lengths = pc.list_value_length(array.dictionary)
        lengths = pc.take(dictionary_lengths, array.indices)
    else:
        lengths = pc.list_value_length(array)
    if lengths.null_count:
        lengths = pc.fill_null(lengths, 0)
    values = torch.from_numpy(
        lengths.to_numpy(zero_copy_only=False).copy()
    ).to(dtype=torch.long)
    if sequence.max_length is not None:
        values.clamp_(max=sequence.max_length)
    return values


def _table_effective_sequence_lengths(
    config: AppConfig,
    table: object,
    metric: str = "max",
) -> Tensor:
    """Return the configured per-row sequence-work metric."""

    result = torch.zeros(table.num_rows, dtype=torch.long)
    for sequence in config.sequences:
        if not sequence.fields:
            continue
        values = _table_sequence_lengths(config, sequence, table)
        if metric == "sum":
            result.add_(values)
        else:
            result = torch.maximum(result, values)
    return result


def _iter_length_bucketed_tables(
    config: AppConfig,
    split_name: str,
    shard_rank: int,
    shard_world_size: int,
    require_labels: bool = True,
) -> Iterator[object]:
    """Group rows by sequence length before one vectorized padding operation."""

    reader = _split_reader(config, split_name)
    buckets = reader.length_buckets
    preserve_request_groups = reader.deduplicate_request_features
    if not buckets or not config.sequences:
        tables = _iter_shuffled_candidate_tables(
            config,
            split_name,
            shard_rank=shard_rank,
            shard_world_size=shard_world_size,
            require_labels=require_labels,
        )
        if preserve_request_groups:
            yield from _iter_group_preserving_batches(
                tables,
                config.training.batch_size,
            )
        else:
            for table in tables:
                yield from _slice_table(table, config.training.batch_size)
        return

    pa, _pc, _ds, _pq = _require_pyarrow()
    finite_boundaries = [
        bucket.max_length
        for bucket in buckets
        if bucket.max_length is not None
    ]
    boundaries = torch.tensor(finite_boundaries, dtype=torch.long)
    buffered: list[list[object]] = [[] for _ in buckets]
    buffered_rows = [0] * len(buckets)

    for table in _iter_shuffled_candidate_tables(
        config,
        split_name,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
        require_labels=require_labels,
    ):
        if preserve_request_groups:
            # Context and UPS are identical across one request group. Reading
            # the first row avoids repeating the same nine sequence lengths for
            # every candidate.
            lengths = _table_effective_sequence_lengths(
                config,
                table.slice(0, 1),
                metric=reader.length_bucket_metric,
            )
            bucket_index = int(
                torch.bucketize(lengths, boundaries, right=False).item()
            )
            bucket = buckets[bucket_index]
            remaining = table
            while remaining.num_rows > bucket.batch_size:
                if buffered_rows[bucket_index]:
                    yield pa.concat_tables(buffered[bucket_index])
                    buffered[bucket_index] = []
                    buffered_rows[bucket_index] = 0
                yield remaining.slice(0, bucket.batch_size)
                remaining = remaining.slice(bucket.batch_size)
            if not remaining.num_rows:
                continue
            if (
                buffered_rows[bucket_index]
                and buffered_rows[bucket_index] + remaining.num_rows
                > bucket.batch_size
            ):
                yield pa.concat_tables(buffered[bucket_index])
                buffered[bucket_index] = []
                buffered_rows[bucket_index] = 0
            buffered[bucket_index].append(remaining)
            buffered_rows[bucket_index] += remaining.num_rows
            if buffered_rows[bucket_index] == bucket.batch_size:
                yield pa.concat_tables(buffered[bucket_index])
                buffered[bucket_index] = []
                buffered_rows[bucket_index] = 0
            continue

        lengths = _table_effective_sequence_lengths(
            config,
            table,
            metric=reader.length_bucket_metric,
        )
        assignments = torch.bucketize(lengths, boundaries, right=False)
        for bucket_index, bucket in enumerate(buckets):
            selected = torch.nonzero(
                assignments == bucket_index, as_tuple=False
            ).flatten()
            if not selected.numel():
                continue
            selected_table = _safe_table_take(table, selected)
            buffered[bucket_index].append(selected_table)
            buffered_rows[bucket_index] += selected_table.num_rows
            if buffered_rows[bucket_index] < bucket.batch_size:
                continue
            combined = pa.concat_tables(buffered[bucket_index])
            offset = 0
            while combined.num_rows - offset >= bucket.batch_size:
                yield combined.slice(offset, bucket.batch_size)
                offset += bucket.batch_size
            remainder = combined.slice(offset)
            buffered[bucket_index] = [remainder] if remainder.num_rows else []
            buffered_rows[bucket_index] = remainder.num_rows

    for bucket_index in range(len(buckets)):
        if not buffered_rows[bucket_index]:
            continue
        yield pa.concat_tables(buffered[bucket_index])


def _iter_batch_tables(
    config: AppConfig,
    split_name: str,
    shard_rank: int,
    shard_world_size: int,
    require_labels: bool = True,
) -> Iterator[object]:
    yield from _iter_length_bucketed_tables(
        config,
        split_name,
        shard_rank,
        shard_world_size,
        require_labels,
    )


def _feature_batch_tensor_bytes(batch: FeatureBatch) -> int:
    def visit(value: Any) -> int:
        if isinstance(value, Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(visit(item) for item in value)
        return 0

    return (
        visit(batch.features)
        + visit(batch.labels)
        + visit(batch.label_mask)
        + visit(batch.scenario_id)
    )


def _max_bag_length(table: object, source: str) -> int:
    """Conservative per-column bag length without unifying dictionaries.

    Multi-chunk ``dictionary<list<int64>>`` columns cannot always
    ``combine_chunks()`` (Arrow lacks nested-dictionary unification). For the
    prefetch byte budget it is enough to take the max list length per chunk;
    inspecting a dictionary's full value set may slightly over-estimate when
    some entries are unreferenced, which matches the conservative reservation.
    """

    pa, pc, _ds, _pq = _require_pyarrow()
    chunked = table[source]
    chunks = getattr(chunked, "chunks", None)
    if chunks is None:
        chunks = (chunked,)
    maximum = 0
    for chunk in chunks:
        if pa.types.is_dictionary(chunk.type):
            lengths = pc.list_value_length(chunk.dictionary)
        else:
            lengths = pc.list_value_length(chunk)
        if lengths.null_count:
            lengths = pc.fill_null(lengths, 0)
        chunk_max = pc.max(lengths).as_py()
        maximum = max(maximum, int(chunk_max or 0))
    return maximum


def _estimate_prepared_batch_bytes(config: AppConfig, table: object) -> int:
    """Conservative Arrow-plus-tensor reservation for the prefetch queue."""

    rows = int(table.num_rows)
    tensor_bytes = 0
    bag_max_lengths: dict[str, int] = {}
    for feature in config.features:
        if feature.kind == "categorical" and feature.pooling == "mean":
            max_length = bag_max_lengths.get(feature.source)
            if max_length is None:
                try:
                    max_length = _max_bag_length(table, feature.source)
                except (KeyError, TypeError, AttributeError):
                    max_length = feature.max_length or 1
                bag_max_lengths[feature.source] = max_length
            if feature.max_length is not None:
                max_length = min(max_length, feature.max_length)
            tensor_bytes += rows * (8 + max_length * 8)
        else:
            tensor_bytes += rows * feature.dimension * (8 if feature.kind == "categorical" else 4)
    for sequence in config.sequences:
        if not sequence.fields:
            continue
        lengths = _table_sequence_lengths(config, sequence, table)
        padded_length = int(lengths.max().item()) if lengths.numel() else 0
        tensor_bytes += rows * 8
        for field in sequence.fields:
            element_bytes = 8 if field.kind == "categorical" else 4 * field.dimension
            tensor_bytes += rows * padded_length * element_bytes
    # Labels, masks, scenario IDs, and a margin for Python/allocator metadata.
    tensor_bytes += rows * (4 * max(1, len(config.task_names)) + 16)
    arrow_bytes = int(getattr(table, "nbytes", 0))
    return max(1, arrow_bytes + tensor_bytes + tensor_bytes // 8)


def _prepare_feature_batch(
    config: AppConfig,
    split: ParquetSplitConfig,
    table: object,
    vocab_maps: dict[str, dict[str, int]],
    require_labels: bool,
    pin_memory: bool,
    coalesce_pinned_tensors: bool,
    include_group_id: bool,
) -> FeatureBatch:
    batch = table_to_feature_batch(
        config,
        table,
        vocab_maps,
        require_labels=require_labels,
        include_group_id=include_group_id,
        split=split,
    )
    return (
        pin_feature_batch(
            batch,
            coalesce_tensors=coalesce_pinned_tensors,
        )
        if pin_memory
        else batch
    )


def _split_reader(config: AppConfig, split_name: str) -> ReaderConfig:
    split = config.data.train if split_name == "train" else config.data.test
    if split is None:
        raise ValueError(f"split {split_name!r} is not configured")
    return split.reader


def iter_feature_batches(
    config: AppConfig,
    split_name: str,
    vocab_maps: dict[str, dict[str, int]],
    require_labels: bool,
    shard_rank: int = 0,
    shard_world_size: int = 1,
    pin_memory: bool = False,
    include_group_id: bool = True,
) -> Iterator[FeatureBatch]:
    split = config.data.train if split_name == "train" else config.data.test
    if split is None:
        raise ValueError(f"split {split_name!r} is not configured")
    reader = _split_reader(config, split_name)
    pin_memory = reader.pin_memory and pin_memory
    coalesce_pinned_tensors = reader.coalesce_pinned_tensors and pin_memory
    table_iter = _iter_batch_tables(
        config,
        split_name,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
        require_labels=require_labels,
    )

    if reader.prefetch_batches <= 0:
        for table in table_iter:
            yield _prepare_feature_batch(
                config,
                split,
                table,
                vocab_maps,
                require_labels,
                pin_memory,
                coalesce_pinned_tensors,
                include_group_id,
            )
        return

    max_pending = max(1, reader.prefetch_batches)
    worker_count = min(max_pending, max(1, reader.num_workers))
    pending: deque[tuple[Future[FeatureBatch], int]] = deque()
    pending_bytes = 0
    buffered_table: object | None = None
    buffered_reservation = 0
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="mdl-reader") as executor:
        exhausted = False
        while pending or not exhausted or buffered_table is not None:
            while len(pending) < max_pending and not exhausted:
                if buffered_table is None:
                    try:
                        table = next(table_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    reservation = _estimate_prepared_batch_bytes(config, table)
                else:
                    table = buffered_table
                    reservation = buffered_reservation
                    buffered_table = None
                    buffered_reservation = 0
                if (
                    pending
                    and pending_bytes + reservation > reader.max_prefetch_bytes
                ):
                    buffered_table = table
                    buffered_reservation = reservation
                    break
                pending.append(
                    (
                        executor.submit(
                            _prepare_feature_batch,
                            config,
                            split,
                            table,
                            vocab_maps,
                            require_labels,
                            pin_memory,
                            coalesce_pinned_tensors,
                            include_group_id,
                        ),
                        reservation,
                    )
                )
                pending_bytes += reservation
            if not pending:
                if buffered_table is not None:
                    # A single oversized batch is admitted to guarantee progress.
                    table = buffered_table
                    reservation = buffered_reservation
                    buffered_table = None
                    buffered_reservation = 0
                    pending.append(
                        (
                            executor.submit(
                                _prepare_feature_batch,
                                config,
                                split,
                                table,
                                vocab_maps,
                                require_labels,
                                pin_memory,
                                coalesce_pinned_tensors,
                                include_group_id,
                            ),
                            reservation,
                        )
                    )
                    pending_bytes += reservation
                else:
                    break
            future, reservation = pending.popleft()
            batch = future.result()
            actual_bytes = _feature_batch_tensor_bytes(batch)
            if actual_bytes > reservation:
                pending_bytes += actual_bytes - reservation
                reservation = actual_bytes
            yield batch
            pending_bytes -= reservation


@dataclass(frozen=True)
class _DevicePrefetchItem:
    host_batch: FeatureBatch | None = None
    batch: FeatureBatch | None = None
    ready: torch.cuda.Event | None = None
    error: BaseException | None = None
    done: bool = False


def _record_feature_batch_stream(
    batch: FeatureBatch,
    stream: torch.cuda.Stream,
) -> None:
    """Associate prefetched allocations with the consuming CUDA stream."""

    def record(value: Any) -> None:
        if isinstance(value, Tensor) and value.device.type == "cuda":
            value.record_stream(stream)
        elif isinstance(value, dict):
            for child in value.values():
                record(child)

    record(batch.features)
    record(batch.labels)
    record(batch.label_mask)
    record(batch.scenario_id)
    for buffer in batch._packed_buffers:
        record(buffer)


class _DevicePrefetchIterator:
    """Prepare and copy future batches on a dedicated CUDA stream/thread."""

    def __init__(
        self,
        iterator: Iterator[FeatureBatch],
        device: torch.device,
        depth: int,
    ) -> None:
        if device.type != "cuda" or depth <= 0:
            raise ValueError("device prefetch requires CUDA and positive depth")
        self.iterator = iterator
        self.device = device
        self.stop_event = threading.Event()
        self.queue: queue.Queue[_DevicePrefetchItem] = queue.Queue(maxsize=depth)
        self.thread = threading.Thread(
            target=self._worker,
            name="mdl-cuda-prefetch",
            daemon=True,
        )
        self.thread.start()

    def __iter__(self) -> "_DevicePrefetchIterator":
        return self

    def _put(self, item: _DevicePrefetchItem) -> bool:
        while not self.stop_event.is_set():
            try:
                self.queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _worker(self) -> None:
        try:
            torch.cuda.set_device(self.device)
            transfer_stream = torch.cuda.Stream(device=self.device)
            while not self.stop_event.is_set():
                try:
                    host_batch = next(self.iterator)
                except StopIteration:
                    self._put(_DevicePrefetchItem(done=True))
                    return
                with torch.cuda.stream(transfer_stream):
                    device_batch = move_feature_batch(
                        host_batch,
                        self.device,
                        non_blocking=True,
                    )
                    ready = torch.cuda.Event()
                    ready.record(transfer_stream)
                if not self._put(
                    _DevicePrefetchItem(
                        host_batch=host_batch,
                        batch=device_batch,
                        ready=ready,
                    )
                ):
                    return
        except BaseException as error:
            self._put(_DevicePrefetchItem(error=error))
        finally:
            close = getattr(self.iterator, "close", None)
            if callable(close):
                close()

    def _next_item(self) -> _DevicePrefetchItem:
        item = self.queue.get()
        if item.error is not None:
            self.close()
            raise item.error
        if item.done:
            self.close()
            raise StopIteration
        if item.batch is None or item.ready is None:
            self.close()
            raise RuntimeError("invalid CUDA-prefetch queue item")
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_event(item.ready)
        _record_feature_batch_stream(item.batch, current_stream)
        return item

    def __next__(self) -> FeatureBatch:
        item = self._next_item()
        assert item.batch is not None
        return item.batch

    def next_with_host(self) -> tuple[FeatureBatch, FeatureBatch]:
        """Return matching host/device views for pre-update evaluation replay."""

        item = self._next_item()
        if item.host_batch is None or item.batch is None:
            self.close()
            raise RuntimeError("CUDA-prefetch item did not retain its host batch")
        return item.host_batch, item.batch

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not threading.current_thread():
            self.thread.join(timeout=5.0)


def _classify_model_parameters(model: nn.Module) -> _ParameterGroups:
    """Separate optimizer ownership from native sparse-gradient ownership.

    All ``nn.Embedding`` parameters retain the repository's existing Adagrad
    optimizer assignment. Only embeddings constructed with ``sparse=True``
    need to bypass DDP's reducer, since standard NCCL cannot all-reduce their
    COO gradients.
    """

    embedding_ids: set[int] = set()
    sparse_gradient_ids: set[int] = set()
    sharded_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, ShardedEmbedding):
            module_parameter_ids = {
                id(parameter) for parameter in module.parameters(recurse=False)
            }
            embedding_ids.update(module_parameter_ids)
            sharded_ids.update(module_parameter_ids)
            continue
        if not isinstance(module, nn.Embedding):
            continue
        module_parameter_ids = {
            id(parameter) for parameter in module.parameters(recurse=False)
        }
        embedding_ids.update(module_parameter_ids)
        if module.sparse:
            sparse_gradient_ids.update(module_parameter_ids)

    dense: list[nn.Parameter] = []
    embeddings: list[nn.Parameter] = []
    sharded: list[nn.Parameter] = []
    sparse_sync: list[_NamedSparseParameter] = []
    sharded_ignore: list[_NamedSparseParameter] = []
    seen_sparse_ids: set[int] = set()
    seen_sharded_ids: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in sharded_ids:
            sharded.append(parameter)
        elif parameter_id in embedding_ids:
            embeddings.append(parameter)
        else:
            dense.append(parameter)
        if parameter_id in sharded_ids:
            if parameter_id not in seen_sharded_ids:
                sharded_ignore.append(
                    _NamedSparseParameter(name=name, parameter=parameter)
                )
                seen_sharded_ids.add(parameter_id)
            continue
        if parameter_id not in sparse_gradient_ids:
            continue
        if parameter_id in seen_sparse_ids:
            continue
        if parameter.ndim != 2:
            raise ValueError(
                f"row-sparse embedding parameter {name!r} must be two-dimensional"
            )
        sparse_sync.append(_NamedSparseParameter(name=name, parameter=parameter))
        seen_sparse_ids.add(parameter_id)

    missing_sparse_ids = sparse_gradient_ids - seen_sparse_ids
    if missing_sparse_ids:
        raise RuntimeError("failed to resolve names for sparse embedding parameters")
    return _ParameterGroups(
        dense_optimizer=tuple(dense),
        embedding_optimizer=tuple(embeddings),
        sparse_sync=tuple(sparse_sync),
        sharded_optimizer=tuple(sharded),
        sharded_ddp_ignore=tuple(sharded_ignore),
    )


def _partition_embedding_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Compatibility wrapper returning the two optimizer parameter groups."""

    groups = _classify_model_parameters(model)
    return (
        list(groups.dense_optimizer),
        list(groups.embedding_optimizer) + list(groups.sharded_optimizer),
    )


def _build_dense_optimizer(
    parameters: list[nn.Parameter],
    config: AppConfig,
    device: torch.device,
) -> torch.optim.Optimizer:
    """Construct RMSprop and enable fused execution only when implemented."""

    kwargs: dict[str, Any] = {
        "lr": config.training.lr_dense,
        "alpha": config.training.rmsprop_alpha,
        "momentum": config.training.rmsprop_momentum,
    }
    fused_requested = (
        getattr(config.training, "fused_dense_optimizer", False)
        and device.type == "cuda"
    )
    try:
        fused_supported = (
            "fused" in inspect.signature(torch.optim.RMSprop).parameters
        )
    except (TypeError, ValueError):
        fused_supported = False
    if fused_requested and fused_supported:
        kwargs["fused"] = True
    elif fused_requested and is_main_process():
        print(
            "Dense optimizer | optimizer=RMSprop fused_requested=true "
            "fused_supported=false"
        )
    return torch.optim.RMSprop(parameters, **kwargs)


def _sparse_parameter_descriptors(
    sparse_parameters: tuple[_NamedSparseParameter, ...],
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(
        (ref.name, tuple(ref.parameter.shape), str(ref.parameter.dtype))
        for ref in sparse_parameters
    )


def _validate_sharded_embedding_metadata(
    context: DistributedContext,
    model: nn.Module,
) -> None:
    modules = sorted(sharded_embedding_modules(model), key=lambda item: item.table_name)
    descriptors = tuple(
        (
            module.table_name,
            module.num_embeddings,
            module.embedding_dim,
            module.padding_idx,
            module.shard_spec.strategy,
            module.shard_spec.cyclic_offset,
            module.shard_spec.table_owner,
            module.shard_spec.world_size,
            str(module.weight.dtype),
        )
        for module in modules
    )
    if len({item[0] for item in descriptors}) != len(descriptors):
        raise RuntimeError("sharded embedding table names must be unique after alias resolution")
    if not context.enabled:
        return
    gathered: list[object | None] = [None] * context.world_size
    torch_dist.all_gather_object(gathered, descriptors)
    if any(item != descriptors for item in gathered):
        raise RuntimeError(
            "sharded embedding metadata or ownership plan differs across ranks"
        )


@torch.no_grad()
def _synchronize_sparse_parameter_replicas(
    context: DistributedContext,
    sparse_parameters: tuple[_NamedSparseParameter, ...],
) -> None:
    """Validate sparse table metadata and broadcast complete rank-0 replicas."""

    if not context.enabled or not sparse_parameters:
        return
    descriptors = _sparse_parameter_descriptors(sparse_parameters)
    gathered: list[object | None] = [None] * context.world_size
    torch_dist.all_gather_object(gathered, descriptors)
    if any(item != descriptors for item in gathered):
        raise RuntimeError(
            "sparse embedding metadata differs across ranks; names, shapes, and dtypes "
            "must match before replicated DDP training"
        )
    for ref in sparse_parameters:
        torch_dist.broadcast(ref.parameter, src=0)


def _exclude_sparse_parameters_from_ddp(
    forward_model: nn.Module,
    sparse_parameters: tuple[_NamedSparseParameter, ...],
) -> None:
    """Tell DDP to leave COO parameters to the replicated sparse synchronizer."""

    if not sparse_parameters:
        return
    sparse_ids = {id(ref.parameter) for ref in sparse_parameters}
    ignored_names = [
        name
        for name, parameter in forward_model.named_parameters()
        if id(parameter) in sparse_ids
    ]
    resolved_ids = {
        id(parameter)
        for name, parameter in forward_model.named_parameters()
        if name in ignored_names
    }
    if resolved_ids != sparse_ids:
        raise RuntimeError(
            "failed to map sparse embedding parameters through the compiled model wrapper"
        )
    ignore_helper = getattr(
        DistributedDataParallel,
        "_set_params_and_buffers_to_ignore_for_model",
        None,
    )
    if ignore_helper is None:
        raise RuntimeError(
            "this PyTorch version cannot exclude sparse embedding parameters from DDP; "
            "use a supported torch>=2.2 build or set embedding_sparse_gradients=false"
        )
    ignore_helper(forward_model, ignored_names)


def _build_sparse_group_specs(
    sparse_parameters: tuple[_NamedSparseParameter, ...],
) -> tuple[_SparseGroupSpec, ...]:
    grouped: dict[tuple[torch.dtype, int], list[_NamedSparseParameter]] = {}
    for ref in sorted(sparse_parameters, key=lambda item: item.name):
        parameter = ref.parameter
        key = (parameter.dtype, int(parameter.shape[1]))
        grouped.setdefault(key, []).append(ref)

    specs: list[_SparseGroupSpec] = []
    for (dtype, embedding_dim), refs in grouped.items():
        offset = 0
        tables: list[_SparseTableSpec] = []
        for ref in refs:
            tables.append(_SparseTableSpec(ref=ref, row_offset=offset))
            offset += int(ref.parameter.shape[0])
        specs.append(
            _SparseGroupSpec(
                embedding_dim=embedding_dim,
                dtype=dtype,
                tables=tuple(tables),
                total_rows=offset,
            )
        )
    return tuple(specs)


class _ReplicatedSparseGradientSynchronizer:
    """Synchronize only touched embedding rows using dense NCCL collectives."""

    def __init__(
        self,
        context: DistributedContext,
        sparse_parameters: tuple[_NamedSparseParameter, ...],
    ) -> None:
        self.context = context
        self.sparse_parameters = tuple(
            sorted(sparse_parameters, key=lambda item: item.name)
        )
        self.groups = _build_sparse_group_specs(self.sparse_parameters)

    @staticmethod
    def _empty_sparse_gradient(parameter: nn.Parameter) -> Tensor:
        return torch.sparse_coo_tensor(
            torch.empty((1, 0), dtype=torch.long, device=parameter.device),
            torch.empty(
                (0, int(parameter.shape[1])),
                dtype=parameter.dtype,
                device=parameter.device,
            ),
            size=tuple(parameter.shape),
            dtype=parameter.dtype,
            device=parameter.device,
            is_coalesced=True,
        )

    @staticmethod
    def _local_group_gradient(
        group: _SparseGroupSpec,
        rank_active: bool,
    ) -> tuple[Tensor, Tensor]:
        first_parameter = group.tables[0].ref.parameter
        device = first_parameter.device
        if not rank_active:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(
                    (0, group.embedding_dim),
                    dtype=group.dtype,
                    device=device,
                ),
            )

        encoded_rows: list[Tensor] = []
        values: list[Tensor] = []
        for table in group.tables:
            parameter = table.ref.parameter
            grad = parameter.grad
            if grad is None:
                continue
            if not grad.is_sparse or grad.layout != torch.sparse_coo:
                raise RuntimeError(
                    f"expected a COO gradient for sparse embedding {table.ref.name!r}"
                )
            grad = grad.coalesce()
            if grad.sparse_dim() != 1 or grad.dense_dim() != 1:
                raise RuntimeError(
                    f"sparse embedding {table.ref.name!r} must have one sparse row dimension"
                )
            grad_values = grad.values()
            if grad_values.shape[0] == 0:
                continue
            rows = grad.indices()[0] + table.row_offset
            encoded_rows.append(rows)
            values.append(grad_values)

        if not encoded_rows:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(
                    (0, group.embedding_dim),
                    dtype=group.dtype,
                    device=device,
                ),
            )
        return torch.cat(encoded_rows), torch.cat(values)

    @staticmethod
    def _assign_group_gradient(
        group: _SparseGroupSpec,
        encoded_rows: Tensor,
        values: Tensor,
        globally_present: set[str],
    ) -> int:
        if encoded_rows.numel() == 0:
            for table in group.tables:
                parameter = table.ref.parameter
                parameter.grad = (
                    _ReplicatedSparseGradientSynchronizer._empty_sparse_gradient(parameter)
                    if table.ref.name in globally_present
                    else None
                )
            return 0

        virtual_grad = torch.sparse_coo_tensor(
            encoded_rows.unsqueeze(0),
            values,
            size=(group.total_rows, group.embedding_dim),
            dtype=group.dtype,
            device=values.device,
        ).coalesce()
        global_rows = virtual_grad.indices()[0]
        global_values = virtual_grad.values()
        for table in group.tables:
            parameter = table.ref.parameter
            start = table.row_offset
            stop = start + int(parameter.shape[0])
            selected = (global_rows >= start) & (global_rows < stop)
            table_rows = global_rows[selected] - start
            if table_rows.numel() == 0:
                parameter.grad = (
                    _ReplicatedSparseGradientSynchronizer._empty_sparse_gradient(parameter)
                    if table.ref.name in globally_present
                    else None
                )
                continue
            table_values = global_values[selected]
            parameter.grad = torch.sparse_coo_tensor(
                table_rows.unsqueeze(0),
                table_values,
                size=tuple(parameter.shape),
                dtype=parameter.dtype,
                device=parameter.device,
                is_coalesced=True,
            )
        return int(global_rows.numel())

    @torch.no_grad()
    def synchronize(self, rank_active: bool = True) -> _SparseSyncStats:
        if not self.context.enabled or not self.groups:
            return _SparseSyncStats()

        local_gradients = [
            self._local_group_gradient(group, rank_active)
            for group in self.groups
        ]
        local_counts = torch.tensor(
            [int(rows.numel()) for rows, _values in local_gradients],
            dtype=torch.long,
            device=self.context.device,
        )
        local_presence = torch.tensor(
            [
                int(rank_active and ref.parameter.grad is not None)
                for ref in self.sparse_parameters
            ],
            dtype=torch.long,
            device=self.context.device,
        )
        local_metadata = torch.cat([local_counts, local_presence])
        gathered_metadata = [
            torch.empty_like(local_metadata) for _ in range(self.context.world_size)
        ]
        torch_dist.all_gather(gathered_metadata, local_metadata)
        metadata_by_rank = torch.stack(gathered_metadata).cpu().tolist()
        group_count = len(self.groups)
        counts_by_rank = [items[:group_count] for items in metadata_by_rank]
        globally_present = {
            ref.name
            for parameter_index, ref in enumerate(self.sparse_parameters)
            if any(
                int(items[group_count + parameter_index]) != 0
                for items in metadata_by_rank
            )
        }

        local_row_count = int(local_counts.sum().item())
        global_row_count = 0
        logical_payload_bytes = (
            self.context.world_size
            * local_metadata.numel()
            * local_metadata.element_size()
        )
        for group_index, (group, local_gradient) in enumerate(
            zip(self.groups, local_gradients)
        ):
            counts = [int(rank_counts[group_index]) for rank_counts in counts_by_rank]
            max_rows = max(counts)
            if max_rows == 0:
                self._assign_group_gradient(
                    group,
                    local_gradient[0],
                    local_gradient[1],
                    globally_present,
                )
                continue

            local_rows, local_values = local_gradient
            padded_rows = torch.zeros(
                max_rows,
                dtype=torch.long,
                device=local_rows.device,
            )
            padded_values = torch.zeros(
                (max_rows, group.embedding_dim),
                dtype=group.dtype,
                device=local_values.device,
            )
            padded_rows[: local_rows.numel()] = local_rows
            padded_values[: local_values.shape[0]] = local_values
            gathered_rows = [torch.empty_like(padded_rows) for _ in counts]
            gathered_values = [torch.empty_like(padded_values) for _ in counts]
            torch_dist.all_gather(gathered_rows, padded_rows)
            torch_dist.all_gather(gathered_values, padded_values)
            encoded_rows = torch.cat(
                [rows[:count] for rows, count in zip(gathered_rows, counts)]
            )
            values = torch.cat(
                [items[:count] for items, count in zip(gathered_values, counts)]
            )
            values.div_(float(self.context.world_size))
            global_row_count += self._assign_group_gradient(
                group,
                encoded_rows,
                values,
                globally_present,
            )
            logical_payload_bytes += self.context.world_size * max_rows * (
                padded_rows.element_size()
                + group.embedding_dim * padded_values.element_size()
            )

        return _SparseSyncStats(
            local_rows=local_row_count,
            global_rows=global_row_count,
            logical_payload_bytes=logical_payload_bytes,
        )


def _maybe_compile_model(config: AppConfig, model: nn.Module) -> nn.Module:
    if not config.runtime.compile:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("runtime.compile requires torch.compile support")
    compile_mode = getattr(config.runtime, "compile_mode", "default")
    if compile_mode == "default":
        return torch.compile(model)
    return torch.compile(model, mode=compile_mode)


def _prepare_forward_model(
    config: AppConfig,
    base_model: nn.Module,
    context: DistributedContext,
    ddp_ignored: tuple[_NamedSparseParameter, ...] = (),
) -> nn.Module:
    """Wrap DDP before compile so reducer buckets remain overlap boundaries."""

    forward_model: nn.Module = base_model
    if context.enabled:
        _exclude_sparse_parameters_from_ddp(base_model, ddp_ignored)
        ddp_config = getattr(config.training, "ddp", DDPConfig())
        forward_model = DistributedDataParallel(
            base_model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            output_device=(
                context.local_rank if context.device.type == "cuda" else None
            ),
            find_unused_parameters=ddp_config.find_unused_parameters,
            static_graph=ddp_config.static_graph,
            gradient_as_bucket_view=ddp_config.gradient_as_bucket_view,
            bucket_cap_mb=ddp_config.bucket_cap_mb,
        )
    # Compiling the wrapper lets Dynamo's DDPOptimizer split the backward graph
    # at reducer bucket boundaries instead of delaying all reductions until a
    # monolithic compiled backward has completed.
    return _maybe_compile_model(config, forward_model)


def _autocast_dtype(config: AppConfig, device: torch.device) -> torch.dtype | None:
    if config.runtime.precision == "fp32":
        return None
    if config.runtime.precision == "bf16":
        if device.type in {"cuda", "cpu"}:
            return torch.bfloat16
        return None
    if config.runtime.precision == "fp16":
        if device.type == "cuda":
            return torch.float16
        return None
    raise ValueError(f"unsupported runtime.precision {config.runtime.precision!r}")


def _autocast_context(config: AppConfig, device: torch.device):
    dtype = _autocast_dtype(config, device)
    if dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def _make_grad_scaler(config: AppConfig, device: torch.device):
    if config.runtime.precision != "fp16" or device.type != "cuda":
        return _NoOpGradScaler()
    return torch.amp.GradScaler(
        device="cuda",
        enabled=True,
    )


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _non_blocking_transfer(config: AppConfig, split_name: str, device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    return _split_reader(config, split_name).pin_memory


def _batch_input_token_count(batch: FeatureBatch) -> int:
    """Count valid sequence events without materializing padding masks.

    The count is deliberately an input-data metric: every sequence contributes
    the sum of its configured ``lengths`` tensor. Models without sequence inputs
    fall back to one token per row so ``tokens/s`` remains well-defined.
    """

    total = 0
    found_sequence = False
    for value in batch.features.values():
        if not isinstance(value, dict):
            continue
        lengths = value.get("lengths")
        if not isinstance(lengths, Tensor):
            continue
        found_sequence = True
        row_indices = value.get("row_indices")
        if isinstance(row_indices, Tensor):
            lengths = lengths.index_select(0, row_indices.to(lengths.device).long())
        total += int(lengths.detach().sum().cpu().item())
    if found_sequence:
        return total
    return int(batch.scenario_id.size(0))


def _batch_padded_token_slots(batch: FeatureBatch) -> int:
    """Count dense sequence slots, including padding, across all sequences."""

    total = 0
    found_sequence = False
    for value in batch.features.values():
        if not isinstance(value, dict):
            continue
        lengths = value.get("lengths")
        fields = value.get("fields")
        if not isinstance(lengths, Tensor) or not isinstance(fields, dict):
            continue
        found_sequence = True
        padded_length = 0
        for field_value in fields.values():
            if isinstance(field_value, Tensor) and field_value.dim() >= 2:
                padded_length = int(field_value.size(1))
                break
        row_indices = value.get("row_indices")
        logical_rows = (
            int(row_indices.numel()) if isinstance(row_indices, Tensor) else int(lengths.numel())
        )
        total += logical_rows * padded_length
    if found_sequence:
        return total
    return int(batch.scenario_id.size(0))


def _resolve_lr_decay_steps(config: AppConfig, max_steps: int | None) -> int | None:
    if config.training.lr_schedule == "constant":
        return None
    if config.training.lr_decay_steps is not None:
        return config.training.lr_decay_steps
    if max_steps is not None:
        return max_steps
    raise ValueError(
        "training.lr_decay_steps is required for cosine when train --max-steps is not set"
    )


def _lr_schedule_multiplier(config: AppConfig, step: int, decay_steps: int | None) -> float:
    warmup_steps = config.training.lr_warmup_steps
    if warmup_steps > 0 and step <= warmup_steps:
        return float(step) / float(warmup_steps)
    if config.training.lr_schedule == "constant":
        return 1.0
    if config.training.lr_schedule != "cosine":
        raise ValueError(f"unsupported lr_schedule {config.training.lr_schedule!r}")
    if decay_steps is None:
        raise RuntimeError("cosine lr_schedule requires resolved decay_steps")
    if decay_steps <= warmup_steps:
        return min(1.0, float(step) / float(max(warmup_steps, 1)))

    progress = float(step - warmup_steps) / float(decay_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_ratio = config.training.lr_min_ratio
    return min_ratio + (1.0 - min_ratio) * cosine


def _set_optimizer_lrs(
    optimizers: list[torch.optim.Optimizer],
    base_lrs: list[list[float]],
    multiplier: float,
) -> None:
    for optimizer, optimizer_base_lrs in zip(optimizers, base_lrs):
        for group, base_lr in zip(optimizer.param_groups, optimizer_base_lrs):
            group["lr"] = base_lr * multiplier


def _active_rank_count(context: DistributedContext, rank_active: bool) -> int:
    if not context.enabled:
        return int(rank_active)
    device = torch.device("cpu") if context.control_group is not None else context.device
    value = torch.tensor(int(rank_active), dtype=torch.long, device=device)
    torch_dist.all_reduce(
        value,
        op=torch_dist.ReduceOp.SUM,
        group=context.control_group,
    )
    return int(value.item())


def _tensor_nbytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _log_sparse_replica_memory(
    context: DistributedContext,
    sparse_parameters: tuple[_NamedSparseParameter, ...],
    embedding_optimizer: torch.optim.Optimizer | None,
) -> None:
    if context.rank != 0 or not sparse_parameters:
        return
    total_weight_bytes = 0
    total_state_bytes = 0
    for ref in sparse_parameters:
        weight_bytes = _tensor_nbytes(ref.parameter)
        state = (
            embedding_optimizer.state.get(ref.parameter, {})
            if embedding_optimizer is not None
            else {}
        )
        accumulator = state.get("sum")
        state_bytes = _tensor_nbytes(accumulator) if isinstance(accumulator, Tensor) else weight_bytes
        total_weight_bytes += weight_bytes
        total_state_bytes += state_bytes
        print(
            "Sparse replica | "
            f"name={ref.name} shape={tuple(ref.parameter.shape)} "
            f"weight_mib={weight_bytes / (1024 ** 2):.2f} "
            f"optimizer_state_mib={state_bytes / (1024 ** 2):.2f}"
        )
    per_rank_bytes = total_weight_bytes + total_state_bytes
    print(
        "Sparse replica total | "
        f"tables={len(sparse_parameters)} "
        f"per_rank_mib={per_rank_bytes / (1024 ** 2):.2f} "
        f"world_size={context.world_size} "
        f"job_replica_mib={per_rank_bytes * context.world_size / (1024 ** 2):.2f} "
        "sharded=false"
    )


def _log_sharded_embedding_memory(
    context: DistributedContext,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    sparse_optimizer: str = "adagrad",
) -> None:
    modules = sharded_embedding_modules(model)
    if not modules:
        return
    local_tables: list[dict[str, Any]] = []
    for module in modules:
        state = optimizer.state.get(module.weight, {}) if optimizer is not None else {}
        accumulator = state.get("sum")
        local_tables.append(
            {
                "name": module.table_name,
                "strategy": module.shard_spec.strategy,
                "global_rows": module.num_embeddings,
                "local_rows": int(module.weight.size(0)),
                "weight_bytes": _tensor_nbytes(module.weight),
                "state_bytes": (
                    _tensor_nbytes(accumulator)
                    if isinstance(accumulator, Tensor)
                    else 0
                ),
            }
        )
    gathered: list[object] = [local_tables]
    if context.enabled:
        gathered = [None] * context.world_size
        torch_dist.all_gather_object(gathered, local_tables)
    if context.rank != 0:
        return
    layout = (
        "rowwise" if sparse_optimizer == "rowwise_adagrad" else "full"
    )
    print(
        "Sharded embedding memory | "
        f"sparse_optimizer={sparse_optimizer} "
        f"optimizer_state_layout={layout} "
        f"embedding_weight_dtype="
        f"{getattr(model, 'embedding_weight_dtype', torch.float32)}"
    )
    for rank, rank_tables_raw in enumerate(gathered):
        rank_tables = list(rank_tables_raw or [])
        weight_bytes = sum(int(item["weight_bytes"]) for item in rank_tables)
        state_bytes = sum(int(item["state_bytes"]) for item in rank_tables)
        print(
            "Sharded embedding memory | "
            f"rank={rank} tables={len(rank_tables)} "
            f"weight_gib={weight_bytes / (1024 ** 3):.5f} "
            f"optimizer_state_gib={state_bytes / (1024 ** 3):.5f} "
            f"total_gib={(weight_bytes + state_bytes) / (1024 ** 3):.5f}"
        )
        for item in rank_tables:
            print(
                "Sharded embedding table | "
                f"rank={rank} name={item['name']} strategy={item['strategy']} "
                f"rows={item['local_rows']}/{item['global_rows']}"
            )


def _mark_sparse_invariant_checks_explicitly_disabled() -> None:
    checker = getattr(torch.sparse, "check_sparse_tensor_invariants", None)
    if checker is not None and not checker.is_enabled():
        checker.disable()


@torch.no_grad()
def _gradient_values(parameters: list[nn.Parameter]) -> list[Tensor]:
    grads: list[Tensor] = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad
        if grad.is_sparse:
            grad = grad.coalesce()
            parameter.grad = grad
            grads.append(grad._values())
        else:
            grads.append(grad)
    return grads


def _gradient_squared_norm(grads: list[Tensor], device: torch.device) -> Tensor:
    total = torch.zeros((), dtype=torch.float32, device=device)
    for grad in grads:
        values = grad.detach()
        if values.dtype in {torch.float16, torch.bfloat16}:
            values = values.float()
        norm = torch.linalg.vector_norm(values, 2.0).to(device=device, dtype=torch.float32)
        total.add_(norm.square())
    return total


def _scale_gradients(grads: list[Tensor], coefficient: Tensor) -> None:
    if not grads:
        return
    first = grads[0]
    foreach_compatible = all(
        grad.device == first.device and grad.dtype == first.dtype
        for grad in grads
    )
    if foreach_compatible and hasattr(torch, "_foreach_mul_"):
        try:
            torch._foreach_mul_(grads, coefficient.to(device=first.device, dtype=first.dtype))
            return
        except (RuntimeError, TypeError):
            pass
    for grad in grads:
        grad.mul_(coefficient.to(device=grad.device, dtype=grad.dtype))


@torch.no_grad()
def _clip_grad_norm(parameters: list[nn.Parameter], max_norm: float) -> Tensor:
    grads = _gradient_values(parameters)
    if not grads:
        return torch.tensor(0.0)
    total_norm = _gradient_squared_norm(grads, grads[0].device).sqrt()
    # Clamping and applying on-device avoids the Python truth-value conversion
    # that otherwise synchronizes CUDA every training step.
    clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    _scale_gradients(grads, clip_coef)
    return total_norm


@torch.no_grad()
def _clip_sparse_grad_norm(
    replicated_parameters: list[nn.Parameter],
    sharded_parameters: list[nn.Parameter],
    max_norm: float,
) -> Tensor:
    """Clip one logical sparse group, reducing sharded norm squares globally.

    Replicated embedding gradients are already identical after DDP/sparse-row
    synchronization and therefore count once. Each sharded parameter contains a
    disjoint portion of the logical tables, so its squared norm is summed across
    ranks before one common coefficient is applied.
    """

    replicated_grads = _gradient_values(replicated_parameters)
    sharded_grads = _gradient_values(sharded_parameters)
    all_grads = [*replicated_grads, *sharded_grads]
    parameter = next(
        (
            item
            for item in [*replicated_parameters, *sharded_parameters]
            if item.device is not None
        ),
        None,
    )
    if parameter is None:
        return torch.tensor(0.0)
    device = parameter.device
    replicated_squared = _gradient_squared_norm(replicated_grads, device)
    sharded_squared = _gradient_squared_norm(sharded_grads, device)
    if (
        sharded_parameters
        and torch_dist.is_available()
        and torch_dist.is_initialized()
    ):
        torch_dist.all_reduce(sharded_squared, op=torch_dist.ReduceOp.SUM)
    total_norm = (replicated_squared + sharded_squared).sqrt()
    coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    _scale_gradients(all_grads, coefficient)
    return total_norm


@torch.no_grad()
def _step_sparse_moe_controllers(
    module: nn.Module,
    *,
    rank_active: bool = True,
    active_rank_count: int | None = None,
) -> None:
    for item in module.modules():
        if not isinstance(item, SparseMoEPerTokenFFN):
            continue
        active_ratio = item.active_ratio(item.regularization_coefficient).clone()
        if not rank_active:
            active_ratio.zero_()
        if torch_dist.is_available() and torch_dist.is_initialized():
            torch_dist.all_reduce(active_ratio, op=torch_dist.ReduceOp.SUM)
            divisor = active_rank_count
            if divisor is None:
                divisor = torch_dist.get_world_size()
            if divisor <= 0:
                raise RuntimeError("sparse MoE controller requires at least one active rank")
            active_ratio.div_(float(divisor))
        item.step_regularization_controller(active_ratio)


def _loss_terms_from_batch(
    output: dict[str, Tensor],
    batch: FeatureBatch,
    moe_loss_weight: float = 0.0,
    loss_reduction: str = "sum",
    rank_active: bool = True,
    active_rank_count: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if batch.labels is None:
        raise ValueError("training batch must contain labels")
    logits = output["logits"]
    if logits.shape != batch.labels.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} does not match labels {tuple(batch.labels.shape)}"
        )
    element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        batch.labels,
        reduction="none",
    )
    if batch.label_mask is None:
        task_numerators = element_loss.sum(dim=0)
        task_counts = element_loss.new_full(
            (element_loss.size(1),),
            float(element_loss.size(0)),
        )
        if not rank_active:
            # Preserve the replayed forward graph on an exhausted rank while
            # contributing no samples to either reduction.
            task_numerators = task_numerators * 0.0
            task_counts.zero_()
    else:
        weights = batch.label_mask.to(
            device=logits.device,
            dtype=element_loss.dtype,
        )
        if not rank_active:
            weights = torch.zeros_like(weights)
        task_numerators = (element_loss * weights).sum(dim=0)
        task_counts = weights.sum(dim=0)

    distributed = torch_dist.is_available() and torch_dist.is_initialized()
    if loss_reduction == "sum":
        # DDP averages gradients across ranks. Multiplying each local sum by the
        # world size makes the averaged gradient equal the global paper sum.
        world_size = float(torch_dist.get_world_size()) if distributed else 1.0
        prediction_loss = task_numerators.sum() * world_size
    elif loss_reduction == "mean_per_task":
        if distributed:
            global_counts = task_counts.detach().clone()
            torch_dist.all_reduce(global_counts, op=torch_dist.ReduceOp.SUM)
            world_size = float(torch_dist.get_world_size())
            task_scale = torch.where(
                global_counts > 0,
                world_size / global_counts.clamp_min(1.0),
                torch.zeros_like(global_counts),
            )
        else:
            task_scale = torch.where(
                task_counts > 0,
                task_counts.clamp_min(1.0).reciprocal(),
                torch.zeros_like(task_counts),
            )
        prediction_loss = (task_numerators * task_scale).sum()
    else:
        raise ValueError("loss_reduction must be sum or mean_per_task")
    moe_loss = output.get("moe_regularization_loss")
    total_loss = prediction_loss
    if moe_loss is not None and moe_loss_weight > 0.0:
        moe_scale = 1.0 if rank_active else 0.0
        if distributed:
            active_ranks = active_rank_count
            if active_ranks is None:
                active_ranks = torch_dist.get_world_size()
            if active_ranks <= 0:
                raise RuntimeError("MoE regularization requires at least one active rank")
            moe_scale *= float(torch_dist.get_world_size()) / float(active_ranks)
        total_loss = total_loss + moe_loss_weight * moe_loss * moe_scale
    # Aggregation averages this already task-balanced scalar across ranks.
    return total_loss, total_loss.detach(), total_loss.new_ones(())


def _loss_from_batch(output: dict[str, Tensor], batch: FeatureBatch) -> Tensor:
    return _loss_terms_from_batch(output, batch)[0]


def _aggregate_train_result(
    context: DistributedContext,
    local_result: TrainResult,
    last_loss_numerator: float,
    last_loss_denominator: float,
) -> TrainResult:
    if not context.enabled or not torch_dist.is_initialized():
        return local_result

    sum_values = torch.tensor(
        [
            float(local_result.rows),
            float(last_loss_numerator),
            float(last_loss_denominator),
            local_result.last_loss if local_result.steps > 0 else 0.0,
            1.0 if local_result.steps > 0 else 0.0,
        ],
        dtype=torch.float64,
        device=context.device,
    )
    max_values = torch.tensor(
        [float(local_result.steps), float(local_result.elapsed_seconds)],
        dtype=torch.float64,
        device=context.device,
    )
    torch_dist.all_reduce(sum_values, op=torch_dist.ReduceOp.SUM)
    torch_dist.all_reduce(max_values, op=torch_dist.ReduceOp.MAX)

    global_denominator = float(sum_values[2].item())
    if global_denominator > 0.0:
        last_loss = float((sum_values[1] / sum_values[2]).item())
    elif float(sum_values[4].item()) > 0.0:
        last_loss = float((sum_values[3] / sum_values[4]).item())
    else:
        last_loss = 0.0
    return TrainResult(
        steps=int(max_values[0].item()),
        rows=int(sum_values[0].item()),
        last_loss=last_loss,
        elapsed_seconds=float(max_values[1].item()),
    )


def train_mdl(
    config: AppConfig,
    max_steps: int | None = None,
    save_checkpoint: bool = True,
    log_steps: bool = True,
    step_observer: TrainStepObserver | None = None,
    run_quick_eval: bool = True,
) -> TrainResult:
    if config.training.sparse_update_mode == "external_parameter_server":
        config = resolve_auto_scenarios(config)
        adapter = _load_external_train_adapter(config.training.sparse_parameter_server_adapter)
        return _coerce_train_result(adapter(config=config, max_steps=max_steps))

    context = _setup_distributed(config)
    batch_iterator: Iterator[FeatureBatch] | None = None
    try:
        device = context.device
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
            torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
            torch.set_float32_matmul_precision(
                "high" if config.runtime.allow_tf32 else "highest"
            )
        # Fail before scenario discovery / Parquet scans when flash lacks varlen.
        attention_runtime = _attention_runtime_description(config, device)
        if log_steps and context.rank == 0:
            print(f"Attention backend | {attention_runtime}")
        config = _resolve_distributed_auto_scenarios(config, context)
        vocab_maps = load_vocab_maps(config)
        base_model = build_model(config, vocab_maps).to(device)
        _validate_sharded_embedding_metadata(context, base_model)
        parameter_groups = _classify_model_parameters(base_model)
        _synchronize_sparse_parameter_replicas(
            context,
            parameter_groups.sparse_sync,
        )
        ddp_ignored = (
            *parameter_groups.sparse_sync,
            *parameter_groups.sharded_ddp_ignore,
        )
        ddp_config = getattr(config.training, "ddp", DDPConfig())
        ddp_auditor = _DDPGraphAuditor(
            base_model,
            ignored_parameter_ids={id(ref.parameter) for ref in ddp_ignored},
            max_steps=ddp_config.audit_steps,
        )
        model = _prepare_forward_model(
            config,
            base_model,
            context,
            ddp_ignored,
        )

        dense_params = list(parameter_groups.dense_optimizer)
        replicated_embedding_params = list(parameter_groups.embedding_optimizer)
        sharded_embedding_params = list(parameter_groups.sharded_optimizer)
        sparse_params = replicated_embedding_params + sharded_embedding_params
        dense_optimizer: torch.optim.Optimizer | None = None
        embedding_optimizer: torch.optim.Optimizer | None = None
        sharded_embedding_optimizer: torch.optim.Optimizer | None = None
        optimizers: list[torch.optim.Optimizer] = []
        if dense_params:
            dense_optimizer = _build_dense_optimizer(
                dense_params,
                config,
                device,
            )
            optimizers.append(dense_optimizer)
        if replicated_embedding_params:
            _mark_sparse_invariant_checks_explicitly_disabled()
            sparse_lr = config.training.lr_sparse or config.training.lr_dense
            embedding_optimizer = torch.optim.Adagrad(
                replicated_embedding_params,
                lr=sparse_lr,
                lr_decay=config.training.adagrad_lr_decay,
                weight_decay=config.training.adagrad_weight_decay,
                initial_accumulator_value=config.training.adagrad_initial_accumulator_value,
                eps=config.training.adagrad_eps,
            )
            optimizers.append(embedding_optimizer)
        if sharded_embedding_params:
            _mark_sparse_invariant_checks_explicitly_disabled()
            sparse_lr = config.training.lr_sparse or config.training.lr_dense
            optimizer_kwargs = {
                "lr": sparse_lr,
                "lr_decay": config.training.adagrad_lr_decay,
                "weight_decay": config.training.adagrad_weight_decay,
                "initial_accumulator_value": (
                    config.training.adagrad_initial_accumulator_value
                ),
                "eps": config.training.adagrad_eps,
            }
            if config.training.sparse_optimizer == "rowwise_adagrad":
                sharded_embedding_optimizer = ShardedRowWiseAdagrad(
                    sharded_embedding_params,
                    **optimizer_kwargs,
                )
            else:
                sharded_embedding_optimizer = ShardedAdagrad(
                    sharded_embedding_params,
                    **optimizer_kwargs,
                )
            optimizers.append(sharded_embedding_optimizer)
        sparse_synchronizer = _ReplicatedSparseGradientSynchronizer(
            context,
            parameter_groups.sparse_sync,
        )
        if log_steps:
            _log_sparse_replica_memory(
                context,
                parameter_groups.sparse_sync,
                embedding_optimizer,
            )
            _log_sharded_embedding_memory(
                context,
                base_model,
                sharded_embedding_optimizer,
                sparse_optimizer=config.training.sparse_optimizer,
            )
        optimizer_base_lrs = [
            [float(group["lr"]) for group in optimizer.param_groups]
            for optimizer in optimizers
        ]
        lr_decay_steps = _resolve_lr_decay_steps(config, max_steps)
        scaler = _make_grad_scaler(config, device)
        non_blocking = _non_blocking_transfer(config, "train", device)
        quick_eval = getattr(config.training, "quick_eval", QuickEvalConfig())

        steps = 0
        rows = 0
        last_loss = 0.0
        last_loss_numerator = 0.0
        last_loss_denominator = 0.0
        last_loss_tensor: Tensor | None = None
        last_loss_numerator_tensor: Tensor | None = None
        last_loss_denominator_tensor: Tensor | None = None
        model.train()
        _sync_device(device)
        start = perf_counter()
        host_batch_iterator = iter(
            iter_feature_batches(
                config,
                "train",
                vocab_maps,
                require_labels=True,
                shard_rank=context.rank,
                shard_world_size=context.world_size,
                pin_memory=non_blocking,
                include_group_id=False,
            )
        )
        device_prefetch_depth = (
            config.data.train.reader.device_prefetch_batches
            if device.type == "cuda"
            else 0
        )
        batches_on_device = device_prefetch_depth > 0
        batch_iterator = (
            _DevicePrefetchIterator(
                host_batch_iterator,
                device,
                device_prefetch_depth,
            )
            if batches_on_device
            else host_batch_iterator
        )
        pending_train_batches: deque[FeatureBatch | None] = deque()
        last_device_batch: FeatureBatch | None = None
        while max_steps is None or steps < max_steps:
            tracing = step_observer is not None
            if tracing:
                _sync_device(device)
            step_started = perf_counter() if tracing else 0.0
            dataloader_started = step_started
            if pending_train_batches:
                local_batch = pending_train_batches.popleft()
                local_batch_on_device = False
            else:
                try:
                    local_batch = next(batch_iterator)
                except StopIteration:
                    local_batch = None
                local_batch_on_device = batches_on_device
            dataloader_wait_seconds = (
                perf_counter() - dataloader_started if tracing else 0.0
            )
            rank_active = local_batch is not None
            active_ranks = _active_rank_count(context, rank_active)
            if active_ranks == 0:
                break
            if steps == 0 and context.enabled and active_ranks != context.world_size:
                raise RuntimeError(
                    "replicated sparse DDP requires every rank to provide an initial batch; "
                    "reduce world_size or choose a finer reader.shard_unit"
                )
            h2d_started = perf_counter() if tracing else 0.0
            if rank_active:
                if local_batch is None:
                    raise AssertionError("active rank is missing its training batch")
                batch = (
                    local_batch
                    if local_batch_on_device
                    else move_feature_batch(
                        local_batch,
                        device,
                        non_blocking=non_blocking,
                    )
                )
                last_device_batch = batch
            else:
                if last_device_batch is None:
                    raise RuntimeError("inactive rank has no batch available for zero-loss replay")
                batch = last_device_batch
            if tracing:
                _sync_device(device)
            h2d_seconds = perf_counter() - h2d_started if tracing else 0.0

            lr_multiplier = _lr_schedule_multiplier(config, steps + 1, lr_decay_steps)
            _set_optimizer_lrs(optimizers, optimizer_base_lrs, lr_multiplier)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            forward_started = perf_counter() if tracing else 0.0
            with _autocast_context(config, device):
                output = model(batch.features, batch.scenario_id)
                loss, loss_numerator, loss_denominator = _loss_terms_from_batch(
                    output,
                    batch,
                    moe_loss_weight=config.model.sparse_moe_loss_weight,
                    loss_reduction=config.training.loss_reduction,
                    rank_active=rank_active,
                    active_rank_count=active_ranks,
                )
            if tracing:
                _sync_device(device)
            forward_seconds = perf_counter() - forward_started if tracing else 0.0
            backward_started = perf_counter() if tracing else 0.0
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            ddp_auditor.observe()
            if tracing:
                _sync_device(device)
            backward_seconds = perf_counter() - backward_started if tracing else 0.0
            sparse_sync_started = perf_counter() if tracing else 0.0
            sparse_sync_stats = sparse_synchronizer.synchronize(rank_active=rank_active)
            sharded_stats = consume_sharded_embedding_stats(base_model)
            if sharded_stats:
                sparse_sync_stats = _SparseSyncStats(
                    local_rows=(
                        sparse_sync_stats.local_rows
                        + sum(item.local_unique_ids for item in sharded_stats)
                    ),
                    global_rows=(
                        sparse_sync_stats.global_rows
                        + sum(item.owner_unique_ids for item in sharded_stats)
                    ),
                    logical_payload_bytes=(
                        sparse_sync_stats.logical_payload_bytes
                        + sum(item.total_communication_bytes for item in sharded_stats)
                    ),
                )
            if tracing:
                _sync_device(device)
            sparse_sync_seconds = perf_counter() - sparse_sync_started if tracing else 0.0
            optimizer_started = perf_counter() if tracing else 0.0
            if scaler.is_enabled():
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)
            _step_sparse_moe_controllers(
                base_model,
                rank_active=rank_active,
                active_rank_count=active_ranks,
            )
            if config.training.dense_clip_norm is not None and dense_params:
                _clip_grad_norm(dense_params, config.training.dense_clip_norm)
            if config.training.sparse_clip_norm is not None and sparse_params:
                _clip_sparse_grad_norm(
                    replicated_embedding_params,
                    sharded_embedding_params,
                    config.training.sparse_clip_norm,
                )
            if scaler.is_enabled():
                for optimizer in optimizers:
                    scaler.step(optimizer)
                scaler.update()
            else:
                for optimizer in optimizers:
                    optimizer.step()
            if tracing:
                _sync_device(device)
            optimizer_seconds = perf_counter() - optimizer_started if tracing else 0.0
            steps += 1
            if rank_active:
                rows += int(batch.scenario_id.size(0))
            last_loss_tensor = loss.detach()
            last_loss_numerator_tensor = loss_numerator.detach()
            last_loss_denominator_tensor = loss_denominator.detach()
            if step_observer is not None:
                step_observer(
                    TrainStepTrace(
                        step=steps,
                        rank_active=rank_active,
                        active_ranks=active_ranks,
                        rows=(int(batch.scenario_id.size(0)) if rank_active else 0),
                        input_tokens=(_batch_input_token_count(batch) if rank_active else 0),
                        padded_token_slots=(
                            _batch_padded_token_slots(batch) if rank_active else 0
                        ),
                        step_seconds=perf_counter() - step_started,
                        dataloader_wait_seconds=dataloader_wait_seconds,
                        h2d_seconds=h2d_seconds,
                        forward_seconds=forward_seconds,
                        backward_seconds=backward_seconds,
                        sparse_sync_seconds=sparse_sync_seconds,
                        optimizer_seconds=optimizer_seconds,
                        sparse_local_rows=sparse_sync_stats.local_rows,
                        sparse_global_rows=sparse_sync_stats.global_rows,
                        sparse_payload_bytes=sparse_sync_stats.logical_payload_bytes,
                    )
                )
            should_log = (
                log_steps
                and context.rank == 0
                and steps % config.training.log_every_steps == 0
            )
            if should_log:
                last_loss = float(last_loss_tensor.float().cpu().item())
                payload_mib = sparse_sync_stats.logical_payload_bytes / (1024 ** 2)
                valid_tokens = _batch_input_token_count(batch) if rank_active else 0
                padded_slots = _batch_padded_token_slots(batch) if rank_active else 0
                padding_ratio = (
                    1.0 - valid_tokens / padded_slots if padded_slots > 0 else 0.0
                )
                print(
                    f"Train step | step={steps} | loss={last_loss:.6f} "
                    f"active_ranks={active_ranks}/{context.world_size} "
                    f"padding_ratio={padding_ratio:.4f} "
                    f"sparse_local_rows={sparse_sync_stats.local_rows} "
                    f"sparse_global_rows={sparse_sync_stats.global_rows} "
                    f"sparse_payload_mib={payload_mib:.2f}"
                )
            if (
                run_quick_eval
                and quick_eval.enabled
                and steps % quick_eval.every_steps == 0
                and (
                    quick_eval.split != "train"
                    or not pending_train_batches
                )
                and (
                    quick_eval.split != "train"
                    or max_steps is None
                    or steps < max_steps
                )
            ):
                quick_eval_batch_limit = quick_eval.max_batches
                if quick_eval.split == "train" and max_steps is not None:
                    quick_eval_batch_limit = min(
                        quick_eval_batch_limit,
                        max_steps - steps,
                    )
                quick_eval_result, staged_batches = _run_training_quick_eval(
                    config,
                    model,
                    vocab_maps,
                    context,
                    quick_eval,
                    fallback_batch=last_device_batch,
                    training_batch_iterator=(
                        batch_iterator if quick_eval.split == "train" else None
                    ),
                    training_batches_on_device=batches_on_device,
                    max_batches=quick_eval_batch_limit,
                )
                pending_train_batches.extend(staged_batches)
                # Evaluation forwards also touch sharded-embedding diagnostics;
                # keep them out of the following training step's trace/log.
                consume_sharded_embedding_stats(base_model)
                if context.rank == 0:
                    _print_training_quick_eval(
                        steps,
                        quick_eval,
                        quick_eval_result,
                    )
        audit_report = ddp_auditor.report(context)
        if log_steps and context.rank == 0 and audit_report is not None:
            print(f"DDP graph audit | {audit_report}")
        _sync_device(device)
        elapsed = perf_counter() - start

        if last_loss_tensor is not None:
            last_loss = float(last_loss_tensor.float().cpu().item())
            assert last_loss_numerator_tensor is not None
            assert last_loss_denominator_tensor is not None
            last_loss_numerator = float(
                last_loss_numerator_tensor.float().cpu().item()
            )
            last_loss_denominator = float(
                last_loss_denominator_tensor.float().cpu().item()
            )

        local_result = TrainResult(steps=steps, rows=rows, last_loss=last_loss, elapsed_seconds=elapsed)
        result = _aggregate_train_result(
            context,
            local_result,
            last_loss_numerator,
            last_loss_denominator,
        )

        if save_checkpoint and config.training.save_checkpoint and config.training.checkpoint_path:
            save_model_checkpoint(
                config,
                base_model,
                config.training.checkpoint_path,
                rank=context.rank,
                world_size=context.world_size,
                sharded_optimizer=sharded_embedding_optimizer,
            )
        return result
    finally:
        if batch_iterator is not None:
            close = getattr(batch_iterator, "close", None)
            if callable(close):
                close()
        _cleanup_distributed(context)


def _binary_auc(scores: Tensor, labels: Tensor) -> float | None:
    """Exact rank-based binary AUC with average ranks for tied scores."""

    scores = scores.detach().float().flatten().cpu()
    labels = labels.detach().float().flatten().cpu()
    if scores.numel() != labels.numel():
        raise ValueError("AUC scores and labels must have the same length")
    if scores.numel() == 0:
        return None
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("AUC scores must be finite")
    if not bool(((labels == 0.0) | (labels == 1.0)).all()):
        raise ValueError("AUC labels must be binary")
    positive_count = int((labels == 1.0).sum().item())
    negative_count = int(labels.numel() - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None

    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    _unique, inverse, counts = torch.unique_consecutive(
        sorted_scores,
        return_inverse=True,
        return_counts=True,
    )
    ranks = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    rank_sums = torch.zeros(counts.numel(), dtype=torch.float64)
    rank_sums.scatter_add_(0, inverse, ranks)
    average_ranks = rank_sums / counts.to(torch.float64)
    positive_rank_sum = average_ranks[inverse][sorted_labels == 1.0].sum()
    auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc.item())


def _group_auc(scores: Tensor, labels: Tensor, group_ids: list[str]) -> float | None:
    """Unweighted mean AUC over groups containing both label classes."""

    if scores.numel() != len(group_ids) or labels.numel() != len(group_ids):
        raise ValueError("group AUC inputs must have matching lengths")
    indices_by_group: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        indices_by_group.setdefault(group_id, []).append(index)
    values: list[float] = []
    for indices in indices_by_group.values():
        index_tensor = torch.tensor(indices, dtype=torch.long)
        value = _binary_auc(scores[index_tensor], labels[index_tensor])
        if value is not None:
            values.append(value)
    return None if not values else float(sum(values) / len(values))


class _StreamingHistogramAUC:
    """Bounded-memory AUC using deterministic score bins."""

    def __init__(self, bins: int) -> None:
        if bins < 2:
            raise ValueError("AUC histogram requires at least two bins")
        self.bins = bins
        self.histogram = torch.zeros(2, bins, dtype=torch.float64)
        self.positives = self.histogram[0]
        self.negatives = self.histogram[1]

    def update(self, scores: Tensor, labels: Tensor) -> None:
        scores = scores.detach().float().flatten().cpu()
        labels = labels.detach().float().flatten().cpu()
        if scores.numel() != labels.numel():
            raise ValueError("AUC scores and labels must have the same length")
        if not scores.numel():
            return
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("AUC scores must be finite")
        if not bool(((labels == 0.0) | (labels == 1.0)).all()):
            raise ValueError("AUC labels must be binary")
        indices = torch.clamp(
            torch.floor(scores.clamp(0.0, 1.0) * self.bins).long(),
            max=self.bins - 1,
        )
        self.positives += torch.bincount(
            indices[labels == 1.0], minlength=self.bins
        ).to(torch.float64)
        self.negatives += torch.bincount(
            indices[labels == 0.0], minlength=self.bins
        ).to(torch.float64)

    def compute(self) -> float | None:
        positive_count = float(self.positives.sum().item())
        negative_count = float(self.negatives.sum().item())
        if positive_count == 0.0 or negative_count == 0.0:
            return None
        negatives_below = torch.cumsum(self.negatives, dim=0) - self.negatives
        concordant = (
            self.positives * (negatives_below + 0.5 * self.negatives)
        ).sum()
        return float((concordant / (positive_count * negative_count)).item())

    def counts(self) -> tuple[int, int, int]:
        positives = int(self.positives.sum().item())
        negatives = int(self.negatives.sum().item())
        return positives + negatives, positives, negatives


def _all_reduce_cpu_sum_(tensor: Tensor, context: DistributedContext) -> None:
    """Sum a CPU metric tensor even when the data process group is NCCL."""

    if not context.enabled:
        return
    if context.control_group is not None or context.device.type == "cpu":
        torch_dist.all_reduce(
            tensor,
            op=torch_dist.ReduceOp.SUM,
            group=context.control_group,
        )
        return
    device_value = tensor.to(context.device)
    torch_dist.all_reduce(device_value, op=torch_dist.ReduceOp.SUM)
    tensor.copy_(device_value.cpu())


def _reduce_evaluation_histograms(
    context: DistributedContext,
    accumulators: list[list[_StreamingHistogramAUC]],
    rows: int,
) -> int:
    if not context.enabled:
        return rows
    for task_accumulators in accumulators:
        for accumulator in task_accumulators:
            _all_reduce_cpu_sum_(accumulator.histogram, context)
    row_count = torch.tensor(rows, dtype=torch.long)
    _all_reduce_cpu_sum_(row_count, context)
    return int(row_count.item())


def _run_training_quick_eval(
    config: AppConfig,
    model: nn.Module,
    vocab_maps: dict[str, dict[str, int]],
    context: DistributedContext,
    quick_eval: QuickEvalConfig,
    *,
    fallback_batch: FeatureBatch | None,
    training_batch_iterator: Iterator[FeatureBatch] | None = None,
    training_batches_on_device: bool = False,
    max_batches: int | None = None,
) -> tuple[QuickEvalResult, tuple[FeatureBatch | None, ...]]:
    """Evaluate upcoming train batches or a deterministic held-out prefix.

    When ``quick_eval.split`` is ``train``, batches are consumed from the main
    training iterator and returned untouched so the caller can train those exact
    batches immediately afterward. No separate training reader is created.
    """

    split = (
        config.data.train if quick_eval.split == "train" else config.data.test
    )
    if split is None:
        raise ValueError(
            f"quick evaluation split {quick_eval.split!r} is not configured"
        )

    retain_batches = quick_eval.split == "train"
    if retain_batches and training_batch_iterator is None:
        raise ValueError(
            "training quick evaluation requires the main training batch iterator"
        )
    batch_limit = quick_eval.max_batches if max_batches is None else max_batches
    if batch_limit <= 0:
        raise ValueError("quick-evaluation max_batches must be positive")

    accumulators = [
        [_StreamingHistogramAUC(quick_eval.auc_bins)]
        for _ in config.task_names
    ]
    rows = 0
    local_batches = 0
    batch_iterator: Iterator[FeatureBatch] | None = None
    owns_batch_iterator = False
    staged_batches: list[FeatureBatch | None] = []
    replay_batch = fallback_batch
    was_training = model.training
    started = perf_counter()
    model.eval()
    try:
        non_blocking = _non_blocking_transfer(
            config,
            quick_eval.split,
            context.device,
        )
        if retain_batches:
            assert training_batch_iterator is not None
            batch_iterator = training_batch_iterator
        else:
            batch_iterator = iter(
                iter_feature_batches(
                    config,
                    quick_eval.split,
                    vocab_maps,
                    require_labels=True,
                    shard_rank=context.rank,
                    shard_world_size=context.world_size,
                    pin_memory=non_blocking,
                    include_group_id=False,
                )
            )
            owns_batch_iterator = True
        with torch.no_grad():
            while True:
                prefetched_device_batch: FeatureBatch | None = None
                if local_batches >= batch_limit:
                    local_batch = None
                else:
                    try:
                        if retain_batches and training_batches_on_device:
                            if not isinstance(
                                batch_iterator,
                                _DevicePrefetchIterator,
                            ):
                                raise RuntimeError(
                                    "device-resident training batches require the "
                                    "CUDA prefetch iterator"
                                )
                            (
                                local_batch,
                                prefetched_device_batch,
                            ) = batch_iterator.next_with_host()
                        else:
                            local_batch = next(batch_iterator)
                        local_batches += 1
                    except StopIteration:
                        local_batch = None
                rank_active = local_batch is not None
                active_ranks = _active_rank_count(context, rank_active)
                if active_ranks == 0:
                    break
                if retain_batches:
                    staged_batches.append(local_batch)
                if rank_active:
                    if local_batch is None:
                        raise AssertionError(
                            "active quick-evaluation rank is missing its batch"
                        )
                    batch = (
                        prefetched_device_batch
                        if prefetched_device_batch is not None
                        else move_feature_batch(
                            local_batch,
                            context.device,
                            non_blocking=non_blocking,
                        )
                    )
                    replay_batch = batch
                else:
                    if replay_batch is None:
                        raise RuntimeError(
                            "quick evaluation requires every rank to have either an "
                            "evaluation batch or a previous training batch for replay"
                        )
                    batch = replay_batch

                with _autocast_context(config, context.device):
                    logits = model(batch.features, batch.scenario_id)["logits"]
                if not rank_active:
                    continue
                if batch.labels is None:
                    raise RuntimeError(
                        "quick-evaluation batch did not contain labels"
                    )
                probabilities = torch.sigmoid(logits.float()).cpu()
                labels = batch.labels.float().cpu()
                label_mask = (
                    None
                    if batch.label_mask is None
                    else batch.label_mask.bool().cpu()
                )
                rows += int(labels.size(0))
                for task_index in range(len(config.task_names)):
                    if label_mask is None:
                        task_scores = probabilities[:, task_index]
                        task_labels = labels[:, task_index]
                    else:
                        valid = label_mask[:, task_index]
                        task_scores = probabilities[valid, task_index]
                        task_labels = labels[valid, task_index]
                    accumulators[task_index][0].update(
                        task_scores,
                        task_labels,
                    )

        rows = _reduce_evaluation_histograms(context, accumulators, rows)
        metrics: dict[str, dict[str, float | int | None]] = {}
        for task_index, task_name in enumerate(config.task_names):
            accumulator = accumulators[task_index][0]
            examples, positives, negatives = accumulator.counts()
            metrics[task_name] = {
                "auc": accumulator.compute(),
                "examples": examples,
                "positives": positives,
                "negatives": negatives,
            }
        _sync_device(context.device)
        return (
            QuickEvalResult(
                rows=rows,
                metrics=metrics,
                elapsed_seconds=perf_counter() - started,
            ),
            tuple(staged_batches),
        )
    finally:
        if owns_batch_iterator and batch_iterator is not None:
            close = getattr(batch_iterator, "close", None)
            if callable(close):
                close()
        model.train(was_training)


def _print_training_quick_eval(
    step: int,
    quick_eval: QuickEvalConfig,
    result: QuickEvalResult,
) -> None:
    print(
        f"Quick eval | step={step} split={quick_eval.split} rows={result.rows} "
        f"staged_for_training={str(quick_eval.split == 'train').lower()} "
        f"max_batches_per_rank={quick_eval.max_batches} "
        f"elapsed_seconds={result.elapsed_seconds:.6f}"
    )
    for task_name, metrics in result.metrics.items():
        auc = metrics["auc"]
        formatted_auc = "NA" if auc is None else f"{float(auc):.8f}"
        print(
            f"Quick eval task | step={step} task={task_name} auc={formatted_auc} "
            f"examples={metrics['examples']} positives={metrics['positives']} "
            f"negatives={metrics['negatives']}"
        )


class _DiskBackedGroupAUC:
    """Aggregate sparse (group, score-bin) counts without retaining predictions."""

    def __init__(self, bins: int) -> None:
        self.bins = bins
        self.temporary = tempfile.TemporaryDirectory(prefix="mdl-group-auc-")
        path = Path(self.temporary.name) / "groups.sqlite3"
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute(
            """
            CREATE TABLE counts (
                task INTEGER NOT NULL,
                scenario INTEGER NOT NULL,
                group_id TEXT NOT NULL,
                score_bin INTEGER NOT NULL,
                positives INTEGER NOT NULL,
                negatives INTEGER NOT NULL,
                PRIMARY KEY (task, scenario, group_id, score_bin)
            ) WITHOUT ROWID
            """
        )

    def add(
        self,
        task_index: int,
        group_ids: list[str],
        scores: Tensor,
        labels: Tensor,
        scenario_membership: Tensor,
    ) -> None:
        score_values = scores.detach().float().flatten().cpu()
        label_values = labels.detach().long().flatten().cpu()
        memberships = scenario_membership.detach().bool().cpu()
        if (
            len(group_ids) != score_values.numel()
            or label_values.numel() != score_values.numel()
            or memberships.size(0) != score_values.numel()
        ):
            raise ValueError("group AUC batch inputs must have matching rows")
        score_bins = torch.clamp(
            torch.floor(score_values.clamp(0.0, 1.0) * self.bins).long(),
            max=self.bins - 1,
        ).tolist()
        records: list[tuple[int, int, str, int, int, int]] = []
        membership_rows = memberships.tolist()
        for group_id, score_bin, label, member_row in zip(
            group_ids,
            score_bins,
            label_values.tolist(),
            membership_rows,
        ):
            positive = int(label == 1)
            negative = 1 - positive
            records.append(
                (task_index, -1, str(group_id), score_bin, positive, negative)
            )
            records.extend(
                (task_index, scenario, str(group_id), score_bin, positive, negative)
                for scenario, active in enumerate(member_row)
                if active
            )
        self.connection.executemany(
            """
            INSERT INTO counts(task, scenario, group_id, score_bin, positives, negatives)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(task, scenario, group_id, score_bin) DO UPDATE SET
                positives = positives + excluded.positives,
                negatives = negatives + excluded.negatives
            """,
            records,
        )

    @staticmethod
    def _finish_group(rows: list[tuple[int, int, int]]) -> float | None:
        positive_count = sum(item[1] for item in rows)
        negative_count = sum(item[2] for item in rows)
        if positive_count == 0 or negative_count == 0:
            return None
        negatives_below = 0
        concordant = 0.0
        for _score_bin, positives, negatives in rows:
            concordant += positives * (negatives_below + 0.5 * negatives)
            negatives_below += negatives
        return concordant / (positive_count * negative_count)

    def compute(self, task_index: int, scenario: int) -> float | None:
        cursor = self.connection.execute(
            """
            SELECT group_id, score_bin, positives, negatives
            FROM counts
            WHERE task = ? AND scenario = ?
            ORDER BY group_id, score_bin
            """,
            (task_index, scenario),
        )
        values: list[float] = []
        current_group: str | None = None
        group_rows: list[tuple[int, int, int]] = []
        for group_id, score_bin, positives, negatives in cursor:
            if current_group is not None and group_id != current_group:
                value = self._finish_group(group_rows)
                if value is not None:
                    values.append(value)
                group_rows = []
            current_group = group_id
            group_rows.append((score_bin, positives, negatives))
        if group_rows:
            value = self._finish_group(group_rows)
            if value is not None:
                values.append(value)
        return None if not values else float(sum(values) / len(values))

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()
        self.temporary.cleanup()


def _load_inference_model(
    config: AppConfig,
    device: torch.device,
    checkpoint_path: str | None,
    allow_random_init: bool,
    context: DistributedContext | None = None,
) -> tuple[nn.Module, dict[str, dict[str, int]]]:
    vocab_maps = load_vocab_maps(config)
    base_model = build_model(config, vocab_maps).to(device)
    if context is not None:
        _validate_sharded_embedding_metadata(context, base_model)
    resolved_checkpoint_path = checkpoint_path or config.training.checkpoint_path
    if resolved_checkpoint_path is None and not allow_random_init:
        raise ValueError(
            "evaluation requires a checkpoint; pass --checkpoint-path, set "
            "training.checkpoint_path, or pass --allow-random-init explicitly"
        )
    if resolved_checkpoint_path is not None:
        load_model_checkpoint(
            config,
            base_model,
            resolved_checkpoint_path,
            device=device,
        )
    model = _maybe_compile_model(config, base_model)
    base_model.eval()
    model.eval()
    return model, vocab_maps


@torch.no_grad()
def evaluate_mdl(
    config: AppConfig,
    split_name: str = "test",
    checkpoint_path: str | None = None,
    max_batches: int | None = None,
    allow_random_init: bool = False,
    group_metric_name: str | None = None,
    auc_bins: int = 65536,
) -> EvaluateResult:
    context = _setup_distributed(config)
    batch_iterator: Iterator[FeatureBatch] | None = None
    grouped_auc: _DiskBackedGroupAUC | None = None
    try:
        attention_runtime = _attention_runtime_description(config, context.device)
        if context.rank == 0:
            print(f"Attention backend | {attention_runtime}")
        config = _resolve_distributed_auto_scenarios(config, context)
        if split_name not in {"train", "test"}:
            raise ValueError("evaluation split must be train or test")
        if group_metric_name not in {None, "qauc", "uauc"}:
            raise ValueError("group_metric_name must be null, qauc, or uauc")
        if auc_bins < 2:
            raise ValueError("auc_bins must be at least 2")
        if context.enabled and group_metric_name is not None:
            raise ValueError(
                "distributed evaluation currently supports overall/per-scene AUC; "
                "qauc/uauc require single-process evaluation"
            )
        split = config.data.train if split_name == "train" else config.data.test
        if split is None:
            raise ValueError(f"split {split_name!r} is not configured")
        if list(split.labels) != config.task_names:
            raise ValueError(
                f"data.{split_name}.labels must declare the training tasks in the same order: "
                + ", ".join(config.task_names)
            )
        if group_metric_name is not None and split.group_id is None:
            raise ValueError(
                f"data.{split_name}.group_id is required for {group_metric_name.upper()}"
            )

        device = context.device
        model, vocab_maps = _load_inference_model(
            config,
            device,
            checkpoint_path,
            allow_random_init,
            context=context,
        )
        scenario_count = len(config.scenarios.names)
        auc_accumulators = [
            [_StreamingHistogramAUC(auc_bins) for _ in range(scenario_count + 1)]
            for _ in config.task_names
        ]
        grouped_auc = (
            _DiskBackedGroupAUC(auc_bins)
            if group_metric_name is not None
            else None
        )
        rows = 0
        non_blocking = _non_blocking_transfer(config, split_name, device)
        batch_iterator = iter(
            iter_feature_batches(
                config,
                split_name,
                vocab_maps,
                require_labels=True,
                shard_rank=context.rank,
                shard_world_size=context.world_size,
                pin_memory=non_blocking,
                include_group_id=group_metric_name is not None,
            )
        )
        local_batches = 0
        last_device_batch: FeatureBatch | None = None
        while True:
            if max_batches is not None and local_batches >= max_batches:
                local_batch = None
            else:
                try:
                    local_batch = next(batch_iterator)
                    local_batches += 1
                except StopIteration:
                    local_batch = None
            rank_active = local_batch is not None
            active_ranks = _active_rank_count(context, rank_active)
            if active_ranks == 0:
                break
            if last_device_batch is None and active_ranks != context.world_size:
                raise RuntimeError(
                    "sharded distributed evaluation requires every rank to provide an "
                    "initial test batch; reduce world_size or use a finer reader.shard_unit"
                )
            if rank_active:
                if local_batch is None:
                    raise AssertionError("active evaluation rank is missing its batch")
                batch = move_feature_batch(
                    local_batch,
                    device,
                    non_blocking=non_blocking,
                )
                last_device_batch = batch
            else:
                if last_device_batch is None:
                    raise RuntimeError("inactive evaluation rank has no replay batch")
                batch = last_device_batch
            with _autocast_context(config, device):
                logits = model(batch.features, batch.scenario_id)["logits"]
            if not rank_active:
                continue
            if batch.labels is None:
                raise RuntimeError("evaluation batch did not contain labels")
            probabilities = torch.sigmoid(logits.float()).cpu()
            labels = batch.labels.float().cpu()
            label_mask = (
                None
                if batch.label_mask is None
                else batch.label_mask.bool().cpu()
            )
            raw_scenarios = batch.scenario_id.cpu()
            if raw_scenarios.ndim == 1:
                scenario_membership = torch.nn.functional.one_hot(
                    raw_scenarios.long(),
                    num_classes=scenario_count,
                ).bool()
            else:
                scenario_membership = raw_scenarios.bool()
            rows += int(labels.size(0))
            for task_index in range(len(config.task_names)):
                if label_mask is None:
                    task_scores = probabilities[:, task_index]
                    task_labels = labels[:, task_index]
                    task_scenarios = scenario_membership
                    task_groups = batch.group_id
                else:
                    valid = label_mask[:, task_index]
                    task_scores = probabilities[valid, task_index]
                    task_labels = labels[valid, task_index]
                    task_scenarios = scenario_membership[valid]
                    task_groups = [
                        group_id
                        for group_id, keep in zip(batch.group_id, valid.tolist())
                        if keep
                    ]
                auc_accumulators[task_index][0].update(
                    task_scores, task_labels
                )
                for scenario in range(scenario_count):
                    selected = task_scenarios[:, scenario]
                    auc_accumulators[task_index][scenario + 1].update(
                        task_scores[selected], task_labels[selected]
                    )
                if grouped_auc is not None:
                    grouped_auc.add(
                        task_index,
                        task_groups,
                        task_scores,
                        task_labels,
                        task_scenarios,
                    )

        rows = _reduce_evaluation_histograms(
            context,
            auc_accumulators,
            rows,
        )
        metrics: dict[str, dict[str, float | int | None]] = {}
        for task_index, task_name in enumerate(config.task_names):
            overall = auc_accumulators[task_index][0]
            total, positives, negatives = overall.counts()
            values: dict[str, float | int | None] = {
                "auc": overall.compute(),
                "examples": total,
                "positives": positives,
                "negatives": negatives,
            }
            if grouped_auc is not None and group_metric_name is not None:
                values[group_metric_name] = grouped_auc.compute(task_index, -1)
            for scenario, scenario_name in enumerate(config.scenarios.names):
                accumulator = auc_accumulators[task_index][scenario + 1]
                scenario_total, scenario_positives, scenario_negatives = accumulator.counts()
                prefix = f"scene_{scenario_name}"
                scenario_auc = accumulator.compute()
                values[f"{prefix}_auc"] = scenario_auc
                values[f"{prefix}_examples"] = scenario_total
                values[f"{prefix}_positives"] = scenario_positives
                values[f"{prefix}_negatives"] = scenario_negatives
                if scenario_total > 0 and scenario_auc is None:
                    logger.warning(
                        "AUC is undefined for task=%s scene=%s: examples=%d positives=%d negatives=%d",
                        task_name,
                        scenario_name,
                        scenario_total,
                        scenario_positives,
                        scenario_negatives,
                    )
                if grouped_auc is not None and group_metric_name is not None:
                    values[f"{prefix}_{group_metric_name}"] = grouped_auc.compute(
                        task_index,
                        scenario,
                    )
            metrics[task_name] = values
        return EvaluateResult(
            rows=rows,
            group_metric_name=group_metric_name,
            metrics=metrics,
            auc_histogram_bins=auc_bins,
        )
    finally:
        if batch_iterator is not None:
            close = getattr(batch_iterator, "close", None)
            if callable(close):
                close()
        if grouped_auc is not None:
            grouped_auc.close()
        _cleanup_distributed(context)


@torch.no_grad()
def predict_mdl(
    config: AppConfig,
    checkpoint_path: str | None = None,
    output_path: str | None = None,
    max_batches: int | None = None,
    allow_random_init: bool = False,
) -> PredictResult:
    device = _select_device(config)
    attention_runtime = _attention_runtime_description(config, device)
    print(f"Attention backend | {attention_runtime}")
    config = resolve_auto_scenarios(config)
    pa, _pc, _ds, pq = _require_pyarrow()
    vocab_maps = load_vocab_maps(config)
    base_model = build_model(config, vocab_maps).to(device)
    resolved_checkpoint_path = checkpoint_path or config.training.checkpoint_path
    if resolved_checkpoint_path is None and not allow_random_init:
        raise ValueError(
            "prediction requires a checkpoint; pass --checkpoint-path, set "
            "training.checkpoint_path, or pass --allow-random-init explicitly"
        )
    if resolved_checkpoint_path is not None:
        load_model_checkpoint(
            config,
            base_model,
            resolved_checkpoint_path,
            device=device,
        )
    model = _maybe_compile_model(config, base_model)
    base_model.eval()
    model.eval()

    rows: list[dict[str, object]] = []
    split = config.data.test
    if split is None:
        raise ValueError("prediction requires data.test")
    score_columns = {
        task: f"{task}{split.prediction_score_suffix}"
        for task in config.task_names
    }
    output_names = {"group_id", *split.prediction_keys, *score_columns.values()}
    expected_name_count = 1 + len(split.prediction_keys) + len(score_columns)
    if len(output_names) != expected_name_count:
        raise ValueError(
            "prediction key and score output column names must be unique and must not use group_id"
        )
    seen_candidate_keys: set[tuple[object, ...]] = set()
    non_blocking = _non_blocking_transfer(config, "test", device)
    for batch_index, batch in enumerate(
        iter_feature_batches(
            config,
            "test",
            vocab_maps,
            require_labels=False,
            pin_memory=non_blocking,
        )
    ):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_feature_batch(batch, device, non_blocking=non_blocking)
        with _autocast_context(config, device):
            logits = model(batch.features, batch.scenario_id)["logits"]
        probabilities = torch.sigmoid(logits.float()).cpu().tolist()
        for row_index, (group_id, scores) in enumerate(
            zip(batch.group_id, probabilities)
        ):
            row = {"group_id": group_id}
            for output_name in split.prediction_keys:
                values = batch.prediction_keys.get(output_name)
                if values is None or len(values) != len(probabilities):
                    raise RuntimeError(
                        f"prediction batch is missing aligned key {output_name!r}"
                    )
                row[output_name] = values[row_index]
            if split.prediction_keys:
                identity = tuple(row[name] for name in split.prediction_keys)
                try:
                    duplicate = identity in seen_candidate_keys
                except TypeError as error:
                    raise ValueError(
                        "prediction keys must be scalar/hashable values"
                    ) from error
                if duplicate:
                    raise ValueError(
                        "prediction candidate key is not unique: "
                        + repr(dict(zip(split.prediction_keys, identity)))
                    )
                seen_candidate_keys.add(identity)
            row.update(
                {
                    score_columns[task]: float(score)
                    for task, score in zip(config.task_names, scores)
                }
            )
            rows.append(row)

    path = Path(output_path) if output_path else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        columns: dict[str, list[object]] = {
            "group_id": [row["group_id"] for row in rows]
        }
        for output_name in split.prediction_keys:
            columns[output_name] = [row[output_name] for row in rows]
        for task in config.task_names:
            score_column = score_columns[task]
            columns[score_column] = [row[score_column] for row in rows]
        pq.write_table(pa.table(columns), path)
    return PredictResult(rows=len(rows), output_path=path)
