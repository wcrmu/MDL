from __future__ import annotations

from bisect import bisect_left
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import timedelta
from importlib import import_module
import inspect
import logging
import math
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import queue
import sqlite3
import tempfile
import threading
from time import perf_counter, time_ns
from typing import Any, Callable, Iterator, MutableMapping

import numpy as np
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
    PreparedAxisBatch,
    PreparedBatchTable,
    SourceRegistry,
    _adapter_request_level_sources,
    _coalesce_feature_batch,
    _column_array,
    _require_pyarrow,
    _safe_table_take,
    axis_batch_to_feature_batch,
    build_packed_request_plan,
    build_request_deduplication_from_pack,
    discover_scenario_values,
    iter_adapted_axis_bundles,
    iter_flat_tables,
    iter_length_bucketed_packs,
    materialize_packed_blocks,
    move_feature_batch,
    pin_feature_batch,
    prepare_packed_arrow_axis_batch,
    prepare_packed_axis_batch,
    publish_direct_pipeline_stats,
    request_group_blocks_from_adapted_table,
    request_group_blocks_from_arrow_source,
    request_group_blocks_from_axis_bundle,
    reset_direct_pipeline_stats,
    resolve_auto_scenarios,
    run_feature_cardinality_audit,
    table_to_feature_batch,
)
from .features import load_vocab_maps
from .embeddings import (
    ShardedEmbedding,
    consume_sharded_embedding_stats,
    sharded_embedding_modules,
)
from .model import build_model
from .modules.attention import varlen_attention_available, varlen_attention_backend
from .modules.mlp import SparseMoEPerTokenFFN
from .optim import ShardedAdagrad, ShardedRowWiseAdagrad


logger = logging.getLogger(__name__)

_CONTROL_PROCESS_GROUP: torch_dist.ProcessGroup | None = None
# Cold scenario discovery on HDFS can exceed the default 10-minute store wait
# before the first collective. Keep process-group timeouts generous so peers
# survive a slow rank-0 metadata pass; discovery itself is also optimized.
_PROCESS_GROUP_TIMEOUT = timedelta(minutes=60)
_BATCH_SEQUENCE_WORK_COLUMN = "__mdl_batch_sequence_work"


def _varlen_attention_reasons(config: AppConfig) -> tuple[str, ...]:
    """Human-readable reasons this config needs ``flash_attn`` varlen Flash."""

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
    """True when strict flash would execute the packed flash-attn varlen path."""

    return bool(_varlen_attention_reasons(config))


def _needs_padded_sdpa_flash(config: AppConfig) -> bool:
    """True when strict flash would also exercise ordinary padded SDPA Flash.

    LONGER / OneTrans S-streams use flash-attn varlen. Ordinary padded FlashAttention is
    only required when MDL constructs ``DomainAwareAttention`` for enabled
    task/scenario feature interactions. Plain RankMixer token mixing does not
    use padded Flash, so both capabilities are independent.
    """

    model = config.model
    if model.name not in {
        "mdl_rankmixer",
        "mdl_onetrans",
        "mdl_mixformer",
    }:
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

    requested = getattr(config.runtime, "attention_backend", "flash")
    reasons = _varlen_attention_reasons(config)
    needs_varlen = _requires_varlen_attention(config)
    needs_padded = _needs_padded_sdpa_flash(config)
    varlen_api = varlen_attention_available()
    varlen_backend = varlen_attention_backend() if varlen_api else None
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
                "flash_attn.flash_attn_varlen_func for "
                f"{detail}, but that API is unavailable. Install a Dao-AILab "
                "flash-attn build compatible with the deployed PyTorch/CUDA "
                "runtime, or explicitly use runtime.attention_backend='sdpa'."
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
            f"varlen_backend={varlen_backend or 'unavailable'} "
            f"strict=true device={device} precision={config.runtime.precision}"
        )
    return (
        f"requested={requested} resolved=padded_sdpa "
        f"kernel_policy=runtime_dispatch "
        f"flash_path_requires_varlen={needs_varlen} "
        f"flash_path_requires_padded_sdpa={needs_padded} "
        f"varlen_api_available={varlen_api} "
        f"varlen_backend={varlen_backend or 'unavailable'} "
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
    """Phase timings for one training step.

    ``train_mdl`` defaults to synchronized phase boundaries for diagnostic
    precision. Throughput benchmarks disable those synchronizations so the
    observer cannot manufacture GPU bubbles; in that mode phase values are CPU
    enqueue/wait times while aggregate throughput and GPU utilization remain
    representative of normal training.
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


def _build_model_on_device(
    config: AppConfig,
    vocab_maps: dict[str, dict[str, int]],
    device: torch.device,
) -> nn.Module:
    """Construct large models directly on their destination accelerator.

    Industrial embedding shards can occupy tens of GiB per rank. Building them
    on CPU and then calling ``to(cuda)`` leaves every GPU idle during CPU RNG
    initialization, doubles transient host memory, and performs one enormous
    serial H2D copy. The device context makes parameter initialization happen
    independently on each rank's GPU while the final ``to`` catches any module
    that explicitly requested CPU storage.
    """

    if device.type == "cuda":
        with torch.device(device):
            model = build_model(config, vocab_maps)
        return model.to(device)
    return build_model(config, vocab_maps).to(device)


def _dev_shm_free_bytes() -> int:
    try:
        st = os.statvfs("/dev/shm")
    except OSError:
        return 0
    return int(st.f_bavail) * int(st.f_frsize)


def _local_cuda_p2p_accessible() -> bool | None:
    """Probe whether every visible CUDA device pair can use P2P.

    Returns:
        True: all pairs report peer access (NCCL can use P2P/NVLink).
        False: at least one pair cannot (must fall back to SHM/NET).
        None: CUDA unavailable, probe failed, or this process only sees one
        GPU while other local ranks own peers (cannot claim P2P is healthy).
    """

    if not torch.cuda.is_available():
        return None
    device_count = int(torch.cuda.device_count())
    local_world = _env_int("LOCAL_WORLD_SIZE", _env_int("WORLD_SIZE", 1))
    if device_count <= 1:
        # torchrun remaps each rank to a single visible GPU, so peer access
        # cannot be probed here. Treat multi-GPU local jobs as inconclusive
        # unless the parent launcher already decided (env already set).
        return True if local_world <= 1 else None
    try:
        for source in range(device_count):
            for peer in range(device_count):
                if source == peer:
                    continue
                if not torch.cuda.can_device_access_peer(source, peer):
                    return False
    except Exception:  # noqa: BLE001 - probe must never crash launch
        return None
    return True


def _configure_nccl_runtime_env(
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Use CUDA P2P when healthy; otherwise tell NCCL to fall back safely.

    - P2P OK → leave NCCL defaults so NVLink/P2P stays enabled.
    - P2P broken → ``NCCL_IGNORE_DISABLED_P2P=1`` + ``NCCL_P2P_DISABLE=1``
      so NCCL skips doomed P2P and uses SHM/NET.
    - Inconclusive (e.g. each torchrun rank only sees one GPU) → only
      ``NCCL_IGNORE_DISABLED_P2P=1``: still try P2P when the fabric allows
      it, but do not abort if the driver/NVLink reports disabled P2P.
    - Explicit env exports always win.

    ``environ`` defaults to ``os.environ``; the DDP launcher may pass a copied
    dict so child processes inherit the decision.
    """

    env: MutableMapping[str, str] = os.environ if environ is None else environ
    if "NCCL_IGNORE_DISABLED_P2P" in env:
        return
    accessible = _local_cuda_p2p_accessible()
    if accessible is True:
        if is_main_process():
            logger.info(
                "CUDA P2P accessible across %d visible GPU(s); NCCL will use P2P",
                int(torch.cuda.device_count()),
            )
        return
    if accessible is False:
        env["NCCL_IGNORE_DISABLED_P2P"] = "1"
        # Skip doomed P2P attempts; SHM/NET path is the working transport here.
        env.setdefault("NCCL_P2P_DISABLE", "1")
        if is_main_process():
            logger.warning(
                "CUDA P2P is not accessible between some visible GPUs; "
                "enabling NCCL_IGNORE_DISABLED_P2P=1 and NCCL_P2P_DISABLE=%s "
                "so NCCL falls back to SHM/NET",
                env.get("NCCL_P2P_DISABLE", "1"),
            )
        return
    # Probe inconclusive: keep the historical safe default.
    env.setdefault("NCCL_IGNORE_DISABLED_P2P", "1")
    if is_main_process():
        logger.info(
            "CUDA P2P probe inconclusive; defaulting NCCL_IGNORE_DISABLED_P2P=%s",
            env.get("NCCL_IGNORE_DISABLED_P2P"),
        )


def _resolve_process_group_backend(device: torch.device) -> str:
    """Pick a process-group backend that can actually initialize here.

    NCCL 2.2x still allocates ~32MiB/rank under ``/dev/shm`` during init even
    when ``NCCL_SHM_DISABLE=1``. Containers with the default 64MiB shm therefore
    cannot form a 2-rank NCCL group. Fall back to Gloo (CPU collectives; CUDA
    tensors are staged in the embedding all-to-all helpers).
    """

    forced = os.environ.get("MDL_DIST_BACKEND", "").strip().lower()
    if forced in {"nccl", "gloo"}:
        return forced
    if device.type != "cuda":
        return "gloo"
    shm_free = _dev_shm_free_bytes()
    # Two ranks need ~33MiB each for NCCL's SHM segments on this stack.
    if shm_free and shm_free < 256 * 1024 * 1024:
        logger.warning(
            "Using Gloo process group because /dev/shm free=%.1fMiB is too "
            "small for NCCL (need host --shm-size>=1g for NCCL multi-GPU). "
            "Set MDL_DIST_BACKEND=nccl to force NCCL.",
            shm_free / (1024 * 1024),
        )
        return "gloo"
    return "nccl"


def _setup_distributed(config: AppConfig) -> DistributedContext:
    global _CONTROL_PROCESS_GROUP
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    enabled = world_size > 1
    device = _select_device(config, local_rank if enabled else None)
    initialized_here = False

    if enabled and not torch_dist.is_initialized():
        if device.type == "cuda":
            _configure_nccl_runtime_env()
        backend = _resolve_process_group_backend(device)
        torch_dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=_PROCESS_GROUP_TIMEOUT,
        )
        initialized_here = True
        if rank == 0:
            logger.info("Initialized process group backend=%s world_size=%d", backend, world_size)

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


def _resolve_distributed_cardinality_audit(
    config: AppConfig,
    context: DistributedContext,
    split_name: str,
) -> AppConfig:
    """Sample scalar/bag cardinalities on each rank, merge, fail once if needed."""

    # A few focused optimizer/DDP tests intentionally use a minimal config
    # double and inject batches directly. Real AppConfig instances always have
    # data splits and ReaderConfig objects, so skipping here only preserves that
    # narrow dependency-injection boundary.
    data = getattr(config, "data", None)
    if data is None:
        return config
    split = data.train if split_name == "train" else data.test
    if split is None or not hasattr(split, "reader"):
        return config
    if split.reader.effective_cardinality_audit_raw_rows() <= 0:
        return config
    auditor = run_feature_cardinality_audit(
        config,
        split_name,
        shard_rank=context.rank,
        shard_world_size=context.world_size if context.enabled else 1,
        process_group=(
            context.control_group
            if context.enabled and context.control_group is not None
            else None
        ),
    )
    if auditor is not None and context.rank == 0:
        logger.info(
            "Feature cardinality audit passed for split %s (raw_rows_seen=%s)",
            split_name,
            auditor.raw_rows_seen,
        )
    return config


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
    """Yield one table per request while preserving candidate order.

    Candidate-major agg output can interleave requests. Reordering the whole
    Arrow table once is substantially cheaper than taking 169 wide columns
    separately for every non-contiguous request.
    """

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

    groups = list(positions_by_request.values())
    if all(
        positions == list(range(positions[0], positions[0] + len(positions)))
        for positions in groups
    ):
        for positions in groups:
            yield table.slice(positions[0], len(positions))
        return

    reordered = _safe_table_take(
        table,
        [position for positions in groups for position in positions],
    )
    offset = 0
    for positions in groups:
        yield reordered.slice(offset, len(positions))
        offset += len(positions)


def _shuffle_table_groups(
    tables: list[object],
    generator: torch.Generator,
) -> list[object]:
    if len(tables) <= 1:
        return tables
    permutation = torch.randperm(len(tables), generator=generator).tolist()
    return [tables[index] for index in permutation]


def _concat_batch_tables(pa: Any, tables: list[object]) -> object:
    """Concatenate and coalesce request slices before tensorization.

    Slicing dictionary-encoded request lists keeps the source batch's complete
    dictionary attached to every small request chunk. Concatenating dozens of
    those chunks can make a 2 MiB logical batch appear hundreds of MiB wide and
    forces every feature encoder to revisit the duplicated dictionaries.
    Decode only the selected indices at this boundary; request deduplication
    still happens before tensorization.
    """

    if not tables:
        raise ValueError("cannot concatenate an empty batch-table list")
    if len(tables) == 1:
        return tables[0]

    def dictionary_storage_key(dictionary: Any) -> tuple[Any, ...]:
        return (
            str(dictionary.type),
            len(dictionary),
            dictionary.offset,
            tuple(
                None if buffer is None else (buffer.address, buffer.size)
                for buffer in dictionary.buffers()
            ),
        )

    def _decode_dictionary_columns(table: object) -> object:
        """Materialize dictionary columns so mixed list/dict schemas can concat."""

        arrays: list[Any] = []
        changed = False
        for column_index, name in enumerate(table.column_names):
            column = table[name]
            if not any(
                pa.types.is_dictionary(chunk.type) for chunk in column.chunks
            ):
                arrays.append(column.combine_chunks())
                continue
            changed = True
            decoded = [
                chunk.dictionary_decode()
                if pa.types.is_dictionary(chunk.type)
                else chunk
                for chunk in column.chunks
            ]
            arrays.append(
                decoded[0] if len(decoded) == 1 else pa.concat_arrays(decoded)
            )
        if not changed:
            return table
        return pa.Table.from_arrays(arrays, names=table.column_names)

    # Request slices produced from one adapted scanner table share the same
    # dictionary buffers. Group those slices by source and decode one combined
    # index vector per source/column instead of calling dictionary_decode for
    # every tiny request slice (hundreds of calls per wide batch).
    dictionary_column_index: int | None = None
    for column_index in range(tables[0].num_columns):
        if all(
            table[column_index].num_chunks == 1
            and pa.types.is_dictionary(table[column_index].chunk(0).type)
            for table in tables
        ):
            dictionary_column_index = column_index
            break

    if dictionary_column_index is not None and all(
        all(column.num_chunks == 1 for column in table.columns)
        for table in tables
    ):
        source_groups: dict[tuple[Any, ...], list[int]] = {}
        for table_index, table in enumerate(tables):
            dictionary = table[dictionary_column_index].chunk(0).dictionary
            storage_key = dictionary_storage_key(dictionary)
            source_groups.setdefault(storage_key, []).append(table_index)

        arrays: list[Any] = []
        for column_index in range(tables[0].num_columns):
            source_arrays: list[Any] = []
            for table_indices in source_groups.values():
                chunks = [
                    tables[table_index][column_index].chunk(0)
                    for table_index in table_indices
                ]
                if pa.types.is_dictionary(chunks[0].type):
                    dictionary = chunks[0].dictionary
                    # Source groups were built from slices of one adapted
                    # table, so every column shares its dictionary within the
                    # group. Check the last slice as a cheap defensive guard.
                    if (
                        len(chunks) == 1
                        or dictionary_storage_key(chunks[-1].dictionary)
                        == dictionary_storage_key(dictionary)
                    ):
                        indices = (
                            chunks[0].indices
                            if len(chunks) == 1
                            else pa.concat_arrays([chunk.indices for chunk in chunks])
                        )
                        source_arrays.append(
                            pa.DictionaryArray.from_arrays(
                                indices,
                                dictionary,
                            ).dictionary_decode()
                        )
                    else:
                        source_arrays.append(
                            pa.concat_arrays(
                                [chunk.dictionary_decode() for chunk in chunks]
                            )
                        )
                else:
                    source_arrays.append(
                        chunks[0]
                        if len(chunks) == 1
                        else pa.concat_arrays(chunks)
                    )
            arrays.append(
                source_arrays[0]
                if len(source_arrays) == 1
                else pa.concat_arrays(source_arrays)
            )
        return pa.Table.from_arrays(arrays, names=tables[0].column_names)

    # Scanner batches can interleave plain list<> and dictionary<list<>>
    # encodings for the same logical column. Normalize before concat_tables.
    try:
        schemas_match = all(table.schema.equals(tables[0].schema) for table in tables[1:])
    except Exception:
        schemas_match = False
    if not schemas_match:
        tables = [_decode_dictionary_columns(table) for table in tables]

    combined = pa.concat_tables(tables)
    if all(column.num_chunks <= 1 for column in combined.columns):
        return combined
    arrays = []
    for column in combined.columns:
        if any(pa.types.is_dictionary(chunk.type) for chunk in column.chunks):
            decoded = [
                chunk.dictionary_decode()
                if pa.types.is_dictionary(chunk.type)
                else chunk
                for chunk in column.chunks
            ]
            arrays.append(
                decoded[0] if len(decoded) == 1 else pa.concat_arrays(decoded)
            )
        else:
            arrays.append(column.combine_chunks())
    return pa.Table.from_arrays(arrays, names=combined.column_names)


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
    def request_groups() -> Iterator[object]:
        for table in candidate_source:
            if not table.num_rows:
                continue
            if (
                reader.deduplicate_request_features
                and reader.length_buckets
                and config.sequences
                and all(
                    not sequence.fields
                    or sequence.fields[0].source in table.column_names
                    for sequence in config.sequences
                )
            ):
                if _BATCH_SEQUENCE_WORK_COLUMN in table.column_names:
                    raise ValueError(
                        f"input data uses reserved column {_BATCH_SEQUENCE_WORK_COLUMN!r}"
                    )
                pa, _pc, _ds, _pq = _require_pyarrow()
                lengths = _table_effective_sequence_lengths(
                    config,
                    table,
                    metric=reader.length_bucket_metric,
                )
                table = table.append_column(
                    _BATCH_SEQUENCE_WORK_COLUMN,
                    pa.array(lengths.numpy(), type=pa.int64()),
                )
            yield from _request_group_tables(split, table)

    source = request_groups()
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
        if table.num_rows > reader.shuffle_buffer_rows:
            yield from _shuffle_table_groups(buffered_groups, generator)
            buffered_groups = []
            buffered_rows = 0
            yield table
            continue
        while (
            buffered_groups
            and buffered_rows + table.num_rows > reader.shuffle_buffer_rows
        ):
            selected_index = int(
                torch.randint(
                    len(buffered_groups),
                    (),
                    generator=generator,
                ).item()
            )
            selected = buffered_groups[selected_index]
            buffered_groups[selected_index] = buffered_groups[-1]
            buffered_groups.pop()
            buffered_rows -= selected.num_rows
            yield selected
        buffered_groups.append(table)
        buffered_rows += table.num_rows
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
                yield _concat_batch_tables(pa, buffered)
                buffered = []
                buffered_rows = 0
            yield table.slice(0, batch_size)
            table = table.slice(batch_size)
        if not table.num_rows:
            continue
        if buffered_rows and buffered_rows + table.num_rows > batch_size:
            yield _concat_batch_tables(pa, buffered)
            buffered = []
            buffered_rows = 0
        buffered.append(table)
        buffered_rows += table.num_rows
        if buffered_rows == batch_size:
            yield _concat_batch_tables(pa, buffered)
            buffered = []
            buffered_rows = 0
    if buffered_rows:
        yield _concat_batch_tables(pa, buffered)


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
            # the vectorized metric attached before grouping avoids repeating
            # nine Arrow length kernels for every individual request.
            if _BATCH_SEQUENCE_WORK_COLUMN in table.column_names:
                effective_length = int(
                    table[_BATCH_SEQUENCE_WORK_COLUMN][0].as_py()
                )
            else:
                effective_length = int(
                    _table_effective_sequence_lengths(
                        config,
                        table.slice(0, 1),
                        metric=reader.length_bucket_metric,
                    ).item()
                )
            bucket_index = bisect_left(finite_boundaries, effective_length)
            bucket = buckets[bucket_index]
            remaining = table
            while remaining.num_rows > bucket.batch_size:
                if buffered_rows[bucket_index]:
                    yield _concat_batch_tables(pa, buffered[bucket_index])
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
                yield _concat_batch_tables(pa, buffered[bucket_index])
                buffered[bucket_index] = []
                buffered_rows[bucket_index] = 0
            buffered[bucket_index].append(remaining)
            buffered_rows[bucket_index] += remaining.num_rows
            if buffered_rows[bucket_index] == bucket.batch_size:
                yield _concat_batch_tables(pa, buffered[bucket_index])
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
            combined = _concat_batch_tables(pa, buffered[bucket_index])
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
        yield _concat_batch_tables(pa, buffered[bucket_index])


def _iter_batch_tables_direct(
    config: AppConfig,
    split_name: str,
    shard_rank: int,
    shard_world_size: int,
    require_labels: bool = True,
) -> Iterator[object]:
    """Axis-separated adapt → descriptor pack → direct FeatureBatch payload.

    Skips candidate-flat materialization of whole scanner tables. Adapted
    payloads stay as :class:`AdaptedAxisBundle` or :class:`ArrowAxisSource`
    under a :class:`SourceRegistry` until every referencing block has been
    packed; then the source is released. The adapter-parquet path yields
    :class:`PreparedAxisBatch` without rebuilding Arrow. ``flat_parquet``
    remains a transitional narrow-Arrow fallback.
    """

    reader = _split_reader(config, split_name)
    split = config.data.train if split_name == "train" else config.data.test
    if split is None:
        raise ValueError(f"split {split_name!r} is not configured")
    if not reader.deduplicate_request_features:
        raise ValueError(
            "reader.agg_direct_mode requires reader.deduplicate_request_features=true"
        )
    if split.request_id is None:
        raise ValueError("reader.agg_direct_mode requires split.request_id")

    adapter_options = {} if split.adapter is None else split.adapter.options
    request_level_sources = _adapter_request_level_sources(adapter_options)
    sequence_sources = {
        field.source
        for sequence in config.sequences
        for field in sequence.fields
    }

    use_axis = split.format == "adapter_parquet"
    use_arrow_axis = use_axis and reader.agg_direct_mode == "direct_arrow"
    registry = SourceRegistry()
    reset_direct_pipeline_stats()

    def blocks() -> Iterator[Any]:
        if use_axis:
            producer_queue = 2 if reader.prefetch_batches > 0 else 1
            bundle_iter = iter_adapted_axis_bundles(
                config,
                split_name,
                shard_rank=shard_rank,
                shard_world_size=shard_world_size,
                require_labels=require_labels,
                producer_queue_size=producer_queue,
                arrow_axis=use_arrow_axis,
            )
            for bundle in bundle_iter:
                source_id = registry.put(bundle)
                if use_arrow_axis:
                    group_blocks = request_group_blocks_from_arrow_source(
                        bundle,
                        source_id=source_id,
                        sequences=config.sequences,
                        length_bucket_metric=reader.length_bucket_metric,
                    )
                else:
                    group_blocks = request_group_blocks_from_axis_bundle(
                        bundle,
                        source_id=source_id,
                        sequences=config.sequences,
                        length_bucket_metric=reader.length_bucket_metric,
                    )
                if not group_blocks:
                    # put() leaves refcount 0; release(0) drops the empty payload.
                    registry.release(source_id, 0)
                    continue
                registry.acquire(source_id, len(group_blocks))
                yield from group_blocks
            return

        # Transitional fallback for flat_parquet: still uses adapted tables.
        for source_id, table in enumerate(
            iter_candidate_tables(
                config,
                split_name,
                shard_rank=shard_rank,
                shard_world_size=shard_world_size,
                require_labels=require_labels,
            )
        ):
            if not table.num_rows:
                continue
            source_id = registry.put(table)
            group_blocks = request_group_blocks_from_adapted_table(
                table,
                source_id=source_id,
                request_id_column=split.request_id,
                sequences=config.sequences,
                length_bucket_metric=reader.length_bucket_metric,
            )
            registry.acquire(source_id, len(group_blocks))
            yield from group_blocks

    try:
        for pack in iter_length_bucketed_packs(
            blocks(),
            buckets=reader.length_buckets,
            default_batch_size=config.training.batch_size,
            shuffle_buffer_rows=reader.shuffle_buffer_rows,
            shuffle_seed=reader.shuffle_seed,
            shard_rank=shard_rank,
        ):
            packed = build_packed_request_plan(pack)
            registry.observe_pack(packed.blocks)
            sources_in_pack: dict[int, int] = {}
            source_releases_in_pack: dict[int, int] = {}
            for block in packed.blocks:
                sources_in_pack[block.source_id] = (
                    sources_in_pack.get(block.source_id, 0) + 1
                )
                if block.releases_source_reference:
                    source_releases_in_pack[block.source_id] = (
                        source_releases_in_pack.get(block.source_id, 0) + 1
                    )

            if use_axis:
                retained = {
                    source_id: registry.get(source_id)
                    for source_id in sources_in_pack
                }
                candidate_request_columns = sorted(
                    {
                        *(
                            [split.request_id]
                            if split.request_id is not None
                            else []
                        ),
                        *(
                            [split.group_id]
                            if split.group_id is not None
                            else []
                        ),
                        *(
                            [config.scenarios.source]
                            if config.scenarios.source is not None
                            else []
                        ),
                        *split.prediction_keys.values(),
                    }
                )
                if use_arrow_axis:
                    prepared = prepare_packed_arrow_axis_batch(
                        retained,
                        packed,
                        sequences=config.sequences,
                        request_id_column=split.request_id,
                        candidate_request_columns=candidate_request_columns,
                    )
                else:
                    prepared = prepare_packed_axis_batch(
                        retained,
                        packed,
                        sequences=config.sequences,
                        request_id_column=split.request_id,
                        candidate_request_columns=candidate_request_columns,
                    )
            else:
                source_tables = {
                    source_id: registry.get(source_id)
                    for source_id in sources_in_pack
                }
                candidate_table = materialize_packed_blocks(
                    source_tables, packed.blocks
                )
                request_columns = sorted(
                    {
                        *(
                            [split.request_id]
                            if split.request_id is not None
                            else []
                        ),
                        *request_level_sources,
                        *sequence_sources,
                    }
                )
                request_dedup = build_request_deduplication_from_pack(
                    packed,
                    source_tables,
                    columns=request_columns,
                )
                prepared = PreparedBatchTable(
                    table=candidate_table,
                    request_deduplication=request_dedup,
                )

            try:
                yield prepared
            finally:
                for source_id, count in source_releases_in_pack.items():
                    registry.release(source_id, count)
    finally:
        publish_direct_pipeline_stats(registry.snapshot_stats())


def _iter_batch_tables(
    config: AppConfig,
    split_name: str,
    shard_rank: int,
    shard_world_size: int,
    require_labels: bool = True,
) -> Iterator[object]:
    reader = _split_reader(config, split_name)
    if (
        reader.agg_direct_mode in {"direct", "direct_arrow", "compare"}
        and reader.deduplicate_request_features
    ):
        yield from _iter_batch_tables_direct(
            config,
            split_name,
            shard_rank,
            shard_world_size,
            require_labels,
        )
        return
    for table in _iter_length_bucketed_tables(
        config,
        split_name,
        shard_rank,
        shard_world_size,
        require_labels,
    ):
        if _BATCH_SEQUENCE_WORK_COLUMN in table.column_names:
            table = table.drop_columns([_BATCH_SEQUENCE_WORK_COLUMN])
        yield table


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


def _assert_feature_batch_equal(
    legacy: FeatureBatch,
    direct: FeatureBatch,
    *,
    batch_index: int,
) -> None:
    """Strict runtime oracle with a path and first differing tensor element."""

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(left, Tensor):
            if not isinstance(right, Tensor):
                raise AssertionError(
                    f"{path}: legacy is Tensor, direct is "
                    f"{type(right).__name__}"
                )
            if left.dtype != right.dtype:
                raise AssertionError(
                    f"{path}: dtype legacy={left.dtype}, direct={right.dtype}"
                )
            if left.shape != right.shape:
                raise AssertionError(
                    f"{path}: shape legacy={tuple(left.shape)}, "
                    f"direct={tuple(right.shape)}"
                )
            if left.dtype.is_floating_point:
                equal = torch.isclose(
                    left,
                    right,
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
            else:
                equal = left == right
            if bool(equal.all().item()):
                return
            first = torch.nonzero(~equal, as_tuple=False)[0].tolist()
            index = tuple(int(item) for item in first)
            raise AssertionError(
                f"{path}{index}: legacy={left[index].item()!r}, "
                f"direct={right[index].item()!r}"
            )
        if isinstance(left, dict):
            if not isinstance(right, dict):
                raise AssertionError(
                    f"{path}: legacy is dict, direct is "
                    f"{type(right).__name__}"
                )
            if set(left) != set(right):
                raise AssertionError(
                    f"{path}: key difference "
                    f"{sorted(set(left).symmetric_difference(right))}"
                )
            for key in left:
                compare(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, (list, tuple)):
            if not isinstance(right, type(left)):
                raise AssertionError(
                    f"{path}: container type legacy={type(left).__name__}, "
                    f"direct={type(right).__name__}"
                )
            if len(left) != len(right):
                raise AssertionError(
                    f"{path}: length legacy={len(left)}, direct={len(right)}"
                )
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if left != right:
            raise AssertionError(f"{path}: legacy={left!r}, direct={right!r}")

    try:
        for attribute in (
            "features",
            "labels",
            "label_mask",
            "scenario_id",
            "group_id",
            "prediction_keys",
        ):
            compare(
                getattr(legacy, attribute),
                getattr(direct, attribute),
                attribute,
            )
    except AssertionError as error:
        raise AssertionError(
            f"agg_direct compare mismatch in batch {batch_index}: {error}"
        ) from error


def _config_with_reader_mode(
    config: AppConfig,
    split_name: str,
    mode: str,
) -> AppConfig:
    split = config.data.train if split_name == "train" else config.data.test
    if split is None:
        raise ValueError(f"split {split_name!r} is not configured")
    updated_split = replace(
        split,
        reader=replace(split.reader, agg_direct_mode=mode),
    )
    updated_data = (
        replace(config.data, train=updated_split)
        if split_name == "train"
        else replace(config.data, test=updated_split)
    )
    return replace(config, data=updated_data)


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

    if isinstance(table, PreparedAxisBatch):
        tensor_bytes = table.request_row_indices.numel() * 8
        for feature in config.features:
            # Metadata columns may be broadcast onto candidates as well as
            # retained on requests. Treat those as candidate-major here; the
            # overestimate is intentional for queue admission.
            request_level = (
                feature.source in table.request_values
                and feature.source not in table.candidate_values
            )
            axis_rows = table.n_requests if request_level else table.n_candidates
            if feature.kind == "categorical" and feature.pooling == "mean":
                values = (
                    table.request_values
                    if request_level
                    else table.candidate_values
                ).get(feature.source, ())
                max_length = (
                    int(feature.max_length)
                    if feature.max_length is not None
                    else max(
                        (
                            len(value)
                            for value in values
                            if isinstance(value, (list, tuple))
                        ),
                        default=0,
                    )
                )
                tensor_bytes += axis_rows * (8 + max_length * 8)
            else:
                element_bytes = 8 if feature.kind == "categorical" else 4
                tensor_bytes += axis_rows * feature.dimension * element_bytes
        for sequence in config.sequences:
            plan = table.sequence_plans.get(sequence.name)
            padded_length = (
                int(plan.compacted_lengths.max())
                if plan is not None and plan.compacted_lengths.size
                else 0
            )
            tensor_bytes += table.n_requests * 8
            for field in sequence.fields:
                element_bytes = 8 if field.kind == "categorical" else 4
                tensor_bytes += (
                    table.n_requests
                    * padded_length
                    * field.dimension
                    * element_bytes
                )
        tensor_bytes += table.n_candidates * (
            4 * max(1, len(config.task_names)) + 16
        )
        # Axis payload consists of Python scalar/list references retained until
        # tensorization. Two tensor footprints plus 25% allocator headroom is a
        # conservative queue reservation without rescanning every nested value.
        return max(1, tensor_bytes * 2 + tensor_bytes // 4)

    request_table = None
    if isinstance(table, PreparedBatchTable):
        if table.request_deduplication is not None:
            request_table = table.request_deduplication[0]
        table = table.table

    rows = int(table.num_rows)
    tensor_bytes = 0
    bag_max_lengths: dict[str, int] = {}
    for feature in config.features:
        if feature.kind == "categorical" and feature.pooling == "mean":
            if feature.max_length is not None:
                # The configured truncation limit is already a conservative
                # tensor bound. Scanning every bag column with Arrow kernels
                # just to discover a smaller value can cost more than
                # tensorizing the batch itself on very wide schemas.
                max_length = feature.max_length
            else:
                max_length = bag_max_lengths.get(feature.source)
                if max_length is None:
                    source_table = table
                    if (
                        request_table is not None
                        and feature.source in request_table.column_names
                    ):
                        source_table = request_table
                    try:
                        max_length = _max_bag_length(source_table, feature.source)
                    except (KeyError, TypeError, AttributeError):
                        max_length = 1
                    bag_max_lengths[feature.source] = max_length
            tensor_bytes += rows * (8 + max_length * 8)
        else:
            tensor_bytes += rows * feature.dimension * (8 if feature.kind == "categorical" else 4)
    for sequence in config.sequences:
        if not sequence.fields:
            continue
        source = sequence.fields[0].source
        length_table = table
        if request_table is not None and source in request_table.column_names:
            length_table = request_table
        lengths = _table_sequence_lengths(config, sequence, length_table)
        padded_length = int(lengths.max().item()) if lengths.numel() else 0
        tensor_bytes += rows * 8
        for field in sequence.fields:
            element_bytes = 8 if field.kind == "categorical" else 4 * field.dimension
            tensor_bytes += rows * padded_length * element_bytes
    # Labels, masks, scenario IDs, and a margin for Python/allocator metadata.
    tensor_bytes += rows * (4 * max(1, len(config.task_names)) + 16)
    arrow_bytes = int(getattr(table, "nbytes", 0))
    if request_table is not None:
        arrow_bytes += int(getattr(request_table, "nbytes", 0))
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
    if isinstance(table, PreparedAxisBatch):
        batch = axis_batch_to_feature_batch(
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

    request_deduplication = None
    candidate_table = table
    if isinstance(table, PreparedBatchTable):
        candidate_table = table.table
        request_deduplication = table.request_deduplication
    batch = table_to_feature_batch(
        config,
        candidate_table,
        vocab_maps,
        require_labels=require_labels,
        include_group_id=include_group_id,
        split=split,
        **(
            {"request_deduplication": request_deduplication}
            if isinstance(table, PreparedBatchTable)
            else {}
        ),
    )
    return (
        pin_feature_batch(
            batch,
            coalesce_tensors=coalesce_pinned_tensors,
        )
        if pin_memory
        else batch
    )


def _schedule_overlapped_host_prepare(batch_iterator: object) -> None:
    """Fetch+prepare the next host batch while CUDA backward kernels drain."""

    schedule_fetch = getattr(batch_iterator, "schedule_fetch", None)
    if callable(schedule_fetch):
        schedule_fetch()
    schedule_next = getattr(batch_iterator, "schedule_next", None)
    if callable(schedule_next):
        schedule_next()


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
    if reader.agg_direct_mode == "compare":
        # Direct v1 deliberately falls back to legacy when request-axis
        # deduplication is disabled. There is no distinct direct result to
        # compare in that mode.
        if not reader.deduplicate_request_features:
            return iter_feature_batches(
                _config_with_reader_mode(config, split_name, "legacy"),
                split_name,
                vocab_maps,
                require_labels,
                shard_rank=shard_rank,
                shard_world_size=shard_world_size,
                pin_memory=pin_memory,
                include_group_id=include_group_id,
            )
        return _iter_compare_feature_batches(
            config,
            split_name,
            vocab_maps,
            require_labels=require_labels,
            shard_rank=shard_rank,
            shard_world_size=shard_world_size,
            pin_memory=pin_memory,
            include_group_id=include_group_id,
        )

    pin_memory = reader.pin_memory and pin_memory
    coalesce_pinned_tensors = reader.coalesce_pinned_tensors and pin_memory
    # Prefer a spawn child for pack+tensorize: shared-memory FeatureBatches
    # avoid GIL fights with wide-batch forward (threaded prepare regresses).
    # May be wrapped later by ``_DevicePrefetchIterator`` for H2D overlap.
    if reader.host_prepare_prefetch > 0:
        return _ProcessHostPrepareIterator(
            config,
            split_name,
            vocab_maps,
            require_labels=require_labels,
            shard_rank=shard_rank,
            shard_world_size=shard_world_size,
            pin_memory=pin_memory,
            coalesce_pinned_tensors=coalesce_pinned_tensors,
            include_group_id=include_group_id,
            queue_size=int(reader.host_prepare_prefetch),
        )

    table_iter = _iter_batch_tables(
        config,
        split_name,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
        require_labels=require_labels,
    )

    def _prepare(table: object) -> FeatureBatch:
        return _prepare_feature_batch(
            config,
            split,
            table,
            vocab_maps,
            require_labels,
            pin_memory,
            coalesce_pinned_tensors,
            include_group_id,
        )

    # Same-thread overlap with leftover CUDA backward when process prefetch is off.
    if reader.overlap_host_prepare and reader.device_prefetch_batches == 0:
        return _OverlappedHostPrepareIterator(table_iter, _prepare)
    return _iter_sync_prepared_feature_batches(table_iter, _prepare)


def _iter_sync_prepared_feature_batches(
    table_iter: Iterator[object],
    prepare_fn: Callable[[object], FeatureBatch],
) -> Iterator[FeatureBatch]:
    for table in table_iter:
        yield prepare_fn(table)


def _iter_compare_feature_batches(
    config: AppConfig,
    split_name: str,
    vocab_maps: dict[str, dict[str, int]],
    *,
    require_labels: bool,
    shard_rank: int,
    shard_world_size: int,
    pin_memory: bool,
    include_group_id: bool,
) -> Iterator[FeatureBatch]:
    legacy_iter = iter_feature_batches(
        _config_with_reader_mode(config, split_name, "legacy"),
        split_name,
        vocab_maps,
        require_labels,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
        pin_memory=pin_memory,
        include_group_id=include_group_id,
    )
    direct_iter = iter_feature_batches(
        _config_with_reader_mode(config, split_name, "direct"),
        split_name,
        vocab_maps,
        require_labels,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
        pin_memory=pin_memory,
        include_group_id=include_group_id,
    )
    sentinel = object()
    batch_index = 0
    try:
        while True:
            legacy_batch = next(legacy_iter, sentinel)
            direct_batch = next(direct_iter, sentinel)
            if legacy_batch is sentinel and direct_batch is sentinel:
                break
            if legacy_batch is sentinel or direct_batch is sentinel:
                exhausted = "legacy" if legacy_batch is sentinel else "direct"
                raise AssertionError(
                    "agg_direct compare batch-count mismatch: "
                    f"{exhausted} ended before batch {batch_index}"
                )
            assert isinstance(legacy_batch, FeatureBatch)
            assert isinstance(direct_batch, FeatureBatch)
            _assert_feature_batch_equal(
                legacy_batch,
                direct_batch,
                batch_index=batch_index,
            )
            # Compare mode is an oracle gate; train/evaluate on the legacy
            # result until the caller explicitly selects direct.
            yield legacy_batch
            batch_index += 1
    finally:
        close = getattr(legacy_iter, "close", None)
        if callable(close):
            close()
        close = getattr(direct_iter, "close", None)
        if callable(close):
            close()


# Prefer zero-copy shared-memory FeatureBatch IPC when /dev/shm is large enough
# for a deep host-prepare queue (prod nodes often expose hundreds of GiB). Tiny
# containers (default Docker 64MiB) stay on anonymous memfd.
_HOST_PREPARE_SHARE_SHM_BYTES = 2 * 1024 * 1024 * 1024


def _configure_host_prepare_tensor_sharing() -> Path:
    """Scratch dir for host-prepare ``file_system`` tensor IPC.

    Prefer ``/dev/shm`` when it has headroom so ``share`` mode can use
    ``file_system`` sharing (reliable with spawn+Queue). ``file_descriptor``
    sharing breaks across spawn Queue unpickle once the child's resource
    sharer socket is gone. Tiny containers keep using ``/tmp``.
    """

    override = os.environ.get("MDL_TORCH_SHARE_DIR")
    if override:
        share_dir = Path(override)
    elif _dev_shm_free_bytes() >= _HOST_PREPARE_SHARE_SHM_BYTES:
        share_dir = Path("/dev/shm/mdl-torch-shm")
    else:
        share_dir = Path("/tmp/mdl-torch-shm")
    share_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(share_dir)
    os.environ["TEMP"] = str(share_dir)
    os.environ["TMP"] = str(share_dir)
    tempfile.tempdir = str(share_dir)
    return share_dir


def _host_prepare_ipc_mode(
    environ: MutableMapping[str, str] | None = None,
) -> str:
    """Pick host-prepare IPC transport: ``share`` (zero-copy) or ``memfd``.

    Override with ``MDL_HOST_PREPARE_IPC=share|memfd|auto`` (default auto).
    ``share`` needs enough ``/dev/shm`` for pinned ``share_memory_`` queues;
    otherwise we keep the memfd path that works under 64MiB containers.
    """

    env = os.environ if environ is None else environ
    forced = str(env.get("MDL_HOST_PREPARE_IPC", "auto")).strip().lower()
    if forced in {"share", "memfd"}:
        return forced
    shm_free = _dev_shm_free_bytes()
    if shm_free >= _HOST_PREPARE_SHARE_SHM_BYTES:
        return "share"
    return "memfd"


def _share_cpu_tensor_tree(value: Any) -> Any:
    """Move CPU tensor storages into shared memory for ForkingPickler IPC."""

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("host-prepare IPC requires CPU tensors")
        return value.share_memory_()
    if isinstance(value, dict):
        return {key: _share_cpu_tensor_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_share_cpu_tensor_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_share_cpu_tensor_tree(child) for child in value)
    return value


def _share_feature_batch_for_ipc(batch: FeatureBatch) -> FeatureBatch:
    """Share every tensor storage so mp.Queue transfers handles, not multi-GiB blobs."""

    batch.features = _share_cpu_tensor_tree(batch.features)
    if batch.labels is not None:
        batch.labels = _share_cpu_tensor_tree(batch.labels)
    if batch.label_mask is not None:
        batch.label_mask = _share_cpu_tensor_tree(batch.label_mask)
    batch.scenario_id = _share_cpu_tensor_tree(batch.scenario_id)
    if batch._packed_buffers:
        batch._packed_buffers = tuple(
            _share_cpu_tensor_tree(buffer) for buffer in batch._packed_buffers
        )
    return batch


def _encode_feature_batch_views(value: Any, buffers: tuple[Tensor, ...]) -> Any:
    if isinstance(value, dict):
        return {
            key: _encode_feature_batch_views(child, buffers)
            for key, child in value.items()
        }
    if isinstance(value, torch.Tensor):
        data_ptr = value.untyped_storage().data_ptr()
        for index, buffer in enumerate(buffers):
            if buffer.untyped_storage().data_ptr() == data_ptr:
                return (
                    "_view",
                    index,
                    tuple(int(dim) for dim in value.size()),
                    tuple(int(step) for step in value.stride()),
                    int(value.storage_offset()),
                )
        raise ValueError("feature tensor is not a view into coalesced packed buffers")
    return value


def _decode_feature_batch_views(value: Any, buffers: tuple[Tensor, ...]) -> Any:
    if isinstance(value, dict):
        return {
            key: _decode_feature_batch_views(child, buffers)
            for key, child in value.items()
        }
    if isinstance(value, tuple) and value and value[0] == "_view":
        _, index, size, stride, offset = value
        return buffers[int(index)].as_strided(size, stride, int(offset))
    return value


_DTYPE_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}


def _tensor_to_raw_bytes(buffer: Tensor) -> bytes:
    """Copy one coalesced CPU buffer into a standalone bytes object."""

    contiguous = buffer.detach().contiguous()
    if contiguous.dtype == torch.bfloat16:
        return contiguous.view(torch.uint16).numpy().tobytes()
    array = contiguous.numpy().reshape(-1)
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    return array.tobytes()


def _raw_bytes_to_tensor(
    raw: memoryview | bytes,
    *,
    dtype: torch.dtype,
    pin_memory: bool,
) -> Tensor:
    if dtype == torch.bfloat16:
        numel = len(raw) // 2
        tensor = torch.empty(numel, dtype=torch.bfloat16, pin_memory=pin_memory)
        tensor.view(torch.uint16).numpy()[:] = np.frombuffer(raw, dtype=np.uint16)
        return tensor
    np_dtype = _DTYPE_TO_NUMPY.get(dtype)
    if np_dtype is None:
        raise TypeError(f"unsupported packed dtype for host-prepare IPC: {dtype}")
    array = np.frombuffer(raw, dtype=np_dtype)
    tensor = torch.empty(array.shape[0], dtype=dtype, pin_memory=pin_memory)
    tensor.numpy()[:] = array
    return tensor


def _spill_feature_batch_for_ipc(batch: FeatureBatch, share_dir: Path) -> dict[str, Any]:
    """Pack coalesced buffers into an anonymous memfd Queue payload.

    Writes each buffer sequentially into the memfd (no giant ``b"".join`` peak)
    so large batches stay within tiny-container memory headroom.
    """

    del share_dir
    if not batch._packed_buffers:
        raise ValueError("host-prepare IPC requires coalesced _packed_buffers")
    import mmap
    from multiprocessing.reduction import DupFd

    chunks = [_tensor_to_raw_bytes(buffer) for buffer in batch._packed_buffers]
    buffer_records: list[tuple[str, int, int]] = []
    offset = 0
    for buffer, chunk in zip(batch._packed_buffers, chunks):
        buffer_records.append((str(buffer.dtype), len(chunk), offset))
        offset += len(chunk)
    total = offset
    fd = os.memfd_create(f"mdl-host-prep-{time_ns()}", 0)
    try:
        os.ftruncate(fd, total)
        mapped = mmap.mmap(fd, total)
        try:
            cursor = 0
            for chunk in chunks:
                end = cursor + len(chunk)
                mapped[cursor:end] = chunk
                cursor = end
            mapped.flush()
        finally:
            mapped.close()
        payload = {
            "fd": DupFd(fd),
            "size": total,
            "buffers": buffer_records,
            "features": _encode_feature_batch_views(batch.features, batch._packed_buffers),
            "labels": _encode_feature_batch_views(batch.labels, batch._packed_buffers),
            "label_mask": _encode_feature_batch_views(batch.label_mask, batch._packed_buffers),
            "scenario_id": _encode_feature_batch_views(batch.scenario_id, batch._packed_buffers),
            "group_id": batch.group_id,
            "prediction_keys": batch.prediction_keys,
        }
    finally:
        os.close(fd)
    return payload


def _load_feature_batch_from_ipc(
    payload: dict[str, Any],
    *,
    pin_memory: bool,
) -> FeatureBatch:
    import mmap

    fd = payload["fd"].detach()
    size = int(payload["size"])
    try:
        mapped = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        try:
            buffers: list[Tensor] = []
            for dtype_name, nbytes, offset in payload["buffers"]:
                dtype = getattr(torch, dtype_name.removeprefix("torch."))
                # mmap slice returns bytes (a copy); safe to close afterward.
                raw = mapped[offset : offset + nbytes]
                buffers.append(_raw_bytes_to_tensor(raw, dtype=dtype, pin_memory=pin_memory))
            buffer_tuple = tuple(buffers)
            labels = payload["labels"]
            label_mask = payload["label_mask"]
            return FeatureBatch(
                features=_decode_feature_batch_views(payload["features"], buffer_tuple),
                labels=None if labels is None else _decode_feature_batch_views(labels, buffer_tuple),
                label_mask=(
                    None if label_mask is None else _decode_feature_batch_views(label_mask, buffer_tuple)
                ),
                scenario_id=_decode_feature_batch_views(payload["scenario_id"], buffer_tuple),
                group_id=list(payload["group_id"]),
                prediction_keys=dict(payload["prediction_keys"]),
                _packed_buffers=buffer_tuple,
            )
        finally:
            mapped.close()
    finally:
        os.close(fd)


def _host_prepare_process_main(
    queue: Any,
    config: AppConfig,
    split_name: str,
    vocab_maps: dict[str, dict[str, int]],
    *,
    require_labels: bool,
    shard_rank: int,
    shard_world_size: int,
    coalesce_tensors: bool,
    include_group_id: bool,
    ipc_mode: str,
    pin_memory: bool,
) -> None:
    """Child entry: pack+tensorize and push FeatureBatches to the train process.

    - ``memfd``: hide CUDA, coalesce unpinned, spill via anonymous memfd (tiny shm).
    - ``share``: keep CUDA visible so we can pin in-child, then ``share_memory_``
      so the parent receives already-pinned handles (large ``/dev/shm``).
    """

    os.environ["MDL_HOST_PREPARE_PROCESS"] = "1"
    use_share = ipc_mode == "share"
    shm_free = _dev_shm_free_bytes()
    shm_ok_for_pin_share = shm_free >= _HOST_PREPARE_SHARE_SHM_BYTES
    if not use_share or not shm_ok_for_pin_share:
        # Memfd, or forced share under tiny /dev/shm: pin on the parent and
        # hide CUDA so the child never steals a torchrun-remapped device.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    share_dir = _configure_host_prepare_tensor_sharing()
    if use_share:
        try:
            import torch.multiprocessing as torch_mp

            # Always ``file_system`` for spawn+Queue: ``file_descriptor`` needs
            # the child's resource_sharer socket during parent unpickle, which
            # races and raises FileNotFoundError. With large /dev/shm the
            # share dir lives there so this stays a tmpfs mmap path.
            torch_mp.set_sharing_strategy("file_system")
        except (RuntimeError, ValueError, AttributeError):
            pass
    try:
        import psutil

        n_cpu = int(psutil.cpu_count(logical=True) or 4)
        prepare_cores = min(max(16, n_cpu // 3), n_cpu)
        psutil.Process().cpu_affinity(list(range(prepare_cores)))
    except Exception:
        prepare_cores = 16
    try:
        split = config.data.train if split_name == "train" else config.data.test
        if split is None:
            raise ValueError(f"split {split_name!r} is not configured")
        # Pin-in-child needs CUDA + enough /dev/shm for pinned share_memory_.
        child_pin = bool(
            pin_memory and use_share and shm_ok_for_pin_share and torch.cuda.is_available()
        )
        table_iter = _iter_batch_tables(
            config,
            split_name,
            shard_rank=shard_rank,
            shard_world_size=shard_world_size,
            require_labels=require_labels,
        )
        try:
            for table in table_iter:
                batch = _prepare_feature_batch(
                    config,
                    split,
                    table,
                    vocab_maps,
                    require_labels,
                    False,
                    False,
                    include_group_id,
                )
                if coalesce_tensors:
                    batch = _coalesce_feature_batch(batch, pin_memory=child_pin)
                elif child_pin:
                    batch = pin_feature_batch(batch, coalesce_tensors=False)
                if use_share:
                    queue.put(_share_feature_batch_for_ipc(batch))
                else:
                    queue.put(_spill_feature_batch_for_ipc(batch, share_dir))
            queue.put(None)
        finally:
            close = getattr(table_iter, "close", None)
            if callable(close):
                close()
    except BaseException as error:  # noqa: BLE001 - propagate to parent
        # Pickling a bare exception across spawn drops the child traceback.
        # Embed it in the message so train logs show the real int(None)/etc site.
        import traceback

        queue.put(
            RuntimeError(
                f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
            )
        )


class _ProcessHostPrepareIterator:
    """Yield FeatureBatches prepared in a spawn child.

    IPC auto-selects by ``/dev/shm`` size (see ``_host_prepare_ipc_mode``):

    - large shm → ``share_memory_`` + optional in-child pin (zero-copy to train)
    - tiny shm → anonymous memfd; parent materializes pinned packed buffers

    Override with ``MDL_HOST_PREPARE_IPC=share|memfd|auto``.
    """

    _SENTINEL = None

    def __init__(
        self,
        config: AppConfig,
        split_name: str,
        vocab_maps: dict[str, dict[str, int]],
        *,
        require_labels: bool,
        shard_rank: int,
        shard_world_size: int,
        pin_memory: bool,
        coalesce_pinned_tensors: bool,
        include_group_id: bool,
        queue_size: int,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("host_prepare_prefetch queue_size must be positive")
        del coalesce_pinned_tensors
        self._pin_memory = bool(pin_memory)
        self._ipc_mode = _host_prepare_ipc_mode()
        self._closed = False
        _configure_host_prepare_tensor_sharing()
        if self._ipc_mode == "share":
            try:
                import torch.multiprocessing as torch_mp

                # Match the child: file_system under /dev/shm (or TMPDIR).
                torch_mp.set_sharing_strategy("file_system")
            except (RuntimeError, ValueError, AttributeError):
                pass
        if is_main_process():
            logger.info(
                "host-prepare IPC mode=%s (shm_free=%.1fMiB, pin_memory=%s)",
                self._ipc_mode,
                (_dev_shm_free_bytes() or 0) / (1024 * 1024),
                self._pin_memory,
            )
        self._ctx = mp.get_context("spawn")
        self._queue: Any = self._ctx.Queue(maxsize=int(queue_size))
        self._process = self._ctx.Process(
            target=_host_prepare_process_main,
            kwargs={
                "queue": self._queue,
                "config": config,
                "split_name": split_name,
                "vocab_maps": vocab_maps,
                "require_labels": require_labels,
                "shard_rank": shard_rank,
                "shard_world_size": shard_world_size,
                "coalesce_tensors": True,
                "include_group_id": include_group_id,
                "ipc_mode": self._ipc_mode,
                "pin_memory": self._pin_memory,
            },
            name=f"mdl-host-prepare-{split_name}",
            daemon=False,
        )
        self._process.start()
        try:
            import psutil

            n_cpu = int(psutil.cpu_count(logical=True) or 4)
            prepare_cores = min(max(16, n_cpu // 3), n_cpu)
            if prepare_cores < n_cpu:
                psutil.Process().cpu_affinity(list(range(prepare_cores, n_cpu)))
        except Exception:
            pass

    def __iter__(self) -> "_ProcessHostPrepareIterator":
        return self

    def __next__(self) -> FeatureBatch:
        if self._closed:
            raise StopIteration
        item = self._queue.get()
        if item is self._SENTINEL:
            self.close()
            raise StopIteration
        if isinstance(item, BaseException):
            self.close()
            raise RuntimeError("host prepare process failed") from item
        if isinstance(item, FeatureBatch):
            # Shared-memory path: child already pinned when CUDA was available.
            if self._pin_memory and item._packed_buffers:
                if all(buffer.is_pinned() for buffer in item._packed_buffers):
                    return item
                return pin_feature_batch(item, coalesce_tensors=False)
            if self._pin_memory:
                return pin_feature_batch(item, coalesce_tensors=False)
            return item
        if not isinstance(item, dict):
            self.close()
            raise TypeError(
                f"host prepare process returned {type(item).__name__}, "
                "expected FeatureBatch or memfd payload dict"
            )
        return _load_feature_batch_from_ipc(item, pin_memory=self._pin_memory)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        try:
            self._queue.close()
        except Exception:
            pass
        try:
            self._queue.join_thread()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class _OverlappedHostPrepareIterator:
    """Yield FeatureBatches while hiding prepare under leftover CUDA backward.

    Training should call ``schedule_fetch`` then ``schedule_next`` after
    backward. Both run on the train thread: the table iterator is a generator
    that owns a ProcessPool, and threaded prepare under the optimizer was
    measured to inflate optimizer time via GIL contention on RankMixer.
    """

    def __init__(
        self,
        table_iter: Iterator[object],
        prepare_fn: Callable[[object], FeatureBatch],
    ) -> None:
        self._table_iter = table_iter
        self._prepare_fn = prepare_fn
        self._ready: FeatureBatch | None = None
        self._pending_table: object | None = None
        self._exhausted = False
        self._fill_ready()

    def __iter__(self) -> "_OverlappedHostPrepareIterator":
        return self

    def __next__(self) -> FeatureBatch:
        if self._ready is None:
            self._fill_ready()
        if self._ready is None:
            raise StopIteration
        batch = self._ready
        self._ready = None
        return batch

    def schedule_fetch(self) -> None:
        """Pull the next packed table without tensorizing it yet."""

        if (
            self._ready is not None
            or self._pending_table is not None
            or self._exhausted
        ):
            return
        try:
            self._pending_table = next(self._table_iter)
        except StopIteration:
            self._exhausted = True
            self._pending_table = None

    def schedule_next(self) -> None:
        """Tensorize/pin the pending table; fetch first if needed."""

        if self._ready is not None or self._exhausted:
            return
        if self._pending_table is None:
            self.schedule_fetch()
        if self._pending_table is None:
            return
        table = self._pending_table
        self._pending_table = None
        self._ready = self._prepare_fn(table)

    def _fill_ready(self) -> None:
        if self._exhausted:
            self._ready = None
            return
        if self._pending_table is not None:
            table = self._pending_table
            self._pending_table = None
            self._ready = self._prepare_fn(table)
            return
        try:
            table = next(self._table_iter)
        except StopIteration:
            self._exhausted = True
            self._ready = None
            return
        self._ready = self._prepare_fn(table)

    def close(self) -> None:
        close = getattr(self._table_iter, "close", None)
        if callable(close):
            close()


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
        self.device = (
            device
            if device.index is not None
            else torch.device("cuda", torch.cuda.current_device())
        )
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
            # The worker owns the host iterator and may still be inside Parquet
            # decode/tensorization when training reaches max_steps. Do not let
            # the interpreter tear down CUDA/Arrow while that work is live:
            # doing so can abort the process after an otherwise successful
            # final step. Scanner IO has its own timeouts, so a full join is
            # the safe lifecycle boundary here.
            self.thread.join()


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


def _bounded_optimizer_param_groups(
    parameters: list[nn.Parameter],
    bucket_bytes: int,
) -> list[dict[str, list[nn.Parameter]]]:
    """Keep foreach workspace bounded while preserving parameter order."""

    if bucket_bytes <= 0:
        raise ValueError("optimizer bucket_bytes must be positive")
    groups: list[dict[str, list[nn.Parameter]]] = []
    current: list[nn.Parameter] = []
    current_bytes = 0
    for parameter in parameters:
        parameter_bytes = parameter.numel() * parameter.element_size()
        if current and current_bytes + parameter_bytes > bucket_bytes:
            groups.append({"params": current})
            current = []
            current_bytes = 0
        current.append(parameter)
        current_bytes += parameter_bytes
    if current:
        groups.append({"params": current})
    return groups


def _build_dense_optimizer(
    parameters: list[nn.Parameter],
    config: AppConfig,
    device: torch.device,
) -> torch.optim.Optimizer:
    """Construct RMSprop with an explicit speed-vs-peak-memory policy."""

    kwargs: dict[str, Any] = {
        "lr": config.training.lr_dense,
        "alpha": config.training.rmsprop_alpha,
        "momentum": config.training.rmsprop_momentum,
    }
    fused_requested = (
        getattr(config.training, "fused_dense_optimizer", False)
        and device.type == "cuda"
    )
    foreach_bucket_mb = int(
        getattr(config.training, "dense_optimizer_foreach_bucket_mb", 0)
    )
    bounded_foreach_requested = foreach_bucket_mb > 0 and device.type == "cuda"
    try:
        optimizer_parameters = inspect.signature(torch.optim.RMSprop).parameters
    except (TypeError, ValueError):
        optimizer_parameters = {}
    fused_supported = "fused" in optimizer_parameters
    foreach_supported = "foreach" in optimizer_parameters
    optimizer_input: Any = parameters
    if fused_requested and fused_supported:
        kwargs["fused"] = True
    elif fused_requested:
        if foreach_supported:
            # RMSprop has no fused CUDA implementation in supported PyTorch
            # releases. Multi-tensor foreach avoids launching one optimizer
            # kernel per dense parameter and is the fastest available path.
            kwargs["foreach"] = True
            if is_main_process():
                print(
                    "Dense optimizer | optimizer=RMSprop "
                    "implementation=foreach fused_supported=false"
                )
        elif is_main_process():
            print(
                "Dense optimizer | optimizer=RMSprop implementation=scalar "
                "fused_supported=false foreach_supported=false"
            )
    elif bounded_foreach_requested and foreach_supported:
        bucket_bytes = foreach_bucket_mb * 1024 * 1024
        optimizer_input = _bounded_optimizer_param_groups(
            parameters,
            bucket_bytes,
        )
        kwargs["foreach"] = True
        if is_main_process():
            print(
                "Dense optimizer | optimizer=RMSprop "
                "implementation=bounded_foreach "
                f"bucket_mb={foreach_bucket_mb} buckets={len(optimizer_input)}"
            )
    elif foreach_supported:
        # Leaving foreach=None lets PyTorch silently auto-select the multi-tensor
        # path on CUDA. Explicit False is required for the low-peak-memory mode.
        kwargs["foreach"] = False
        if device.type == "cuda" and is_main_process():
            print(
                "Dense optimizer | optimizer=RMSprop implementation=scalar "
                "memory_efficient=true"
            )
    return torch.optim.RMSprop(optimizer_input, **kwargs)


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
    # Prefer the CPU control group so metadata exchange does not depend on NCCL
    # / NVLink / P2P being healthy before the first training step.
    torch_dist.all_gather_object(
        gathered,
        descriptors,
        group=context.control_group,
    )
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
    # RankMixer: fuse dense blocks+logits only. Compiling the full module pulls
    # sharded-embedding host splits into inductor CUDAGraphs and thrashs.
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    compile_dense = getattr(raw, "compile_dense_backbone", None)
    if callable(compile_dense) and getattr(config.model, "name", None) == "rankmixer":
        compile_dense()
        return model
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

    # CUDA-graph the dense RankMixer stack before DDP installs reducer hooks on
    # parameters; make_graphed_callables bwd capture is incompatible with those hooks.
    if (
        bool(getattr(config.runtime, "cuda_graph_backbone", False))
        and context.device.type == "cuda"
        and hasattr(base_model, "prewarm_cuda_graph_backbone")
    ):
        base_model.prewarm_cuda_graph_backbone(context.device)
        # Freeze further captures before DDP reducer hooks are installed.
        if hasattr(base_model, "_cuda_graph_backbone_capture_allowed"):
            base_model._cuda_graph_backbone_capture_allowed = False

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
    # make_graphed_callables forbids autocast caching during capture/replay.
    cache_enabled = not bool(getattr(config.runtime, "cuda_graph_backbone", False))
    return torch.amp.autocast(
        device_type=device.type,
        dtype=dtype,
        cache_enabled=cache_enabled,
    )


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


def _gradient_sync_context(model: nn.Module, *, synchronize: bool):
    """Suppress DDP dense reductions for intermediate accumulation passes."""

    if synchronize:
        return nullcontext()
    no_sync = getattr(model, "no_sync", None)
    if not callable(no_sync):
        original = getattr(model, "_orig_mod", None)
        no_sync = getattr(original, "no_sync", None)
    if not callable(no_sync):
        raise RuntimeError(
            "gradient accumulation under DDP requires a model.no_sync() context"
        )
    return no_sync()


def _coalesce_accumulated_sparse_gradients(
    parameters: list[nn.Parameter],
) -> None:
    """Bound duplicate COO storage while several micro-batches accumulate."""

    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None and gradient.is_sparse and not gradient.is_coalesced():
            parameter.grad = gradient.coalesce()


def _non_blocking_transfer(config: AppConfig, split_name: str, device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    return _split_reader(config, split_name).pin_memory


def _sequence_lengths_for_batch_rows(value: dict[str, Any]) -> Tensor | None:
    """Return per-row sequence lengths aligned to the batch (after row_indices)."""

    lengths = value.get("lengths")
    fields = value.get("fields")
    # Categorical bags also carry ``lengths`` but are feature lookups, not
    # sequence events. Requiring the sequence ``fields`` mapping keeps the
    # token numerator aligned with the padded-slot denominator.
    if not isinstance(lengths, Tensor) or not isinstance(fields, dict):
        return None
    row_indices = value.get("row_indices")
    if isinstance(row_indices, Tensor):
        lengths = lengths.index_select(0, row_indices.to(lengths.device).long())
    return lengths


def _sequence_padded_width(value: dict[str, Any], lengths: Tensor) -> int:
    """Rectangular pad width for one sequence feature.

    Prefer the dense ``[rows, pad, ...]`` field width. When fields are compact
    1D (direct-path dim-1 tensors), fall back to ``max(lengths)`` so the
    denominator never under-counts the numerator (which previously made
    ``padding_ratio`` negative).
    """

    padded_length = 0
    fields = value.get("fields")
    if isinstance(fields, dict):
        for field_value in fields.values():
            if isinstance(field_value, Tensor) and field_value.dim() >= 2:
                padded_length = max(padded_length, int(field_value.size(1)))
    if lengths.numel():
        padded_length = max(padded_length, int(lengths.max().item()))
    return padded_length


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
        lengths = _sequence_lengths_for_batch_rows(value)
        if lengths is None:
            continue
        found_sequence = True
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
        lengths = _sequence_lengths_for_batch_rows(value)
        if lengths is None:
            continue
        found_sequence = True
        padded_length = _sequence_padded_width(value, lengths)
        total += int(lengths.numel()) * padded_length
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
    return _start_active_rank_count(context, rank_active).wait()


class _ActiveRankCountHandle:
    """Async world-active-rank reduction; wait before loss scaling / exit checks."""

    __slots__ = ("_value", "_work")

    def __init__(self, value: Tensor, work: Any | None) -> None:
        self._value = value
        self._work = work

    def wait(self) -> int:
        if self._work is not None:
            self._work.wait()
        return int(self._value.item())


def _start_active_rank_count(
    context: DistributedContext,
    rank_active: bool,
) -> _ActiveRankCountHandle:
    """Kick off active-rank allreduce so H2D can overlap the host collective."""

    if not context.enabled:
        value = torch.tensor(int(rank_active), dtype=torch.long)
        return _ActiveRankCountHandle(value, None)
    # Prefer CPU + Gloo control group so this never inserts a CUDA device sync
    # into the training critical path.
    if context.control_group is not None:
        value = torch.tensor(int(rank_active), dtype=torch.long, device="cpu")
        work = torch_dist.all_reduce(
            value,
            op=torch_dist.ReduceOp.SUM,
            group=context.control_group,
            async_op=True,
        )
        return _ActiveRankCountHandle(value, work)
    value = torch.tensor(
        int(rank_active), dtype=torch.long, device=context.device
    )
    work = torch_dist.all_reduce(
        value,
        op=torch_dist.ReduceOp.SUM,
        async_op=True,
    )
    return _ActiveRankCountHandle(value, work)


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


def _request_row_indices_from_batch(
    batch: FeatureBatch,
    candidate_count: int,
    device: torch.device,
) -> Tensor:
    """Resolve the shared candidate→request map carried by RLB feature payloads."""

    resolved: Tensor | None = None
    for value in batch.features.values():
        if not isinstance(value, dict):
            continue
        candidate = value.get("row_indices")
        if not isinstance(candidate, Tensor) or candidate.ndim != 1:
            continue
        if candidate.numel() != candidate_count:
            continue
        candidate = candidate.to(device=device, dtype=torch.long)
        if resolved is None:
            resolved = candidate
            continue
        if resolved.data_ptr() != candidate.data_ptr() and not torch.equal(
            resolved,
            candidate,
        ):
            raise ValueError(
                "request-level loss requires one consistent row_indices mapping"
            )
    if resolved is None:
        raise ValueError(
            "mean_per_request_per_task requires request-deduplicated features "
            "with candidate-to-request row_indices"
        )
    if resolved.numel() and bool((resolved < 0).any()):
        raise ValueError("request row_indices must be non-negative")
    return resolved


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
        weights = torch.ones_like(element_loss)
    else:
        weights = batch.label_mask.to(
            device=logits.device,
            dtype=element_loss.dtype,
        )
    if not rank_active:
        # Preserve the replayed forward graph on an exhausted rank while
        # contributing no samples to either reduction.
        weights = torch.zeros_like(weights)

    if loss_reduction == "mean_per_request_per_task":
        row_indices = _request_row_indices_from_batch(
            batch,
            element_loss.size(0),
            logits.device,
        )
        _request_ids, inverse_rows = torch.unique(
            row_indices,
            sorted=True,
            return_inverse=True,
        )
        request_count = int(_request_ids.numel())
        request_numerators = element_loss.new_zeros(
            request_count,
            element_loss.size(1),
        ).index_add(
            0,
            inverse_rows,
            element_loss * weights,
        )
        request_counts = element_loss.new_zeros(
            request_count,
            element_loss.size(1),
        ).index_add(
            0,
            inverse_rows,
            weights,
        )
        valid_requests = request_counts > 0
        request_means = torch.where(
            valid_requests,
            request_numerators / request_counts.clamp_min(1.0),
            torch.zeros_like(request_numerators),
        )
        task_numerators = request_means.sum(dim=0)
        task_counts = valid_requests.sum(dim=0).to(dtype=element_loss.dtype)
    else:
        task_numerators = (element_loss * weights).sum(dim=0)
        task_counts = weights.sum(dim=0)

    distributed = torch_dist.is_available() and torch_dist.is_initialized()
    if loss_reduction == "sum":
        # DDP averages gradients across ranks. Multiplying each local sum by the
        # world size makes the averaged gradient equal the global paper sum.
        world_size = float(torch_dist.get_world_size()) if distributed else 1.0
        prediction_loss = task_numerators.sum() * world_size
    elif loss_reduction in {"mean_per_task", "mean_per_request_per_task"}:
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
        raise ValueError(
            "loss_reduction must be sum, mean_per_task, "
            "or mean_per_request_per_task"
        )
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
    training_started_observer: Callable[[], None] | None = None,
    synchronize_step_observer: bool = True,
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
        config = _resolve_distributed_cardinality_audit(config, context, "train")
        vocab_maps = load_vocab_maps(config)
        base_model = _build_model_on_device(config, vocab_maps, device)
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
        gradient_accumulation_steps = int(
            getattr(config.training, "gradient_accumulation_steps", 1)
        )
        if gradient_accumulation_steps <= 0:
            raise ValueError(
                "training.gradient_accumulation_steps must be a positive integer"
            )
        configured_batch_per_rank = int(getattr(config.training, "batch_size", 1))
        runtime_effective_global_batch = (
            configured_batch_per_rank
            * context.world_size
            * gradient_accumulation_steps
        )
        if log_steps and context.rank == 0:
            print(
                "Batching | "
                f"configured_batch_per_rank={configured_batch_per_rank} "
                f"runtime_world_size={context.world_size} "
                f"gradient_accumulation_steps={gradient_accumulation_steps} "
                f"runtime_effective_global_batch={runtime_effective_global_batch}"
            )

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
        if training_started_observer is not None:
            training_started_observer()
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
        train_data = getattr(getattr(config, "data", None), "train", None)
        train_reader = getattr(train_data, "reader", None)
        device_prefetch_depth = (
            int(getattr(train_reader, "device_prefetch_batches", 0))
            if device.type == "cuda"
            else 0
        )
        host_prepare_depth = int(
            getattr(train_reader, "host_prepare_prefetch", 0)
        )
        if device_prefetch_depth > 0 and host_prepare_depth <= 0:
            # Device-prefetch thread would call in-process prepare → GIL fight.
            logger.warning(
                "reader.device_prefetch_batches=%d without host_prepare_prefetch "
                "overlaps in-process FeatureBatch prepare with training; prefer "
                "host_prepare_prefetch>0 so the CUDA thread only does H2D.",
                device_prefetch_depth,
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
        last_trace_batch: FeatureBatch | None = None
        accumulation_index = 0
        window_rank_active = False
        window_rows = 0
        window_input_tokens = 0
        window_padded_token_slots = 0
        window_loss_numerator: Tensor | None = None
        window_loss_denominator: Tensor | None = None
        window_task_monitors: list[_StreamingTaskMonitor] | None = None
        window_step_started = 0.0
        window_dataloader_wait_seconds = 0.0
        window_h2d_seconds = 0.0
        window_forward_seconds = 0.0
        window_backward_seconds = 0.0
        while max_steps is None or steps < max_steps:
            observing = step_observer is not None
            tracing = observing and synchronize_step_observer
            if tracing:
                _sync_device(device)
            if accumulation_index == 0:
                window_rank_active = False
                window_rows = 0
                window_input_tokens = 0
                window_padded_token_slots = 0
                window_loss_numerator = None
                window_loss_denominator = None
                window_task_monitors = None
                # Always time the window: Train-step logs need wait_ratio even
                # when no benchmark step_observer is attached.
                window_step_started = perf_counter()
                window_dataloader_wait_seconds = 0.0
                window_h2d_seconds = 0.0
                window_forward_seconds = 0.0
                window_backward_seconds = 0.0
            dataloader_started = perf_counter()
            trace_batch: FeatureBatch | None = None
            collect_batch_stats = observing or (
                log_steps
                and context.rank == 0
                and (steps + 1) % config.training.log_every_steps == 0
            )
            if pending_train_batches:
                local_batch = pending_train_batches.popleft()
                local_batch_on_device = False
                if collect_batch_stats:
                    trace_batch = local_batch
            else:
                try:
                    if (
                        collect_batch_stats
                        and batches_on_device
                        and isinstance(batch_iterator, _DevicePrefetchIterator)
                    ):
                        trace_batch, local_batch = batch_iterator.next_with_host()
                    else:
                        local_batch = next(batch_iterator)
                        if collect_batch_stats and not batches_on_device:
                            trace_batch = local_batch
                except StopIteration:
                    local_batch = None
                local_batch_on_device = batches_on_device
            dataloader_wait_seconds = perf_counter() - dataloader_started
            rank_active = local_batch is not None
            # Kick the Gloo/NCCL active-rank reduction immediately so pinned H2D
            # can overlap it. Wait only after the host→device copy is queued.
            active_rank_handle = _start_active_rank_count(context, rank_active)
            h2d_started = perf_counter() if observing else 0.0
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
                if trace_batch is not None:
                    last_trace_batch = trace_batch
                active_ranks = active_rank_handle.wait()
            else:
                active_ranks = active_rank_handle.wait()
                if active_ranks == 0:
                    if accumulation_index > 0:
                        for optimizer in optimizers:
                            optimizer.zero_grad(set_to_none=True)
                        consume_sharded_embedding_stats(base_model)
                        if log_steps and context.rank == 0:
                            print(
                                "Gradient accumulation | "
                                f"dropped_incomplete_micro_batches={accumulation_index} "
                                f"required={gradient_accumulation_steps}"
                            )
                    break
                if last_device_batch is None:
                    raise RuntimeError("inactive rank has no batch available for zero-loss replay")
                batch = last_device_batch
                trace_batch = last_trace_batch
            if (
                steps == 0
                and accumulation_index == 0
                and context.enabled
                and active_ranks != context.world_size
            ):
                raise RuntimeError(
                    "replicated sparse DDP requires every rank to provide an initial batch; "
                    "reduce world_size or choose a finer reader.shard_unit"
                )
            if tracing:
                _sync_device(device)
            h2d_seconds = perf_counter() - h2d_started if observing else 0.0

            if accumulation_index == 0:
                lr_multiplier = _lr_schedule_multiplier(
                    config,
                    steps + 1,
                    lr_decay_steps,
                )
                _set_optimizer_lrs(optimizers, optimizer_base_lrs, lr_multiplier)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
            synchronize_gradients = (
                accumulation_index + 1 == gradient_accumulation_steps
            )
            # DDP's static-graph reducer must observe one fully synchronized
            # iteration before no_sync() is safe; otherwise current PyTorch
            # releases trip an internal expect_autograd_hooks assertion. Only
            # the first optimizer window pays the extra reductions.
            static_graph_warmup = (
                context.enabled and ddp_config.static_graph and steps == 0
            )
            forward_started = perf_counter() if observing else 0.0
            with _gradient_sync_context(
                model,
                synchronize=(
                    not context.enabled
                    or synchronize_gradients
                    or static_graph_warmup
                ),
            ):
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
                    backward_loss = loss
                    if config.training.loss_reduction != "sum":
                        backward_loss = loss / float(gradient_accumulation_steps)
                if tracing:
                    _sync_device(device)
                forward_seconds = (
                    perf_counter() - forward_started if observing else 0.0
                )
                backward_started = perf_counter() if observing else 0.0
                if scaler.is_enabled():
                    scaler.scale(backward_loss).backward()
                else:
                    backward_loss.backward()
                if tracing:
                    _sync_device(device)
                backward_seconds = (
                    perf_counter() - backward_started if observing else 0.0
                )
            if gradient_accumulation_steps > 1 and sparse_params:
                _coalesce_accumulated_sparse_gradients(sparse_params)

            stats_batch = trace_batch if trace_batch is not None else batch
            window_rank_active = window_rank_active or rank_active
            if rank_active:
                micro_batch_rows = int(batch.scenario_id.size(0))
                window_rows += micro_batch_rows
                if collect_batch_stats:
                    window_input_tokens += _batch_input_token_count(stats_batch)
                    window_padded_token_slots += _batch_padded_token_slots(stats_batch)
                if (
                    collect_batch_stats
                    and batch.labels is not None
                    and "logits" in output
                ):
                    if window_task_monitors is None:
                        window_task_monitors = [
                            _StreamingTaskMonitor() for _ in config.task_names
                        ]
                    logits_f = output["logits"].detach().float()
                    labels_f = batch.labels.detach().float()
                    mask = (
                        None
                        if batch.label_mask is None
                        else batch.label_mask.detach().bool()
                    )
                    for task_index, monitor in enumerate(window_task_monitors):
                        task_logits = logits_f[:, task_index]
                        task_labels = labels_f[:, task_index]
                        if mask is not None:
                            valid = mask[:, task_index]
                            task_logits = task_logits[valid]
                            task_labels = task_labels[valid]
                        monitor.update(task_logits, task_labels)
            detached_numerator = loss_numerator.detach()
            detached_denominator = loss_denominator.detach()
            window_loss_numerator = (
                detached_numerator
                if window_loss_numerator is None
                else window_loss_numerator + detached_numerator
            )
            if config.training.loss_reduction == "sum":
                window_loss_denominator = detached_denominator
            else:
                window_loss_denominator = (
                    detached_denominator
                    if window_loss_denominator is None
                    else window_loss_denominator + detached_denominator
                )
            window_dataloader_wait_seconds += dataloader_wait_seconds
            window_h2d_seconds += h2d_seconds
            window_forward_seconds += forward_seconds
            window_backward_seconds += backward_seconds
            accumulation_index += 1
            # Hide next host fetch+prepare under leftover CUDA backward work.
            # Process-prefetch materialize runs on its own thread — skip here.
            if not isinstance(batch_iterator, _ProcessHostPrepareIterator):
                _schedule_overlapped_host_prepare(batch_iterator)
            if not synchronize_gradients:
                continue

            ddp_auditor.observe()
            window_active_ranks = _active_rank_count(context, window_rank_active)
            sparse_sync_started = perf_counter() if observing else 0.0
            sparse_sync_stats = sparse_synchronizer.synchronize(
                rank_active=window_rank_active
            )
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
            sparse_sync_seconds = perf_counter() - sparse_sync_started if observing else 0.0
            optimizer_started = perf_counter() if observing else 0.0
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
            optimizer_seconds = perf_counter() - optimizer_started if observing else 0.0
            steps += 1
            rows += window_rows
            if window_loss_numerator is None or window_loss_denominator is None:
                raise AssertionError("completed accumulation window has no loss")
            last_loss_numerator_tensor = window_loss_numerator
            last_loss_denominator_tensor = window_loss_denominator
            last_loss_tensor = (
                last_loss_numerator_tensor
                / last_loss_denominator_tensor.clamp_min(1.0)
            )
            accumulation_index = 0
            if step_observer is not None:
                step_observer(
                    TrainStepTrace(
                        step=steps,
                        rank_active=window_rank_active,
                        active_ranks=window_active_ranks,
                        rows=window_rows,
                        input_tokens=window_input_tokens,
                        padded_token_slots=window_padded_token_slots,
                        step_seconds=perf_counter() - window_step_started,
                        dataloader_wait_seconds=window_dataloader_wait_seconds,
                        h2d_seconds=window_h2d_seconds,
                        forward_seconds=window_forward_seconds,
                        backward_seconds=window_backward_seconds,
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
                padding_ratio = (
                    max(
                        0.0,
                        1.0 - window_input_tokens / window_padded_token_slots,
                    )
                    if window_padded_token_slots > 0
                    else 0.0
                )
                window_step_seconds = max(
                    perf_counter() - window_step_started, 1.0e-9
                )
                wait_ratio = window_dataloader_wait_seconds / window_step_seconds
                task_stats = (
                    [monitor.compute() for monitor in window_task_monitors]
                    if window_task_monitors is not None
                    else []
                )
                task_loss_parts = [
                    f"{task_name}={_format_optional_float(stats['loss'])}"
                    for task_name, stats in zip(config.task_names, task_stats)
                ]
                task_loss_suffix = (
                    (" | " + " ".join(task_loss_parts)) if task_loss_parts else ""
                )
                print(
                    f"Train step | step={steps} | logloss={last_loss:.6f}"
                    f"{task_loss_suffix} "
                    f"active_ranks={window_active_ranks}/{context.world_size} "
                    f"micro_batches={gradient_accumulation_steps} "
                    f"local_rows={window_rows} "
                    f"runtime_effective_global_batch="
                    f"{runtime_effective_global_batch} "
                    f"padding_ratio={padding_ratio:.4f} "
                    f"dataloader_wait_s={window_dataloader_wait_seconds:.4f} "
                    f"dataloader_wait_ratio={wait_ratio:.4f} "
                    f"sparse_local_rows={sparse_sync_stats.local_rows} "
                    f"sparse_global_rows={sparse_sync_stats.global_rows} "
                    f"sparse_payload_mib={payload_mib:.2f}"
                )
                # Window-aggregated per-task moments for collapse detection.
                for task_name, stats in zip(config.task_names, task_stats):
                    print(
                        f"Train task | step={steps} task={task_name} "
                        f"logloss={_format_optional_float(stats['loss'])} "
                        f"prob_mean={_format_optional_float(stats['prob_mean'])} "
                        f"logit_mean={_format_optional_float(stats['logit_mean'])} "
                        f"logit_std={_format_optional_float(stats['logit_std'])}"
                    )
                    warning_parts = _task_monitor_warning_parts(
                        prob_mean=(
                            None
                            if stats["prob_mean"] is None
                            else float(stats["prob_mean"])
                        ),
                    )
                    if warning_parts:
                        print(
                            f"Train task warning | step={steps} task={task_name} "
                            + " ".join(warning_parts)
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
                        (max_steps - steps) * gradient_accumulation_steps,
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


class _StreamingCOPC:
    """Calibration metric: COPC = sum(pred) / sum(label). Closer to 1 is better."""

    def __init__(self) -> None:
        # [sum_predictions, sum_labels]
        self.totals = torch.zeros(2, dtype=torch.float64)

    def update(self, scores: Tensor, labels: Tensor) -> None:
        scores = scores.detach().float().flatten().cpu()
        labels = labels.detach().float().flatten().cpu()
        if scores.numel() != labels.numel():
            raise ValueError("COPC scores and labels must have the same length")
        if not scores.numel():
            return
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("COPC scores must be finite")
        self.totals[0] += float(scores.sum().item())
        self.totals[1] += float(labels.sum().item())

    def compute(self) -> float | None:
        predicted = float(self.totals[0].item())
        labeled = float(self.totals[1].item())
        if labeled <= 0.0:
            return None
        return predicted / labeled


# Collapse / miscalibration heuristics for train and quick-eval monitors.
_MONITOR_PROB_MEAN_WARN = 0.9
_MONITOR_COPC_HIGH_WARN = 2.0
_MONITOR_COPC_LOW_WARN = 0.5
_MONITOR_AUC_WARN = 0.55


class _StreamingTaskMonitor:
    """Stream per-task BCE loss and logit/prob moments for collapse detection."""

    def __init__(self) -> None:
        # [count, loss_sum, logit_sum, logit_sq_sum, prob_sum]
        self.totals = torch.zeros(5, dtype=torch.float64)

    def update(self, logits: Tensor, labels: Tensor) -> None:
        logits = logits.detach().float().flatten().cpu()
        labels = labels.detach().float().flatten().cpu()
        if logits.numel() != labels.numel():
            raise ValueError("task monitor logits/labels must have the same length")
        if not logits.numel():
            return
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("task monitor logits must be finite")
        probabilities = torch.sigmoid(logits)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )
        self.totals[0] += float(logits.numel())
        self.totals[1] += float(losses.sum().item())
        self.totals[2] += float(logits.sum().item())
        self.totals[3] += float(logits.square().sum().item())
        self.totals[4] += float(probabilities.sum().item())

    def compute(self) -> dict[str, float | None]:
        count = float(self.totals[0].item())
        if count <= 0.0:
            return {
                "loss": None,
                "prob_mean": None,
                "logit_mean": None,
                "logit_std": None,
            }
        logit_mean = float(self.totals[2].item()) / count
        logit_second = float(self.totals[3].item()) / count
        variance = max(logit_second - logit_mean * logit_mean, 0.0)
        return {
            "loss": float(self.totals[1].item()) / count,
            "prob_mean": float(self.totals[4].item()) / count,
            "logit_mean": logit_mean,
            "logit_std": math.sqrt(variance),
        }


def _task_monitor_stats_from_batch(
    logits: Tensor,
    labels: Tensor,
    label_mask: Tensor | None,
    task_count: int,
) -> list[dict[str, float | None]]:
    """Compute per-task monitor stats for one local batch (no distributed reduce)."""

    if logits.ndim != 2 or labels.ndim != 2:
        raise ValueError("logits/labels must have shape [batch, tasks]")
    if logits.shape != labels.shape:
        raise ValueError("logits and labels shapes must match")
    if logits.size(1) != task_count:
        raise ValueError(
            f"expected {task_count} tasks, got logits width {logits.size(1)}"
        )
    stats: list[dict[str, float | None]] = []
    logits_f = logits.detach().float()
    labels_f = labels.detach().float()
    mask = None if label_mask is None else label_mask.detach().bool()
    for task_index in range(task_count):
        monitor = _StreamingTaskMonitor()
        task_logits = logits_f[:, task_index]
        task_labels = labels_f[:, task_index]
        if mask is not None:
            valid = mask[:, task_index]
            task_logits = task_logits[valid]
            task_labels = task_labels[valid]
        monitor.update(task_logits, task_labels)
        stats.append(monitor.compute())
    return stats


def _task_monitor_warning_parts(
    *,
    prob_mean: float | None = None,
    copc: float | None = None,
    auc: float | None = None,
) -> list[str]:
    warnings: list[str] = []
    if prob_mean is not None and prob_mean > _MONITOR_PROB_MEAN_WARN:
        warnings.append(f"prob_mean={prob_mean:.4f}>{_MONITOR_PROB_MEAN_WARN}")
    if copc is not None and copc > _MONITOR_COPC_HIGH_WARN:
        warnings.append(f"copc={copc:.4f}>{_MONITOR_COPC_HIGH_WARN}")
    if copc is not None and 0.0 <= copc < _MONITOR_COPC_LOW_WARN:
        warnings.append(f"copc={copc:.4f}<{_MONITOR_COPC_LOW_WARN}")
    if auc is not None and auc < _MONITOR_AUC_WARN:
        warnings.append(f"auc={auc:.4f}<{_MONITOR_AUC_WARN}")
    return warnings


def _format_optional_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


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


def _reduce_evaluation_copc(
    context: DistributedContext,
    accumulators: list[_StreamingCOPC],
) -> None:
    if not context.enabled:
        return
    for accumulator in accumulators:
        _all_reduce_cpu_sum_(accumulator.totals, context)


def _reduce_evaluation_task_monitors(
    context: DistributedContext,
    accumulators: list[_StreamingTaskMonitor],
) -> None:
    if not context.enabled:
        return
    for accumulator in accumulators:
        _all_reduce_cpu_sum_(accumulator.totals, context)


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
    copc_accumulators = [_StreamingCOPC() for _ in config.task_names]
    task_monitors = [_StreamingTaskMonitor() for _ in config.task_names]
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
                logits_cpu = logits.float().cpu()
                probabilities = torch.sigmoid(logits_cpu)
                labels = batch.labels.float().cpu()
                label_mask = (
                    None
                    if batch.label_mask is None
                    else batch.label_mask.bool().cpu()
                )
                rows += int(labels.size(0))
                for task_index in range(len(config.task_names)):
                    if label_mask is None:
                        task_logits = logits_cpu[:, task_index]
                        task_scores = probabilities[:, task_index]
                        task_labels = labels[:, task_index]
                    else:
                        valid = label_mask[:, task_index]
                        task_logits = logits_cpu[valid, task_index]
                        task_scores = probabilities[valid, task_index]
                        task_labels = labels[valid, task_index]
                    accumulators[task_index][0].update(
                        task_scores,
                        task_labels,
                    )
                    copc_accumulators[task_index].update(
                        task_scores,
                        task_labels,
                    )
                    task_monitors[task_index].update(task_logits, task_labels)

        rows = _reduce_evaluation_histograms(context, accumulators, rows)
        _reduce_evaluation_copc(context, copc_accumulators)
        _reduce_evaluation_task_monitors(context, task_monitors)
        metrics: dict[str, dict[str, float | int | None]] = {}
        for task_index, task_name in enumerate(config.task_names):
            accumulator = accumulators[task_index][0]
            examples, positives, negatives = accumulator.counts()
            monitor = task_monitors[task_index].compute()
            metrics[task_name] = {
                "auc": accumulator.compute(),
                "copc": copc_accumulators[task_index].compute(),
                "loss": monitor["loss"],
                "prob_mean": monitor["prob_mean"],
                "logit_mean": monitor["logit_mean"],
                "logit_std": monitor["logit_std"],
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
        copc = metrics.get("copc")
        auc_value = None if auc is None else float(auc)
        copc_value = None if copc is None else float(copc)
        prob_mean = metrics.get("prob_mean")
        prob_mean_value = None if prob_mean is None else float(prob_mean)
        print(
            f"Quick eval task | step={step} task={task_name} "
            f"auc={_format_optional_float(auc_value, 8)} "
            f"copc={_format_optional_float(copc_value, 8)} "
            f"logloss={_format_optional_float(metrics.get('loss'))} "
            f"prob_mean={_format_optional_float(prob_mean_value)} "
            f"logit_mean={_format_optional_float(metrics.get('logit_mean'))} "
            f"logit_std={_format_optional_float(metrics.get('logit_std'))} "
            f"examples={metrics['examples']} positives={metrics['positives']} "
            f"negatives={metrics['negatives']}"
        )
        warning_parts = _task_monitor_warning_parts(
            prob_mean=prob_mean_value,
            copc=copc_value,
            auc=auc_value,
        )
        if warning_parts:
            print(
                f"Quick eval warning | step={step} task={task_name} "
                + " ".join(warning_parts)
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
        config = _resolve_distributed_cardinality_audit(config, context, split_name)
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
