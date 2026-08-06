from __future__ import annotations

from bisect import bisect_left
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import timedelta
from importlib import import_module
import errno
import gc
import inspect
import logging
import math
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import queue
import shutil
import sqlite3
import tempfile
import threading
from time import perf_counter, time, time_ns
from typing import Any, Callable, Iterator, MutableMapping

# Torchrun workers import this module before main()'s allocator bootstrap.
# Set both env names before ``import torch`` so expandable segments are active
# for the first CUDA caching-allocator init (cuts near-capacity fragmentation).
_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:256"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOC_CONF)
os.environ.setdefault("PYTORCH_ALLOC_CONF", _ALLOC_CONF)

import numpy as np
import torch
import torch.distributed as torch_dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from .config import (
    AppConfig,
    DDPConfig,
    FixedTestEvalConfig,
    ParquetSplitConfig,
    ReaderConfig,
)
from .checkpoint import (
    check_staging_space,
    CheckpointUploader,
    DataCursor,
    DEFAULT_SHARD_CHUNK_BYTES,
    estimate_staging_space,
    fetch_checkpoint_for_rank,
    load_model_checkpoint,
    load_training_checkpoint,
    open_run_store,
    resolve_resume_checkpoint,
    save_model_checkpoint,
    stage_training_checkpoint,
    StagingSpaceEstimate,
    step_directory_name,
)
from .dataloader import (
    arrow_pool_bytes,
    FeatureBatch,
    PreparedAxisBatch,
    PreparedBatchTable,
    RemoteIoStallError,
    ScanCursorChannel,
    ScanResumePlan,
    SourceRegistry,
    _adapter_request_level_sources,
    _coalesce_feature_batch,
    _column_array,
    _map_feature_value,
    _require_pyarrow,
    _safe_table_take,
    abort_rank_for_remote_io_stall,
    axis_batch_to_feature_batch,
    build_packed_request_plan,
    build_request_deduplication_from_pack,
    discover_parquet_inputs,
    discover_scenario_values,
    io_progress_pulses,
    is_remote_io_stall_error,
    iter_adapted_axis_bundles,
    iter_flat_tables,
    iter_length_bucketed_packs,
    limit_malloc_arenas,
    materialize_packed_blocks,
    move_feature_batch,
    pin_feature_batch,
    process_peak_resident_bytes,
    process_resident_bytes,
    privatize_shared_feature_batch,
    prepare_packed_arrow_axis_batch,
    prepare_packed_axis_batch,
    publish_direct_pipeline_stats,
    request_group_blocks_from_adapted_table,
    request_group_blocks_from_arrow_source,
    request_group_blocks_from_axis_bundle,
    note_scan_emitted_rows,
    reset_direct_pipeline_stats,
    resolve_auto_scenarios,
    run_feature_cardinality_audit,
    scan_resume_rewind,
    scan_split_key,
    set_io_progress_hook,
    set_scan_cursor_channel,
    set_scan_resume_plan,
    table_to_feature_batch,
    trim_process_heap,
)
from .features import load_vocab_maps
from .embeddings import (
    ShardedEmbedding,
    consume_sharded_embedding_stats,
    set_embedding_stats_host_sync,
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


class _StepWatchdog:
    """Terminate the process if training stops making optimizer-step progress.

    Silent hangs (blocked HDFS JNI / NCCL waits) leave GPU memory allocated and
    print no traceback. A daemon watchdog converts that into a hard exit so the
    job can be restarted.

    Call :meth:`beat` at phase boundaries (``dataloader`` / ``forward`` / …)
    so a stall dump names where the main thread last made progress instead of
    only saying "no optimizer step".
    """

    def __init__(self, timeout_sec: float, *, rank: int = 0) -> None:
        self._timeout_sec = float(timeout_sec)
        self._rank = int(rank)
        self._last_progress = perf_counter()
        self._phase = "init"
        self._detail = ""
        self._allocated_bytes: int | None = None
        self._reserved_bytes: int | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="mdl-step-watchdog",
            daemon=True,
        )

    def start(self) -> None:
        self.beat("init")
        self._thread.start()
        if self._rank == 0:
            logger.info(
                "step watchdog armed: exit if no optimizer step for %.0fs",
                self._timeout_sec,
            )

    def beat(
        self,
        phase: str | None = None,
        detail: str = "",
        *,
        allocated_bytes: int | None = None,
        reserved_bytes: int | None = None,
    ) -> None:
        with self._lock:
            self._last_progress = perf_counter()
            if phase is not None:
                self._phase = str(phase)
            if detail:
                self._detail = str(detail)
            if allocated_bytes is not None:
                self._allocated_bytes = int(allocated_bytes)
            if reserved_bytes is not None:
                self._reserved_bytes = int(reserved_bytes)

    def stop(self) -> None:
        self._stop.set()

    def _snapshot(self) -> tuple[float, str, str, int | None, int | None]:
        with self._lock:
            return (
                perf_counter() - self._last_progress,
                self._phase,
                self._detail,
                self._allocated_bytes,
                self._reserved_bytes,
            )

    def _run(self) -> None:
        # Poll often enough to notice stalls promptly without waking every second.
        interval = min(30.0, max(5.0, self._timeout_sec / 60.0))
        while not self._stop.wait(interval):
            stalled, phase, detail, allocated, reserved = self._snapshot()
            if stalled < self._timeout_sec:
                continue
            parts = [
                f"Train step watchdog: no progress for {stalled:.0f}s "
                f"(limit={self._timeout_sec:.0f}s) on rank={self._rank}",
                f"last_phase={phase}",
            ]
            if detail:
                parts.append(f"detail={detail}")
            if allocated is not None:
                parts.append(f"cuda_allocated_mib={allocated / (1024 ** 2):.1f}")
            if reserved is not None:
                parts.append(f"cuda_reserved_mib={reserved / (1024 ** 2):.1f}")
            parts.append(
                "exiting to escape a likely dataloader/NCCL hang "
                "(peer ranks should fail faster with "
                "TORCH_NCCL_ASYNC_ERROR_HANDLING=1)"
            )
            message = "; ".join(parts)
            logger.error(message)
            print(message, flush=True)
            # Hard exit: exceptions in this thread cannot unwind the hung main
            # thread blocked in JNI/NCCL.
            os._exit(70)


def _step_watchdog_beat(
    watchdog: _StepWatchdog | None,
    phase: str,
    *,
    detail: str = "",
    device: torch.device | None = None,
) -> None:
    """Record phase progress for the step watchdog (no-op when disarmed)."""

    if watchdog is None:
        return
    allocated: int | None = None
    reserved: int | None = None
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        try:
            allocated = int(torch.cuda.memory_allocated(device))
            reserved = int(torch.cuda.memory_reserved(device))
        except Exception:  # noqa: BLE001 - diagnostics must not break training
            allocated = None
            reserved = None
    watchdog.beat(
        phase,
        detail,
        allocated_bytes=allocated,
        reserved_bytes=reserved,
    )


def _abort_rank_for_cuda_oom(
    error: BaseException,
    *,
    rank: int,
    steps: int,
    detail: str = "",
) -> None:
    """Hard-exit after CUDA OOM so torchrun restarts; peers rely on NCCL async."""

    parts = [
        f"CUDA_OOM: aborting rank={rank} after step={steps}",
        str(error).split("\n", 1)[0],
    ]
    if detail:
        parts.append(detail)
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            parts.append(
                f"cuda_allocated_mib={allocated:.1f} cuda_reserved_mib={reserved:.1f}"
            )
        except Exception:  # noqa: BLE001
            pass
    parts.append(
        "peers should surface NCCL errors via TORCH_NCCL_ASYNC_ERROR_HANDLING; "
        "resume from checkpoint after restart"
    )
    message = "; ".join(parts)
    logger.error(message)
    print(message, flush=True)
    os._exit(70)


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
    only required when a model constructs ``DomainAwareAttention``: MDL does so
    for enabled task/scenario feature interactions, and the equal-readout
    control does so for its per-task queries. Plain RankMixer token mixing does
    not use padded Flash, so both capabilities are independent.
    """

    model = config.model
    if getattr(model, "readout", "default") == "task_query":
        return True
    if model.name not in {
        "mdl_rankmixer",
        "mdl_onetrans",
        "mdl_mixformer",
    }:
        return False
    if model.name == "mdl_onetrans" and model.first_domain_sequence_layer == 0:
        # Every Domain block reads the variable-length ``[Q_S; NS]`` pool, so
        # no fixed-width ``DomainAwareAttention`` is built at all.
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
class FixedTestEvalResult:
    rows: int
    metrics: dict[str, dict[str, float | int | None]]
    elapsed_seconds: float
    files: int


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
    *,
    prefer_collective_bw: bool = False,
    hbm_caps_min_world_size: int = 6,
) -> None:
    """Use CUDA P2P when healthy; otherwise tell NCCL to fall back safely.

    - Always default ``TORCH_NCCL_ASYNC_ERROR_HANDLING=1`` so a dead peer
      fails collectives quickly instead of waiting for the step watchdog.
    - P2P OK → leave NCCL P2P defaults so NVLink/P2P stays enabled.
    - P2P broken → ``NCCL_IGNORE_DISABLED_P2P=1`` + ``NCCL_P2P_DISABLE=1``
      so NCCL skips doomed P2P and uses SHM/NET.
    - Inconclusive (e.g. each torchrun rank only sees one GPU) → only
      ``NCCL_IGNORE_DISABLED_P2P=1``: still try P2P when the fabric allows
      it, but do not abort if the driver/NVLink reports disabled P2P.
    - Explicit env exports always win.

    ``prefer_collective_bw`` (RankMixer-family): keep larger NCCL buffers /
    full channel count so emb A2A is not BW-starved on 6–8 GPU. OneTrans-style
    models keep the tighter HBM caps. ``hbm_caps_min_world_size`` is 4 for
    OneTrans (long-S remat fights NCCL scratch for the same 80 GiB) and 6 for
    other non-RankMixer models.

    ``environ`` defaults to ``os.environ``; the DDP launcher may pass a copied
    dict so child processes inherit the decision.
    """

    env: MutableMapping[str, str] = os.environ if environ is None else environ
    # Fail collectives when a peer dies/errors instead of blocking until the
    # 600s step watchdog. Explicit exports always win over this default.
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    # Read WORLD_SIZE from the target mapping (launcher may pass a child env
    # before torchrun rewrites the parent process environ).
    try:
        world_size = int(env.get("WORLD_SIZE", "1") or "1")
    except ValueError:
        world_size = 1
    # Emb-bound RankMixer needs larger scratch from 2-GPU up. OneTrans applies
    # tighter HBM caps from 4-GPU up; other models keep the historical ≥6 gate.
    if prefer_collective_bw and world_size >= 2:
        env.setdefault("NCCL_CUMEM_ENABLE", "0")
        env.setdefault("NCCL_BUFFSIZE", str(8 * 1024 * 1024))
        env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")
    elif world_size >= max(2, int(hbm_caps_min_world_size)):
        env.setdefault("NCCL_CUMEM_ENABLE", "0")
        env.setdefault("NCCL_BUFFSIZE", str(2 * 1024 * 1024))
        env.setdefault("NCCL_MAX_NCHANNELS", "4")
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


def _is_rankmixer_family(config: AppConfig) -> bool:
    return str(getattr(config.model, "name", "")) in {"rankmixer", "mdl_rankmixer"}


def _is_onetrans_family(config: AppConfig) -> bool:
    return str(getattr(config.model, "name", "")) in {"onetrans", "mdl_onetrans"}


def _local_world_size() -> int:
    """Ranks co-located on this node (torchrun ``LOCAL_WORLD_SIZE``)."""

    local = _env_int("LOCAL_WORLD_SIZE", 0)
    if local > 0:
        return local
    return max(1, _env_int("WORLD_SIZE", 1))


def _apply_local_rank_cpu_affinity(role: str) -> list[int]:
    """Partition host CPUs across local ranks so 6/8-GPU prepare does not collide.

    Every rank used to pin its host-prepare child to ``0..n_cpu//3`` and the
    train parent to the remainder — on 6–8 GPU that means six prepare children
    fighting over the same cores while train threads also overlap. Slice the
    machine by ``LOCAL_RANK`` instead.
    """

    try:
        import psutil
    except ImportError:
        return []
    n_cpu = int(psutil.cpu_count(logical=True) or 4)
    local_world = _local_world_size()
    local_rank = max(0, min(_env_int("LOCAL_RANK", 0), local_world - 1))
    slice_size = max(1, n_cpu // local_world)
    start = local_rank * slice_size
    end = n_cpu if local_rank == local_world - 1 else start + slice_size
    slice_cores = list(range(start, end))
    if not slice_cores:
        return []
    if role == "host_prepare":
        # Pack/tensorize is CPU-heavy: take ~2/3 of the rank slice.
        n_prep = max(2, (len(slice_cores) * 2) // 3)
        cores = slice_cores[:n_prep]
    elif role == "train":
        n_prep = max(2, (len(slice_cores) * 2) // 3)
        cores = slice_cores[n_prep:] or slice_cores[-1:]
    else:
        cores = slice_cores
    try:
        psutil.Process().cpu_affinity(cores)
    except Exception:
        return []
    return cores


def _small_hbm_cuda_device(*, threshold_gib: float = 32.0) -> bool:
    """True when the visible CUDA device looks like ≤``threshold_gib`` HBM."""

    if not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        total = float(getattr(props, "total_memory", 0) or 0)
    except Exception:  # noqa: BLE001 - probe must never crash launch
        return False
    return total > 0.0 and total <= float(threshold_gib) * (1024**3)


def _local_batch_scale_for_world_size(
    world_size: int,
    *,
    rankmixer_family: bool = False,
) -> float:
    """Scale per-rank batch so large world sizes keep HBM headroom for NCCL.

    Override with ``MDL_LOCAL_BATCH_SCALE`` (e.g. ``0.75`` / ``1.0``).
    RankMixer-family keeps full batch at 8 GPUs by default on large HBM; the
    small-HBM mild derate is applied in ``_apply_world_size_training_profile``.
    """

    forced = os.environ.get("MDL_LOCAL_BATCH_SCALE", "").strip()
    if forced:
        scale = float(forced)
        if scale <= 0.0:
            raise ValueError("MDL_LOCAL_BATCH_SCALE must be positive")
        return scale
    if world_size <= 6:
        return 1.0
    if world_size <= 8:
        # OneTrans 8×1024 OOMed in backward → 0.75. RankMixer-family keeps
        # full per-rank batch so dense/emb work can hide HDFS/A2A bubbles.
        return 1.0 if rankmixer_family else 0.75
    return max(0.5, 6.0 / float(world_size))


def _scale_int_batch(value: int, scale: float) -> int:
    return max(8, int(round(int(value) * float(scale))))


def _scale_reader_batches(reader: ReaderConfig, scale: float) -> ReaderConfig:
    if abs(scale - 1.0) < 1.0e-9 or not reader.length_buckets:
        return reader
    buckets = tuple(
        replace(bucket, batch_size=_scale_int_batch(bucket.batch_size, scale))
        for bucket in reader.length_buckets
    )
    return replace(reader, length_buckets=buckets)


def _apply_world_size_training_profile(
    config: AppConfig,
    world_size: int,
) -> AppConfig:
    """Derate local batch / prefetch and widen DDP buckets for multi-GPU jobs.

    Aggressive multi-GPU defaults kick in at ``world_size >= 2`` so 2/3/4-GPU
    local DDP gets emb/prefetch/bucket treatment (not only 6/8-GPU).
    Override via env:
    - ``MDL_LOCAL_BATCH_SCALE``: per-rank batch multiplier
    - ``MDL_GROUPED_EMB_MAX_OUTPUT_MIB``: emb A2A chunk cap
    """

    if world_size <= 1:
        return config
    rankmixer_family = _is_rankmixer_family(config)
    multi_gpu = world_size >= 2
    model_name = str(getattr(config.model, "name", ""))
    accessible = _local_cuda_p2p_accessible()
    if accessible is True:
        p2p_ok = True
    elif accessible is False:
        p2p_ok = False
    else:
        p2p_ok = os.environ.get("NCCL_P2P_DISABLE", "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }
    small_hbm = _small_hbm_cuda_device()
    scale = _local_batch_scale_for_world_size(
        world_size, rankmixer_family=rankmixer_family
    )
    # ≤32 GiB 7–8 GPU: mild RankMixer-family derate next to graph/NCCL pools.
    # Large-HBM (H100) keeps scale=1.0 for util protect.
    if (
        rankmixer_family
        and small_hbm
        and 6 < world_size <= 8
        and not os.environ.get("MDL_LOCAL_BATCH_SCALE", "").strip()
        and abs(scale - 1.0) < 1.0e-9
    ):
        scale = 0.9
    # Plain RankMixer is emb-A2A–bound: on no-P2P multi-GPU, mild upscale
    # lengthens dense work so it can hide collectives (4×4090 A/B: 1.42).
    if (
        abs(scale - 1.0) < 1.0e-9
        and model_name == "rankmixer"
        and multi_gpu
        and not p2p_ok
        and not os.environ.get("MDL_LOCAL_BATCH_SCALE", "").strip()
    ):
        scale = 1.42
    training = config.training
    data = config.data
    runtime = config.runtime
    model = config.model
    onetrans_family = _is_onetrans_family(config)
    full_remat = str(getattr(runtime, "activation_checkpoint", "none")) == "full"
    if abs(scale - 1.0) >= 1.0e-9:
        old_bs = int(training.batch_size)
        new_bs = _scale_int_batch(old_bs, scale)
        training = replace(training, batch_size=new_bs)
        train_split = data.train
        test_split = data.test
        if train_split is not None:
            train_split = replace(
                train_split,
                reader=_scale_reader_batches(train_split.reader, scale),
            )
        if test_split is not None:
            test_split = replace(
                test_split,
                reader=_scale_reader_batches(test_split.reader, scale),
            )
        data = replace(data, train=train_split, test=test_split)

    # Cap emb A2A staging. Explicit launcher exports always win.
    if multi_gpu:
        if rankmixer_family and p2p_ok and not small_hbm:
            emb_cap = "1024"
        elif rankmixer_family:
            # No-P2P / 24GB: medium chunks — 768+ regressed sps on 4×4090.
            emb_cap = "512"
        elif onetrans_family and full_remat:
            # Full remat already fights activations for HBM; keep emb scratch
            # smaller than the default 512 MiB OneTrans staging buffer.
            emb_cap = "256" if world_size >= 4 else "384"
        else:
            emb_cap = "384" if world_size >= 8 else "512"
        os.environ.setdefault("MDL_GROUPED_EMB_MAX_OUTPUT_MIB", emb_cap)

    # OneTrans full-remat HBM rescue (infra only; numerics unchanged):
    # compact backbone packing, drop the extra device FeatureBatch, drop the
    # request-cache layer ``s_input`` tape that full remat otherwise retains,
    # and tighten projector chunking.
    if onetrans_family and full_remat:
        runtime_updates: dict[str, object] = {}
        if getattr(runtime, "varlen_packing", "fixed") != "compact":
            runtime_updates["varlen_packing"] = "compact"
        chunk_tokens = int(
            getattr(runtime, "sequence_projection_chunk_tokens", 0) or 0
        )
        if chunk_tokens <= 0 or chunk_tokens > 32768:
            runtime_updates["sequence_projection_chunk_tokens"] = 32768
        if runtime_updates:
            runtime = replace(runtime, **runtime_updates)
        if bool(getattr(model, "use_request_cache", False)):
            model = replace(model, use_request_cache=False)

    # MDL-RankMixer CUDA-graph pools are huge on ≤32 GiB; disable graph.
    if (
        small_hbm
        and bool(runtime.cuda_graph_backbone)
        and model_name == "mdl_rankmixer"
    ):
        runtime = replace(runtime, cuda_graph_backbone=False)
        if is_main_process():
            print(
                "Multi-GPU profile | disabled cuda_graph_backbone for "
                "mdl_rankmixer on ≤32GiB GPU (graph private pools OOM)",
                flush=True,
            )

    train = data.train
    if train is not None:
        reader = train.reader
        reader_changed = False
        if (
            world_size >= 8
            and reader.device_prefetch_batches > 0
            and not rankmixer_family
        ):
            reader = replace(reader, device_prefetch_batches=0)
            reader_changed = True
        # OneTrans full remat: even one prefetched FeatureBatch of long S is a
        # second live copy beside the remat working set. Drop it from 2-GPU up.
        if (
            onetrans_family
            and full_remat
            and multi_gpu
            and reader.device_prefetch_batches > 0
        ):
            reader = replace(reader, device_prefetch_batches=0)
            reader_changed = True
        # Deepen device prefetch only when P2P + HBM can absorb it.
        deepen_device = (
            rankmixer_family
            and multi_gpu
            and p2p_ok
            and not small_hbm
            and 0 < reader.device_prefetch_batches < 2
        )
        # Keep prior ≥6-GPU RankMixer deepen even when P2P probe is unclear
        # (matches existing 6/8-GPU util profile tests).
        if (
            not deepen_device
            and rankmixer_family
            and world_size >= 6
            and 0 < reader.device_prefetch_batches < 2
        ):
            deepen_device = True
        if deepen_device:
            reader = replace(reader, device_prefetch_batches=2)
            reader_changed = True
        if (
            small_hbm
            and bool(runtime.cuda_graph_backbone)
            and model_name == "mdl_rankmixer"
            and reader.device_prefetch_batches > 1
        ):
            reader = replace(reader, device_prefetch_batches=1)
            reader_changed = True
        # Deep host-prepare / Arrow prefetch for all multi-GPU families.
        # RankMixer-family util-protect mock: host=10/pf=6.
        if rankmixer_family and multi_gpu:
            target_host = 10
            target_prefetch = 6
        elif multi_gpu:
            target_host = 6
            target_prefetch = 4
        else:
            target_host = reader.host_prepare_prefetch
            target_prefetch = reader.prefetch_batches
        if (
            reader.host_prepare_prefetch > 0
            and reader.host_prepare_prefetch < target_host
        ):
            reader = replace(reader, host_prepare_prefetch=target_host)
            reader_changed = True
        if multi_gpu and 0 < reader.prefetch_batches < target_prefetch:
            reader = replace(reader, prefetch_batches=target_prefetch)
            reader_changed = True
        if (
            rankmixer_family
            and multi_gpu
            and float(reader.hdfs_op_timeout) > 15.0
        ):
            reader = replace(reader, hdfs_op_timeout=15.0)
            reader_changed = True
        if reader_changed:
            train = replace(train, reader=reader)
            data = replace(data, train=train)

    ddp = training.ddp
    if world_size >= 8:
        target_bucket = 250.0 if rankmixer_family else 125.0
    elif multi_gpu:
        target_bucket = 250.0 if rankmixer_family else 100.0
    else:
        target_bucket = ddp.bucket_cap_mb
    if multi_gpu and ddp.bucket_cap_mb < target_bucket:
        training = replace(
            training,
            ddp=replace(ddp, bucket_cap_mb=target_bucket),
        )

    if (
        training is config.training
        and data is config.data
        and runtime is config.runtime
        and model is config.model
    ):
        if is_main_process() and multi_gpu:
            print(
                "Multi-GPU profile | "
                f"world_size={world_size} local_batch_scale={scale:.3f} "
                f"batch_per_rank={config.training.batch_size} "
                f"device_prefetch="
                f"{0 if config.data.train is None else config.data.train.reader.device_prefetch_batches} "
                f"host_prepare="
                f"{0 if config.data.train is None else config.data.train.reader.host_prepare_prefetch} "
                f"ddp_bucket_cap_mb={config.training.ddp.bucket_cap_mb:g} "
                f"emb_max_output_mib="
                f"{os.environ.get('MDL_GROUPED_EMB_MAX_OUTPUT_MIB', '')} "
                f"p2p={int(p2p_ok)} small_hbm={int(small_hbm)} "
                f"cuda_graph={int(bool(runtime.cuda_graph_backbone))} "
                f"activation_checkpoint={runtime.activation_checkpoint} "
                f"varlen_packing={runtime.varlen_packing} "
                f"request_cache={int(bool(model.use_request_cache))}",
                flush=True,
            )
        return config
    updated = replace(
        config, training=training, data=data, runtime=runtime, model=model
    )
    if is_main_process():
        train_reader = updated.data.train.reader if updated.data.train else None
        print(
            "Multi-GPU profile | "
            f"world_size={world_size} local_batch_scale={scale:.3f} "
            f"batch_per_rank={updated.training.batch_size} "
            f"device_prefetch="
            f"{0 if train_reader is None else train_reader.device_prefetch_batches} "
            f"host_prepare="
            f"{0 if train_reader is None else train_reader.host_prepare_prefetch} "
            f"ddp_bucket_cap_mb={updated.training.ddp.bucket_cap_mb:g} "
            f"emb_max_output_mib="
            f"{os.environ.get('MDL_GROUPED_EMB_MAX_OUTPUT_MIB', '')} "
            f"p2p={int(p2p_ok)} small_hbm={int(small_hbm)} "
            f"cuda_graph={int(bool(updated.runtime.cuda_graph_backbone))} "
            f"activation_checkpoint={updated.runtime.activation_checkpoint} "
            f"varlen_packing={updated.runtime.varlen_packing} "
            f"request_cache={int(bool(updated.model.use_request_cache))}",
            flush=True,
        )
    return updated


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
            # WORLD_SIZE is already set by torchrun here; apply NCCL HBM/BW
            # knobs before ProcessGroupNCCL allocates channel/scratch buffers.
            _configure_nccl_runtime_env(
                prefer_collective_bw=_is_rankmixer_family(config),
                hbm_caps_min_world_size=(
                    4 if _is_onetrans_family(config) else 6
                ),
            )
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
    if not (context.initialized_here and torch_dist.is_initialized()):
        return

    def _destroy() -> None:
        global _CONTROL_PROCESS_GROUP
        if _CONTROL_PROCESS_GROUP is not None:
            torch_dist.destroy_process_group(_CONTROL_PROCESS_GROUP)
            _CONTROL_PROCESS_GROUP = None
        torch_dist.destroy_process_group()

    # A peer stuck in host-prepare teardown can make NCCL destroy block forever
    # after we already printed the local SIGKILL warning.
    worker = threading.Thread(
        target=_destroy,
        name="mdl-destroy-process-group",
        daemon=True,
    )
    worker.start()
    worker.join(timeout=30.0)
    if worker.is_alive():
        logger.warning(
            "destroy_process_group timed out after 30s; abandoning distributed cleanup"
        )
        print(
            "destroy_process_group timed out after 30s; abandoning to avoid hang",
            flush=True,
        )


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


def _evenly_spaced_file_uris(
    uris: list[str],
    limit: int,
) -> tuple[str, ...]:
    """Choose a deterministic subset that spans the full sorted time range."""

    if limit <= 0:
        raise ValueError("fixed-test file limit must be positive")
    if len(uris) <= limit:
        return tuple(uris)
    if limit == 1:
        return (uris[len(uris) // 2],)
    last = len(uris) - 1
    return tuple(uris[(index * last) // (limit - 1)] for index in range(limit))


def _prepare_fixed_test_eval(
    config: AppConfig,
    context: DistributedContext,
) -> AppConfig:
    """Freeze one representative test manifest and tune its forward-only reader."""

    evaluation = config.training.fixed_test_eval
    if not evaluation.enabled:
        return config
    test = config.data.test
    if test is None:
        raise ValueError("fixed-test evaluation requires data.test")
    test.require_inputs("test")
    file_limit = int(evaluation.files_per_rank) * max(1, context.world_size)
    payload: list[dict[str, Any] | None] = [None]
    if not context.enabled or context.rank == 0:
        try:
            refs = discover_parquet_inputs(
                test.inputs,
                remote_list_timeout_sec=(
                    float(test.reader.hdfs_open_timeout)
                    if any(
                        str(item).startswith(("hdfs://", "viewfs://"))
                        for item in test.inputs
                    )
                    else None
                ),
            )
            available = len(refs)
            if (
                test.reader.shard_unit == "file"
                and available < context.world_size
            ):
                raise ValueError(
                    "fixed test window contains fewer Parquet files than ranks: "
                    f"files={available} ranks={context.world_size}"
                )
            selected = _evenly_spaced_file_uris(
                [ref.canonical_uri for ref in refs],
                file_limit,
            )
            payload[0] = {
                "paths": list(selected),
                "available": available,
                "error": None,
            }
        except Exception as error:  # Broadcast the failure so peers do not hang.
            payload[0] = {"paths": None, "available": None, "error": str(error)}
    if context.enabled:
        torch_dist.broadcast_object_list(
            payload,
            src=0,
            group=context.control_group,
        )
    result = payload[0]
    if not isinstance(result, dict):
        raise RuntimeError("fixed-test manifest broadcast returned an invalid payload")
    failure = result.get("error")
    if failure:
        raise RuntimeError(f"fixed-test manifest discovery failed: {failure}")
    paths = result.get("paths")
    if not isinstance(paths, list) or not paths:
        raise RuntimeError("fixed-test manifest discovery returned no Parquet files")

    # Evaluation has no backward activations, so training-sized batches improve
    # GPU occupancy without exceeding the already-proven training batch budget.
    # Keep only one CUDA-prefetched batch and leave deeper buffering in host RAM.
    train_reader = config.data.train.reader
    test_reader = replace(
        test.reader,
        length_buckets=train_reader.length_buckets,
        shuffle_buffer_rows=0,
        prefetch_batches=max(
            test.reader.prefetch_batches,
            min(train_reader.prefetch_batches, 4),
        ),
        host_prepare_prefetch=max(
            test.reader.host_prepare_prefetch,
            min(train_reader.host_prepare_prefetch, 4),
        ),
        device_prefetch_batches=min(test.reader.device_prefetch_batches, 1),
    )
    frozen_test = replace(
        test,
        inputs=tuple(paths),
        reader=test_reader,
        prediction_keys={},
    )
    prepared = replace(config, data=replace(config.data, test=frozen_test))
    if context.rank == 0:
        print(
            "Fixed test manifest | "
            f"files_selected={len(paths)} files_available={result.get('available')} "
            f"files_per_rank={evaluation.files_per_rank} "
            f"world_size={context.world_size}",
            flush=True,
        )
    return prepared


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
    tensor_max_length = getattr(
        sequence, "tensor_max_length", sequence.max_length
    )
    if tensor_max_length is not None:
        values.clamp_(max=tensor_max_length)
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
        registry.clear()


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


def _feature_batch_row_count(batch: FeatureBatch) -> int:
    """Rows in one prepared batch, counted the way the training loop counts them."""

    scenario_id = getattr(batch, "scenario_id", None)
    if scenario_id is None:
        return 0
    return int(scenario_id.size(0))


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
    scan_cursor: ScanCursorChannel | None = None,
    scan_resume_plan: ScanResumePlan | None = None,
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
                scan_cursor=scan_cursor,
                scan_resume_plan=scan_resume_plan,
            )
        # Compare mode runs two readers over the same rows, so neither may own
        # the cursor. Checkpoints still resume the step and the weights; the
        # input scan restarts from the beginning of the shard.
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
            scan_cursor=scan_cursor,
            scan_resume_plan=scan_resume_plan,
        )

    # In-process reader: the scanner runs here, so install the cursor and the
    # resume offset on this process instead of handing them to a child.
    split_key: str | None = None
    if scan_cursor is not None or scan_resume_plan is not None:
        split_key = scan_split_key(
            split,
            shard_rank=shard_rank,
            shard_world_size=shard_world_size,
        )
        if scan_cursor is not None:
            set_scan_cursor_channel(scan_cursor, split_key=split_key)
        set_scan_resume_plan(scan_resume_plan)

    table_iter = _iter_batch_tables(
        config,
        split_name,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
        require_labels=require_labels,
    )

    def _prepare(table: object) -> FeatureBatch:
        batch = _prepare_feature_batch(
            config,
            split,
            table,
            vocab_maps,
            require_labels,
            pin_memory,
            coalesce_pinned_tensors,
            include_group_id,
        )
        if split_key is not None:
            note_scan_emitted_rows(
                _feature_batch_row_count(batch), split_key=split_key
            )
        return batch

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
    ``auto`` prefers ``share`` when ``/dev/shm`` has headroom (zero-copy Queue
    handles — critical for RankMixer util on 2–4 GPU). Tiny containers fall
    back to ``memfd``. Parent always privatizes/pins via the recycled pool so
    ``share`` no longer ratchets RSS the way the old pin-before-share path did.
    """

    env = os.environ if environ is None else environ
    forced = str(env.get("MDL_HOST_PREPARE_IPC", "auto")).strip().lower()
    if forced in {"share", "memfd"}:
        return forced
    shm_free = _dev_shm_free_bytes()
    if shm_free >= _HOST_PREPARE_SHARE_SHM_BYTES:
        return "share"
    return "memfd"


def _release_cached_host_allocator_memory() -> None:
    """Return idle CUDA host-allocator slabs to the OS when the runtime allows.

    Variable-length ``pin_memory()`` traffic otherwise leaves the host caching
    allocator at its high-water mark so container RSS climbs for hours after
    FeatureBatch Python refs are already gone.
    """

    try:
        empty = getattr(torch._C, "_host_emptyCache", None)
        if empty is None:
            empty = getattr(torch._C, "_accelerator_emptyHostCache", None)
        if callable(empty):
            empty()
    except Exception:
        pass


class _PinnedPoolLease:
    """Owns one pooled pinned-buffer slot until the FeatureBatch is collected."""

    __slots__ = ("_pool", "_slot", "_released")

    def __init__(
        self,
        pool: "_PinnedHostBufferPool",
        slot: dict[torch.dtype, Tensor],
    ) -> None:
        self._pool = pool
        self._slot = slot
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        slot = self._slot
        self._slot = {}
        pool = self._pool
        self._pool = None  # type: ignore[assignment]
        if slot is not None and pool is not None:
            pool._release(slot)

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def _pinned_pool_max_slot_bytes_from_env(
    environ: MutableMapping[str, str] | None = None,
) -> int | None:
    """Optional hard cap on idle pinned bytes retained per dtype buffer.

    Set ``MDL_PINNED_POOL_MAX_SLOT_BYTES`` (positive int). Unset / empty keeps
    the default soft policy (sliding high-water shrink only).
    """

    env = os.environ if environ is None else environ
    raw = str(env.get("MDL_PINNED_POOL_MAX_SLOT_BYTES", "")).strip()
    if not raw:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError("MDL_PINNED_POOL_MAX_SLOT_BYTES must be positive")
    return value


class _PinnedHostBufferPool:
    """Recycle pinned host storages across variable-length FeatureBatches.

    Fresh ``torch.empty(..., pin_memory=True)`` per batch lets the CUDA caching
    host allocator ratchet RSS with every new size class. Outstanding batches
    exclusive-lease a small set of buffers; when the lease returns, oversized
    idle slabs are dropped against a *sliding* recent high-water mark so an
    occasional long pack cannot pin container RSS at the historical peak.
    """

    def __init__(
        self,
        *,
        max_free_slots: int = 4,
        recent_window: int = 256,
        shrink_factor: float = 2.0,
        max_slot_bytes: int | None = None,
    ) -> None:
        self._free: list[dict[torch.dtype, Tensor]] = []
        self._lock = threading.Lock()
        self._max_free_slots = max(1, int(max_free_slots))
        self._recent_window = max(8, int(recent_window))
        self._shrink_factor = max(1.0, float(shrink_factor))
        if max_slot_bytes is None:
            max_slot_bytes = _pinned_pool_max_slot_bytes_from_env()
        self._max_slot_bytes = (
            None if max_slot_bytes is None else max(1, int(max_slot_bytes))
        )
        self._recent: dict[torch.dtype, deque[int]] = {}
        self._checkouts = 0
        self._reuses = 0
        self._grows = 0
        self._shrinks = 0

    def checkout(
        self,
        specs: list[tuple[torch.dtype, int]],
    ) -> tuple[list[Tensor], _PinnedPoolLease]:
        with self._lock:
            slot = self._free.pop() if self._free else {}
            reused = bool(slot)
            if reused:
                self._reuses += 1
            self._checkouts += 1
            for dtype, numel in specs:
                if numel <= 0:
                    raise ValueError("pinned pool numel must be positive")
                self._note_request_locked(dtype, int(numel))
        views: list[Tensor] = []
        grew = False
        for dtype, numel in specs:
            need = int(numel)
            buf = slot.get(dtype)
            if buf is None or int(buf.numel()) < need:
                alloc = self._alloc_numel(dtype, need)
                if buf is not None:
                    grew = True
                buf = torch.empty(alloc, dtype=dtype, pin_memory=True)
                slot[dtype] = buf
            views.append(buf.narrow(0, 0, need))
        if grew:
            with self._lock:
                self._grows += 1
            # Previous smaller slab is now unreferenced; return it to the OS.
            _release_cached_host_allocator_memory()
        return views, _PinnedPoolLease(self, slot)

    def _note_request_locked(self, dtype: torch.dtype, numel: int) -> None:
        window = self._recent.get(dtype)
        if window is None:
            window = deque(maxlen=self._recent_window)
            self._recent[dtype] = window
        window.append(int(numel))

    def _recent_high_water(self, dtype: torch.dtype) -> int:
        window = self._recent.get(dtype)
        if not window:
            return 0
        return max(window)

    def _alloc_numel(self, dtype: torch.dtype, need: int) -> int:
        # Modest headroom (12.5%) cuts realloc chatter without the old 25% ratchet.
        alloc = max(int(need), int(need * 9 // 8))
        if self._max_slot_bytes is None:
            return alloc
        elem = max(1, int(torch.empty((), dtype=dtype).element_size()))
        cap = max(1, int(self._max_slot_bytes) // elem)
        # Always satisfy the live batch; skip headroom when already over the
        # idle retention cap so release-time trim can reclaim promptly.
        if need >= cap:
            return need
        return min(alloc, cap)

    def _trim_slot_locked(self, slot: dict[torch.dtype, Tensor]) -> bool:
        """Drop idle buffers far above the sliding high-water / byte cap."""

        trimmed = False
        for dtype, buf in list(slot.items()):
            hwm = self._recent_high_water(dtype)
            drop = False
            if hwm > 0 and int(buf.numel()) > int(self._shrink_factor * hwm):
                drop = True
            elif self._max_slot_bytes is not None:
                bytes_per = max(1, int(buf.element_size()))
                if int(buf.numel()) * bytes_per > int(self._max_slot_bytes):
                    drop = True
            if drop:
                del slot[dtype]
                trimmed = True
                self._shrinks += 1
        return trimmed

    def _release(self, slot: dict[torch.dtype, Tensor]) -> None:
        reclaim = False
        with self._lock:
            reclaim = self._trim_slot_locked(slot)
            if slot and len(self._free) < self._max_free_slots:
                self._free.append(slot)
            else:
                # Empty after trim, or free-list full: drop so pages can return.
                reclaim = True
                del slot
        if reclaim:
            _release_cached_host_allocator_memory()

    def idle_bytes(self) -> int:
        """Pinned bytes parked in the free list (the part that can ratchet)."""

        with self._lock:
            return sum(
                _tensor_nbytes(buffer)
                for slot in self._free
                for buffer in slot.values()
            )


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


def _pin_feature_batch_with_pool(
    batch: FeatureBatch,
    *,
    pool: _PinnedHostBufferPool | None,
) -> FeatureBatch:
    """Clone shared/unpinned coalesced buffers into a recycled pinned lease."""

    if pool is None or not batch._packed_buffers:
        return pin_feature_batch(batch, coalesce_tensors=False)
    specs = [
        (buffer.dtype, int(buffer.numel())) for buffer in batch._packed_buffers
    ]
    views, lease = pool.checkout(specs)
    for dst, src in zip(views, batch._packed_buffers):
        dst.copy_(src.detach().reshape(-1))
    pinned_by_dtype = {buffer.dtype: buffer for buffer in views}

    def pin_view(tensor: Tensor) -> Tensor:
        base = pinned_by_dtype[tensor.dtype]
        return base.as_strided(
            tensor.size(),
            tensor.stride(),
            tensor.storage_offset(),
        )

    return FeatureBatch(
        features={
            key: _map_feature_value(value, pin_view)
            for key, value in batch.features.items()
        },
        labels=None if batch.labels is None else pin_view(batch.labels),
        label_mask=(
            None if batch.label_mask is None else pin_view(batch.label_mask)
        ),
        scenario_id=pin_view(batch.scenario_id),
        group_id=batch.group_id,
        prediction_keys=batch.prediction_keys,
        _packed_buffers=tuple(views),
        _keepalive=lease,
    )


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


def _write_tensor_into_mmap(mapped: Any, offset: int, buffer: Tensor) -> int:
    """Copy one CPU tensor into ``mapped`` at ``offset``; return bytes written."""

    contiguous = buffer.detach().contiguous()
    if contiguous.dtype == torch.bfloat16:
        array = contiguous.view(torch.uint16).numpy().reshape(-1)
    else:
        array = contiguous.numpy().reshape(-1)
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
    raw = memoryview(array).cast("B")
    end = offset + len(raw)
    mapped[offset:end] = raw
    return len(raw)


def _create_anonymous_ipc_fd(name: str) -> int:
    """Create an anonymous, send_handle-compatible file descriptor."""

    create_memfd = getattr(os, "memfd_create", None)
    if callable(create_memfd):
        return int(create_memfd(name, 0))
    # Some Linux Python builds omit os.memfd_create even when the kernel has
    # tmpfs. TemporaryFile is already unlinked; dup keeps its backing alive
    # after the Python file object closes.
    shm_dir = (
        "/dev/shm"
        if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)
        else None
    )
    with tempfile.TemporaryFile(prefix=f"{name}-", dir=shm_dir) as temporary:
        return os.dup(temporary.fileno())


def _spill_feature_batch_for_ipc(batch: FeatureBatch) -> tuple[dict[str, Any], int]:
    """Pack coalesced buffers into an anonymous FD + metadata payload.

    Returns ``(metadata, memfd)``. The caller owns ``memfd`` and must pass it
    to ``_publish_memfd_payload`` or ``_load_feature_batch_from_ipc``; both
    close it.

    Cross-process transfer must NOT use ``multiprocessing.reduction.DupFd`` /
    ``resource_sharer``: detaching those tokens races with child exit and shows
    up in production as ``ConnectionResetError`` or ``FileNotFoundError`` after
    many thousands of steps. Use a dedicated Pipe + ``send_handle`` instead.
    """

    if not batch._packed_buffers:
        raise ValueError("host-prepare IPC requires coalesced _packed_buffers")
    import mmap

    sizes = [
        int(buffer.numel()) * int(buffer.element_size())
        for buffer in batch._packed_buffers
    ]
    total = int(sum(sizes))
    buffer_records: list[tuple[str, int, int]] = []
    offset = 0
    for buffer, nbytes in zip(batch._packed_buffers, sizes):
        buffer_records.append((str(buffer.dtype), int(nbytes), offset))
        offset += int(nbytes)
    fd = _create_anonymous_ipc_fd(f"mdl-host-prep-{time_ns()}")
    try:
        os.ftruncate(fd, total)
        mapped = mmap.mmap(fd, total)
        try:
            cursor = 0
            for buffer, nbytes in zip(batch._packed_buffers, sizes):
                written = _write_tensor_into_mmap(mapped, cursor, buffer)
                if written != nbytes:
                    raise RuntimeError(
                        f"IPC write size mismatch: wrote {written}, expected {nbytes}"
                    )
                cursor += written
            mapped.flush()
        finally:
            mapped.close()
        payload = {
            "size": total,
            "buffers": buffer_records,
            "features": _encode_feature_batch_views(batch.features, batch._packed_buffers),
            "labels": _encode_feature_batch_views(batch.labels, batch._packed_buffers),
            "label_mask": _encode_feature_batch_views(batch.label_mask, batch._packed_buffers),
            "scenario_id": _encode_feature_batch_views(batch.scenario_id, batch._packed_buffers),
            "group_id": batch.group_id,
            "prediction_keys": batch.prediction_keys,
        }
    except BaseException:
        os.close(fd)
        raise
    return payload, fd


def _load_feature_batch_from_ipc(
    payload: dict[str, Any],
    *,
    fd: int,
    pin_memory: bool,
    pinned_pool: _PinnedHostBufferPool | None = None,
) -> FeatureBatch:
    import mmap

    try:
        size = int(payload["size"])
        mapped = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        try:
            records = list(payload["buffers"])
            dtypes: list[torch.dtype] = []
            raw_slices: list[Any] = []
            specs: list[tuple[torch.dtype, int]] = []
            for dtype_name, nbytes, offset in records:
                dtype = getattr(torch, dtype_name.removeprefix("torch."))
                dtypes.append(dtype)
                # mmap slice returns bytes (a copy); safe to close afterward.
                raw = mapped[offset : offset + int(nbytes)]
                raw_slices.append(raw)
                itemsize = 2 if dtype == torch.bfloat16 else dtype.itemsize
                specs.append((dtype, int(nbytes) // int(itemsize)))
            lease: _PinnedPoolLease | None = None
            if pin_memory and pinned_pool is not None:
                buffers, lease = pinned_pool.checkout(specs)
                for buffer, raw, dtype in zip(buffers, raw_slices, dtypes):
                    if dtype == torch.bfloat16:
                        buffer.view(torch.uint16).numpy()[:] = np.frombuffer(
                            raw, dtype=np.uint16
                        )
                    else:
                        np_dtype = _DTYPE_TO_NUMPY.get(dtype)
                        if np_dtype is None:
                            raise TypeError(
                                f"unsupported packed dtype for host-prepare IPC: {dtype}"
                            )
                        buffer.numpy()[:] = np.frombuffer(raw, dtype=np_dtype)
            else:
                buffers = [
                    _raw_bytes_to_tensor(raw, dtype=dtype, pin_memory=pin_memory)
                    for raw, dtype in zip(raw_slices, dtypes)
                ]
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
                _keepalive=lease,
            )
        finally:
            mapped.close()
    finally:
        os.close(fd)


def _memfd_handle_channel(ctx: Any) -> tuple[Any, Any]:
    """Create the Unix socketpair required by ``send_handle``/``recv_handle``."""

    # On POSIX, Pipe(duplex=False) is os.pipe(), which cannot carry SCM_RIGHTS.
    # Duplex is deliberate even though ownership makes this channel one-way.
    return ctx.Pipe(duplex=True)


def _publish_memfd_payload(
    queue: Any,
    fd_conn: Any,
    parent_pid: int,
    payload: dict[str, Any],
    memfd: int,
) -> None:
    """Transfer one memfd, then publish its paired metadata."""

    from multiprocessing.reduction import send_handle

    try:
        # The parent must never observe metadata for an fd that failed to send.
        send_handle(fd_conn, memfd, int(parent_pid))
        _queue_put_interruptible(queue, payload)
    finally:
        os.close(memfd)


def _wait_for_host_prepare_terminal_ack(conn: Any | None) -> None:
    """Keep share-memory files alive until the parent consumes the terminal item."""

    if conn is None:
        return
    try:
        while True:
            if conn.poll(0.5):
                try:
                    conn.recv_bytes()
                except EOFError:
                    pass
                return
    except (EOFError, OSError):
        # Parent closed or exited before acknowledging; teardown owns cleanup.
        return


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
    progress_mtime: Any | None = None,
    fd_conn: Any | None = None,
    parent_pid: int | None = None,
    terminal_ack_conn: Any | None = None,
    scan_cursor_storage: Any | None = None,
    scan_resume_plan: ScanResumePlan | None = None,
    scan_cursor_split_key: str | None = None,
) -> None:
    """Child entry: pack+tensorize and push FeatureBatches to the train process.

    - ``memfd``: hide CUDA, coalesce unpinned, spill via anonymous memfd (tiny shm).
      File descriptors travel on ``fd_conn`` via ``send_handle`` (not DupFd).
    - ``share``: keep CUDA visible so we can pin in-child, then ``share_memory_``
      so the parent receives already-pinned handles (large ``/dev/shm``).
    """

    def _beat(note: str = "") -> None:
        del note  # reserved for optional debug logging
        if progress_mtime is None:
            return
        try:
            progress_mtime.value = time()
        except Exception:
            pass

    # Own a process group so the parent can SIGTERM/SIGKILL the whole tree
    # (including adapter ProcessPool workers) on idle/startup stall.
    try:
        os.setsid()
    except OSError:
        try:
            os.setpgrp()
        except OSError:
            pass
    _install_host_prepare_shutdown_handlers()
    set_io_progress_hook(_beat)
    # The scanner runs here, so this is where the resume offset is applied and
    # where the position the parent checkpoints is published from.
    if scan_cursor_storage is not None:
        set_scan_cursor_channel(
            ScanCursorChannel(scan_cursor_storage),
            split_key=scan_cursor_split_key,
        )
    if scan_resume_plan is not None:
        set_scan_resume_plan(scan_resume_plan)
    _beat("child-start")

    try:
        _host_prepare_process_body(
            queue,
            config,
            split_name,
            vocab_maps,
            require_labels=require_labels,
            shard_rank=shard_rank,
            shard_world_size=shard_world_size,
            coalesce_tensors=coalesce_tensors,
            include_group_id=include_group_id,
            ipc_mode=ipc_mode,
            pin_memory=pin_memory,
            progress_beat=_beat,
            fd_conn=fd_conn,
            parent_pid=parent_pid,
            terminal_ack_conn=terminal_ack_conn,
        )
    finally:
        set_io_progress_hook(None)
        set_scan_cursor_channel(None)
        set_scan_resume_plan(None)
        if fd_conn is not None:
            try:
                fd_conn.close()
            except Exception:
                pass
        if terminal_ack_conn is not None:
            try:
                terminal_ack_conn.close()
            except Exception:
                pass


def _host_prepare_process_body(
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
    progress_beat: Callable[[str], None],
    fd_conn: Any | None = None,
    parent_pid: int | None = None,
    terminal_ack_conn: Any | None = None,
) -> None:
    _beat = progress_beat
    os.environ["MDL_HOST_PREPARE_PROCESS"] = "1"
    use_share = ipc_mode == "share"
    # CPU-only child: never touch a torchrun-remapped CUDA device. Pinning is
    # done in the parent after privatizing IPC buffers (avoids pinned pages in
    # /dev/shm that ratchet container RSS across long runs).
    del pin_memory
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if use_share:
        _configure_host_prepare_tensor_sharing()
        try:
            import torch.multiprocessing as torch_mp

            # Always ``file_system`` for spawn+Queue: ``file_descriptor`` needs
            # the child's resource_sharer socket during parent unpickle, which
            # races and raises FileNotFoundError. With large /dev/shm the
            # share dir lives there so this stays a tmpfs mmap path.
            torch_mp.set_sharing_strategy("file_system")
        except (RuntimeError, ValueError, AttributeError):
            pass
    # Inherit LOCAL_RANK from the train parent and take this rank's CPU slice
    # so 6–8 co-located prepare children do not all fight over cores 0..N/3.
    _apply_local_rank_cpu_affinity("host_prepare")
    limit_malloc_arenas()
    try:
        split = config.data.train if split_name == "train" else config.data.test
        if split is None:
            raise ValueError(f"split {split_name!r} is not configured")
        _beat("before-table-iter")
        table_iter = _iter_batch_tables(
            config,
            split_name,
            shard_rank=shard_rank,
            shard_world_size=shard_world_size,
            require_labels=require_labels,
        )
        _beat("after-table-iter-open")
        try:
            produced = 0
            for table in table_iter:
                _beat("table")
                with io_progress_pulses(15.0):
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
                        batch = _coalesce_feature_batch(batch, pin_memory=False)
                if use_share:
                    _queue_put_interruptible(
                        queue, _share_feature_batch_for_ipc(batch)
                    )
                else:
                    if fd_conn is None or parent_pid is None:
                        raise RuntimeError(
                            "memfd host-prepare IPC requires fd_conn + parent_pid"
                        )
                    payload, memfd = _spill_feature_batch_for_ipc(batch)
                    _publish_memfd_payload(
                        queue, fd_conn, parent_pid, payload, memfd
                    )
                _beat("batch-queued")
                # Queued rows are rows the trainer will count, so this is what
                # the checkpoint compares its own row total against.
                note_scan_emitted_rows(_feature_batch_row_count(batch))
                # Free the child's handle promptly so shared IPC files / Arrow
                # arenas are not pinned by the previous loop iteration.
                del batch
                produced += 1
                # Release every batch: industrial Arrow/HDFS pools otherwise
                # climb for the whole epoch and dominate pod Working Set.
                try:
                    import pyarrow as pa

                    pa.default_memory_pool().release_unused()
                except Exception:
                    pass
                if produced % 64 == 0:
                    gc.collect()
                    # release_unused() only reaches Arrow's free lists; without
                    # this the freed heap tops stay charged to the pod.
                    trim_process_heap()
            _queue_put_interruptible(queue, None)
            _wait_for_host_prepare_terminal_ack(terminal_ack_conn)
        except BaseException as error:  # noqa: BLE001 - classify teardown vs real IO
            # Only swallow parent-closed-queue teardown. A bare ``except OSError``
            # previously hid HDFS/shm failures: the child exited with an empty
            # queue and the parent only saw "exited without a terminal queue item".
            if _is_host_prepare_ipc_teardown_error(error):
                return
            raise
        finally:
            close = getattr(table_iter, "close", None)
            if callable(close):
                close()
    except BaseException as error:  # noqa: BLE001 - propagate to parent
        if _is_host_prepare_ipc_teardown_error(error):
            return
        if isinstance(error, RemoteIoStallError):
            try:
                _queue_put_interruptible(queue, error)
                _wait_for_host_prepare_terminal_ack(terminal_ack_conn)
            except Exception:
                pass
            return
        # Pickling a bare exception across spawn drops the child traceback.
        # Embed it in the message so train logs show the real int(None)/etc site.
        import traceback

        try:
            _queue_put_interruptible(
                queue,
                RuntimeError(
                    f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
                ),
            )
            _wait_for_host_prepare_terminal_ack(terminal_ack_conn)
        except Exception:
            # Parent may already have closed the queue during teardown.
            pass


def _install_host_prepare_shutdown_handlers() -> None:
    """Make SIGTERM exit the host-prepare child promptly.

    Default Python signal handling only runs between bytecode instructions. A
    child blocked inside ``Queue.put`` / CUDA pin can outlive the parent's
    SIGTERM grace window and force a noisy SIGKILL. ``os._exit`` skips orderly
    interpreter teardown; the parent already abandoned the IPC queue.
    """

    import signal

    def _exit_immediately(signum: int, _frame: object | None) -> None:
        os._exit(128 + int(signum))

    signal.signal(signal.SIGTERM, _exit_immediately)
    signal.signal(signal.SIGINT, _exit_immediately)


def _is_host_prepare_ipc_teardown_error(error: BaseException) -> bool:
    """True when the parent already tore down the host-prepare IPC queue.

    Mid-run HDFS/shm ``OSError`` must NOT match: those need to be reported on
    the queue so the train rank fails with a real root cause instead of
    ``host prepare process exited without a terminal queue item``.
    """

    if isinstance(error, (BrokenPipeError, EOFError)):
        return True
    if not isinstance(error, OSError):
        return False
    errno_value = getattr(error, "errno", None)
    if errno_value in {errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED}:
        return True
    text = str(error).lower()
    return (
        "broken pipe" in text
        or "connection reset" in text
        or "handle is closed" in text
    )


def _queue_put_interruptible(
    ipc_queue: Any, item: object, *, timeout: float = 0.5
) -> None:
    """``Queue.put`` that returns to Python often enough for SIGTERM handlers."""

    while True:
        try:
            ipc_queue.put(item, timeout=max(0.05, float(timeout)))
            return
        except queue.Full:
            continue


def _wait_process_exit(process: Any, timeout_sec: float) -> bool:
    """Wait until ``process`` exits; return False on timeout (never hang)."""

    deadline = perf_counter() + max(0.0, float(timeout_sec))
    while True:
        try:
            alive = bool(process.is_alive())
        except Exception:
            return True
        if not alive:
            try:
                process.join(timeout=0.05)
            except Exception:
                pass
            return True
        remaining = deadline - perf_counter()
        if remaining <= 0.0:
            return False
        try:
            process.join(timeout=min(0.2, remaining))
        except Exception:
            return False


def _terminate_process_group(
    process: Any,
    *,
    grace_sec: float = 2.0,
    kill_grace_sec: float = 0.5,
    label: str = "host-prepare",
) -> None:
    """SIGTERM the child's process group, then SIGKILL if it refuses to exit.

    Never blocks longer than ``grace_sec + kill_grace_sec``. A child stuck in
    uninterruptible HDFS/JNI (D-state) can ignore even SIGKILL until the kernel
    wakes it; abandon after the kill grace so training teardown cannot hang.
    """

    import signal

    if process is None:
        return
    try:
        alive = bool(process.is_alive())
    except Exception:
        alive = False
    if not alive:
        try:
            process.join(timeout=0.1)
        except Exception:
            pass
        return

    pid = getattr(process, "pid", None)
    killed_group = False
    if pid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(int(pid), signal.SIGTERM)
            killed_group = True
        except (ProcessLookupError, PermissionError, OSError) as error:
            logger.warning(
                "%s killpg(SIGTERM) failed for pid=%s (%s); falling back to terminate()",
                label,
                pid,
                error,
            )
    if not killed_group:
        try:
            process.terminate()
        except Exception as error:
            logger.warning("%s terminate() failed: %s", label, error)

    if _wait_process_exit(process, grace_sec):
        return

    logger.warning(
        "%s still alive after %.1fs SIGTERM; sending SIGKILL",
        label,
        grace_sec,
    )
    print(
        f"{label} still alive after {grace_sec:.1f}s SIGTERM; sending SIGKILL",
        flush=True,
    )
    killed_group = False
    if pid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(int(pid), signal.SIGKILL)
            killed_group = True
        except (ProcessLookupError, PermissionError, OSError):
            killed_group = False
    if not killed_group:
        try:
            process.kill()
        except Exception as error:
            logger.warning("%s kill() failed: %s", label, error)
    if _wait_process_exit(process, kill_grace_sec):
        return
    # D-state / unreapable child: do not block training or destroy_process_group.
    logger.warning(
        "%s pid=%s still alive after SIGKILL (%.1fs); abandoning process",
        label,
        pid,
        kill_grace_sec,
    )
    print(
        f"{label} pid={pid} still alive after SIGKILL; abandoning to avoid hang",
        flush=True,
    )


def _close_process_queue(ipc_queue: Any) -> None:
    """Discard an IPC queue without waiting for its feeder thread.

    This is teardown after the producer process has already been terminated.
    ``Queue.join_thread()`` has no timeout and can wait forever when that
    producer died in JNI or while flushing a partially written payload.  The
    parent also retains both pipe endpoints even though it only consumes from
    the queue, so close them explicitly to wake a device-prefetch thread that
    may still be blocked in ``Queue.get()``.
    """

    cancel_join = getattr(ipc_queue, "cancel_join_thread", None)
    if callable(cancel_join):
        try:
            cancel_join()
        except Exception:
            pass
    try:
        ipc_queue.close()
    except Exception:
        pass
    for endpoint_name in ("_reader", "_writer"):
        endpoint = getattr(ipc_queue, endpoint_name, None)
        close_endpoint = getattr(endpoint, "close", None)
        if callable(close_endpoint):
            try:
                close_endpoint()
            except Exception:
                pass


class _ProcessHostPrepareIterator:
    """Yield FeatureBatches prepared in a spawn child.

    IPC auto-selects by ``/dev/shm`` size (see ``_host_prepare_ipc_mode``):

    - large shm → ``share_memory_`` + optional in-child pin (zero-copy to train)
    - tiny shm → anonymous memfd; parent materializes pinned packed buffers

    Override with ``MDL_HOST_PREPARE_IPC=share|memfd|auto``.

    Parent enforces startup/idle timeouts against the child. Timers measure
    silence since the last child heartbeat (HDFS list/footer/adapt) or
    delivered batch. A hung JNI read that stops Python heartbeats will not
    wait for the global step watchdog: the parent kills the process group and
    aborts the rank with ``REMOTE_IO_STALL``.
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
        scan_resume_plan: ScanResumePlan | None = None,
        scan_cursor: ScanCursorChannel | None = None,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("host_prepare_prefetch queue_size must be positive")
        del coalesce_pinned_tensors
        self._pin_memory = bool(pin_memory)
        self._ipc_mode = _host_prepare_ipc_mode()
        self._closed = False
        # Recycle pinned pages across the prefetch depth. Cap free slots so a
        # deep host_prepare queue cannot multiply peak-sized idle slabs; the
        # pool itself also shrinks against a sliding high-water mark.
        self._pinned_pool = (
            _PinnedHostBufferPool(
                max_free_slots=min(4, max(2, int(queue_size))),
            )
            if self._pin_memory
            else None
        )
        split = config.data.train if split_name == "train" else config.data.test
        reader = split.reader if split is not None else None
        self._startup_timeout_sec = (
            None
            if reader is None
            else getattr(reader, "host_prepare_startup_timeout_sec", 300.0)
        )
        self._idle_timeout_sec = (
            None
            if reader is None
            else getattr(reader, "host_prepare_idle_timeout_sec", 300.0)
        )
        self._started_at = perf_counter()
        self._last_progress_at = self._started_at
        self._received_item = False
        # Wall-clock heartbeat shared with the child. Updated during HDFS
        # discover / footer / adapt work so slow-but-alive startup is not
        # mistaken for a JNI hang (which would stop heartbeats too once the
        # child can no longer run Python).
        self._ctx = mp.get_context("spawn")
        self._progress_mtime = self._ctx.Value("d", time())
        # Reader lives in the child, so the checkpointer reads its scan position
        # out of shared memory rather than through the batch queue.
        self._scan_cursor = scan_cursor
        self._scan_resume_plan = scan_resume_plan
        # Only share-mode needs torch file_system IPC under /dev/shm.
        if self._ipc_mode == "share":
            _configure_host_prepare_tensor_sharing()
            try:
                import torch.multiprocessing as torch_mp

                # Match the child: file_system under /dev/shm (or TMPDIR).
                torch_mp.set_sharing_strategy("file_system")
            except (RuntimeError, ValueError, AttributeError):
                pass
        # Platform trainjob logs often capture stdout only (Train step | …),
        # not the Python logging handlers — print so IPC mode is searchable.
        ipc_message = (
            f"host-prepare IPC mode={self._ipc_mode} "
            f"shm_free_mib={(_dev_shm_free_bytes() or 0) / (1024 * 1024):.1f} "
            f"pin_memory={self._pin_memory} "
            f"pinned_pool={'on' if self._pinned_pool is not None else 'off'} "
            f"startup_timeout={self._startup_timeout_sec} "
            f"idle_timeout={self._idle_timeout_sec}"
        )
        logger.info("%s", ipc_message)
        if is_main_process():
            print(ipc_message, flush=True)
        self._queue: Any = self._ctx.Queue(maxsize=int(queue_size))
        # memfd fds travel on this Pipe via send_handle/recv_handle so we never
        # depend on the child's multiprocessing.resource_sharer socket.
        self._fd_recv: Any | None = None
        fd_send: Any | None = None
        parent_pid: int | None = None
        if self._ipc_mode == "memfd":
            self._fd_recv, fd_send = _memfd_handle_channel(self._ctx)
            parent_pid = int(os.getpid())
        # file_system share handles are owned by the producer's torch-shm
        # manager. Keep that producer alive until the parent has consumed the
        # FIFO terminal item, which proves all earlier batches were unpickled
        # and privatized.
        terminal_ack_recv: Any | None = None
        self._terminal_ack_send: Any | None = None
        if self._ipc_mode == "share":
            terminal_ack_recv, self._terminal_ack_send = self._ctx.Pipe(
                duplex=False
            )
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
                "progress_mtime": self._progress_mtime,
                "fd_conn": fd_send,
                "parent_pid": parent_pid,
                "terminal_ack_conn": terminal_ack_recv,
                "scan_cursor_storage": (
                    None if self._scan_cursor is None else self._scan_cursor.storage
                ),
                "scan_resume_plan": scan_resume_plan,
                "scan_cursor_split_key": (
                    None
                    if split is None
                    else scan_split_key(
                        split,
                        shard_rank=shard_rank,
                        shard_world_size=shard_world_size,
                    )
                ),
            },
            name=f"mdl-host-prepare-{split_name}",
            daemon=False,
        )
        self._process.start()
        # Parent only needs the recv end; close our copy of the send end so the
        # pipe drains / EOFs when the child exits.
        if fd_send is not None:
            try:
                fd_send.close()
            except Exception:
                pass
        if terminal_ack_recv is not None:
            try:
                terminal_ack_recv.close()
            except Exception:
                pass
        # Train parent keeps the complementary slice of this LOCAL_RANK's CPUs.
        _apply_local_rank_cpu_affinity("train")

    def __iter__(self) -> "_ProcessHostPrepareIterator":
        return self

    def scan_position(self) -> Any | None:
        """Return the child reader's latest published scan position."""

        return None if self._scan_cursor is None else self._scan_cursor.read()

    def _mark_progress(self) -> None:
        self._received_item = True
        self._last_progress_at = perf_counter()

    def memory_report(self) -> str:
        """Reader-side memory counters that only this iterator can see.

        The child does all the HDFS/Arrow work, so a climb in ``child_rss_mib``
        against a flat parent points somewhere completely different from the
        reverse — worth separating rather than reading one total.
        """

        fields: list[str] = []
        child_pid = getattr(self._process, "pid", None)
        child_rss = None if child_pid is None else process_resident_bytes(child_pid)
        if child_rss is not None:
            fields.append(f"child_rss_mib={child_rss / _MIB:.1f}")
        if self._pinned_pool is not None:
            fields.append(
                f"pinned_idle_mib={self._pinned_pool.idle_bytes() / _MIB:.1f}"
            )
        return " ".join(fields)

    def _delivery_silence_sec(self, now: float) -> float:
        """Seconds since the last *delivered* batch, ignoring heartbeats."""

        return now - self._last_progress_at

    def _startup_silence_sec(self, now: float) -> float:
        """Seconds since the freshest child heartbeat or delivered batch.

        Only meaningful before the first batch: HDFS list/footer/adapt work is
        legitimately long and heartbeat-only, and a JNI hang stops the beats.
        """

        silence = self._delivery_silence_sec(now)
        progress_mtime = getattr(self, "_progress_mtime", None)
        if progress_mtime is None:
            return silence
        try:
            beat_age = time() - float(progress_mtime.value)
        except Exception:
            return silence
        if beat_age < 0:
            beat_age = 0.0
        return min(silence, beat_age)

    def _raise_if_child_stalled(self) -> None:
        now = perf_counter()
        if (
            not self._received_item
            and self._startup_timeout_sec is not None
            and self._startup_silence_sec(now) >= float(self._startup_timeout_sec)
        ):
            self._abort_stalled_child(
                RemoteIoStallError(
                    f"host-prepare startup exceeded "
                    f"{float(self._startup_timeout_sec):.0f}s without progress"
                )
            )
        # Deliberately heartbeat-blind. Once batches have flowed, "child is alive
        # but delivering nothing" is precisely the state to abort on; honouring
        # heartbeats here made this timer unreachable, because the child beats
        # per record batch and every 15s from io_progress_pulses.
        if (
            self._received_item
            and self._idle_timeout_sec is not None
            and self._delivery_silence_sec(now) >= float(self._idle_timeout_sec)
        ):
            self._abort_stalled_child(
                RemoteIoStallError(
                    f"host-prepare delivered no batch for "
                    f"{float(self._idle_timeout_sec):.0f}s while still alive"
                )
            )

    def _abort_stalled_child(self, error: RemoteIoStallError) -> None:
        logger.error("%s", error)
        # Logging handlers may be buffered by the platform. Emit the root cause
        # before teardown so a subsequent native/IPC cleanup failure cannot
        # leave SIGKILL as the only visible line.
        print(str(error), flush=True)
        # Watchdog abort must not spend seconds on graceful SIGTERM: close the
        # IPC, SIGKILL immediately, then hard-exit the rank.
        self._closed = True
        try:
            _close_process_queue(self._queue)
        except Exception:
            pass
        import signal

        process = self._process
        pid = getattr(process, "pid", None)
        try:
            if pid is not None and hasattr(os, "killpg"):
                os.killpg(int(pid), signal.SIGKILL)
            elif process is not None:
                process.kill()
        except Exception:
            pass
        abort_rank_for_remote_io_stall(error)

    def __next__(self) -> FeatureBatch:
        return self.next_within(None)

    def next_within(self, budget_sec: float | None) -> Any:
        """Next batch, or ``BATCH_NOT_READY`` once ``budget_sec`` elapses.

        Reporting starvation lets the caller sit out one step rather than block
        the whole world; the batch stays queued and arrives on a later step.
        """

        if self._closed:
            raise StopIteration
        deadline = _budget_deadline(budget_sec)
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._closed:
                    raise StopIteration
                try:
                    alive = bool(self._process.is_alive())
                except Exception:
                    alive = False
                if not alive and self._queue.empty():
                    exitcode = getattr(self._process, "exitcode", None)
                    self.close()
                    # Treat as retryable remote/IO stall: platform launchers map
                    # exit 70 to restart. Include exitcode so SIGKILL (-9) /
                    # native crashes are distinguishable from swallowed errors.
                    raise RemoteIoStallError(
                        "host-prepare process exited without a terminal queue "
                        f"item (exitcode={exitcode!r}); child likely crashed in "
                        "native/HDFS code or failed before reporting an error"
                    )
                self._raise_if_child_stalled()
                if deadline is not None and perf_counter() >= deadline:
                    return BATCH_NOT_READY
                continue
            except (OSError, ValueError, EOFError, RuntimeError) as error:
                self.close()
                raise RuntimeError(
                    "host prepare IPC queue closed while waiting for a batch"
                ) from error
            break
        self._mark_progress()
        if item is self._SENTINEL:
            self._ack_share_terminal()
            self.close()
            raise StopIteration
        if isinstance(item, RemoteIoStallError):
            self._ack_share_terminal()
            self.close()
            abort_rank_for_remote_io_stall(item)
        if isinstance(item, BaseException):
            self._ack_share_terminal()
            self.close()
            if is_remote_io_stall_error(item):
                abort_rank_for_remote_io_stall(item)
            raise RuntimeError("host prepare process failed") from item
        if isinstance(item, FeatureBatch):
            # share_memory_ IPC: always clone off shared storages in the parent.
            # Returning already-pinned shared buffers used to keep /dev/shm files
            # alive across steps and ratchet parent RSS.
            try:
                if self._pin_memory:
                    return _pin_feature_batch_with_pool(
                        item, pool=self._pinned_pool
                    )
                return privatize_shared_feature_batch(item)
            finally:
                del item
        if not isinstance(item, dict):
            self.close()
            raise TypeError(
                f"host prepare process returned {type(item).__name__}, "
                "expected FeatureBatch or memfd payload dict"
            )
        try:
            memfd = self._recv_memfd_handle()
            return _load_feature_batch_from_ipc(
                item,
                pin_memory=self._pin_memory,
                pinned_pool=self._pinned_pool,
                fd=memfd,
            )
        except BaseException:
            self.close()
            raise
        finally:
            del item

    def _recv_memfd_handle(self) -> int:
        """Receive the next memfd from the child Pipe (paired with Queue meta)."""

        from multiprocessing.reduction import recv_handle

        conn = self._fd_recv
        if conn is None:
            raise RuntimeError("memfd host-prepare IPC pipe was not created")
        while True:
            try:
                if conn.poll(0.5):
                    return int(recv_handle(conn))
            except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as error:
                raise RemoteIoStallError(
                    "host-prepare memfd handle transfer failed "
                    f"({type(error).__name__}: {error}); child likely exited "
                    "before send_handle completed"
                ) from error
            if self._closed:
                raise RemoteIoStallError(
                    "host-prepare closed while waiting for memfd handle"
                )
            try:
                alive = bool(self._process.is_alive())
            except Exception:
                alive = False
            if not alive:
                raise RemoteIoStallError(
                    "host-prepare process exited while parent waited for memfd "
                    f"handle (exitcode={getattr(self._process, 'exitcode', None)!r})"
                )
            self._raise_if_child_stalled()

    def _ack_share_terminal(self) -> None:
        conn = getattr(self, "_terminal_ack_send", None)
        if conn is None:
            return
        try:
            conn.send_bytes(b"done")
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
            self._terminal_ack_send = None
        # This is the clean terminal path. Give the child a short opportunity
        # to exit normally before close() applies the bounded kill policy.
        try:
            self._process.join(timeout=1.0)
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        terminal_ack_send = getattr(self, "_terminal_ack_send", None)
        if terminal_ack_send is not None:
            try:
                terminal_ack_send.close()
            except Exception:
                pass
            self._terminal_ack_send = None
        # Unblock a child stuck in Queue.put (full prefetch) before SIGTERM so
        # the grace window is spent on real teardown, not a deadlocked feeder.
        _close_process_queue(self._queue)
        fd_recv = getattr(self, "_fd_recv", None)
        if fd_recv is not None:
            try:
                fd_recv.close()
            except Exception:
                pass
            self._fd_recv = None
        _terminate_process_group(
            self._process,
            grace_sec=2.0,
            kill_grace_sec=0.5,
            label="host-prepare",
        )

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


@dataclass
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


# Host teardown actively closes the process queue before joining this thread,
# so a long HDFS/open timeout no longer has to be paid here. Keep a bounded
# fallback for CUDA calls that cannot be interrupted from Python.
_DEVICE_PREFETCH_JOIN_TIMEOUT_SEC = 30.0

_MIB = 1024 * 1024


def _host_memory_report(iterator: Any) -> str:
    """Host memory counters for the periodic log; empty when unreadable.

    Host RSS is the number the container OOM killer acts on, and until now
    nothing on the training path recorded it — a multi-hour climb could only be
    seen from outside, with no way to tell which layer was holding the memory.
    """

    fields: list[str] = []
    resident = process_resident_bytes()
    if resident is not None:
        fields.append(f"rank_rss_mib={resident / _MIB:.1f}")
    peak = process_peak_resident_bytes()
    if peak is not None:
        fields.append(f"rank_peak_rss_mib={peak / _MIB:.1f}")
    arrow_bytes = arrow_pool_bytes()
    if arrow_bytes is not None:
        fields.append(f"rank_arrow_mib={arrow_bytes / _MIB:.1f}")
    reporter = getattr(iterator, "memory_report", None)
    if reporter is not None:
        try:
            reader_fields = str(reporter())
        except Exception:  # noqa: BLE001 - diagnostics must not break training
            reader_fields = ""
        if reader_fields:
            fields.append(reader_fields)
    return " ".join(fields)


class _BatchNotReady:
    """Sentinel: the reader has no batch *yet*, but is not exhausted.

    Distinct from ``StopIteration``. A starved rank sits out one step and picks
    the batch up later; an exhausted rank is done for good. Collapsing the two
    would make a transient world-wide stall look like the end of the epoch.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "BATCH_NOT_READY"


BATCH_NOT_READY = _BatchNotReady()


def _budget_deadline(budget_sec: float | None) -> float | None:
    return None if budget_sec is None else perf_counter() + float(budget_sec)


def _next_batch_within(iterator: Any, budget_sec: float | None) -> Any:
    """Read the next batch, reporting starvation instead of blocking forever.

    Iterators with no budgeted read (in-process paths with no remote reader
    behind them) keep the plain blocking contract.
    """

    if budget_sec is None:
        return next(iterator)
    reader = getattr(iterator, "next_within", None)
    if reader is None:
        return next(iterator)
    return reader(budget_sec)


class _DevicePrefetchIterator:
    """Prepare and copy future batches on a dedicated CUDA stream/thread."""

    def __init__(
        self,
        iterator: Iterator[FeatureBatch],
        device: torch.device,
        depth: int,
        *,
        join_timeout_sec: float = _DEVICE_PREFETCH_JOIN_TIMEOUT_SEC,
    ) -> None:
        if device.type != "cuda" or depth <= 0:
            raise ValueError("device prefetch requires CUDA and positive depth")
        self.iterator = iterator
        self.device = (
            device
            if device.index is not None
            else torch.device("cuda", torch.cuda.current_device())
        )
        self._join_timeout_sec = float(join_timeout_sec)
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

    def _next_item(self, deadline: float | None = None) -> _DevicePrefetchItem | None:
        """Pop the next ready item, or ``None`` once ``deadline`` passes."""

        while True:
            try:
                item = self.queue.get(timeout=0.5)
            except queue.Empty:
                if self.stop_event.is_set():
                    raise StopIteration
                if not self.thread.is_alive() and self.queue.empty():
                    raise RuntimeError(
                        "CUDA prefetch worker exited without a terminal queue item"
                    )
                if deadline is not None and perf_counter() >= deadline:
                    return None
                continue
            break
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
        return self.next_within(None)

    def next_within(self, budget_sec: float | None) -> Any:
        """Next device batch, or ``BATCH_NOT_READY`` once the budget expires."""

        item = self._next_item(_budget_deadline(budget_sec))
        if item is None:
            return BATCH_NOT_READY
        assert item.batch is not None
        device_batch = item.batch
        # H2D has completed (wait_event in _next_item). Drop the host FeatureBatch
        # immediately so recycled pinned leases return to the pool instead of
        # lingering until the next GC cycle.
        item.host_batch = None
        item.batch = None
        return device_batch

    def next_with_host(self) -> tuple[FeatureBatch, FeatureBatch]:
        """Return matching host/device views for pre-update evaluation replay."""

        return self.next_with_host_within(None)

    def next_with_host_within(self, budget_sec: float | None) -> Any:
        """Host/device view pair, or ``BATCH_NOT_READY`` once the budget expires."""

        item = self._next_item(_budget_deadline(budget_sec))
        if item is None:
            return BATCH_NOT_READY
        if item.host_batch is None or item.batch is None:
            self.close()
            raise RuntimeError("CUDA-prefetch item did not retain its host batch")
        host_batch = item.host_batch
        device_batch = item.batch
        item.host_batch = None
        item.batch = None
        return host_batch, device_batch

    def close(self) -> None:
        self.stop_event.set()
        # Unblock a worker currently inside next(self.iterator) before waiting
        # for it. The old order waited up to 180 seconds first, while the host
        # iterator could only be closed from that same worker's finally block.
        close_iterator = getattr(self.iterator, "close", None)
        if callable(close_iterator):
            try:
                close_iterator()
            except Exception as error:
                logger.warning("device-prefetch host iterator close failed: %s", error)
        if self.thread is threading.current_thread():
            return
        # Prefer a clean join so CUDA/Arrow teardown does not race the worker.
        # Cap the wait: an abandoned HDFS JNI thread must not hang process exit
        # after the step watchdog has already been (or is about to be) stopped.
        timeout = max(0.01, self._join_timeout_sec)
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            logger.warning(
                "abandoning CUDA prefetch worker after %.1fs join timeout "
                "(daemon thread may still be blocked in host iterator / HDFS)",
                timeout,
            )


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
    ddp_config = getattr(config.training, "ddp", DDPConfig())
    if (
        bool(getattr(config.runtime, "cuda_graph_backbone", False))
        and context.device.type == "cuda"
        and hasattr(base_model, "prewarm_cuda_graph_backbone")
    ):
        base_model.prewarm_cuda_graph_backbone(context.device)
        # Freeze further captures before DDP reducer hooks are installed.
        # Non-prewarmed shapes may eager-fallback; wrappers register the live
        # dense modules so graph/eager share one DDP parameter surface.
        if hasattr(base_model, "_cuda_graph_backbone_capture_allowed"):
            base_model._cuda_graph_backbone_capture_allowed = False
        if is_main_process():
            pool = getattr(base_model, "_cuda_graph_backbone_pool", {}) or {}
            captured_trainable_params = int(
                getattr(base_model, "_cuda_graph_backbone_parameter_count", 0)
            )
            print(
                "CUDA graph backbone | "
                f"prewarmed_shapes={len(pool)} "
                f"captured_trainable_params={captured_trainable_params} "
                f"static_graph={bool(ddp_config.static_graph)}",
                flush=True,
            )

    forward_model: nn.Module = base_model
    if context.enabled:
        _exclude_sparse_parameters_from_ddp(base_model, ddp_ignored)
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
    """Blocking active-rank count for paths that must consume every row.

    Evaluation cannot sit out a starved rank without silently dropping test
    rows, so it keeps the plain blocking contract and treats "no batch" as
    exhausted.
    """

    supply = _start_rank_supply_count(
        context,
        rank_active=rank_active,
        rank_exhausted=not rank_active,
    ).wait()
    return supply.active


@dataclass(frozen=True)
class _RankSupply:
    """How many peers had a batch, and how many will never have one again."""

    active: int
    exhausted: int


class _RankSupplyHandle:
    """Async supply-state reduction; wait before loss scaling / exit checks."""

    __slots__ = ("_value", "_work")

    def __init__(self, value: Tensor, work: Any | None) -> None:
        self._value = value
        self._work = work

    def wait(self) -> _RankSupply:
        if self._work is not None:
            self._work.wait()
        counts = self._value.tolist()
        return _RankSupply(active=int(counts[0]), exhausted=int(counts[1]))


def _supply_verdict(supply: _RankSupply, world_size: int) -> str:
    """Classify a step for a rank that has no batch of its own.

    ``"stop"`` only once every rank is exhausted. ``"retry"`` when nobody has
    data but somebody may still get some: a world-wide reader stall is not the
    end of the epoch, and stopping there would silently truncate training.
    ``"replay"`` when peers are training, so this rank owes them a zero-loss
    step to keep the DDP collectives aligned.
    """

    if supply.exhausted >= world_size:
        return "stop"
    if supply.active == 0:
        return "retry"
    return "replay"


def _start_rank_supply_count(
    context: DistributedContext,
    *,
    rank_active: bool,
    rank_exhausted: bool,
) -> _RankSupplyHandle:
    """Kick off the supply allreduce so H2D can overlap the host collective.

    Two counters rather than one flag: a rank without a batch is either
    *starved* (slow reader, retry later) or *exhausted* (end of its shard).
    Collapsing them would make a world-wide transient stall indistinguishable
    from the end of the epoch, and training would stop early and silently.
    """

    counts = [int(rank_active), int(rank_exhausted)]
    if not context.enabled:
        return _RankSupplyHandle(torch.tensor(counts, dtype=torch.long), None)
    # Prefer CPU + Gloo control group so this never inserts a CUDA device sync
    # into the training critical path.
    if context.control_group is not None:
        value = torch.tensor(counts, dtype=torch.long, device="cpu")
        work = torch_dist.all_reduce(
            value,
            op=torch_dist.ReduceOp.SUM,
            group=context.control_group,
            async_op=True,
        )
        return _RankSupplyHandle(value, work)
    value = torch.tensor(counts, dtype=torch.long, device=context.device)
    work = torch_dist.all_reduce(
        value,
        op=torch_dist.ReduceOp.SUM,
        async_op=True,
    )
    return _RankSupplyHandle(value, work)


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
    task_loss_weights: Tensor | tuple[float, ...] | None = None,
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

    if task_loss_weights is None:
        task_weight_tensor = torch.ones_like(task_numerators)
    elif isinstance(task_loss_weights, Tensor):
        task_weight_tensor = task_loss_weights.to(
            device=task_numerators.device,
            dtype=task_numerators.dtype,
        )
    else:
        raw_task_weights = tuple(float(weight) for weight in task_loss_weights)
        if any(
            not math.isfinite(weight) or weight < 0.0
            for weight in raw_task_weights
        ):
            raise ValueError(
                "task_loss_weights must contain finite non-negative values"
            )
        task_weight_tensor = task_numerators.new_tensor(raw_task_weights)
    if (
        task_weight_tensor.ndim != 1
        or task_weight_tensor.numel() != element_loss.size(1)
    ):
        raise ValueError(
            "task_loss_weights must contain exactly one weight per logits task"
        )

    distributed = torch_dist.is_available() and torch_dist.is_initialized()
    if loss_reduction == "sum":
        # DDP averages gradients across ranks. Multiplying each local sum by the
        # world size makes the averaged gradient equal the global paper sum.
        world_size = float(torch_dist.get_world_size()) if distributed else 1.0
        prediction_loss = (task_numerators * task_weight_tensor).sum() * world_size
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
        prediction_loss = (
            task_numerators * task_scale * task_weight_tensor
        ).sum()
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


def _configured_task_loss_weights(config: AppConfig) -> tuple[float, ...]:
    ordered = getattr(config, "ordered_task_loss_weights", None)
    if ordered is not None:
        return tuple(float(weight) for weight in ordered)
    configured = getattr(config.training, "task_loss_weights", {})
    return tuple(
        float(configured.get(task_name, 1.0))
        for task_name in config.task_names
    )


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


def _checkpoint_plan_message(config: AppConfig) -> str:
    """One-line summary of whether periodic HDFS checkpoints will run."""

    settings = getattr(config.training, "checkpoint", None)
    if settings is None or not settings.enabled:
        return (
            "Checkpointing | disabled "
            "(training.checkpoint.dir is unset; periodic HDFS saves and "
            "resume are off)"
        )
    run_name = settings.run_name or config.model.name
    return (
        "Checkpointing | enabled "
        f"dir={settings.dir} run_name={run_name} "
        f"every_steps={settings.every_steps} keep_last={settings.keep_last} "
        f"save_on_exit={settings.save_on_exit} resume={settings.resume} "
        f"async_upload={settings.async_upload} data_resume={settings.data_resume}"
    )


class _CheckpointCoordinator:
    """Periodic resumable checkpoints for one rank of a training run.

    Saving stages this rank's files on local disk and hands them to a background
    uploader, so a multi-GiB step costs the training loop a local write rather
    than an HDFS round trip. Resuming restores weights, optimizer accumulators,
    the global step, and the rank's input-scan position.
    """

    def __init__(
        self,
        settings: Any,
        store: Any,
        uploader: CheckpointUploader,
        context: DistributedContext,
        *,
        scan_cursor: ScanCursorChannel | None,
        staging_root: Path,
        run_name: str,
        log_steps: bool,
    ) -> None:
        self._settings = settings
        self._store = store
        self._uploader = uploader
        self._context = context
        self._staging_root = staging_root
        self._run_name = run_name
        self._log = log_steps and context.rank == 0
        self._chunk_bytes = int(
            getattr(settings, "shard_chunk_bytes", DEFAULT_SHARD_CHUNK_BYTES)
        )
        self._estimate: StagingSpaceEstimate | None = None
        self.scan_cursor = scan_cursor
        self.scan_resume_plan: ScanResumePlan | None = None
        self._last_saved_step = -1
        # The reader's emitted-row counter starts at zero in every process, so
        # the reader's lead is only comparable against rows trained since this
        # process resumed.
        self._rows_at_start = 0

    @classmethod
    def create(
        cls,
        config: AppConfig,
        context: DistributedContext,
        log_steps: bool,
    ) -> "_CheckpointCoordinator | None":
        settings = getattr(config.training, "checkpoint", None)
        # Rank-0 always announces the plan *before* touching HDFS. A missing
        # banner used to mean either "dir unset" (silent return) or "open_run_store
        # hung/threw before the success print" — both looked like "checkpoint
        # code never ran" in trainjob logs that only keep the training tail.
        if context.rank == 0:
            print(_checkpoint_plan_message(config), flush=True)
        if settings is None or not settings.enabled:
            return None
        run_name = settings.run_name or config.model.name
        try:
            store = open_run_store(str(settings.dir), run_name)
        except Exception as error:
            if context.rank == 0:
                print(
                    "Checkpointing | FAILED to open run directory "
                    f"dir={settings.dir!r} run_name={run_name!r}: {error}",
                    flush=True,
                )
            raise
        # An explicit staging_dir also pins the path: ``gettempdir()`` follows
        # TMPDIR, which host-prepare repoints at /dev/shm once its shared-memory
        # IPC comes up.
        staging_root = Path(
            settings.staging_dir or tempfile.gettempdir()
        ) / "mdl-checkpoint-staging"
        uploader = CheckpointUploader(
            store,
            rank=context.rank,
            world_size=context.world_size,
            keep_last=settings.keep_last,
            ready_timeout_sec=settings.ready_timeout_sec,
            asynchronous=settings.async_upload,
        )
        if context.rank == 0:
            print(
                "Checkpointing | ready "
                f"run_dir={store.root_uri} "
                f"staging_dir={staging_root} "
                f"staging_default={settings.staging_dir is None} "
                f"every_steps={settings.every_steps} "
                f"keep_last={settings.keep_last} "
                f"async_upload={settings.async_upload} "
                f"data_resume={settings.data_resume} "
                f"resume={settings.resume}",
                flush=True,
            )
        return cls(
            settings,
            store,
            uploader,
            context,
            scan_cursor=(
                ScanCursorChannel.shared(mp.get_context("spawn"))
                if settings.data_resume
                else None
            ),
            staging_root=staging_root,
            run_name=run_name,
            log_steps=log_steps,
        )

    # --- Resume ---

    def _agreed_resume_directory(self) -> str | None:
        """Pick one committed step for the whole job, decided by rank 0.

        Ranks start seconds apart, so each resolving ``auto`` independently could
        split the job across two steps if a commit lands during startup.
        """

        directory: str | None = None
        if self._context.rank == 0:
            checkpoint = resolve_resume_checkpoint(
                self._store,
                self._settings.resume,
            )
            directory = None if checkpoint is None else checkpoint.directory
        if self._context.enabled:
            payload = [directory]
            torch_dist.broadcast_object_list(payload, src=0)
            directory = payload[0]
        return directory

    def restore(
        self,
        config: AppConfig,
        model: nn.Module,
        context: DistributedContext,
        device: torch.device,
        *,
        dense_optimizer: torch.optim.Optimizer | None,
        replicated_sparse_optimizer: torch.optim.Optimizer | None,
        sharded_optimizer: ShardedAdagrad | ShardedRowWiseAdagrad | None,
    ) -> Any | None:
        directory = self._agreed_resume_directory()
        if directory is None:
            if self._log:
                print(
                    f"Checkpoint resume | none found under {self._store.root_uri}; "
                    "starting from a fresh model",
                    flush=True,
                )
            return None

        from .checkpoint import CommittedCheckpoint

        checkpoint = CommittedCheckpoint(
            step=int(directory.split("-")[-1]),
            directory=directory,
            uri=self._store.uri(directory),
        )
        local_dir = Path(
            tempfile.mkdtemp(
                prefix=f"mdl-resume-rank{context.rank}-",
                dir=str(self._staging_root_ready()),
            )
        )
        try:
            fetch_checkpoint_for_rank(
                self._store,
                checkpoint,
                local_dir,
                rank=context.rank,
                world_size=context.world_size,
            )
            resumed = load_training_checkpoint(
                config,
                model,
                local_dir,
                device=device,
                rank=context.rank,
                world_size=context.world_size,
                dense_optimizer=dense_optimizer,
                replicated_sparse_optimizer=replicated_sparse_optimizer,
                sharded_optimizer=sharded_optimizer,
                source_uri=checkpoint.uri,
            )
        finally:
            shutil.rmtree(local_dir, ignore_errors=True)

        self._last_saved_step = resumed.step
        self._rows_at_start = int(resumed.rows)
        self.scan_resume_plan = self._resume_plan(resumed.data_cursor)
        if self._log:
            plan = self.scan_resume_plan
            position = (
                "restart"
                if plan is None
                else f"{plan.work_unit}[{max(0, plan.position - plan.rewind)}] "
                f"(reader_stopped_at={plan.position} rewind={plan.rewind})"
            )
            print(
                f"Checkpoint resume | step={resumed.step} rows={resumed.rows} "
                f"data_position={position} uri={checkpoint.uri}",
                flush=True,
            )
        return resumed

    def _resume_plan(self, cursor: DataCursor | None) -> ScanResumePlan | None:
        if cursor is None or not self._settings.data_resume or cursor.position <= 0:
            return None
        return ScanResumePlan(
            work_unit=cursor.work_unit,
            position=int(cursor.position),
            prefix_digest=cursor.prefix_digest,
            split_key=cursor.split_key,
            # Computed when the checkpoint was written, from the row lead the
            # reader had over the trainer at that moment.
            rewind=max(int(cursor.rewind), int(self._settings.data_resume_rewind)),
        )

    # --- Saving ---

    def due(self, step: int) -> bool:
        every = int(self._settings.every_steps)
        return (
            every > 0
            and step > 0
            and step % every == 0
            and step != self._last_saved_step
        )

    def due_on_exit(self, step: int) -> bool:
        return (
            bool(self._settings.save_on_exit)
            and step > 0
            and step != self._last_saved_step
        )

    def _staging_root_ready(self) -> Path:
        self._staging_root.mkdir(parents=True, exist_ok=True)
        return self._staging_root

    def preflight(
        self,
        model: nn.Module,
        context: DistributedContext,
        *,
        dense_optimizer: torch.optim.Optimizer | None = None,
        replicated_sparse_optimizer: torch.optim.Optimizer | None = None,
        sharded_optimizer: ShardedAdagrad | ShardedRowWiseAdagrad | None = None,
    ) -> None:
        """Fail at startup when staging cannot hold this node's checkpoints.

        Without this the first ``every_steps`` save is where a too-small staging
        filesystem shows up, thousands of steps into a run, as a zip writer error
        that names neither the directory nor the space it wanted.
        """

        if not self._store.is_remote:
            return
        estimate = estimate_staging_space(
            model,
            rank=context.rank,
            world_size=context.world_size,
            dense_optimizer=dense_optimizer,
            replicated_sparse_optimizer=replicated_sparse_optimizer,
            sharded_optimizer=sharded_optimizer,
            chunk_bytes=self._chunk_bytes,
            upload_window=self._uploader.stream_window,
        )
        self._estimate = estimate
        summary = check_staging_space(
            self._staging_root_ready(),
            estimate,
            local_ranks=_local_world_size(),
            enforce=bool(getattr(self._settings, "preflight_staging", True)),
        )
        if self._log:
            print(f"Checkpointing | staging {summary}", flush=True)

    def _data_cursor(
        self,
        config: AppConfig,
        context: DistributedContext,
        rows: int,
    ):
        if self.scan_cursor is None:
            return None
        position = self.scan_cursor.read()
        if position is None:
            return None
        # The reader is ahead of the trainer by whatever the queues hold. Record
        # how far back a restart must start so those read-but-untrained rows are
        # replayed instead of skipped.
        rows_this_run = max(0, int(rows) - self._rows_at_start)
        rewind = scan_resume_rewind(
            position,
            rows_trained=rows_this_run,
            extra_items=int(self._settings.data_resume_rewind),
        )
        return DataCursor(
            work_unit=position.work_unit,
            position=position.position,
            prefix_digest=position.prefix_digest,
            split_key=scan_split_key(
                config.data.train,
                shard_rank=context.rank,
                shard_world_size=context.world_size,
            ),
            rank=context.rank,
            world_size=context.world_size,
            rewind=rewind,
            emitted_rows=int(position.emitted_rows),
            rows_trained=rows_this_run,
        )

    def save(
        self,
        config: AppConfig,
        model: nn.Module,
        context: DistributedContext,
        *,
        step: int,
        rows: int,
        elapsed_seconds: float,
        dense_optimizer: torch.optim.Optimizer | None,
        replicated_sparse_optimizer: torch.optim.Optimizer | None,
        sharded_optimizer: ShardedAdagrad | ShardedRowWiseAdagrad | None,
        watchdog: _StepWatchdog | None = None,
    ) -> None:
        _step_watchdog_beat(
            watchdog,
            "checkpoint_stage",
            detail=f"steps={step}",
            device=context.device,
        )
        started = perf_counter()
        directory = step_directory_name(step)
        if self._store.is_remote:
            staging_dir = self._staging_root_ready() / self._run_name / directory
            cleanup_staging = True
        else:
            # A local run directory is already the destination; staging there
            # turns the "upload" into marker writes instead of a second copy.
            staging_dir = Path(self._store.uri(directory))
            cleanup_staging = False
        if cleanup_staging and self._estimate is not None:
            # Cheap statvfs each time: staging is shared with everything else on
            # the node, so passing the startup check does not keep it roomy.
            check_staging_space(
                self._staging_root,
                self._estimate,
                local_ranks=_local_world_size(),
                enforce=bool(getattr(self._settings, "preflight_staging", True)),
            )
        cursor = self._data_cursor(config, context, rows)

        def _beat() -> None:
            _step_watchdog_beat(
                watchdog,
                "checkpoint_stage_upload_wait",
                detail=f"steps={step}",
                device=context.device,
            )

        staged = stage_training_checkpoint(
            config,
            model,
            staging_dir,
            step=step,
            rows=rows,
            rank=context.rank,
            world_size=context.world_size,
            dense_optimizer=dense_optimizer,
            replicated_sparse_optimizer=replicated_sparse_optimizer,
            sharded_optimizer=sharded_optimizer,
            data_cursor=cursor,
            elapsed_seconds=elapsed_seconds,
            run_name=self._run_name,
            cleanup_staging=cleanup_staging,
            chunk_bytes=self._chunk_bytes,
            # Hand each file over as it lands so staging holds a chunk or two
            # rather than every rank's whole shard.
            publish=self._uploader.stream_publisher(
                staging_dir,
                step,
                enabled=cleanup_staging,
                heartbeat=_beat,
            ),
        )
        self._last_saved_step = step
        self._uploader.submit(staged)
        _step_watchdog_beat(
            watchdog,
            "checkpoint_staged",
            detail=f"steps={step}",
            device=context.device,
        )
        if self._log:
            data_position = (
                "unknown"
                if cursor is None
                else (
                    f"{cursor.work_unit}[{cursor.position}] rewind={cursor.rewind} "
                    f"reader_rows={cursor.emitted_rows} "
                    f"trained_rows={cursor.rows_trained}"
                )
            )
            print(
                f"Checkpoint | step={step} staged_in={perf_counter() - started:.1f}s "
                f"data_position={data_position} uri={self._store.uri(directory)}",
                flush=True,
            )

    def close(self) -> None:
        self._uploader.close(timeout_sec=self._settings.ready_timeout_sec)


def train_mdl(
    config: AppConfig,
    max_steps: int | None = None,
    save_checkpoint: bool = True,
    log_steps: bool = True,
    step_observer: TrainStepObserver | None = None,
    training_started_observer: Callable[[], None] | None = None,
    synchronize_step_observer: bool = True,
    run_fixed_test_eval: bool = True,
) -> TrainResult:
    if config.training.sparse_update_mode == "external_parameter_server":
        config = resolve_auto_scenarios(config)
        adapter = _load_external_train_adapter(config.training.sparse_parameter_server_adapter)
        return _coerce_train_result(adapter(config=config, max_steps=max_steps))

    context = _setup_distributed(config)
    # Partition host CPUs by LOCAL_RANK before host-prepare / adapter pools start.
    _apply_local_rank_cpu_affinity("train")
    # World-size-aware batch/prefetch/NCCL headroom before model+reader start.
    config = _apply_world_size_training_profile(config, context.world_size)
    batch_iterator: Iterator[FeatureBatch] | None = None
    step_watchdog: _StepWatchdog | None = None
    checkpointing: _CheckpointCoordinator | None = None
    steps = 0
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
        fixed_test_eval = getattr(
            config.training,
            "fixed_test_eval",
            FixedTestEvalConfig(enabled=False),
        )
        if (
            run_fixed_test_eval
            and fixed_test_eval.enabled
            and (max_steps is None or max_steps >= fixed_test_eval.every_steps)
        ):
            config = _prepare_fixed_test_eval(config, context)
            fixed_test_eval = config.training.fixed_test_eval
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
        # Captured before any resume: the schedule rewrites group["lr"] every
        # step, so a checkpointed optimizer carries a scheduled lr, not the base.
        optimizer_base_lrs = [
            [float(group["lr"]) for group in optimizer.param_groups]
            for optimizer in optimizers
        ]
        checkpointing = _CheckpointCoordinator.create(config, context, log_steps)
        if checkpointing is not None:
            checkpointing.preflight(
                base_model,
                context,
                dense_optimizer=dense_optimizer,
                replicated_sparse_optimizer=embedding_optimizer,
                sharded_optimizer=sharded_embedding_optimizer,
            )
        resumed = (
            None
            if checkpointing is None
            else checkpointing.restore(
                config,
                base_model,
                context,
                device,
                dense_optimizer=dense_optimizer,
                replicated_sparse_optimizer=embedding_optimizer,
                sharded_optimizer=sharded_embedding_optimizer,
            )
        )
        lr_decay_steps = _resolve_lr_decay_steps(config, max_steps)
        scaler = _make_grad_scaler(config, device)
        ordered_task_loss_weights = _configured_task_loss_weights(config)
        task_loss_weights = torch.tensor(
            ordered_task_loss_weights,
            device=device,
            dtype=torch.float32,
        )
        non_blocking = _non_blocking_transfer(config, "train", device)
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
                f"runtime_effective_global_batch={runtime_effective_global_batch} "
                f"reader_shard_unit={config.data.train.reader.shard_unit}"
            )
            task_weight_summary = " ".join(
                f"{task_name}={weight:g}"
                for task_name, weight in zip(
                    config.task_names,
                    ordered_task_loss_weights,
                )
            )
            print(
                f"Loss | reduction={config.training.loss_reduction} "
                f"task_weights={task_weight_summary}"
            )

        watchdog_sec = getattr(config.training, "step_watchdog_sec", None)
        if watchdog_sec is not None:
            step_watchdog = _StepWatchdog(
                float(watchdog_sec),
                rank=context.rank,
            )
            step_watchdog.start()

        steps = 0 if resumed is None else int(resumed.step)
        rows = 0 if resumed is None else int(resumed.rows)
        # Restarts rebuild DDP and the reader, so "first iteration of this
        # process" is the resumed step, not step zero.
        initial_steps = steps
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
                scan_cursor=(
                    None if checkpointing is None else checkpointing.scan_cursor
                ),
                scan_resume_plan=(
                    None if checkpointing is None else checkpointing.scan_resume_plan
                ),
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
        # A reader that is merely slow must not be fatal: past this budget the
        # rank votes starved, peers keep training, and the batch is picked up on
        # a later step. host_prepare_idle_timeout_sec remains the ceiling that
        # eventually calls a permanently silent reader dead.
        step_batch_budget_sec = getattr(train_reader, "step_batch_budget_sec", 30.0)
        batch_iterator = (
            _DevicePrefetchIterator(
                host_batch_iterator,
                device,
                device_prefetch_depth,
            )
            if batches_on_device
            else host_batch_iterator
        )
        starved_steps = 0
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
            # Sample embedding A2A host stats only on log / observer windows so
            # embedding_collect_stats does not D2H every step.
            sample_embedding_stats = observing or (
                log_steps
                and (steps + 1) % config.training.log_every_steps == 0
            )
            set_embedding_stats_host_sync(
                sample_embedding_stats
                if getattr(config.training, "embedding_collect_stats", False)
                else False
            )
            dataloader_started = perf_counter()
            trace_batch: FeatureBatch | None = None
            collect_batch_stats = observing or (
                log_steps
                and context.rank == 0
                and (steps + 1) % config.training.log_every_steps == 0
            )
            _step_watchdog_beat(
                step_watchdog,
                "dataloader",
                detail=f"steps={steps}",
                device=device,
            )
            # Replicated sparse DDP needs a real batch from every rank on the
            # very first micro-step, so starvation is not an option there yet.
            batch_budget_sec = (
                None
                if steps == initial_steps and accumulation_index == 0
                else step_batch_budget_sec
            )
            try:
                if (
                    collect_batch_stats
                    and batches_on_device
                    and isinstance(batch_iterator, _DevicePrefetchIterator)
                ):
                    read = batch_iterator.next_with_host_within(batch_budget_sec)
                    if read is BATCH_NOT_READY:
                        local_batch = BATCH_NOT_READY
                    else:
                        trace_batch, local_batch = read
                else:
                    local_batch = _next_batch_within(batch_iterator, batch_budget_sec)
                    if (
                        collect_batch_stats
                        and not batches_on_device
                        and local_batch is not BATCH_NOT_READY
                    ):
                        trace_batch = local_batch
            except StopIteration:
                local_batch = None
            local_batch_on_device = batches_on_device
            dataloader_wait_seconds = perf_counter() - dataloader_started
            rank_starved = local_batch is BATCH_NOT_READY
            if rank_starved:
                local_batch = None
                starved_steps += 1
            rank_active = local_batch is not None
            rank_exhausted = not rank_active and not rank_starved
            # Kick the Gloo/NCCL supply reduction immediately so pinned H2D can
            # overlap it. Wait only after the host→device copy is queued.
            _step_watchdog_beat(
                step_watchdog,
                "active_rank_sync",
                detail=(
                    f"steps={steps} rank_active={int(rank_active)} "
                    f"rank_starved={int(rank_starved)}"
                ),
                device=device,
            )
            supply_handle = _start_rank_supply_count(
                context,
                rank_active=rank_active,
                rank_exhausted=rank_exhausted,
            )
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
                active_ranks = supply_handle.wait().active
            else:
                supply = supply_handle.wait()
                active_ranks = supply.active
                verdict = _supply_verdict(
                    supply,
                    context.world_size if context.enabled else 1,
                )
                if verdict == "stop":
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
                if verdict == "retry":
                    if log_steps and context.rank == 0:
                        print(
                            "Dataloader starvation | "
                            f"step={steps} every rank is waiting for data "
                            f"(starved_steps={starved_steps})"
                        )
                    continue
                if last_device_batch is None:
                    raise RuntimeError("inactive rank has no batch available for zero-loss replay")
                batch = last_device_batch
                trace_batch = last_trace_batch
            if (
                steps == initial_steps
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
            local_rows = int(batch.scenario_id.size(0)) if batch is not None else 0
            _step_watchdog_beat(
                step_watchdog,
                "h2d",
                detail=f"steps={steps} local_rows={local_rows}",
                device=device,
            )

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
                context.enabled and ddp_config.static_graph and steps == initial_steps
            )
            forward_started = perf_counter() if observing else 0.0
            _step_watchdog_beat(
                step_watchdog,
                "forward",
                detail=f"steps={steps} local_rows={local_rows}",
                device=device,
            )
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
                        task_loss_weights=task_loss_weights,
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
                _step_watchdog_beat(
                    step_watchdog,
                    "backward",
                    detail=f"steps={steps} local_rows={local_rows}",
                    device=device,
                )
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
            _step_watchdog_beat(
                step_watchdog,
                "sparse_sync",
                detail=f"steps={steps} local_rows={local_rows}",
                device=device,
            )
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
            _step_watchdog_beat(
                step_watchdog,
                "optimizer",
                detail=f"steps={steps} local_rows={local_rows}",
                device=device,
            )
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
            _step_watchdog_beat(
                step_watchdog,
                "step_done",
                detail=(
                    f"steps={steps} local_rows={window_rows} "
                    f"active_ranks={window_active_ranks}"
                ),
                device=device,
            )
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
            # Flush idle pinned-host slabs frequently. With variable-length
            # batches the CUDA caching host allocator otherwise keeps every
            # size-class high-water mark for the whole job (pod RSS climbs
            # linearly for hours even after FeatureBatch refs are gone).
            if device.type == "cuda" and steps > 0 and steps % 10 == 0:
                _release_cached_host_allocator_memory()
                if steps % 100 == 0:
                    gc.collect()
                    # The parent unpickles, privatizes and pins every batch, so
                    # its glibc heap ratchets too; the host allocator flush
                    # above only covers CUDA-pinned slabs.
                    trim_process_heap()
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
                print(
                    f"Train step | step={steps} | logloss={last_loss:.6f} "
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
                    # Otherwise a reader degrading into repeated zero-loss
                    # replays is invisible until it trips the idle ceiling.
                    + (f" starved_steps={starved_steps}" if starved_steps else "")
                    # Keep checkpoint arming visible on the same cadence as
                    # Train step: early banners are easy to miss in truncated
                    # trainjob tails, and a silent disable looks like a code bug.
                    + (
                        f" checkpoint=on(every={config.training.checkpoint.every_steps})"
                        if checkpointing is not None
                        else " checkpoint=off"
                    )
                )
                memory_report = _host_memory_report(host_batch_iterator)
                if memory_report:
                    print(f"Host memory | step={steps} {memory_report}", flush=True)
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
                # Drop the host-side log snapshot so pinned/shared batches from
                # next_with_host do not linger until the next log window.
                last_trace_batch = None
                window_task_monitors = None
                gc.collect()
            if (
                run_fixed_test_eval
                and fixed_test_eval.enabled
                and steps % fixed_test_eval.every_steps == 0
            ):
                fixed_test_result = _run_fixed_test_eval(
                    config,
                    model,
                    vocab_maps,
                    context,
                    fixed_test_eval,
                    fallback_batch=last_device_batch,
                    watchdog=step_watchdog,
                )
                # Evaluation forwards also touch sharded-embedding diagnostics;
                # keep them out of the following training step's trace/log.
                consume_sharded_embedding_stats(base_model)
                if context.rank == 0:
                    _print_fixed_test_eval(
                        steps,
                        fixed_test_eval,
                        fixed_test_result,
                    )
                del fixed_test_result
            if checkpointing is not None and checkpointing.due(steps):
                checkpointing.save(
                    config,
                    base_model,
                    context,
                    step=steps,
                    rows=rows,
                    elapsed_seconds=perf_counter() - start,
                    dense_optimizer=dense_optimizer,
                    replicated_sparse_optimizer=embedding_optimizer,
                    sharded_optimizer=sharded_embedding_optimizer,
                    watchdog=step_watchdog,
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

        if checkpointing is not None and checkpointing.due_on_exit(steps):
            checkpointing.save(
                config,
                base_model,
                context,
                step=steps,
                rows=rows,
                elapsed_seconds=elapsed,
                dense_optimizer=dense_optimizer,
                replicated_sparse_optimizer=embedding_optimizer,
                sharded_optimizer=sharded_embedding_optimizer,
                watchdog=step_watchdog,
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
    except torch.cuda.OutOfMemoryError as error:
        _abort_rank_for_cuda_oom(
            error,
            rank=context.rank,
            steps=steps,
            detail=f"device={device}",
        )
    except RemoteIoStallError as error:
        abort_rank_for_remote_io_stall(error)
    finally:
        # Keep the step watchdog armed through iterator teardown so a hung
        # device/host prefetch close cannot stall silently after training ends.
        try:
            if batch_iterator is not None:
                close = getattr(batch_iterator, "close", None)
                if callable(close):
                    close()
        finally:
            if step_watchdog is not None:
                step_watchdog.stop()
            if checkpointing is not None:
                checkpointing.close()
            _cleanup_distributed(context)


# Reported AUC uses the online ranking evaluator's estimator: a fixed grid of
# `[-eps, 1/(N-1), ..., 1, 1+eps]` probability thresholds, per-threshold
# TP/FP/TN/FN, and trapezoidal integration of the resulting ROC points. N comes
# from config and is 4096 rather than the 3000 used online; the estimator is
# monotone in resolution, so a finer grid only moves the result toward exact.
_AUC_THRESHOLD_EPSILON = 1e-7

# Secondary diagnostic grid, uniform in log-odds instead of in probability. A
# fixed probability grid resolves a task only as finely as the span its scores
# occupy, not as finely as its base rate: predictions inside one threshold step
# read as tied pairs, each counted half, pulling the value toward 0.5 even when
# the ranking is perfect. A 0.3%-rate task with a normal logit spread loses
# ~2e-4 at N=4096, but ~0.04 once its predictions compress into a narrow band.
# The gap between the two numbers separates "ranks worse" from "predicts flat".
_AUC_LOGIT_LIMIT = 20.0
_AUC_DIAGNOSTIC_BINS = 4096


def _production_auc_thresholds(num_thresholds: int) -> Tensor:
    """The online evaluator's threshold grid; kept for reference and tests."""

    interior = torch.arange(1, num_thresholds, dtype=torch.float64) / (
        num_thresholds - 1
    )
    return torch.cat(
        [
            torch.tensor([-_AUC_THRESHOLD_EPSILON], dtype=torch.float64),
            interior,
            torch.tensor([1.0 + _AUC_THRESHOLD_EPSILON], dtype=torch.float64),
        ]
    )


def _auc_logit_bins(logits: Tensor, bins: int) -> Tensor:
    """Map logits onto ``bins`` uniform log-odds buckets over the clamp range."""

    normalized = (
        logits.clamp(-_AUC_LOGIT_LIMIT, _AUC_LOGIT_LIMIT) + _AUC_LOGIT_LIMIT
    ) / (2.0 * _AUC_LOGIT_LIMIT)
    return torch.clamp(torch.floor(normalized * bins).long(), min=0, max=bins - 1)


def _auc_probability_bins(logits: Tensor, num_thresholds: int) -> Tensor:
    """Bucket by the highest online threshold each prediction exceeds.

    The evaluator this mirrors compares every prediction against all `N + 1`
    thresholds to build one confusion matrix per threshold. Bucketing by
    `ceil(p * (N - 1)) - 1` reproduces those counts exactly under the same
    `pred > threshold` rule while staying O(batch) instead of O(batch * N).
    """

    probabilities = torch.sigmoid(logits.double())
    buckets = torch.ceil(probabilities * (num_thresholds - 1)).long() - 1
    return torch.clamp(buckets, min=0, max=num_thresholds - 1)


def _trapezoidal_roc_auc(positives: Tensor, negatives: Tensor) -> float:
    """Integrate the ROC points implied by per-threshold confusion counts."""

    positive_total = float(positives.sum().item())
    negative_total = float(negatives.sum().item())
    # tp/fp at threshold j count everything in bucket j and above, so the
    # per-threshold confusion matrix is a reversed cumulative sum. Both series
    # start at (1, 1) for the -eps threshold and end at (0, 0) for 1 + eps.
    true_positive_rate = torch.cat(
        [
            torch.flip(torch.cumsum(torch.flip(positives, (0,)), 0), (0,)),
            torch.zeros(1, dtype=positives.dtype),
        ]
    ) / positive_total
    false_positive_rate = torch.cat(
        [
            torch.flip(torch.cumsum(torch.flip(negatives, (0,)), 0), (0,)),
            torch.zeros(1, dtype=negatives.dtype),
        ]
    ) / negative_total
    widths = false_positive_rate[:-1] - false_positive_rate[1:]
    heights = (true_positive_rate[:-1] + true_positive_rate[1:]) * 0.5
    return float((widths * heights).sum().item())


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
    """Online-evaluator AUC: fixed probability thresholds plus trapezoid.

    Holds per-bucket positive/negative counts rather than the evaluator's
    ``(4, num_thresholds)`` confusion matrix; the two carry the same
    information, since TP/FP at a threshold are reversed cumulative sums of
    these counts and TN/FN are their complements. Reducing counts across ranks
    is a plain sum either way.
    """

    def __init__(self, bins: int, *, logit_grid: bool = False) -> None:
        if bins < 2:
            raise ValueError("AUC histogram requires at least two bins")
        self.bins = bins
        self.logit_grid = logit_grid
        self.histogram = torch.zeros(2, bins, dtype=torch.float64)
        self.positives = self.histogram[0]
        self.negatives = self.histogram[1]

    def update(self, logits: Tensor, labels: Tensor) -> None:
        logits = logits.detach().float().flatten().cpu()
        labels = labels.detach().float().flatten().cpu()
        if logits.numel() != labels.numel():
            raise ValueError("AUC logits and labels must have the same length")
        if not logits.numel():
            return
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("AUC logits must be finite")
        if not bool(((labels == 0.0) | (labels == 1.0)).all()):
            raise ValueError("AUC labels must be binary")
        indices = (
            _auc_logit_bins(logits, self.bins)
            if self.logit_grid
            else _auc_probability_bins(logits, self.bins)
        )
        self.positives += torch.bincount(
            indices[labels == 1.0], minlength=self.bins
        ).to(torch.float64)
        self.negatives += torch.bincount(
            indices[labels == 0.0], minlength=self.bins
        ).to(torch.float64)

    def compute(self) -> float | None:
        if not float(self.positives.sum().item()) or not float(
            self.negatives.sum().item()
        ):
            return None
        return _trapezoidal_roc_auc(self.positives, self.negatives)

    def occupied_bins(self) -> int:
        """Buckets holding at least one sample; low values mean a coarse grid."""

        return int(((self.positives + self.negatives) > 0).sum().item())

    def counts(self) -> tuple[int, int, int]:
        positives = int(self.positives.sum().item())
        negatives = int(self.negatives.sum().item())
        return positives + negatives, positives, negatives


# Reported AUC and the log-odds diagnostic agree to well under this on any task
# whose predictions actually spread across the threshold grid.
_AUC_GRID_DRIFT_WARN = 0.005

# Collapse / miscalibration heuristics for train and quick-eval monitors.
_MONITOR_PROB_MEAN_WARN = 0.9
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
    auc: float | None = None,
) -> list[str]:
    warnings: list[str] = []
    if prob_mean is not None and prob_mean > _MONITOR_PROB_MEAN_WARN:
        warnings.append(f"prob_mean={prob_mean:.4f}>{_MONITOR_PROB_MEAN_WARN}")
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


def _reduce_evaluation_task_monitors(
    context: DistributedContext,
    accumulators: list[_StreamingTaskMonitor],
) -> None:
    if not context.enabled:
        return
    for accumulator in accumulators:
        _all_reduce_cpu_sum_(accumulator.totals, context)


def _run_fixed_test_eval(
    config: AppConfig,
    model: nn.Module,
    vocab_maps: dict[str, dict[str, int]],
    context: DistributedContext,
    evaluation: FixedTestEvalConfig,
    *,
    fallback_batch: FeatureBatch | None,
    watchdog: _StepWatchdog | None = None,
) -> FixedTestEvalResult:
    """Evaluate every row in the frozen held-out Parquet manifest."""

    split = config.data.test
    if split is None:
        raise ValueError("fixed-test evaluation requires data.test")

    # Second accumulator per task is the log-odds diagnostic grid; it shares the
    # reduce path and only costs one more bincount per batch.
    accumulators = [
        [
            _StreamingHistogramAUC(evaluation.auc_bins),
            _StreamingHistogramAUC(_AUC_DIAGNOSTIC_BINS, logit_grid=True),
        ]
        for _ in config.task_names
    ]
    task_monitors = [_StreamingTaskMonitor() for _ in config.task_names]
    rows = 0
    local_batches = 0
    batch_iterator: Iterator[FeatureBatch] | None = None
    replay_batch = fallback_batch
    was_training = model.training
    started = perf_counter()
    model.eval()
    try:
        non_blocking = _non_blocking_transfer(
            config,
            "test",
            context.device,
        )
        host_iterator = iter(
            iter_feature_batches(
                config,
                "test",
                vocab_maps,
                require_labels=True,
                shard_rank=context.rank,
                shard_world_size=context.world_size,
                pin_memory=non_blocking,
                include_group_id=False,
            )
        )
        device_prefetch_depth = (
            int(split.reader.device_prefetch_batches)
            if context.device.type == "cuda"
            else 0
        )
        batches_on_device = device_prefetch_depth > 0
        batch_iterator = (
            _DevicePrefetchIterator(
                host_iterator,
                context.device,
                device_prefetch_depth,
            )
            if batches_on_device
            else host_iterator
        )
        _step_watchdog_beat(
            watchdog,
            "fixed_test_eval",
            detail="waiting_for_first_batch",
            device=context.device,
        )
        with torch.inference_mode():
            while True:
                try:
                    local_batch = next(batch_iterator)
                except StopIteration:
                    local_batch = None
                if local_batch is not None:
                    local_batches += 1
                    if local_batches == 1 or local_batches % 16 == 0:
                        _step_watchdog_beat(
                            watchdog,
                            "fixed_test_eval",
                            detail=f"local_batches={local_batches}",
                            device=context.device,
                        )
                rank_active = local_batch is not None
                active_ranks = _active_rank_count(context, rank_active)
                if active_ranks == 0:
                    break
                if rank_active:
                    if local_batch is None:
                        raise AssertionError(
                            "active fixed-test rank is missing its batch"
                        )
                    batch = (
                        local_batch
                        if batches_on_device
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
                            "fixed-test evaluation requires every rank to have a "
                            "previous training batch for uneven-tail replay"
                        )
                    batch = replay_batch

                with _autocast_context(config, context.device):
                    logits = model(batch.features, batch.scenario_id)["logits"]
                if not rank_active:
                    continue
                if batch.labels is None:
                    raise RuntimeError("fixed-test batch did not contain labels")
                logits_cpu = logits.float().cpu()
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
                        task_labels = labels[:, task_index]
                    else:
                        valid = label_mask[:, task_index]
                        task_logits = logits_cpu[valid, task_index]
                        task_labels = labels[valid, task_index]
                    for accumulator in accumulators[task_index]:
                        accumulator.update(task_logits, task_labels)
                    task_monitors[task_index].update(task_logits, task_labels)

        _step_watchdog_beat(
            watchdog,
            "fixed_test_eval_reduce",
            detail=f"local_batches={local_batches}",
            device=context.device,
        )
        rows = _reduce_evaluation_histograms(context, accumulators, rows)
        _reduce_evaluation_task_monitors(context, task_monitors)
        metrics: dict[str, dict[str, float | int | None]] = {}
        for task_index, task_name in enumerate(config.task_names):
            accumulator, diagnostic = accumulators[task_index]
            examples, positives, negatives = accumulator.counts()
            monitor = task_monitors[task_index].compute()
            metrics[task_name] = {
                "auc": accumulator.compute(),
                "auc_logit_grid": diagnostic.compute(),
                "auc_occupied_thresholds": accumulator.occupied_bins(),
                "loss": monitor["loss"],
                "prob_mean": monitor["prob_mean"],
                "logit_mean": monitor["logit_mean"],
                "logit_std": monitor["logit_std"],
                "examples": examples,
                "positives": positives,
                "negatives": negatives,
            }
        _sync_device(context.device)
        # Do not retain the last device replay batch across the return path.
        replay_batch = None
        return FixedTestEvalResult(
            rows=rows,
            metrics=metrics,
            elapsed_seconds=perf_counter() - started,
            files=len(split.inputs),
        )
    finally:
        if batch_iterator is not None:
            close = getattr(batch_iterator, "close", None)
            if callable(close):
                close()
        model.train(was_training)


def _print_fixed_test_eval(
    step: int,
    evaluation: FixedTestEvalConfig,
    result: FixedTestEvalResult,
) -> None:
    rows_per_second = (
        result.rows / result.elapsed_seconds
        if result.elapsed_seconds > 0.0
        else 0.0
    )
    print(
        f"Fixed test eval | step={step} rows={result.rows} files={result.files} "
        f"files_per_rank={evaluation.files_per_rank} "
        f"elapsed_seconds={result.elapsed_seconds:.6f} "
        f"rows_per_second={rows_per_second:.2f}"
    )
    for task_name, metrics in result.metrics.items():
        auc = metrics["auc"]
        auc_value = None if auc is None else float(auc)
        prob_mean = metrics.get("prob_mean")
        prob_mean_value = None if prob_mean is None else float(prob_mean)
        print(
            f"Fixed test eval task | step={step} task={task_name} "
            f"auc={_format_optional_float(auc_value, 8)} "
            f"logloss={_format_optional_float(metrics.get('loss'))} "
            f"prob_mean={_format_optional_float(prob_mean_value)} "
            f"logit_mean={_format_optional_float(metrics.get('logit_mean'))} "
            f"logit_std={_format_optional_float(metrics.get('logit_std'))} "
            f"examples={metrics['examples']} positives={metrics['positives']} "
            f"negatives={metrics['negatives']}"
        )
        diagnostic = metrics.get("auc_logit_grid")
        if auc_value is not None and diagnostic is not None:
            drift = float(diagnostic) - auc_value
            # The two grids only diverge when this task's predictions sit inside
            # a handful of threshold steps, so the reported AUC is measuring
            # threshold resolution rather than ranking.
            if abs(drift) > _AUC_GRID_DRIFT_WARN:
                print(
                    f"Fixed test eval warning | step={step} task={task_name} "
                    f"auc_grid_drift={drift:+.6f} "
                    f"auc_logit_grid={float(diagnostic):.8f} "
                    "occupied_thresholds="
                    f"{metrics.get('auc_occupied_thresholds')} "
                    "(predictions span too few thresholds to rank on)"
                )
        warning_parts = _task_monitor_warning_parts(
            prob_mean=prob_mean_value,
            auc=auc_value,
        )
        if warning_parts:
            print(
                f"Fixed test eval warning | step={step} task={task_name} "
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
        logits: Tensor,
        labels: Tensor,
        scenario_membership: Tensor,
    ) -> None:
        logit_values = logits.detach().float().flatten().cpu()
        label_values = labels.detach().long().flatten().cpu()
        memberships = scenario_membership.detach().bool().cpu()
        if (
            len(group_ids) != logit_values.numel()
            or label_values.numel() != logit_values.numel()
            or memberships.size(0) != logit_values.numel()
        ):
            raise ValueError("group AUC batch inputs must have matching rows")
        score_bins = _auc_probability_bins(logit_values, self.bins).tolist()
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
    auc_bins: int = 4096,
) -> EvaluateResult:
    context = _setup_distributed(config)
    _apply_local_rank_cpu_affinity("train")
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
            logits_cpu = logits.float().cpu()
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
                    task_logits = logits_cpu[:, task_index]
                    task_labels = labels[:, task_index]
                    task_scenarios = scenario_membership
                    task_groups = batch.group_id
                else:
                    valid = label_mask[:, task_index]
                    task_logits = logits_cpu[valid, task_index]
                    task_labels = labels[valid, task_index]
                    task_scenarios = scenario_membership[valid]
                    task_groups = [
                        group_id
                        for group_id, keep in zip(batch.group_id, valid.tolist())
                        if keep
                    ]
                auc_accumulators[task_index][0].update(
                    task_logits, task_labels
                )
                for scenario in range(scenario_count):
                    selected = task_scenarios[:, scenario]
                    auc_accumulators[task_index][scenario + 1].update(
                        task_logits[selected], task_labels[selected]
                    )
                if grouped_auc is not None:
                    grouped_auc.add(
                        task_index,
                        task_groups,
                        task_logits,
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
