"""Repeatable data, embedding, compute, and end-to-end benchmarks.

The benchmark surface intentionally reuses the training and dataloader code
paths. It does not maintain a second approximate trainer. Device phase timings
are synchronized only when a benchmark is active; normal training is unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import threading
from time import perf_counter, process_time
from typing import Any, Callable, Literal

import torch
import torch.distributed as torch_dist
from torch import Tensor, nn

from .config import AppConfig, ResolvedVocabEncoding, resolve_categorical_base_input
from .dataloader import FeatureBatch
from .features import load_vocab_maps, vocab_strategy_fingerprint
from .embeddings import ShardedEmbedding
from .optim import ShardedAdagrad
from .model import build_model
from .train import (
    DistributedContext,
    TrainStepTrace,
    _ReplicatedSparseGradientSynchronizer,
    _attention_runtime_description,
    _autocast_context,
    _batch_input_token_count,
    _batch_padded_token_slots,
    _build_dense_optimizer,
    _classify_model_parameters,
    _cleanup_distributed,
    _loss_terms_from_batch,
    _prepare_forward_model,
    _non_blocking_transfer,
    _setup_distributed,
    _step_sparse_moe_controllers,
    _sync_device,
    _synchronize_sparse_parameter_replicas,
    iter_feature_batches,
    train_mdl,
)


BenchmarkMode = Literal["data", "embedding", "compute", "end-to-end"]
IdDistribution = Literal["uniform", "zipf"]


@dataclass(frozen=True)
class BenchmarkOptions:
    """Command-level controls shared by all benchmark modes."""

    mode: BenchmarkMode
    warmup_steps: int = 10
    measured_steps: int = 50
    profile_steps: int = 1
    seed: int = 2025
    batch_size: int | None = None
    sequence_length: int | None = None
    embedding_lookups_per_table: int = 65536
    id_distribution: IdDistribution = "uniform"
    zipf_exponent: float = 1.2
    peak_tflops: float | None = None

    def validate(self) -> None:
        if self.mode not in {"data", "embedding", "compute", "end-to-end"}:
            raise ValueError(f"unsupported benchmark mode {self.mode!r}")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.measured_steps <= 0:
            raise ValueError("measured_steps must be positive")
        if self.profile_steps < 0:
            raise ValueError("profile_steps must be non-negative")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.batch_size is not None and self.mode != "compute":
            raise ValueError("batch_size override is supported only in compute mode")
        if self.sequence_length is not None and self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.embedding_lookups_per_table <= 0:
            raise ValueError("embedding_lookups_per_table must be positive")
        if self.id_distribution not in {"uniform", "zipf"}:
            raise ValueError("id_distribution must be uniform or zipf")
        if self.zipf_exponent <= 1.0:
            raise ValueError("zipf_exponent must be greater than 1")
        if self.peak_tflops is not None and self.peak_tflops <= 0.0:
            raise ValueError("peak_tflops must be positive")


@dataclass(frozen=True)
class ProfilerSummary:
    attention_kernels: tuple[str, ...] = ()
    communication_operators: tuple[str, ...] = ()
    communication_operator_seconds: float | None = None
    estimated_flops_per_step: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class LocalBenchmarkSummary:
    rank: int
    traces: tuple[TrainStepTrace, ...]
    peak_hbm_allocated_bytes: int
    peak_hbm_reserved_bytes: int
    process_peak_rss_bytes: int
    cpu_utilization_percent: float | None
    gpu_utilization_percent: float | None
    host_batch_bytes_peak: int = 0
    profiler: ProfilerSummary = field(default_factory=ProfilerSummary)


@dataclass(frozen=True)
class BenchmarkReport:
    mode: BenchmarkMode
    world_size: int
    warmup_steps: int
    measured_steps: int
    samples: int
    input_tokens: int
    padding_ratio: float
    elapsed_seconds: float
    samples_per_second: float
    tokens_per_second: float
    mean_step_seconds: float
    p95_step_seconds: float
    mean_dataloader_wait_seconds: float
    p95_dataloader_wait_seconds: float
    dataloader_wait_ratio: float
    mean_h2d_seconds: float
    mean_forward_seconds: float
    mean_backward_seconds: float
    mean_sparse_sync_seconds: float
    mean_optimizer_seconds: float
    sparse_payload_bytes_per_step_rank_max: float
    peak_hbm_allocated_bytes_per_rank: tuple[int, ...]
    peak_hbm_reserved_bytes_per_rank: tuple[int, ...]
    process_peak_rss_bytes_per_rank: tuple[int, ...]
    cpu_utilization_percent_per_rank: tuple[float | None, ...]
    gpu_utilization_percent_per_rank: tuple[float | None, ...]
    host_batch_bytes_peak_per_rank: tuple[int, ...]
    attention_kernels: tuple[str, ...]
    communication_operators: tuple[str, ...]
    profiled_communication_operator_seconds_rank_max: float | None
    estimated_flops_per_step_rank_max: float | None
    mfu: float | None
    mfu_method: str | None
    profiler_errors: tuple[str, ...]
    benchmark_options: dict[str, Any]
    environment: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be in [0, 1]")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _process_peak_rss_bytes() -> int:
    # Linux reports KiB, while macOS reports bytes.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _nvidia_smi_device_selector(device: torch.device) -> str:
    """Map a process-local CUDA index back to nvidia-smi's physical selector."""

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        selectors = [item.strip() for item in visible_devices.split(",")]
        selectors = [item for item in selectors if item]
        if 0 <= device_index < len(selectors):
            return selectors[device_index]
    return str(device_index)


class _GpuUtilizationSampler:
    """Best-effort GPU utilization sampling without a mandatory NVML package."""

    def __init__(self, device: torch.device, interval_seconds: float = 0.2) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._values: list[float] = []
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.device.type != "cuda" or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="mdl-benchmark-gpu-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> float | None:
        if self._thread is None:
            return None
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 3.0))
        return _mean(self._values) if self._values else None

    def _run(self) -> None:
        device_selector = _nvidia_smi_device_selector(self.device)
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i",
                        device_selector,
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
                if result.returncode == 0:
                    value = result.stdout.strip().splitlines()[0]
                    self._values.append(float(value))
            except (FileNotFoundError, IndexError, OSError, subprocess.SubprocessError, ValueError):
                return
            self._stop.wait(self.interval_seconds)


class _TraceCollector:
    def __init__(
        self,
        warmup_steps: int,
        measured_steps: int,
        device: torch.device,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.measured_steps = measured_steps
        self.device = device
        self.traces: list[TrainStepTrace] = []
        self.host_batch_bytes_peak = 0
        self._measurement_started = False
        self._cpu_started = 0.0
        self._sampler = _GpuUtilizationSampler(device)
        self._gpu_utilization: float | None = None
        if warmup_steps == 0:
            self._start_measurement()

    def _start_measurement(self) -> None:
        if self._measurement_started:
            return
        self._measurement_started = True
        self._cpu_started = process_time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sampler.start()

    def observe(self, trace: TrainStepTrace, host_batch_bytes: int = 0) -> None:
        if trace.step == self.warmup_steps:
            self._start_measurement()
            return
        if trace.step <= self.warmup_steps:
            return
        if len(self.traces) >= self.measured_steps:
            return
        self.traces.append(trace)
        self.host_batch_bytes_peak = max(self.host_batch_bytes_peak, host_batch_bytes)

    def finish(self, rank: int, profiler: ProfilerSummary) -> LocalBenchmarkSummary:
        if not self._measurement_started:
            self._start_measurement()
        self._gpu_utilization = self._sampler.stop()
        elapsed = sum(trace.step_seconds for trace in self.traces)
        cpu_elapsed = process_time() - self._cpu_started
        cpu_percent = 100.0 * cpu_elapsed / elapsed if elapsed > 0.0 else None
        if self.device.type == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        else:
            peak_allocated = 0
            peak_reserved = 0
        return LocalBenchmarkSummary(
            rank=rank,
            traces=tuple(self.traces),
            peak_hbm_allocated_bytes=peak_allocated,
            peak_hbm_reserved_bytes=peak_reserved,
            process_peak_rss_bytes=_process_peak_rss_bytes(),
            cpu_utilization_percent=cpu_percent,
            gpu_utilization_percent=self._gpu_utilization,
            host_batch_bytes_peak=self.host_batch_bytes_peak,
            profiler=profiler,
        )


def _feature_value_nbytes(value: Any) -> int:
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_feature_value_nbytes(child) for child in value.values())
    return 0


def _feature_batch_nbytes(batch: FeatureBatch) -> int:
    total = _feature_value_nbytes(batch.features)
    for tensor in (batch.labels, batch.label_mask, batch.scenario_id):
        if isinstance(tensor, Tensor):
            total += tensor.numel() * tensor.element_size()
    return total


def _environment(config: AppConfig, context: DistributedContext) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        ).stdout.strip()
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        revision = ""
    cuda_version = torch.version.cuda
    device_name = None
    if context.device.type == "cuda":
        device_name = torch.cuda.get_device_name(context.device)
    return {
        "git_revision": revision or None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": cuda_version,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(context.device),
        "device_name": device_name,
        "world_size": context.world_size,
        "precision": config.runtime.precision,
        "attention_backend_requested": config.runtime.attention_backend,
        "activation_checkpoint": config.runtime.activation_checkpoint,
        "model_name": config.model.name,
        "embedding_distribution": config.training.embedding_distribution,
        "dense_distribution": config.training.dense_distribution,
        "vocab_strategy_hash": vocab_strategy_fingerprint(config),
    }


def _gather_local_summaries(
    context: DistributedContext,
    local: LocalBenchmarkSummary,
) -> list[LocalBenchmarkSummary]:
    if not context.enabled or not torch_dist.is_initialized():
        return [local]
    gathered: list[LocalBenchmarkSummary | None] = [None] * context.world_size
    torch_dist.all_gather_object(gathered, local)
    if any(item is None for item in gathered):
        raise RuntimeError("failed to gather benchmark summaries from every rank")
    return [item for item in gathered if item is not None]


def _rank_max_phase(
    summaries: list[LocalBenchmarkSummary],
    trace_index: int,
    attribute: str,
) -> float:
    return max(
        (
            float(getattr(summary.traces[trace_index], attribute))
            for summary in summaries
            if trace_index < len(summary.traces)
        ),
        default=0.0,
    )


def _build_report(
    config: AppConfig,
    context: DistributedContext,
    options: BenchmarkOptions,
    local: LocalBenchmarkSummary,
) -> BenchmarkReport:
    summaries = _gather_local_summaries(context, local)
    measured_steps = max((len(summary.traces) for summary in summaries), default=0)
    if measured_steps == 0:
        raise RuntimeError(
            "benchmark produced no measured steps; lower warmup_steps or provide more input data"
        )

    step_seconds = [
        _rank_max_phase(summaries, index, "step_seconds")
        for index in range(measured_steps)
    ]
    wait_seconds = [
        _rank_max_phase(summaries, index, "dataloader_wait_seconds")
        for index in range(measured_steps)
    ]
    elapsed = sum(step_seconds)
    samples = sum(trace.rows for summary in summaries for trace in summary.traces)
    input_tokens = sum(
        trace.input_tokens for summary in summaries for trace in summary.traces
    )
    padded_token_slots = sum(
        trace.padded_token_slots
        for summary in summaries
        for trace in summary.traces
    )
    sparse_payload = [
        _rank_max_phase(summaries, index, "sparse_payload_bytes")
        for index in range(measured_steps)
    ]

    attention_kernels = sorted(
        {
            key
            for summary in summaries
            for key in summary.profiler.attention_kernels
        }
    )
    communication_operators = sorted(
        {
            key
            for summary in summaries
            for key in summary.profiler.communication_operators
        }
    )
    profiled_comm = [
        summary.profiler.communication_operator_seconds
        for summary in summaries
        if summary.profiler.communication_operator_seconds is not None
    ]
    profiled_flops = [
        summary.profiler.estimated_flops_per_step
        for summary in summaries
        if summary.profiler.estimated_flops_per_step is not None
    ]
    flops_per_step = max(profiled_flops) if profiled_flops else None
    mfu = None
    mfu_method = None
    if flops_per_step is not None and options.peak_tflops is not None:
        mean_step = _mean(step_seconds)
        if mean_step > 0.0:
            mfu = flops_per_step / (mean_step * options.peak_tflops * 1.0e12)
            mfu_method = "torch_profiler_estimated_flops / configured_peak_tflops"

    profiler_errors = tuple(
        sorted(
            {
                summary.profiler.error
                for summary in summaries
                if summary.profiler.error is not None
            }
        )
    )
    return BenchmarkReport(
        mode=options.mode,
        world_size=context.world_size,
        warmup_steps=options.warmup_steps,
        measured_steps=measured_steps,
        samples=samples,
        input_tokens=input_tokens,
        padding_ratio=(
            1.0 - input_tokens / padded_token_slots
            if padded_token_slots > 0
            else 0.0
        ),
        elapsed_seconds=elapsed,
        samples_per_second=(samples / elapsed if elapsed > 0.0 else 0.0),
        tokens_per_second=(input_tokens / elapsed if elapsed > 0.0 else 0.0),
        mean_step_seconds=_mean(step_seconds),
        p95_step_seconds=_percentile(step_seconds, 0.95),
        mean_dataloader_wait_seconds=_mean(wait_seconds),
        p95_dataloader_wait_seconds=_percentile(wait_seconds, 0.95),
        dataloader_wait_ratio=(sum(wait_seconds) / elapsed if elapsed > 0.0 else 0.0),
        mean_h2d_seconds=_mean(
            [_rank_max_phase(summaries, i, "h2d_seconds") for i in range(measured_steps)]
        ),
        mean_forward_seconds=_mean(
            [_rank_max_phase(summaries, i, "forward_seconds") for i in range(measured_steps)]
        ),
        mean_backward_seconds=_mean(
            [_rank_max_phase(summaries, i, "backward_seconds") for i in range(measured_steps)]
        ),
        mean_sparse_sync_seconds=_mean(
            [_rank_max_phase(summaries, i, "sparse_sync_seconds") for i in range(measured_steps)]
        ),
        mean_optimizer_seconds=_mean(
            [_rank_max_phase(summaries, i, "optimizer_seconds") for i in range(measured_steps)]
        ),
        sparse_payload_bytes_per_step_rank_max=_mean(sparse_payload),
        peak_hbm_allocated_bytes_per_rank=tuple(
            summary.peak_hbm_allocated_bytes for summary in summaries
        ),
        peak_hbm_reserved_bytes_per_rank=tuple(
            summary.peak_hbm_reserved_bytes for summary in summaries
        ),
        process_peak_rss_bytes_per_rank=tuple(
            summary.process_peak_rss_bytes for summary in summaries
        ),
        cpu_utilization_percent_per_rank=tuple(
            summary.cpu_utilization_percent for summary in summaries
        ),
        gpu_utilization_percent_per_rank=tuple(
            summary.gpu_utilization_percent for summary in summaries
        ),
        host_batch_bytes_peak_per_rank=tuple(
            summary.host_batch_bytes_peak for summary in summaries
        ),
        attention_kernels=tuple(attention_kernels),
        communication_operators=tuple(communication_operators),
        profiled_communication_operator_seconds_rank_max=(
            max(profiled_comm) if profiled_comm else None
        ),
        estimated_flops_per_step_rank_max=flops_per_step,
        mfu=mfu,
        mfu_method=mfu_method,
        profiler_errors=profiler_errors,
        benchmark_options=asdict(options),
        environment=_environment(config, context),
    )


def _profile_callable(
    callback: Callable[[], None],
    device: torch.device,
    profile_steps: int,
) -> ProfilerSummary:
    if profile_steps <= 0:
        return ProfilerSummary(error="profiling disabled by profile_steps=0")
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            with_flops=True,
            acc_events=True,
        ) as profiler:
            for _ in range(profile_steps):
                callback()
                profiler.step()
        attention_keys: set[str] = set()
        communication_keys: set[str] = set()
        communication_microseconds = 0.0
        flops = 0.0
        for event in profiler.key_averages():
            key = str(event.key)
            normalized = key.lower()
            if any(token in normalized for token in ("flash", "attention", "scaled_dot_product")):
                attention_keys.add(key)
            if any(
                token in normalized
                for token in ("nccl", "all_reduce", "allreduce", "all_to_all", "alltoall", "c10d")
            ):
                communication_keys.add(key)
                communication_microseconds += float(
                    getattr(
                        event,
                        "device_time_total",
                        getattr(event, "cuda_time_total", 0.0),
                    )
                )
            event_flops = getattr(event, "flops", None)
            if event_flops:
                flops += float(event_flops)
        return ProfilerSummary(
            attention_kernels=tuple(sorted(attention_keys)),
            communication_operators=tuple(sorted(communication_keys)),
            communication_operator_seconds=communication_microseconds / 1.0e6,
            estimated_flops_per_step=(flops / profile_steps if flops > 0.0 else None),
        )
    except Exception as error:  # profiler support varies across secure builds
        return ProfilerSummary(error=f"{type(error).__name__}: {error}")


def _synthetic_vocab_maps(config: AppConfig) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for categorical in config.resolved.categorical_inputs:
        base_input = resolve_categorical_base_input(
            config.resolved.categorical_input_by_name,
            categorical.name,
        )
        if isinstance(base_input.encoding, ResolvedVocabEncoding):
            maps[categorical.name] = {"__synthetic__": 1}
    return maps


def _categorical_size(config: AppConfig, name: str) -> int:
    encoding = resolve_categorical_base_input(
        config.resolved.categorical_input_by_name,
        name,
    ).encoding
    if encoding.encoding == "hash":
        return encoding.num_buckets + 1
    if encoding.encoding == "identity":
        return encoding.num_buckets
    return 2


def _synthetic_ids(
    size: tuple[int, ...],
    num_embeddings: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    if num_embeddings <= 1:
        return torch.zeros(size, dtype=torch.long, device=device)
    return torch.randint(
        1,
        num_embeddings,
        size,
        dtype=torch.long,
        device=device,
        generator=generator,
    )


def _synthetic_feature_batch(
    config: AppConfig,
    device: torch.device,
    batch_size: int,
    sequence_length: int | None,
    seed: int,
) -> FeatureBatch:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    features: dict[str, Any] = {}
    for feature in config.features:
        if feature.kind == "categorical":
            features[feature.name] = _synthetic_ids(
                (batch_size,),
                _categorical_size(config, feature.name),
                device,
                generator,
            )
        else:
            shape = (batch_size, feature.dimension)
            values = torch.randn(shape, generator=generator, device=device)
            features[feature.name] = values[:, 0] if feature.dimension == 1 else values

    for sequence in config.sequences:
        length = sequence_length
        if length is None:
            length = sequence.max_length or 128
        if sequence.max_length is not None and length > sequence.max_length:
            raise ValueError(
                f"synthetic sequence_length={length} exceeds configured max_length="
                f"{sequence.max_length} for {sequence.name!r}"
            )
        tensor_fields: dict[str, Tensor] = {}
        for field_config in sequence.fields:
            qualified = field_config.qualified_name(sequence.name)
            if field_config.kind == "categorical":
                tensor_fields[field_config.name] = _synthetic_ids(
                    (batch_size, length),
                    _categorical_size(config, qualified),
                    device,
                    generator,
                )
            else:
                shape = (batch_size, length, field_config.dimension)
                values = torch.randn(shape, generator=generator, device=device)
                if field_config.dimension == 1:
                    values = values[:, :, 0]
                tensor_fields[field_config.name] = values
        features[sequence.name] = {
            "fields": tensor_fields,
            "lengths": torch.full(
                (batch_size,), length, dtype=torch.long, device=device
            ),
        }

    scenario_count = len(config.scenarios.names)
    scenario_id = torch.randint(
        0,
        scenario_count,
        (batch_size,),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    task_count = len(config.task_names)
    labels = torch.randint(
        0,
        2,
        (batch_size, task_count),
        dtype=torch.long,
        device=device,
        generator=generator,
    ).float()
    return FeatureBatch(
        features=features,
        labels=labels,
        label_mask=torch.ones_like(labels),
        scenario_id=scenario_id,
        group_id=[],
    )


def _id_embedding_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    modules: list[tuple[str, nn.Module]] = []
    seen: set[int] = set()
    for name, module in model.named_modules(remove_duplicate=False):
        if not isinstance(module, (nn.Embedding, ShardedEmbedding)):
            continue
        if ".embeddings." not in f".{name}":
            continue
        if id(module) in seen:
            continue
        modules.append((name, module))
        seen.add(id(module))
    return modules


class _SyntheticEmbedding(nn.Module):
    def __init__(self, source: nn.Embedding | ShardedEmbedding) -> None:
        super().__init__()
        self.embedding_dim = source.embedding_dim
        self.register_buffer(
            "_dtype_anchor",
            torch.empty(0, dtype=source.weight.dtype),
            persistent=False,
        )

    def forward(self, indices: Tensor) -> Tensor:
        return torch.zeros(
            (*indices.shape, self.embedding_dim),
            dtype=self._dtype_anchor.dtype,
            device=indices.device,
        )


def _replace_id_embeddings_with_synthetic(model: nn.Module) -> int:
    replacements: dict[int, _SyntheticEmbedding] = {}
    count = 0
    for name, module in list(model.named_modules(remove_duplicate=False)):
        if not isinstance(module, (nn.Embedding, ShardedEmbedding)):
            continue
        if ".embeddings." not in f".{name}":
            continue
        replacement = replacements.get(id(module))
        if replacement is None:
            replacement = _SyntheticEmbedding(module)
            replacements[id(module)] = replacement
            count += 1
        parent_name, separator, child_name = name.rpartition(".")
        if not separator:
            raise RuntimeError("cannot replace a root embedding module")
        parent = model.get_submodule(parent_name)
        parent._modules[child_name] = replacement
    return count


def _trace(
    *,
    step: int,
    rows: int,
    input_tokens: int,
    padded_token_slots: int | None = None,
    step_seconds: float,
    forward_seconds: float = 0.0,
    backward_seconds: float = 0.0,
    sparse_sync_seconds: float = 0.0,
    optimizer_seconds: float = 0.0,
    dataloader_wait_seconds: float = 0.0,
    h2d_seconds: float = 0.0,
    sparse_local_rows: int = 0,
    sparse_global_rows: int = 0,
    sparse_payload_bytes: int = 0,
    active_ranks: int = 1,
) -> TrainStepTrace:
    return TrainStepTrace(
        step=step,
        rank_active=True,
        active_ranks=active_ranks,
        rows=rows,
        input_tokens=input_tokens,
        padded_token_slots=(
            input_tokens if padded_token_slots is None else padded_token_slots
        ),
        step_seconds=step_seconds,
        dataloader_wait_seconds=dataloader_wait_seconds,
        h2d_seconds=h2d_seconds,
        forward_seconds=forward_seconds,
        backward_seconds=backward_seconds,
        sparse_sync_seconds=sparse_sync_seconds,
        optimizer_seconds=optimizer_seconds,
        sparse_local_rows=sparse_local_rows,
        sparse_global_rows=sparse_global_rows,
        sparse_payload_bytes=sparse_payload_bytes,
    )


def _benchmark_data(
    config: AppConfig,
    context: DistributedContext,
    options: BenchmarkOptions,
) -> LocalBenchmarkSummary:
    vocab_maps = load_vocab_maps(config)
    non_blocking = _non_blocking_transfer(config, "train", context.device)
    iterator = iter(
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
    collector = _TraceCollector(options.warmup_steps, options.measured_steps, context.device)
    total_steps = options.warmup_steps + options.measured_steps
    for step in range(1, total_steps + 1):
        started = perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        elapsed = perf_counter() - started
        collector.observe(
            _trace(
                step=step,
                rows=int(batch.scenario_id.size(0)),
                input_tokens=_batch_input_token_count(batch),
                padded_token_slots=_batch_padded_token_slots(batch),
                step_seconds=elapsed,
                dataloader_wait_seconds=elapsed,
                active_ranks=context.world_size,
            ),
            host_batch_bytes=_feature_batch_nbytes(batch),
        )
    return collector.finish(context.rank, ProfilerSummary())


def _make_embedding_ids(
    module: nn.Embedding | ShardedEmbedding,
    count: int,
    distribution: IdDistribution,
    exponent: float,
    seed: int,
    device: torch.device,
) -> Tensor:
    if module.num_embeddings <= 1:
        return torch.zeros(count, dtype=torch.long, device=device)
    if distribution == "uniform":
        generator = torch.Generator(device=device.type)
        generator.manual_seed(seed)
        return torch.randint(
            1,
            module.num_embeddings,
            (count,),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
    try:
        import numpy as np

        values = np.random.default_rng(seed).zipf(exponent, size=count)
        values = 1 + ((values - 1) % (module.num_embeddings - 1))
        return torch.as_tensor(values, dtype=torch.long, device=device)
    except ImportError as error:
        raise RuntimeError("zipf embedding benchmark requires NumPy") from error


def _benchmark_embedding(
    config: AppConfig,
    context: DistributedContext,
    options: BenchmarkOptions,
) -> LocalBenchmarkSummary:
    sharded_mode = (
        getattr(config.training, "embedding_distribution", "replicated")
        == "sharded"
    )
    model = build_model(config, _synthetic_vocab_maps(config))
    named_embeddings = _id_embedding_modules(model)
    if not named_embeddings:
        raise ValueError("model has no industrial ID embedding tables to benchmark")
    collection = nn.ModuleList([module for _name, module in named_embeddings]).to(context.device)
    del model
    gc.collect()
    parameter_groups = _classify_model_parameters(collection)
    synchronizer: _ReplicatedSparseGradientSynchronizer | None = None
    if not sharded_mode:
        _synchronize_sparse_parameter_replicas(context, parameter_groups.sparse_sync)
        synchronizer = _ReplicatedSparseGradientSynchronizer(
            context, parameter_groups.sparse_sync
        )
    parameters = [parameter for parameter in collection.parameters() if parameter.requires_grad]
    optimizer: torch.optim.Optimizer
    if sharded_mode:
        optimizer = ShardedAdagrad(
            parameters,
            lr=config.training.lr_sparse or config.training.lr_dense,
            lr_decay=config.training.adagrad_lr_decay,
            weight_decay=config.training.adagrad_weight_decay,
            initial_accumulator_value=config.training.adagrad_initial_accumulator_value,
            eps=config.training.adagrad_eps,
        )
    else:
        optimizer = torch.optim.Adagrad(
            parameters,
            lr=config.training.lr_sparse or config.training.lr_dense,
            lr_decay=config.training.adagrad_lr_decay,
            weight_decay=config.training.adagrad_weight_decay,
            initial_accumulator_value=config.training.adagrad_initial_accumulator_value,
            eps=config.training.adagrad_eps,
        )
    ids = [
        _make_embedding_ids(
            module,
            options.embedding_lookups_per_table,
            options.id_distribution,
            options.zipf_exponent,
            options.seed + context.rank * 1009 + index,
            context.device,
        )
        for index, module in enumerate(collection)
    ]
    collector = _TraceCollector(options.warmup_steps, options.measured_steps, context.device)

    def run_step(step: int, collect: bool) -> None:
        _sync_device(context.device)
        step_started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        forward_started = perf_counter()
        with _autocast_context(config, context.device):
            loss = sum(
                module(table_ids).float().square().mean()
                for module, table_ids in zip(collection, ids)
            )
        _sync_device(context.device)
        forward_seconds = perf_counter() - forward_started
        backward_started = perf_counter()
        loss.backward()
        _sync_device(context.device)
        backward_seconds = perf_counter() - backward_started
        sparse_started = perf_counter()
        if synchronizer is not None:
            sparse_stats = synchronizer.synchronize(rank_active=True)
            sparse_local_rows = sparse_stats.local_rows
            sparse_global_rows = sparse_stats.global_rows
            sparse_payload_bytes = sparse_stats.logical_payload_bytes
        else:
            routed_stats = [
                module.consume_communication_stats()
                for module in collection
                if isinstance(module, ShardedEmbedding)
            ]
            sparse_local_rows = sum(item.local_unique_ids for item in routed_stats)
            sparse_global_rows = sum(item.owner_unique_ids for item in routed_stats)
            sparse_payload_bytes = sum(
                item.total_communication_bytes for item in routed_stats
            )
        _sync_device(context.device)
        sparse_seconds = perf_counter() - sparse_started
        optimizer_started = perf_counter()
        optimizer.step()
        _sync_device(context.device)
        optimizer_seconds = perf_counter() - optimizer_started
        if collect:
            collector.observe(
                _trace(
                    step=step,
                    rows=config.training.batch_size,
                    input_tokens=sum(item.numel() for item in ids),
                    step_seconds=perf_counter() - step_started,
                    forward_seconds=forward_seconds,
                    backward_seconds=backward_seconds,
                    sparse_sync_seconds=sparse_seconds,
                    optimizer_seconds=optimizer_seconds,
                    sparse_local_rows=sparse_local_rows,
                    sparse_global_rows=sparse_global_rows,
                    sparse_payload_bytes=sparse_payload_bytes,
                    active_ranks=context.world_size,
                )
            )

    total_steps = options.warmup_steps + options.measured_steps
    for step in range(1, total_steps + 1):
        run_step(step, collect=True)
    profiler = _profile_callable(
        lambda: run_step(total_steps + 1, collect=False),
        context.device,
        options.profile_steps,
    )
    return collector.finish(context.rank, profiler)


def _benchmark_compute(
    config: AppConfig,
    context: DistributedContext,
    options: BenchmarkOptions,
) -> LocalBenchmarkSummary:
    # Compute-only must not briefly materialize industrial embedding tables
    # before replacing their lookups with zero-cost shape-preserving modules.
    model = build_model(
        config,
        _synthetic_vocab_maps(config),
        embedding_size_override=2,
    )
    replaced = _replace_id_embeddings_with_synthetic(model)
    if replaced == 0:
        raise ValueError("compute benchmark could not identify ID embedding modules")
    base_model = model.to(context.device)
    model_for_forward = _prepare_forward_model(config, base_model, context)
    parameters = [parameter for parameter in base_model.parameters() if parameter.requires_grad]
    optimizer = _build_dense_optimizer(
        parameters,
        config,
        context.device,
    )
    batch = _synthetic_feature_batch(
        config,
        context.device,
        options.batch_size or config.training.batch_size,
        options.sequence_length,
        options.seed + context.rank,
    )
    collector = _TraceCollector(options.warmup_steps, options.measured_steps, context.device)
    model_for_forward.train()

    def run_step(step: int, collect: bool) -> None:
        _sync_device(context.device)
        step_started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        forward_started = perf_counter()
        with _autocast_context(config, context.device):
            output = model_for_forward(batch.features, batch.scenario_id)
            loss, _numerator, _denominator = _loss_terms_from_batch(
                output,
                batch,
                moe_loss_weight=config.model.sparse_moe_loss_weight,
                loss_reduction=config.training.loss_reduction,
                rank_active=True,
                active_rank_count=context.world_size,
            )
        _sync_device(context.device)
        forward_seconds = perf_counter() - forward_started
        backward_started = perf_counter()
        loss.backward()
        _sync_device(context.device)
        backward_seconds = perf_counter() - backward_started
        optimizer_started = perf_counter()
        _step_sparse_moe_controllers(
            base_model,
            rank_active=True,
            active_rank_count=context.world_size,
        )
        optimizer.step()
        _sync_device(context.device)
        optimizer_seconds = perf_counter() - optimizer_started
        if collect:
            collector.observe(
                _trace(
                    step=step,
                    rows=int(batch.scenario_id.size(0)),
                    input_tokens=_batch_input_token_count(batch),
                    padded_token_slots=_batch_padded_token_slots(batch),
                    step_seconds=perf_counter() - step_started,
                    forward_seconds=forward_seconds,
                    backward_seconds=backward_seconds,
                    optimizer_seconds=optimizer_seconds,
                    active_ranks=context.world_size,
                )
            )

    total_steps = options.warmup_steps + options.measured_steps
    for step in range(1, total_steps + 1):
        run_step(step, collect=True)
    profiler = _profile_callable(
        lambda: run_step(total_steps + 1, collect=False),
        context.device,
        options.profile_steps,
    )
    return collector.finish(context.rank, profiler)


def _benchmark_end_to_end(
    config: AppConfig,
    context: DistributedContext,
    options: BenchmarkOptions,
) -> LocalBenchmarkSummary:
    collector = _TraceCollector(options.warmup_steps, options.measured_steps, context.device)
    total_steps = options.warmup_steps + options.measured_steps
    train_mdl(
        config,
        max_steps=total_steps,
        save_checkpoint=False,
        log_steps=False,
        step_observer=collector.observe,
    )
    profiler = _profile_callable(
        lambda: train_mdl(
            config,
            max_steps=1,
            save_checkpoint=False,
            log_steps=False,
        ),
        context.device,
        options.profile_steps,
    )
    return collector.finish(context.rank, profiler)


def run_benchmark(config: AppConfig, options: BenchmarkOptions) -> BenchmarkReport:
    """Run one benchmark mode and aggregate a rank-max report."""

    options.validate()
    context = _setup_distributed(config)
    try:
        torch.manual_seed(options.seed + context.rank)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(options.seed + context.rank)
            torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
            torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
            torch.set_float32_matmul_precision(
                "high" if config.runtime.allow_tf32 else "highest"
            )
        if options.mode in {"compute", "end-to-end"}:
            _attention_runtime_description(config, context.device)
        if options.mode == "data":
            local = _benchmark_data(config, context, options)
        elif options.mode == "embedding":
            local = _benchmark_embedding(config, context, options)
        elif options.mode == "compute":
            local = _benchmark_compute(config, context, options)
        else:
            local = _benchmark_end_to_end(config, context, options)
        report = _build_report(config, context, options, local)
        if (
            config.runtime.attention_backend == "flash"
            and options.mode in {"compute", "end-to-end"}
        ):
            if options.profile_steps <= 0:
                raise RuntimeError(
                    "FlashAttention verification requires --profile-steps >= 1"
                )
            if not any("flash" in key.lower() for key in report.attention_kernels):
                details = "; ".join(report.profiler_errors) or "no Flash kernel event"
                raise RuntimeError(
                    "runtime.attention_backend=flash was requested but the profiler did "
                    f"not observe a Flash kernel: {details}"
                )
        return report
    finally:
        _cleanup_distributed(context)


def write_benchmark_report(report: BenchmarkReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(report.to_json() + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
