from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import import_module
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator

import torch
import torch.distributed as torch_dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from .config import AppConfig, ParquetSplitConfig, ReaderConfig
from .dataloader import (
    FeatureBatch,
    _require_pyarrow,
    iter_flat_tables,
    move_feature_batch,
    pin_feature_batch,
    table_to_feature_batch,
)
from .features import load_vocab_maps, vocab_strategy_fingerprint
from .model import build_model
from .modules.mlp import SparseMoEPerTokenFFN


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
class PredictResult:
    rows: int
    output_path: Path | None


@dataclass(frozen=True)
class EvaluateResult:
    rows: int
    group_metric_name: str
    metrics: dict[str, dict[str, float | None]]


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
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    enabled = world_size > 1
    device = _select_device(config, local_rank if enabled else None)
    initialized_here = False

    if enabled and not torch_dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch_dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True

    return DistributedContext(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        initialized_here=initialized_here,
    )


def _cleanup_distributed(context: DistributedContext) -> None:
    if context.initialized_here and torch_dist.is_initialized():
        torch_dist.destroy_process_group()


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
) -> Iterator[object]:
    yield from iter_flat_tables(
        config,
        split_name,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
    )


def _slice_table(table: object, batch_size: int) -> Iterator[object]:
    for offset in range(0, table.num_rows, batch_size):
        yield table.slice(offset, batch_size)


def _iter_batch_tables(
    config: AppConfig,
    split_name: str,
    shard_rank: int,
    shard_world_size: int,
) -> Iterator[object]:
    batch_size = config.training.batch_size
    for table in iter_candidate_tables(
        config,
        split_name,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
    ):
        yield from _slice_table(table, batch_size)


def _prepare_feature_batch(
    config: AppConfig,
    split: ParquetSplitConfig,
    table: object,
    vocab_maps: dict[str, dict[str, int]],
    require_labels: bool,
    pin_memory: bool,
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
    return pin_feature_batch(batch) if pin_memory else batch


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
    table_iter = _iter_batch_tables(
        config,
        split_name,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
    )

    if reader.num_workers <= 0 and reader.prefetch_batches <= 0:
        for table in table_iter:
            yield _prepare_feature_batch(
                config,
                split,
                table,
                vocab_maps,
                require_labels,
                pin_memory,
                include_group_id,
            )
        return

    worker_count = max(1, reader.num_workers)
    max_pending = max(1, reader.prefetch_batches, worker_count)
    pending: deque[Future[FeatureBatch]] = deque()
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="mdl-reader") as executor:
        exhausted = False
        while not exhausted and len(pending) < max_pending:
            try:
                table = next(table_iter)
            except StopIteration:
                exhausted = True
                break
            pending.append(
                executor.submit(
                    _prepare_feature_batch,
                    config,
                    split,
                    table,
                    vocab_maps,
                    require_labels,
                    pin_memory,
                    include_group_id,
                )
            )

        while pending:
            future = pending.popleft()
            if not exhausted:
                try:
                    table = next(table_iter)
                except StopIteration:
                    exhausted = True
                else:
                    pending.append(
                        executor.submit(
                            _prepare_feature_batch,
                            config,
                            split,
                            table,
                            vocab_maps,
                            require_labels,
                            pin_memory,
                            include_group_id,
                        )
                    )
            yield future.result()


def _classify_model_parameters(model: nn.Module) -> _ParameterGroups:
    """Separate optimizer ownership from native sparse-gradient ownership.

    All ``nn.Embedding`` parameters retain the repository's existing Adagrad
    optimizer assignment. Only embeddings constructed with ``sparse=True``
    need to bypass DDP's reducer, since standard NCCL cannot all-reduce their
    COO gradients.
    """

    embedding_ids: set[int] = set()
    sparse_gradient_ids: set[int] = set()
    for module in model.modules():
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
    sparse_sync: list[_NamedSparseParameter] = []
    seen_sparse_ids: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in embedding_ids:
            embeddings.append(parameter)
        else:
            dense.append(parameter)
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
    )


def _partition_embedding_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Compatibility wrapper returning the two optimizer parameter groups."""

    groups = _classify_model_parameters(model)
    return list(groups.dense_optimizer), list(groups.embedding_optimizer)


def _sparse_parameter_descriptors(
    sparse_parameters: tuple[_NamedSparseParameter, ...],
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(
        (ref.name, tuple(ref.parameter.shape), str(ref.parameter.dtype))
        for ref in sparse_parameters
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
    return torch.compile(model)


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
    value = torch.tensor(
        int(rank_active),
        dtype=torch.long,
        device=context.device,
    )
    torch_dist.all_reduce(value, op=torch_dist.ReduceOp.SUM)
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
            f"adagrad_state_mib={state_bytes / (1024 ** 2):.2f}"
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


def _mark_sparse_invariant_checks_explicitly_disabled() -> None:
    checker = getattr(torch.sparse, "check_sparse_tensor_invariants", None)
    if checker is not None and not checker.is_enabled():
        checker.disable()


@torch.no_grad()
def _clip_grad_norm(parameters: list[nn.Parameter], max_norm: float) -> Tensor:
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
    if not grads:
        return torch.tensor(0.0)

    first = grads[0]
    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(grad.detach(), 2.0).to(first.device) for grad in grads]),
        2.0,
    )
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for parameter in parameters:
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                parameter.grad._values().mul_(clip_coef.to(parameter.grad.device))
            else:
                parameter.grad.mul_(clip_coef.to(parameter.grad.device))
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
    if batch.labels is None or batch.label_mask is None:
        raise ValueError("training batch must contain labels and label_mask")
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
    weights = batch.label_mask.to(device=logits.device, dtype=element_loss.dtype)
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
) -> TrainResult:
    if config.training.sparse_update_mode == "external_parameter_server":
        adapter = _load_external_train_adapter(config.training.sparse_parameter_server_adapter)
        return _coerce_train_result(adapter(config=config, max_steps=max_steps))

    context = _setup_distributed(config)
    try:
        device = context.device
        vocab_maps = load_vocab_maps(config)
        base_model = build_model(config, vocab_maps).to(device)
        parameter_groups = _classify_model_parameters(base_model)
        _synchronize_sparse_parameter_replicas(
            context,
            parameter_groups.sparse_sync,
        )
        forward_model = _maybe_compile_model(config, base_model)
        model: nn.Module = forward_model
        if context.enabled:
            _exclude_sparse_parameters_from_ddp(
                forward_model,
                parameter_groups.sparse_sync,
            )
            model = DistributedDataParallel(
                forward_model,
                device_ids=[context.local_rank] if device.type == "cuda" else None,
                output_device=context.local_rank if device.type == "cuda" else None,
                find_unused_parameters=True,
            )

        dense_params = list(parameter_groups.dense_optimizer)
        sparse_params = list(parameter_groups.embedding_optimizer)
        dense_optimizer: torch.optim.Optimizer | None = None
        embedding_optimizer: torch.optim.Optimizer | None = None
        optimizers: list[torch.optim.Optimizer] = []
        if dense_params:
            dense_optimizer = torch.optim.RMSprop(
                dense_params,
                lr=config.training.lr_dense,
                alpha=config.training.rmsprop_alpha,
                momentum=config.training.rmsprop_momentum,
            )
            optimizers.append(dense_optimizer)
        if sparse_params:
            _mark_sparse_invariant_checks_explicitly_disabled()
            sparse_lr = config.training.lr_sparse or config.training.lr_dense
            embedding_optimizer = torch.optim.Adagrad(
                sparse_params,
                lr=sparse_lr,
                lr_decay=config.training.adagrad_lr_decay,
                weight_decay=config.training.adagrad_weight_decay,
                initial_accumulator_value=config.training.adagrad_initial_accumulator_value,
                eps=config.training.adagrad_eps,
            )
            optimizers.append(embedding_optimizer)
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
        optimizer_base_lrs = [
            [float(group["lr"]) for group in optimizer.param_groups]
            for optimizer in optimizers
        ]
        lr_decay_steps = _resolve_lr_decay_steps(config, max_steps)
        scaler = _make_grad_scaler(config, device)
        non_blocking = _non_blocking_transfer(config, "train", device)

        steps = 0
        rows = 0
        last_loss = 0.0
        last_loss_numerator = 0.0
        last_loss_denominator = 0.0
        model.train()
        _sync_device(device)
        start = perf_counter()
        batch_iterator = iter(
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
        last_device_batch: FeatureBatch | None = None
        while max_steps is None or steps < max_steps:
            try:
                local_batch = next(batch_iterator)
            except StopIteration:
                local_batch = None
            rank_active = local_batch is not None
            active_ranks = _active_rank_count(context, rank_active)
            if active_ranks == 0:
                break
            if steps == 0 and context.enabled and active_ranks != context.world_size:
                raise RuntimeError(
                    "replicated sparse DDP requires every rank to provide an initial batch; "
                    "reduce world_size or choose a finer reader.shard_unit"
                )
            if rank_active:
                if local_batch is None:
                    raise AssertionError("active rank is missing its training batch")
                batch = move_feature_batch(
                    local_batch,
                    device,
                    non_blocking=non_blocking,
                )
                last_device_batch = batch
            else:
                if last_device_batch is None:
                    raise RuntimeError("inactive rank has no batch available for zero-loss replay")
                batch = last_device_batch

            lr_multiplier = _lr_schedule_multiplier(config, steps + 1, lr_decay_steps)
            _set_optimizer_lrs(optimizers, optimizer_base_lrs, lr_multiplier)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
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
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            sparse_sync_stats = sparse_synchronizer.synchronize(rank_active=rank_active)
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
                _clip_grad_norm(sparse_params, config.training.sparse_clip_norm)
            if scaler.is_enabled():
                for optimizer in optimizers:
                    scaler.step(optimizer)
                scaler.update()
            else:
                for optimizer in optimizers:
                    optimizer.step()
            steps += 1
            if rank_active:
                rows += int(batch.scenario_id.size(0))
            last_loss = float(loss.detach().cpu().item())
            last_loss_numerator = float(loss_numerator.detach().cpu().item())
            last_loss_denominator = float(loss_denominator.detach().cpu().item())
            if log_steps and context.rank == 0:
                payload_mib = sparse_sync_stats.logical_payload_bytes / (1024 ** 2)
                print(
                    f"Train step | step={steps} | loss={last_loss:.6f} "
                    f"active_ranks={active_ranks}/{context.world_size} "
                    f"sparse_local_rows={sparse_sync_stats.local_rows} "
                    f"sparse_global_rows={sparse_sync_stats.global_rows} "
                    f"sparse_payload_mib={payload_mib:.2f}"
                )
        _sync_device(device)
        elapsed = perf_counter() - start

        local_result = TrainResult(steps=steps, rows=rows, last_loss=last_loss, elapsed_seconds=elapsed)
        result = _aggregate_train_result(
            context,
            local_result,
            last_loss_numerator,
            last_loss_denominator,
        )

        if save_checkpoint and config.training.checkpoint_path and context.rank == 0:
            checkpoint_path = Path(config.training.checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": base_model.state_dict(),
                    "model_name": config.model.name,
                    "task_names": config.task_names,
                    "vocab_strategy_hash": vocab_strategy_fingerprint(config),
                },
                checkpoint_path,
            )
        return result
    finally:
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


def _load_inference_model(
    config: AppConfig,
    device: torch.device,
    checkpoint_path: str | None,
    allow_random_init: bool,
) -> tuple[nn.Module, dict[str, dict[str, int]]]:
    vocab_maps = load_vocab_maps(config)
    base_model = build_model(config, vocab_maps).to(device)
    resolved_checkpoint_path = checkpoint_path or config.training.checkpoint_path
    if resolved_checkpoint_path is None and not allow_random_init:
        raise ValueError(
            "evaluation requires a checkpoint; pass --checkpoint-path, set "
            "training.checkpoint_path, or pass --allow-random-init explicitly"
        )
    if resolved_checkpoint_path is not None:
        checkpoint = torch.load(resolved_checkpoint_path, map_location=device)
        if checkpoint.get("model_name") not in {None, config.model.name}:
            raise ValueError("checkpoint model_name does not match current config")
        checkpoint_task_names = checkpoint.get("task_names")
        if checkpoint_task_names is not None and list(checkpoint_task_names) != config.task_names:
            raise ValueError("checkpoint task_names do not match current config")
        expected_hash = vocab_strategy_fingerprint(config)
        if checkpoint.get("vocab_strategy_hash") != expected_hash:
            raise ValueError("checkpoint vocab_strategy_hash does not match current config")
        base_model.load_state_dict(checkpoint["model_state_dict"])
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
    group_metric_name: str = "qauc",
) -> EvaluateResult:
    if split_name not in {"train", "test"}:
        raise ValueError("evaluation split must be train or test")
    if group_metric_name not in {"qauc", "uauc"}:
        raise ValueError("group_metric_name must be qauc or uauc")
    split = config.data.train if split_name == "train" else config.data.test
    if split is None:
        raise ValueError(f"split {split_name!r} is not configured")
    if list(split.labels) != config.task_names:
        raise ValueError(
            f"data.{split_name}.labels must declare the training tasks in the same order: "
            + ", ".join(config.task_names)
        )
    if split.group_id is None:
        raise ValueError(
            f"data.{split_name}.group_id is required for {group_metric_name.upper()}"
        )

    device = _select_device(config)
    model, vocab_maps = _load_inference_model(
        config,
        device,
        checkpoint_path,
        allow_random_init,
    )
    scores_by_task: list[list[Tensor]] = [[] for _ in config.task_names]
    labels_by_task: list[list[Tensor]] = [[] for _ in config.task_names]
    groups_by_task: list[list[str]] = [[] for _ in config.task_names]
    scenarios_by_task: list[list[Tensor]] = [[] for _ in config.task_names]
    rows = 0
    non_blocking = _non_blocking_transfer(config, split_name, device)
    for batch_index, batch in enumerate(
        iter_feature_batches(
            config,
            split_name,
            vocab_maps,
            require_labels=True,
            pin_memory=non_blocking,
            include_group_id=True,
        )
    ):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_feature_batch(batch, device, non_blocking=non_blocking)
        with _autocast_context(config, device):
            logits = model(batch.features, batch.scenario_id)["logits"]
        if batch.labels is None or batch.label_mask is None:
            raise RuntimeError("evaluation batch did not contain labels")
        probabilities = torch.sigmoid(logits.float()).cpu()
        labels = batch.labels.float().cpu()
        label_mask = batch.label_mask.bool().cpu()
        raw_scenarios = batch.scenario_id.cpu()
        if raw_scenarios.ndim == 1:
            scenario_membership = torch.nn.functional.one_hot(
                raw_scenarios.long(),
                num_classes=len(config.scenarios.names),
            ).bool()
        else:
            scenario_membership = raw_scenarios.bool()
        rows += int(labels.size(0))
        for task_index in range(len(config.task_names)):
            valid = label_mask[:, task_index]
            scores_by_task[task_index].append(probabilities[valid, task_index])
            labels_by_task[task_index].append(labels[valid, task_index])
            scenarios_by_task[task_index].append(scenario_membership[valid])
            groups_by_task[task_index].extend(
                group_id
                for group_id, keep in zip(batch.group_id, valid.tolist())
                if keep
            )

    metrics: dict[str, dict[str, float | None]] = {}
    for task_index, task_name in enumerate(config.task_names):
        task_scores = torch.cat(scores_by_task[task_index]) if scores_by_task[task_index] else torch.empty(0)
        task_labels = torch.cat(labels_by_task[task_index]) if labels_by_task[task_index] else torch.empty(0)
        task_scenarios = (
            torch.cat(scenarios_by_task[task_index])
            if scenarios_by_task[task_index]
            else torch.empty(0, len(config.scenarios.names), dtype=torch.bool)
        )
        task_groups = groups_by_task[task_index]
        values: dict[str, float | None] = {
            "auc": _binary_auc(task_scores, task_labels),
            group_metric_name: _group_auc(task_scores, task_labels, task_groups),
        }
        for scenario in range(len(config.scenarios.names)):
            selected = task_scenarios[:, scenario]
            scenario_groups = [
                group_id
                for group_id, keep in zip(task_groups, selected.tolist())
                if keep
            ]
            values[f"scenario_{scenario}_auc"] = _binary_auc(
                task_scores[selected], task_labels[selected]
            )
            values[f"scenario_{scenario}_{group_metric_name}"] = _group_auc(
                task_scores[selected],
                task_labels[selected],
                scenario_groups,
            )
        metrics[task_name] = values
    return EvaluateResult(
        rows=rows,
        group_metric_name=group_metric_name,
        metrics=metrics,
    )


@torch.no_grad()
def predict_mdl(
    config: AppConfig,
    checkpoint_path: str | None = None,
    output_path: str | None = None,
    max_batches: int | None = None,
    allow_random_init: bool = False,
) -> PredictResult:
    pa, _pc, _ds, pq = _require_pyarrow()
    device = _select_device(config)
    vocab_maps = load_vocab_maps(config)
    base_model = build_model(config, vocab_maps).to(device)
    resolved_checkpoint_path = checkpoint_path or config.training.checkpoint_path
    if resolved_checkpoint_path is None and not allow_random_init:
        raise ValueError(
            "prediction requires a checkpoint; pass --checkpoint-path, set "
            "training.checkpoint_path, or pass --allow-random-init explicitly"
        )
    if resolved_checkpoint_path is not None:
        checkpoint = torch.load(resolved_checkpoint_path, map_location=device)
        if checkpoint.get("model_name") not in {None, config.model.name}:
            raise ValueError("checkpoint model_name does not match current config")
        checkpoint_task_names = checkpoint.get("task_names")
        if checkpoint_task_names is not None and list(checkpoint_task_names) != config.task_names:
            raise ValueError("checkpoint task_names do not match current config")
        expected_hash = vocab_strategy_fingerprint(config)
        if checkpoint.get("vocab_strategy_hash") != expected_hash:
            raise ValueError("checkpoint vocab_strategy_hash does not match current config")
        base_model.load_state_dict(checkpoint["model_state_dict"])
    model = _maybe_compile_model(config, base_model)
    base_model.eval()
    model.eval()

    rows: list[dict[str, object]] = []
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
        for group_id, scores in zip(batch.group_id, probabilities):
            row = {"group_id": group_id}
            row.update({task: float(score) for task, score in zip(config.task_names, scores)})
            rows.append(row)

    path = Path(output_path) if output_path else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        columns: dict[str, list[object]] = {"group_id": [row["group_id"] for row in rows]}
        for task in config.task_names:
            columns[task] = [row[task] for row in rows]
        pq.write_table(pa.table(columns), path)
    return PredictResult(rows=len(rows), output_path=path)
