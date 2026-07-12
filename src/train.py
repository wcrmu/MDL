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
from torch.distributed.algorithms.join import Join
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


def _partition_embedding_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    embedding_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    dense: list[nn.Parameter] = []
    sparse: list[nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in embedding_ids:
            sparse.append(parameter)
        else:
            dense.append(parameter)
    return dense, sparse


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
def _step_sparse_moe_controllers(module: nn.Module) -> None:
    for item in module.modules():
        if not isinstance(item, SparseMoEPerTokenFFN):
            continue
        active_ratio = item.active_ratio(item.regularization_coefficient).clone()
        if torch_dist.is_available() and torch_dist.is_initialized():
            torch_dist.all_reduce(active_ratio, op=torch_dist.ReduceOp.SUM)
            active_ratio.div_(float(torch_dist.get_world_size()))
        item.step_regularization_controller(active_ratio)


def _loss_terms_from_batch(
    output: dict[str, Tensor],
    batch: FeatureBatch,
    moe_loss_weight: float = 0.0,
    loss_reduction: str = "sum",
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
        total_loss = total_loss + moe_loss_weight * moe_loss
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
        forward_model = _maybe_compile_model(config, base_model)
        model: nn.Module = forward_model
        if context.enabled:
            model = DistributedDataParallel(
                forward_model,
                device_ids=[context.local_rank] if device.type == "cuda" else None,
                output_device=context.local_rank if device.type == "cuda" else None,
                find_unused_parameters=True,
            )

        dense_params, sparse_params = _partition_embedding_parameters(base_model)
        optimizers: list[torch.optim.Optimizer] = []
        if dense_params:
            optimizers.append(
                torch.optim.RMSprop(
                    dense_params,
                    lr=config.training.lr_dense,
                    alpha=config.training.rmsprop_alpha,
                    momentum=config.training.rmsprop_momentum,
                )
            )
        if sparse_params:
            _mark_sparse_invariant_checks_explicitly_disabled()
            sparse_lr = config.training.lr_sparse or config.training.lr_dense
            optimizers.append(
                torch.optim.Adagrad(
                    sparse_params,
                    lr=sparse_lr,
                    lr_decay=config.training.adagrad_lr_decay,
                    weight_decay=config.training.adagrad_weight_decay,
                    initial_accumulator_value=config.training.adagrad_initial_accumulator_value,
                    eps=config.training.adagrad_eps,
                )
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
        join_context = Join([model]) if context.enabled else nullcontext()
        _sync_device(device)
        start = perf_counter()
        with join_context:
            for batch in iter_feature_batches(
                config,
                "train",
                vocab_maps,
                require_labels=True,
                shard_rank=context.rank,
                shard_world_size=context.world_size,
                pin_memory=non_blocking,
                include_group_id=False,
            ):
                batch = move_feature_batch(batch, device, non_blocking=non_blocking)
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
                    )
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                else:
                    loss.backward()
                _step_sparse_moe_controllers(base_model)
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
                rows += int(batch.scenario_id.size(0))
                last_loss = float(loss.detach().cpu().item())
                last_loss_numerator = float(loss_numerator.detach().cpu().item())
                last_loss_denominator = float(loss_denominator.detach().cpu().item())
                if log_steps and context.rank == 0:
                    print(f"Train step | step={steps} | loss={last_loss:.6f}")
                if max_steps is not None and steps >= max_steps:
                    break
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
