from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from importlib import import_module
import os
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.distributed as torch_dist
from torch import Tensor, nn
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel

from .config import AppConfig
from .data import AggParquetScanner, ParquetScanner, _require_pyarrow, required_columns_for_split
from .features import vocab_strategy_fingerprint
from .model import build_model
from .tensorize import FeatureBatch, move_feature_batch, table_to_feature_batch
from .vocab import load_vocab_maps


@dataclass(frozen=True)
class TrainResult:
    steps: int
    last_loss: float


@dataclass(frozen=True)
class PredictResult:
    rows: int
    output_path: Path | None


ExternalTrainAdapter = Callable[..., TrainResult | dict[str, Any]]


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
        )
    raise TypeError("external training adapter must return TrainResult or a dict")


def iter_candidate_tables(
    config: AppConfig,
    split_name: str,
    shard_rank: int = 0,
    shard_world_size: int = 1,
) -> Iterator[object]:
    split = config.data.train if split_name == "train" else config.data.test
    if split is None:
        raise ValueError(f"split {split_name!r} is not configured")
    columns = required_columns_for_split(config, split)
    if split.format == "agg_parquet":
        scanner = AggParquetScanner(
            split,
            columns,
            shard_rank=shard_rank,
            shard_world_size=shard_world_size,
        )
        yield from scanner.iter_candidate_tables()
    else:
        scanner = ParquetScanner(
            split,
            columns,
            shard_rank=shard_rank,
            shard_world_size=shard_world_size,
        )
        yield from scanner.iter_tables()


def _slice_table(table: object, batch_size: int) -> Iterator[object]:
    for offset in range(0, table.num_rows, batch_size):
        yield table.slice(offset, batch_size)


def iter_feature_batches(
    config: AppConfig,
    split_name: str,
    vocab_maps: dict[str, dict[str, int]],
    require_labels: bool,
    shard_rank: int = 0,
    shard_world_size: int = 1,
) -> Iterator[FeatureBatch]:
    batch_size = config.training.batch_size
    if split_name == "train" and config.data.train.reader.batch_size_candidates is not None:
        batch_size = config.data.train.reader.batch_size_candidates
    for table in iter_candidate_tables(
        config,
        split_name,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
    ):
        for batch_table in _slice_table(table, batch_size):
            yield table_to_feature_batch(config, batch_table, vocab_maps, require_labels=require_labels)


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


def _loss_from_batch(output: dict[str, Tensor], batch: FeatureBatch) -> Tensor:
    if batch.labels is None or batch.label_mask is None:
        raise ValueError("training batch must contain labels and label_mask")
    logits = output["logits"]
    if logits.shape != batch.labels.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} does not match labels {tuple(batch.labels.shape)}"
        )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        batch.labels,
        reduction="none",
    )
    weights = batch.label_mask.to(device=logits.device, dtype=loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def train_mdl(config: AppConfig, max_steps: int | None = None) -> TrainResult:
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
                    alpha=0.99999,
                    momentum=0.0,
                )
            )
        if sparse_params:
            sparse_lr = config.training.lr_sparse or config.training.lr_dense
            optimizers.append(torch.optim.Adagrad(sparse_params, lr=sparse_lr))

        steps = 0
        last_loss = 0.0
        model.train()
        join_context = Join([model]) if context.enabled else nullcontext()
        with join_context:
            for batch in iter_feature_batches(
                config,
                "train",
                vocab_maps,
                require_labels=True,
                shard_rank=context.rank,
                shard_world_size=context.world_size,
            ):
                batch = move_feature_batch(batch, device)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                output = model(batch.features, batch.scenario_id)
                loss = _loss_from_batch(output, batch)
                loss.backward()
                if config.training.dense_clip_norm is not None and dense_params:
                    _clip_grad_norm(dense_params, config.training.dense_clip_norm)
                if config.training.sparse_clip_norm is not None and sparse_params:
                    _clip_grad_norm(sparse_params, config.training.sparse_clip_norm)
                for optimizer in optimizers:
                    optimizer.step()
                steps += 1
                last_loss = float(loss.detach().cpu().item())
                if context.rank == 0:
                    print(f"Train step | step={steps} | loss={last_loss:.6f}")
                if max_steps is not None and steps >= max_steps:
                    break

        if config.training.checkpoint_path and context.rank == 0:
            checkpoint_path = Path(config.training.checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": base_model.state_dict(),
                    "model_name": config.model.name,
                    "vocab_strategy_hash": vocab_strategy_fingerprint(config.vocab_strategy),
                },
                checkpoint_path,
            )
        return TrainResult(steps=steps, last_loss=last_loss)
    finally:
        _cleanup_distributed(context)


@torch.no_grad()
def predict_mdl(
    config: AppConfig,
    checkpoint_path: str | None = None,
    output_path: str | None = None,
    max_batches: int | None = None,
) -> PredictResult:
    pa, _pc, _ds, pq = _require_pyarrow()
    device = _select_device(config)
    vocab_maps = load_vocab_maps(config)
    base_model = build_model(config, vocab_maps).to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        expected_hash = vocab_strategy_fingerprint(config.vocab_strategy)
        if checkpoint.get("vocab_strategy_hash") != expected_hash:
            raise ValueError("checkpoint vocab_strategy_hash does not match current config")
        base_model.load_state_dict(checkpoint["model_state_dict"])
    model = _maybe_compile_model(config, base_model)
    base_model.eval()
    model.eval()

    rows: list[dict[str, object]] = []
    for batch_index, batch in enumerate(
        iter_feature_batches(config, "test", vocab_maps, require_labels=False)
    ):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_feature_batch(batch, device)
        if config.model.use_request_cache and hasattr(base_model, "precompute_request_cache"):
            request_cache = base_model.precompute_request_cache(batch.features)
            logits = base_model(batch.features, batch.scenario_id, request_cache=request_cache)["logits"]
        else:
            logits = model(batch.features, batch.scenario_id)["logits"]
        probabilities = torch.sigmoid(logits).cpu().tolist()
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
