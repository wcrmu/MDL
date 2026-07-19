"""Atomic model checkpoints for replicated and self-sharded embeddings."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as torch_dist
from torch import Tensor, nn

from .config import AppConfig
from .embeddings import EmbeddingShardSpec, ShardedEmbedding, sharded_embedding_modules
from .features import vocab_strategy_fingerprint
from .optim import ShardedAdagrad, ShardedRowWiseAdagrad


SHARDED_CHECKPOINT_FORMAT = "mdl_sharded_embedding_v1"


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


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def save_model_checkpoint(
    config: AppConfig,
    model: nn.Module,
    path: str | Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    process_group: torch_dist.ProcessGroup | None = None,
    sharded_optimizer: ShardedAdagrad | ShardedRowWiseAdagrad | None = None,
) -> None:
    """Save one replicated file or an atomic manifest plus local shard files."""

    checkpoint_path = Path(path)
    sharded_modules = sharded_embedding_modules(model)
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
        return

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

    tables: dict[str, dict[str, Any]] = {}
    for module in sharded_modules:
        optimizer_state = None
        if sharded_optimizer is not None:
            state = sharded_optimizer.state.get(module.weight)
            if state:
                optimizer_state = _state_to_cpu(state)
        tables[module.table_name] = {
            "weight": module.weight.detach().cpu(),
            "num_embeddings": module.num_embeddings,
            "embedding_dim": module.embedding_dim,
            "padding_idx": module.padding_idx,
            "shard_spec": asdict(module.shard_spec),
            "optimizer_state": optimizer_state,
        }
    rank_file = f"rank-{rank:05d}-of-{world_size:05d}.pt"
    _atomic_torch_save(
        {
            "format": SHARDED_CHECKPOINT_FORMAT,
            "rank": rank,
            "world_size": world_size,
            "tables": tables,
        },
        checkpoint_path / rank_file,
    )

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
                "format": SHARDED_CHECKPOINT_FORMAT,
                "version": 1,
                "world_size": world_size,
                "dense_file": dense_file,
                "rank_files": [
                    f"rank-{item:05d}-of-{world_size:05d}.pt"
                    for item in range(world_size)
                ],
                "tables": table_metadata,
                "training_metadata": {
                    "sparse_optimizer": config.training.sparse_optimizer,
                },
                **_checkpoint_metadata(config),
            },
            checkpoint_path / "manifest.json",
        )
    if world_size > 1:
        torch_dist.barrier(group=process_group)


def _load_sharded_checkpoint(
    config: AppConfig,
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    process_group: torch_dist.ProcessGroup | None,
) -> None:
    manifest_path = checkpoint_path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"sharded checkpoint is missing {manifest_path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != SHARDED_CHECKPOINT_FORMAT:
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
    rank, world_size = (0, 1)
    if torch_dist.is_available() and torch_dist.is_initialized():
        rank = torch_dist.get_rank(process_group)
        world_size = torch_dist.get_world_size(process_group)
    saved_world_size = int(manifest["world_size"])
    rank_files = [checkpoint_path / name for name in manifest["rank_files"]]

    # Same-size restarts need only the local file. Different-size loads stream
    # every saved owner and write directly into the new local rows; no full-table
    # reconstruction is allocated.
    if saved_world_size == world_size:
        payloads = [torch.load(rank_files[rank], map_location=device)]
    else:
        payloads = [torch.load(path, map_location="cpu") for path in rank_files]
    filled = {
        name: torch.zeros(module.weight.size(0), dtype=torch.bool)
        for name, module in modules.items()
    }
    with torch.no_grad():
        for payload in payloads:
            if payload.get("format") != SHARDED_CHECKPOINT_FORMAT:
                raise ValueError("invalid embedding rank shard format")
            saved_rank = int(payload["rank"])
            if int(payload["world_size"]) != saved_world_size:
                raise ValueError("inconsistent world size across embedding shard files")
            for table_name, module in modules.items():
                table = payload["tables"].get(table_name)
                if table is None:
                    raise ValueError(f"rank shard is missing table {table_name!r}")
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
                filled[table_name].index_fill_(0, target_rows.cpu(), True)
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
) -> None:
    """Load a legacy replicated file or a reshardable checkpoint directory."""

    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        _load_sharded_checkpoint(
            config, model, checkpoint_path, device, process_group
        )
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    _validate_checkpoint_metadata(config, checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
