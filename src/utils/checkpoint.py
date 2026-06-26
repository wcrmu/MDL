from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.mdl import ModelConfig


def _serialize_config(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError("checkpoint config must be a dataclass or dict")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    model_config: Any,
    manifest: dict[str, Any] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": _serialize_config(model_config),
        "manifest": manifest,
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    if "model_state_dict" not in payload or "model_config" not in payload:
        raise ValueError("checkpoint must contain model_state_dict and model_config")
    payload["model_config"] = ModelConfig(**payload["model_config"])
    return payload
