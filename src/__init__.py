"""MDL package.

Public names are resolved lazily so ``adapter_workers`` ProcessPool children can
``import src.dataloader`` (adapter only) without pulling torch / CUDA via
``model`` and ``train``. Eager imports here previously made each worker spend
~1.5s+ on torch bootstrap while the parent blocked inside
``ProcessPoolExecutor.submit`` — multi-second stalls that look like a deadlock.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AppConfig",
    "PredictResult",
    "TrainResult",
    "build_model",
    "load_app_config",
    "predict_mdl",
    "train_mdl",
]


def __getattr__(name: str) -> Any:
    if name in {"AppConfig", "load_app_config"}:
        from .config import AppConfig, load_app_config

        globals()["AppConfig"] = AppConfig
        globals()["load_app_config"] = load_app_config
        return globals()[name]
    if name == "build_model":
        from .model import build_model

        globals()["build_model"] = build_model
        return build_model
    if name in {"PredictResult", "TrainResult", "predict_mdl", "train_mdl"}:
        from .train import PredictResult, TrainResult, predict_mdl, train_mdl

        globals()["PredictResult"] = PredictResult
        globals()["TrainResult"] = TrainResult
        globals()["predict_mdl"] = predict_mdl
        globals()["train_mdl"] = train_mdl
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
