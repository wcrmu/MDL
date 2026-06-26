from __future__ import annotations

from typing import Any


def _latest_tokenization(manifest: dict[str, Any]) -> dict[str, Any]:
    tokenization = manifest.get("tokenization")
    if not isinstance(tokenization, dict):
        raise ValueError("manifest must contain tokenization object")
    if tokenization.get("version") != 2 or tokenization.get("kind") != "encoder_registry":
        raise ValueError("tokenization must use version=2 and kind='encoder_registry'")
    return tokenization


def feature_specs_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    tokenization = _latest_tokenization(manifest)
    if "features" not in tokenization:
        raise ValueError("tokenization must contain features")
    return list(tokenization["features"])


def token_specs_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    tokenization = _latest_tokenization(manifest)
    if "token_specs" not in tokenization:
        raise ValueError("tokenization must contain token_specs")
    return list(tokenization["token_specs"])


def scenario_feature_specs_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]] | None:
    tokenization = _latest_tokenization(manifest)
    specs = tokenization.get("scenario_features")
    return None if specs is None else list(specs)


def scenario_token_specs_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]] | None:
    tokenization = _latest_tokenization(manifest)
    specs = tokenization.get("scenario_token_specs")
    return None if specs is None else list(specs)


def task_feature_specs_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]] | None:
    tokenization = _latest_tokenization(manifest)
    specs = tokenization.get("task_features")
    return None if specs is None else list(specs)


def task_token_specs_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]] | None:
    tokenization = _latest_tokenization(manifest)
    specs = tokenization.get("task_token_specs")
    return None if specs is None else list(specs)
