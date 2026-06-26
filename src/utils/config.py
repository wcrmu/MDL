from __future__ import annotations

import json
from pathlib import Path
from typing import Any, MutableMapping


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            return json.load(handle)
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "YAML config files require PyYAML. Install requirements.txt first."
                ) from exc
            data = yaml.safe_load(handle)
            return data or {}
    raise ValueError(f"unsupported config suffix {config_path.suffix!r}")


def deep_update(base: MutableMapping[str, Any], updates: MutableMapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result
