#!/usr/bin/env python
"""Build E2E bench overlays from *current* production configs.

Preserves model contract (mdl_token_state coupled/split, residual_ffn, etc.)
and runtime checkpoint policy. Only applies HBM fixture caps (never raise
hash buckets) and points train inputs at mock Parquet so 24GB cards can run
end-to-end without changing algorithms.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.config import load_app_config


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("rankmixer", "mdl_rankmixer", "onetrans", "mdl_onetrans")
MOCK_INPUTS = [
    str(ROOT / "artifacts" / "mock_parquet_full_2x2500_zstd"),
]
BUCKET_SOURCE = ROOT / "artifacts" / "bench_b512_direct.yaml"
# Absolute ceiling for fixture embeddings on 24GB cards when a name is absent
# from the capped bench map.
DEFAULT_MAX_BUCKETS = 65536


def _feature_bucket_map(path: Path) -> dict[str, int]:
    config = load_app_config(path)
    out: dict[str, int] = {}
    for feature in config.features:
        encoding = feature.encoding
        if encoding is None or encoding.num_buckets is None:
            continue
        out[feature.name] = int(encoding.num_buckets)
    return out


def _cap_num_buckets(current: int, name: str, bucket_map: dict[str, int]) -> int:
    limit = bucket_map.get(name, DEFAULT_MAX_BUCKETS)
    return min(int(current), int(limit))


def _cap_encoding(
    encoding: dict[str, Any],
    name: str,
    bucket_map: dict[str, int],
) -> dict[str, Any]:
    out = dict(encoding)
    if out.get("num_buckets") is not None:
        # Shared tables must follow the base feature's capped width.
        base = out.get("share_with") if out.get("share_embedding") else None
        key = str(base) if base else name
        out["num_buckets"] = _cap_num_buckets(int(out["num_buckets"]), key, bucket_map)
        out.pop("max_id", None)
    return out


def _cap_feature_like(
    item: dict[str, Any],
    bucket_map: dict[str, int],
) -> dict[str, Any]:
    out = dict(item)
    name = str(out.get("name") or "")
    encoding = out.get("encoding")
    if isinstance(encoding, dict):
        out["encoding"] = _cap_encoding(encoding, name, bucket_map)
    return out


def _cap_feature_payload(
    features: list[dict[str, Any]],
    bucket_map: dict[str, int],
) -> list[dict[str, Any]]:
    return [_cap_feature_like(feature, bucket_map) for feature in features]


def _cap_sequence_payload(
    sequences: list[dict[str, Any]],
    bucket_map: dict[str, int],
) -> list[dict[str, Any]]:
    capped: list[dict[str, Any]] = []
    for sequence in sequences:
        item = dict(sequence)
        fields = item.get("fields")
        if isinstance(fields, list):
            item["fields"] = [
                _cap_feature_like(field, bucket_map)
                if isinstance(field, dict)
                else field
                for field in fields
            ]
        capped.append(item)
    return capped


def build_overlay(
    model_name: str,
    *,
    out_dir: Path,
    nproc: int,
    bucket_map: dict[str, int],
) -> Path:
    src = ROOT / "configs" / f"{model_name}.yaml"
    with src.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    payload["features"] = _cap_feature_payload(list(payload["features"]), bucket_map)
    if "sequences" in payload:
        payload["sequences"] = _cap_sequence_payload(
            list(payload["sequences"]),
            bucket_map,
        )
    runtime = dict(payload.get("runtime") or {})
    runtime["nproc_per_node"] = nproc
    runtime["distributed"] = "ddp" if nproc > 1 else "none"
    # Unique ports so parallel model jobs do not collide on 29500.
    runtime["master_port"] = {
        "rankmixer": 29611,
        "mdl_rankmixer": 29612,
        "onetrans": 29613,
        "mdl_onetrans": 29614,
    }.get(model_name, 29500 + hash(model_name) % 1000)
    payload["runtime"] = runtime

    for split_name in ("train", "test"):
        split = (payload.get("data") or {}).get(split_name)
        if not isinstance(split, dict):
            continue
        split = dict(split)
        split["inputs"] = list(MOCK_INPUTS)
        reader = dict(split.get("reader") or {})
        reader.setdefault("host_prepare_prefetch", 3)
        reader.setdefault("device_prefetch_batches", 1)
        reader.setdefault("pin_memory", True)
        reader.setdefault("coalesce_pinned_tensors", True)
        reader.setdefault("agg_direct_mode", "direct")
        split["reader"] = reader
        payload.setdefault("data", {})[split_name] = split

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_current_e2e.yaml"
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "gpu_util_e2e_mock" / "current_coupled_e2e",
    )
    parser.add_argument("--nproc", type=int, default=2)
    args = parser.parse_args()
    bucket_map = _feature_bucket_map(BUCKET_SOURCE)
    for model_name in MODELS:
        path = build_overlay(
            model_name,
            out_dir=args.out_dir,
            nproc=args.nproc,
            bucket_map=bucket_map,
        )
        config = load_app_config(path)
        print(
            model_name,
            "mdl_token_state=",
            getattr(config.model, "mdl_token_state", None),
            "act=",
            config.runtime.activation_checkpoint,
            "graph=",
            config.runtime.cuda_graph_backbone,
            "host_pf=",
            config.data.train.reader.host_prepare_prefetch,
            "dev_pf=",
            config.data.train.reader.device_prefetch_batches,
            "->",
            path,
        )


if __name__ == "__main__":
    main()
