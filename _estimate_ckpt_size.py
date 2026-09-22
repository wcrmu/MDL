#!/usr/bin/env python3
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from scripts.build_production_configs import _embedding_memory_summary
from src.embeddings import EmbeddingTableSpec, embedding_local_bytes, plan_embedding_shards


def load_config(path: Path):
    with path.open() as f:
        return yaml.safe_load(f)


def collect_tables(payload, skip_shared=True):
    tables = []
    for feature in payload.get("features", []):
        if feature.get("kind") != "categorical":
            continue
        enc = feature["encoding"]
        if skip_shared and enc.get("share_embedding"):
            continue
        tables.append(
            (feature["name"], int(enc["num_buckets"]), int(feature["embedding_dim"]))
        )
    for sequence in payload.get("sequences", []):
        for field in sequence["fields"]:
            if field.get("kind") != "categorical":
                continue
            enc = field["encoding"]
            if skip_shared and enc.get("share_embedding"):
                continue
            name = f"{sequence['name']}.{field['name']}"
            tables.append(
                (name, int(enc["num_buckets"]), int(field["embedding_dim"]))
            )
    return tables


def count_aliases(payload):
    n = 0
    for feature in payload.get("features", []):
        if feature.get("kind") == "categorical" and feature["encoding"].get(
            "share_embedding"
        ):
            n += 1
    for sequence in payload.get("sequences", []):
        for field in sequence["fields"]:
            if field.get("kind") == "categorical" and field["encoding"].get(
                "share_embedding"
            ):
                n += 1
    return n


def detailed(path_str: str) -> None:
    path = Path(path_str)
    payload = load_config(path)
    tr = payload["training"]
    dtype = tr.get("embedding_weight_dtype", "bf16")
    sparse = tr.get("sparse_optimizer", "rowwise_adagrad")
    tables = collect_tables(payload, skip_shared=True)
    total_rows = sum(b + 1 for _, b, _ in tables)
    total_wx = sum((b + 1) * d for _, b, d in tables)
    alias = count_aliases(payload)
    weight_es = {"fp32": 4, "bf16": 2}[dtype]
    opt_layout = "rowwise" if sparse == "rowwise_adagrad" else "full"
    w_bytes = sum((b + 1) * d * weight_es for _, b, d in tables)
    if opt_layout == "rowwise":
        s_bytes = sum((b + 1) * 4 for _, b, d in tables)
    else:
        s_bytes = sum((b + 1) * d * 4 for _, b, d in tables)
    gib = 1024**3
    print(f"\n{'=' * 60}\n{path}")
    print(f"unique physical tables: {len(tables)}")
    print(f"share_embedding aliases (not stored): {alias}")
    print(f"SUM num_embeddings (rows) = {total_rows:,}")
    print(f"SUM rows*dim = {total_wx:,}")
    print(f"Global weights: {w_bytes / gib:.6f} GiB  (= rows*dim * {weight_es})")
    print(f"Global rowwise state: {s_bytes / gib:.6f} GiB  (= rows * 4)")
    print(f"Global total: {(w_bytes + s_bytes) / gib:.6f} GiB")
    for gpu in (2, 4):
        mem = _embedding_memory_summary(
            payload,
            gpu_count=gpu,
            budget_gib_per_gpu=9999.0,
            embedding_weight_dtype=dtype,
            sparse_optimizer=sparse,
        )
        print(f"\nworld_size={gpu}:")
        print(f"  uniform split: {(w_bytes + s_bytes) / gpu / gib:.6f} GiB/rank")
        print(
            f"  planned max rank: {mem['planned_weight_plus_state_gib_per_gpu']:.6f} GiB/rank"
        )
        print(f"  per-rank GiB: {mem['planned_weight_plus_state_gib_by_gpu']}")
    table_specs = [
        EmbeddingTableSpec(
            name=n, num_embeddings=b + 1, embedding_dim=d, element_size=weight_es
        )
        for n, b, d in tables
    ]
    plan = plan_embedding_shards(
        table_specs,
        world_size=4,
        strategy=tr["embedding_sharding"]["strategy"],
        table_wise_max_rows=int(tr["embedding_sharding"]["table_wise_max_rows"]),
        optimizer_state_layout=opt_layout,
    )
    tw = sum(1 for s in plan.tables.values() if s.strategy == "table_wise")
    rw = sum(1 for s in plan.tables.values() if s.strategy == "row_wise")
    print(
        f"\nAt ws=4: row_wise={rw}, table_wise={tw} "
        f"(max_rows={tr['embedding_sharding']['table_wise_max_rows']})"
    )


def main() -> None:
    for p in ["configs/mdl_rankmixer_fine.yaml", "configs/rankmixer_fine.yaml"]:
        detailed(str(_REPO / p))
    p1 = load_config(_REPO / "configs/mdl_rankmixer_fine.yaml")
    p2 = load_config(_REPO / "configs/rankmixer_fine.yaml")
    t1 = {n: (b, d) for n, b, d in collect_tables(p1)}
    t2 = {n: (b, d) for n, b, d in collect_tables(p2)}
    diff = {k for k in t1 if k in t2 and t1[k] != t2[k]}
    print(f"\nShape diffs between mdl_rankmixer_fine and rankmixer_fine: {len(diff)}")


if __name__ == "__main__":
    main()
