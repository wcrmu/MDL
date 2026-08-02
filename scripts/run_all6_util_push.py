#!/usr/bin/env python
"""Run end-to-end util/sps push for all 6 models on local multi-GPU.

Builds capped-emb overlays from current production configs, applies util-oriented
infra knobs (deep host/prefetch; model-specific batch scales that fit 24GB),
and reports util + sps.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import importlib.util
import yaml


ROOT = Path(__file__).resolve().parents[1]
# File-sharded DDP needs ≥ nproc parquet files. The x24 tile is the local
# util-protect fixture (plain mock_parquet_full only has 2 files).
MOCK_INPUTS = [
    str(ROOT / "artifacts" / "mock_parquet_full_2x2500_zstd_x24"),
]


def _load_overlay_builder():
    path = ROOT / "scripts" / "build_current_e2e_overlays.py"
    spec = importlib.util.spec_from_file_location("build_current_e2e_overlays", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_overlay = _load_overlay_builder()
BUCKET_SOURCE = _overlay.BUCKET_SOURCE
MODELS = tuple(_overlay.MODELS)
_feature_bucket_map = _overlay._feature_bucket_map
build_overlay = _overlay.build_overlay
PORTS = {
    "rankmixer": 29861,
    "mdl_rankmixer": 29862,
    "onetrans": 29863,
    "mdl_onetrans": 29864,
    "mixformer": 29865,
    "mdl_mixformer": 29866,
}
# Extra request-batch scale on top of production YAML (after emb caps).
# Tuned for 2×24GB no-P2P (matches prior all6_post_opt regime). On 4×4090
# these scales + CUDA-graph pools OOMed; prefer --nproc 2 for local push.
UTIL_BATCH_SCALE = {
    # RankMixer: k@2560 + workers=4 keeps wait~0.005 (util~75 under load).
    # 2.8× re-inflates wait and drops util; stay at post_opt batch.
    "rankmixer": 2.00,  # 1280 → 2560
    "mdl_rankmixer": 1.0,
    # Plain OneTrans: selective + large batch (prod act=none OOMs at util scale).
    "onetrans": 2.00,  # 1408 → ~2816 with selective
    # MDL-OneTrans: act=full + ~2957 (l: util~82 / sps↑ vs post 87.5/1693).
    "mdl_onetrans": 2.10,  # 1408 → ~2957
    # MixFormer: build_current_e2e_overlays already applies 0.90→~460.
    # Extra derate + oversized steps collapse sps (~15×); stay at overlay size.
    "mixformer": 1.0,  # keep ~460 (post_opt control util~95)
    "mdl_mixformer": 1.0,  # keep ~512 + selective
}


def _scale_int(value: int, scale: float) -> int:
    return max(8, int(round(int(value) * float(scale))))


def _apply_batch_scale(payload: dict[str, Any], scale: float) -> None:
    if abs(scale - 1.0) < 1e-9:
        return
    training = dict(payload.get("training") or {})
    if "batch_size" in training:
        training["batch_size"] = _scale_int(int(training["batch_size"]), scale)
        payload["training"] = training
    data = dict(payload.get("data") or {})
    for split_name in ("train", "test"):
        split = data.get(split_name)
        if not isinstance(split, dict):
            continue
        split = dict(split)
        reader = dict(split.get("reader") or {})
        buckets = reader.get("length_buckets")
        if isinstance(buckets, list):
            scaled = []
            for bucket in buckets:
                if not isinstance(bucket, dict) or "batch_size" not in bucket:
                    scaled.append(bucket)
                    continue
                item = dict(bucket)
                item["batch_size"] = _scale_int(int(bucket["batch_size"]), scale)
                scaled.append(item)
            reader["length_buckets"] = scaled
        split["reader"] = reader
        data[split_name] = split
    payload["data"] = data


def _deepen_reader(
    reader: dict[str, Any],
    *,
    host: int,
    prefetch: int,
    device_prefetch: int,
) -> dict[str, Any]:
    out = dict(reader)
    out["host_prepare_prefetch"] = max(int(out.get("host_prepare_prefetch") or 0), host)
    out["prefetch_batches"] = max(int(out.get("prefetch_batches") or 0), prefetch)
    # Cap device prefetch on no-P2P 24GB — depth 2 + graph OOMs RankMixer.
    out["device_prefetch_batches"] = device_prefetch
    out["overlap_host_prepare"] = True
    out["pin_memory"] = True
    out["coalesce_pinned_tensors"] = True
    return out


def _prepare_yaml(
    model: str,
    out_dir: Path,
    *,
    nproc: int,
    bucket_map: dict[str, int],
) -> Path:
    # Start from capped production overlay.
    overlay = build_overlay(
        model, out_dir=out_dir / "overlays", nproc=nproc, bucket_map=bucket_map
    )
    payload = yaml.safe_load(overlay.read_text())
    _apply_batch_scale(payload, float(UTIL_BATCH_SCALE.get(model, 1.0)))

    runtime = dict(payload.get("runtime") or {})
    runtime["nproc_per_node"] = nproc
    runtime["distributed"] = "ddp" if nproc > 1 else "none"
    runtime["master_port"] = PORTS[model]
    if model == "rankmixer":
        # Graph private pools + multi-rank emb staging OOMs 24GB; util push
        # uses eager + deeper host runway (all6_post_opt winner pattern).
        runtime["cuda_graph_backbone"] = False
        runtime["activation_checkpoint"] = "none"
    if model == "mdl_rankmixer":
        runtime["cuda_graph_backbone"] = False
    # Plain OneTrans prod is act=none; selective is required for util-scale
    # batches on 24GB. MDL-OneTrans post_opt util winner kept act=full.
    if model == "onetrans":
        runtime["activation_checkpoint"] = "selective"
        runtime["cuda_graph_backbone"] = False
    if model == "mdl_onetrans":
        runtime["activation_checkpoint"] = "full"
        runtime["cuda_graph_backbone"] = False
    if model == "mdl_mixformer":
        runtime["activation_checkpoint"] = "selective"
    if model in {"mixformer", "mdl_mixformer"}:
        # Prod used to ship 512; that starves SMs (~6× SPS regression vs 8192).
        runtime["sequence_projection_chunk_tokens"] = max(
            int(runtime.get("sequence_projection_chunk_tokens") or 0),
            8192,
        )
    payload["runtime"] = runtime

    rankmixer_family = model in {"rankmixer", "mdl_rankmixer"}
    # Deep host helps RankMixer when CPU is free; on contended hosts it just
    # burns workers. Match post_opt depth for util-critical models.
    if model == "rankmixer":
        host, prefetch = 6, 4
    elif rankmixer_family:
        host, prefetch = 10, 6
    else:
        host, prefetch = 6, 4
    # Plain OneTrans activations are large — shallow device prefetch on 24GB.
    # MDL-OneTrans post_opt util winner used device_prefetch=2.
    if model == "onetrans":
        device_prefetch = 1
    else:
        device_prefetch = 2
    for split_name in ("train", "test"):
        split = (payload.get("data") or {}).get(split_name)
        if not isinstance(split, dict):
            continue
        split = dict(split)
        split["inputs"] = list(MOCK_INPUTS)
        reader = _deepen_reader(
            dict(split.get("reader") or {}),
            host=host,
            prefetch=prefetch,
            device_prefetch=device_prefetch,
        )
        # post_opt util winners used 4 workers; prod YAML often defaults to 2
        # and under-feeds the GPU (rankmixer wait 0.13 → util ~69).
        if model in {
            "rankmixer",
            "mdl_onetrans",
            "mixformer",
            "mdl_mixformer",
        }:
            reader["num_workers"] = max(int(reader.get("num_workers") or 0), 4)
        split["reader"] = reader
        payload.setdefault("data", {})[split_name] = split

    training = dict(payload.get("training") or {})
    fixed_test = dict(training.get("fixed_test_eval") or {})
    fixed_test["enabled"] = False
    training["fixed_test_eval"] = fixed_test
    training["log_every_steps"] = 10
    payload["training"] = training

    out_path = out_dir / f"{model}.yaml"
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out_path


def _summarize(model: str, status: str, json_path: Path, log_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"model": model, "status": status}
    if json_path.exists():
        data = json.loads(json_path.read_text())
        util = data.get("gpu_utilization_percent_per_rank") or []
        env = data.get("environment") or {}
        reader = env.get("data_reader") or {}
        row.update(
            {
                "util_per_rank": [round(float(x), 2) for x in util],
                "util_mean": round(sum(util) / len(util), 2) if util else None,
                "samples_per_second": round(float(data.get("samples_per_second") or 0), 1),
                "dataloader_wait_ratio": round(
                    float(data.get("dataloader_wait_ratio") or 0), 5
                ),
                "mean_step_seconds": round(float(data.get("mean_step_seconds") or 0), 3),
                "mfu": data.get("mfu"),
                "hbm_gib": [
                    round(x / 1024**3, 2)
                    for x in (data.get("peak_hbm_allocated_bytes_per_rank") or [])
                ],
                "host_prepare": reader.get("host_prepare_prefetch"),
                "prefetch_batches": reader.get("prefetch_batches"),
                "device_prefetch": reader.get("device_prefetch_batches"),
                "cuda_graph": env.get("cuda_graph_backbone"),
            }
        )
    elif log_path.exists():
        row["error_tail"] = log_path.read_text(errors="ignore")[-2000:]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "artifacts" / "gpu_util_e2e_mock" / "all6_util_push"),
    )
    parser.add_argument("--warmup-steps", type=int, default=6)
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model in models:
        if model not in MODELS:
            raise SystemExit(f"unknown model {model}")

    bucket_map = _feature_bucket_map(BUCKET_SOURCE)
    results: list[dict[str, Any]] = []
    results_path = out / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    for model in models:
        cfg = _prepare_yaml(
            model, out, nproc=args.nproc, bucket_map=bucket_map
        )
        log_path = out / f"{model}.log"
        json_path = out / f"{model}.json"
        print(f"\n=== {model} nproc={args.nproc} gpus={args.gpus} ===", flush=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
        # Orphaned deleted /dev/shm from other jobs often leaves ~2–3GiB free —
        # enough to pass auto's 2GiB share threshold then thrash. Default memfd
        # for local util push; override with MDL_HOST_PREPARE_IPC=share when shm
        # is truly empty.
        env["MDL_HOST_PREPARE_IPC"] = os.environ.get("MDL_HOST_PREPARE_IPC", "memfd")
        env["MDL_DIST_BACKEND"] = "nccl"
        env["NCCL_P2P_DISABLE"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        for key in (
            "MDL_GROUPED_EMB_MAX_OUTPUT_MIB",
            "MDL_MOCK_IO_STALL_PROB",
        ):
            env.pop(key, None)
        # Avoid stacking overlay batch scale with no-P2P auto 1.42.
        env["MDL_LOCAL_BATCH_SCALE"] = "1.0"
        cmd = [
            sys.executable,
            "-m",
            "src.main",
            "benchmark",
            "--config",
            str(cfg),
            "--mode",
            "end-to-end",
            "--warmup-steps",
            str(args.warmup_steps),
            "--steps",
            str(args.steps),
            "--profile-steps",
            str(args.profile_steps),
            "--peak-tflops",
            "165.2",
            "--nproc-per-node",
            str(args.nproc),
            "--output",
            str(json_path),
        ]
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                cmd, env=env, cwd=str(ROOT), stdout=log_handle, stderr=subprocess.STDOUT
            )
        status = "ok" if completed.returncode == 0 else "fail"
        row = _summarize(model, status, json_path, log_path)
        results.append(row)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"RESULT {model}: status={status} util={row.get('util_mean')} "
            f"sps={row.get('samples_per_second')} wait={row.get('dataloader_wait_ratio')} "
            f"hbm={row.get('hbm_gib')}",
            flush=True,
        )

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("\n===== ALL6 UTIL PUSH =====")
    print(f"{'model':16s} {'status':6s} {'util':>7s} {'sps':>9s} {'wait':>8s}")
    for row in results:
        print(
            f"{row['model']:16s} {row['status']:6s} "
            f"{row.get('util_mean')!s:>7} {row.get('samples_per_second')!s:>9} "
            f"{row.get('dataloader_wait_ratio')!s:>8}"
        )
    print(f"WROTE {summary_path}")
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
