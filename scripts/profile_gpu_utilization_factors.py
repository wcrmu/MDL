#!/usr/bin/env python3
"""Ablate GPU-utilization factors with one real, repeatedly reused Parquet batch.

The script separates input-pipeline stalls from model-side work, then measures
ordinary bag width, sequence width, scalar/sequence encoding, and backbone cost
without changing the batch row count or model topology.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable, Iterator

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import _GpuUtilizationSampler
from src.config import AppConfig, load_app_config
from src.dataloader import FeatureBatch, move_feature_batch
from src.embeddings import consume_sharded_embedding_stats
from src.features import load_vocab_maps
from src.optim import ShardedAdagrad, ShardedRowWiseAdagrad
from src.train import (
    _autocast_context,
    _build_dense_optimizer,
    _build_model_on_device,
    _classify_model_parameters,
    _clip_grad_norm,
    _clip_sparse_grad_norm,
    _loss_terms_from_batch,
    _mark_sparse_invariant_checks_explicitly_disabled,
    _step_sparse_moe_controllers,
    iter_feature_batches,
)


@dataclass(frozen=True)
class AblationResult:
    name: str
    rows: int
    warmup_steps: int
    measured_steps: int
    mean_step_seconds: float
    samples_per_second: float
    gpu_utilization_percent: float | None
    peak_hbm_allocated_gib: float
    peak_hbm_reserved_gib: float


@dataclass
class _OptimizerState:
    optimizers: list[torch.optim.Optimizer]
    dense_params: list[nn.Parameter]
    replicated_embedding_params: list[nn.Parameter]
    sharded_embedding_params: list[nn.Parameter]


def _load_one_host_batch(config: AppConfig) -> FeatureBatch:
    reader = replace(
        config.data.train.reader,
        device_prefetch_batches=0,
        prefetch_batches=1,
    )
    split = replace(config.data.train, reader=reader)
    host_config = replace(config, data=replace(config.data, train=split))
    iterator = iter(
        iter_feature_batches(
            host_config,
            "train",
            load_vocab_maps(host_config),
            require_labels=True,
            pin_memory=False,
            include_group_id=False,
        )
    )
    try:
        return next(iterator)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def _truncate_flat_bag(payload: dict[str, Any], max_length: int) -> dict[str, Any]:
    values = payload["values"]
    lengths = payload["lengths"]
    if not isinstance(values, Tensor) or not isinstance(lengths, Tensor):
        raise TypeError("bag values and lengths must be tensors")
    selected_lengths = lengths.clamp(max=max_length)
    if max_length == 1:
        nonempty = lengths > 0
        offsets = torch.cumsum(lengths, dim=0) - lengths
        selected_values = values.index_select(0, offsets[nonempty].long())
    else:
        positions = torch.arange(values.size(0), device=values.device)
        row_ids = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=lengths.device),
            lengths,
        )
        offsets = torch.cumsum(lengths, dim=0) - lengths
        within_row = positions - offsets.index_select(0, row_ids)
        selected_values = values[within_row < max_length]
    return {
        **payload,
        "values": selected_values,
        "lengths": selected_lengths,
    }


def _truncate_ordinary_bags(
    config: AppConfig,
    batch: FeatureBatch,
    max_length: int,
) -> FeatureBatch:
    features = dict(batch.features)
    for feature in config.features:
        if feature.kind != "categorical" or feature.pooling != "mean":
            continue
        payload = features[feature.name]
        if not isinstance(payload, dict):
            raise TypeError(f"bag feature {feature.name!r} must be a dict")
        features[feature.name] = _truncate_flat_bag(payload, max_length)
    return replace(batch, features=features, _packed_buffers=())


def _truncate_sequences(
    config: AppConfig,
    batch: FeatureBatch,
    max_length: int,
) -> FeatureBatch:
    features = dict(batch.features)
    for sequence in config.sequences:
        payload = features[sequence.name]
        if not isinstance(payload, dict):
            raise TypeError(f"sequence {sequence.name!r} must be a dict")
        fields = {
            name: value[:, :max_length]
            for name, value in payload["fields"].items()
        }
        features[sequence.name] = {
            **payload,
            "fields": fields,
            "lengths": payload["lengths"].clamp(max=max_length),
        }
    return replace(batch, features=features, _packed_buffers=())


def _build_optimizers(
    model: nn.Module,
    config: AppConfig,
    device: torch.device,
) -> _OptimizerState:
    groups = _classify_model_parameters(model)
    optimizers: list[torch.optim.Optimizer] = []
    dense = list(groups.dense_optimizer)
    replicated = list(groups.embedding_optimizer)
    sharded = list(groups.sharded_optimizer)
    if dense:
        optimizers.append(_build_dense_optimizer(dense, config, device))
    if replicated:
        _mark_sparse_invariant_checks_explicitly_disabled()
        optimizers.append(
            torch.optim.Adagrad(
                replicated,
                lr=config.training.lr_sparse or config.training.lr_dense,
                lr_decay=config.training.adagrad_lr_decay,
                weight_decay=config.training.adagrad_weight_decay,
                initial_accumulator_value=(
                    config.training.adagrad_initial_accumulator_value
                ),
                eps=config.training.adagrad_eps,
            )
        )
    if sharded:
        _mark_sparse_invariant_checks_explicitly_disabled()
        optimizer_type = (
            ShardedRowWiseAdagrad
            if config.training.sparse_optimizer == "rowwise_adagrad"
            else ShardedAdagrad
        )
        optimizers.append(
            optimizer_type(
                sharded,
                lr=config.training.lr_sparse or config.training.lr_dense,
                lr_decay=config.training.adagrad_lr_decay,
                weight_decay=config.training.adagrad_weight_decay,
                initial_accumulator_value=(
                    config.training.adagrad_initial_accumulator_value
                ),
                eps=config.training.adagrad_eps,
            )
        )
    return _OptimizerState(
        optimizers=optimizers,
        dense_params=dense,
        replicated_embedding_params=replicated,
        sharded_embedding_params=sharded,
    )


def _execute_step(
    model: nn.Module,
    batch: FeatureBatch,
    config: AppConfig,
    optimizer_state: _OptimizerState,
) -> None:
    for optimizer in optimizer_state.optimizers:
        optimizer.zero_grad(set_to_none=True)
    with _autocast_context(config, batch.scenario_id.device):
        output = model(batch.features, batch.scenario_id)
        loss, _numerator, _denominator = _loss_terms_from_batch(
            output,
            batch,
            moe_loss_weight=config.model.sparse_moe_loss_weight,
            loss_reduction=config.training.loss_reduction,
            rank_active=True,
            active_rank_count=1,
        )
    loss.backward()
    _step_sparse_moe_controllers(
        model,
        rank_active=True,
        active_rank_count=1,
    )
    if config.training.dense_clip_norm is not None:
        _clip_grad_norm(
            optimizer_state.dense_params,
            config.training.dense_clip_norm,
        )
    if config.training.sparse_clip_norm is not None:
        _clip_sparse_grad_norm(
            optimizer_state.replicated_embedding_params,
            optimizer_state.sharded_embedding_params,
            config.training.sparse_clip_norm,
        )
    for optimizer in optimizer_state.optimizers:
        optimizer.step()
    consume_sharded_embedding_stats(model)


def _benchmark_variant(
    name: str,
    model: nn.Module,
    batch: FeatureBatch,
    config: AppConfig,
    optimizer_state: _OptimizerState,
    *,
    warmup_steps: int,
    measured_steps: int,
) -> AblationResult:
    device = batch.scenario_id.device
    for _ in range(warmup_steps):
        _execute_step(model, batch, config, optimizer_state)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    sampler = _GpuUtilizationSampler(device, interval_seconds=0.1)
    sampler.start()
    started = perf_counter()
    for _ in range(measured_steps):
        _execute_step(model, batch, config, optimizer_state)
    torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    utilization = sampler.stop()
    rows = int(batch.scenario_id.numel())
    return AblationResult(
        name=name,
        rows=rows,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        mean_step_seconds=elapsed / measured_steps,
        samples_per_second=rows * measured_steps / elapsed,
        gpu_utilization_percent=utilization,
        peak_hbm_allocated_gib=torch.cuda.max_memory_allocated(device) / (1024**3),
        peak_hbm_reserved_gib=torch.cuda.max_memory_reserved(device) / (1024**3),
    )


def _detach_encoded(encoded: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach() for name, value in encoded.items()}


@contextmanager
def _replace_encoder_forward(
    encoder: nn.Module,
    forward: Callable[..., dict[str, Tensor]],
) -> Iterator[None]:
    original = encoder.forward
    encoder.forward = forward  # type: ignore[method-assign]
    try:
        yield
    finally:
        encoder.forward = original  # type: ignore[method-assign]


def _profile_one_step(
    model: nn.Module,
    batch: FeatureBatch,
    config: AppConfig,
    optimizer_state: _OptimizerState,
) -> dict[str, Any]:
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities) as profiler:
        _execute_step(model, batch, config, optimizer_state)
        torch.cuda.synchronize(batch.scenario_id.device)
    averages = list(profiler.key_averages())

    def device_us(event: Any) -> float:
        return float(
            getattr(
                event,
                "device_time_total",
                getattr(event, "cuda_time_total", 0.0),
            )
        )

    device_events = [
        event
        for event in profiler.events()
        if "cuda" in str(getattr(event, "device_type", "")).lower()
    ]
    device_durations_us = sorted(
        float(getattr(event, "device_time_total", 0.0))
        for event in device_events
    )

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, int(fraction * (len(values) - 1)))
        return values[index]

    top_cuda = sorted(averages, key=device_us, reverse=True)[:20]
    return {
        "cuda_device_event_count": len(device_events),
        "cuda_device_event_total_ms": sum(device_durations_us) / 1000.0,
        "cuda_device_event_mean_us": (
            sum(device_durations_us) / len(device_durations_us)
            if device_durations_us
            else 0.0
        ),
        "cuda_device_event_p50_us": percentile(device_durations_us, 0.50),
        "cuda_device_event_p95_us": percentile(device_durations_us, 0.95),
        "cuda_device_events_under_10us_ratio": (
            sum(value < 10.0 for value in device_durations_us)
            / len(device_durations_us)
            if device_durations_us
            else 0.0
        ),
        "cpu_self_total_ms": sum(
            float(event.self_cpu_time_total) for event in averages
        )
        / 1000.0,
        "operator_call_count_with_cuda": sum(
            int(event.count) for event in averages if device_us(event) > 0.0
        ),
        "top_cuda_operators": [
            {
                "name": str(event.key),
                "calls": int(event.count),
                "cuda_total_ms": device_us(event) / 1000.0,
                "cpu_self_total_ms": float(event.self_cpu_time_total) / 1000.0,
            }
            for event in top_cuda
            if device_us(event) > 0.0
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("artifacts/mock_full_rankmixer_capped_b512_adapter4.yaml"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup_steps < 0 or args.steps <= 0:
        parser.error("warmup-steps must be non-negative and steps must be positive")

    config = load_app_config(args.config)
    host_batch = _load_one_host_batch(config)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("GPU utilization ablation requires CUDA")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    full_batch = move_feature_batch(host_batch, device)
    bag1_batch = _truncate_ordinary_bags(config, full_batch, 1)
    sequence1_batch = _truncate_sequences(config, full_batch, 1)
    both1_batch = _truncate_sequences(config, bag1_batch, 1)

    model = _build_model_on_device(
        config,
        load_vocab_maps(config),
        device,
    )
    model.train()
    optimizer_state = _build_optimizers(model, config, device)
    results: list[AblationResult] = []
    for name, batch in (
        ("full_real_batch", full_batch),
        ("ordinary_bags_length_1", bag1_batch),
        ("sequences_length_1", sequence1_batch),
        ("bags_and_sequences_length_1", both1_batch),
    ):
        result = _benchmark_variant(
            name,
            model,
            batch,
            config,
            optimizer_state,
            warmup_steps=args.warmup_steps,
            measured_steps=args.steps,
        )
        results.append(result)
        print(json.dumps(asdict(result), sort_keys=True), flush=True)

    encoder = getattr(model, "encoder_bank", None)
    if encoder is None:
        raise TypeError("selected model does not expose encoder_bank")
    with torch.no_grad(), _autocast_context(config, device):
        cached_encoded = _detach_encoded(encoder(full_batch.features))
    sequence_names = {sequence.name for sequence in config.sequences}
    cached_sequences = {
        name: value
        for name, value in cached_encoded.items()
        if name in sequence_names
    }

    def ordinary_live(
        features: dict[str, Any],
        request_cache: Any = None,
    ) -> dict[str, Tensor]:
        del request_cache
        encoded = encoder.encode_scalar_features(features)
        encoded.update(cached_sequences)
        return encoded

    with _replace_encoder_forward(encoder, ordinary_live):
        result = _benchmark_variant(
            "ordinary_encoders_live_sequences_cached",
            model,
            full_batch,
            config,
            optimizer_state,
            warmup_steps=args.warmup_steps,
            measured_steps=args.steps,
        )
        results.append(result)
        print(json.dumps(asdict(result), sort_keys=True), flush=True)

    def all_cached(
        features: dict[str, Any],
        request_cache: Any = None,
    ) -> dict[str, Tensor]:
        del features, request_cache
        return cached_encoded

    with _replace_encoder_forward(encoder, all_cached):
        result = _benchmark_variant(
            "all_feature_encoders_cached_backbone_only",
            model,
            full_batch,
            config,
            optimizer_state,
            warmup_steps=args.warmup_steps,
            measured_steps=args.steps,
        )
        results.append(result)
        print(json.dumps(asdict(result), sort_keys=True), flush=True)

    profiler_summary = (
        _profile_one_step(model, full_batch, config, optimizer_state)
        if args.profile
        else None
    )
    payload = {
        "config": str(args.config.resolve()),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "results": [asdict(result) for result in results],
        "profiler": profiler_summary,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print("ABLATION_SUMMARY=" + json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
