#!/usr/bin/env python
"""Count sparse Adagrad tables touched per step and time the step."""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

from src.benchmark import BenchmarkOptions, run_end_to_end_benchmark
from src.config import load_app_config


def main() -> None:
    # Prefer a tiny timed loop via existing train internals if easier:
    from src.config import load_app_config
    from src.features import load_vocab_maps
    from src.train import (
        _build_model,
        _build_optimizers,
        _move_batch_to_device,
        iter_feature_batches,
        _autocast_context,
        _loss_terms_from_batch,
    )
    from src.runtime import resolve_device

    cfg = load_app_config("artifacts/gpu_util_e2e_mock/rankmixer_e2e.yaml")
    device = torch.device("cuda")
    vocab = load_vocab_maps(cfg)
    model = _build_model(cfg, vocab).to(device)
    optimizers = _build_optimizers(cfg, model)
    it = iter_feature_batches(
        cfg, "train", vocab, require_labels=True, pin_memory=True, include_group_id=True
    )
    # warmup
    for _ in range(2):
        batch = next(it)
        batch = _move_batch_to_device(batch, device)
        model.zero_grad(set_to_none=True)
        with _autocast_context(cfg, device):
            out = model(batch.features, batch.scenario_id)
            loss, num, den = _loss_terms_from_batch(
                out, batch, moe_loss_weight=cfg.model.sparse_moe_loss_weight,
                loss_reduction=cfg.training.loss_reduction, rank_active=True,
                active_rank_count=1,
            )
        loss.backward()
        for opt in optimizers:
            opt.step()

    batch = next(it)
    batch = _move_batch_to_device(batch, device)
    model.zero_grad(set_to_none=True)
    with _autocast_context(cfg, device):
        out = model(batch.features, batch.scenario_id)
        loss, num, den = _loss_terms_from_batch(
            out, batch, moe_loss_weight=cfg.model.sparse_moe_loss_weight,
            loss_reduction=cfg.training.loss_reduction, rank_active=True,
            active_rank_count=1,
        )
    loss.backward()
    torch.cuda.synchronize()

    sparse_touched = 0
    sparse_rows = 0
    dense_touched = 0
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    sparse_touched += 1
                    gg = g.coalesce() if not g.is_coalesced() else g
                    sparse_rows += int(gg.indices().size(1))
                else:
                    dense_touched += 1
    print("sparse_tables_touched", sparse_touched, "sparse_rows", sparse_rows, "dense", dense_touched)

    t0 = time.perf_counter()
    for opt in optimizers:
        opt.step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print("optim_step_ms", round((t1 - t0) * 1000, 1))

    # second step timing average
    times = []
    for _ in range(5):
        batch = next(it)
        batch = _move_batch_to_device(batch, device)
        model.zero_grad(set_to_none=True)
        with _autocast_context(cfg, device):
            out = model(batch.features, batch.scenario_id)
            loss, num, den = _loss_terms_from_batch(
                out, batch, moe_loss_weight=cfg.model.sparse_moe_loss_weight,
                loss_reduction=cfg.training.loss_reduction, rank_active=True,
                active_rank_count=1,
            )
        loss.backward()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for opt in optimizers:
            opt.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    print("optim_ms", [round(x, 1) for x in times], "mean", round(sum(times) / len(times), 1))


if __name__ == "__main__":
    main()
