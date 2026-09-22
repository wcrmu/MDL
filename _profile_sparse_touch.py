#!/usr/bin/env python
"""Profile sparse Adagrad touch count and CUDA time."""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

from src.config import load_app_config
from src.embeddings import ShardedEmbedding
from src.features import load_vocab_maps
from src.model import build_model
from src.optim import ShardedRowWiseAdagrad
from src.train import (
    _autocast_context,
    _loss_terms_from_batch,
    iter_feature_batches,
    move_feature_batch,
)


def main() -> None:
    cfg = load_app_config("artifacts/gpu_util_e2e_mock/rankmixer_e2e.yaml")
    vocab = load_vocab_maps(cfg)
    device = torch.device("cuda")
    model = build_model(cfg, vocab).to(device)
    sparse = [p for m in model.modules() if isinstance(m, ShardedEmbedding) for p in m.parameters()]
    dense = [p for p in model.parameters() if p.requires_grad and p not in set(sparse)]
    print("sparse_tables", len(sparse), "dense", len(dense))
    opt_s = ShardedRowWiseAdagrad(sparse, lr=0.05)
    opt_d = torch.optim.RMSprop(dense, lr=1e-3)
    it = iter_feature_batches(
        cfg, "train", vocab, True, pin_memory=True, include_group_id=True
    )
    for _ in range(2):
        batch = move_feature_batch(next(it), device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        with _autocast_context(cfg, device):
            out = model(batch.features, batch.scenario_id)
            loss, _, _ = _loss_terms_from_batch(
                out,
                batch,
                moe_loss_weight=cfg.model.sparse_moe_loss_weight,
                loss_reduction=cfg.training.loss_reduction,
                rank_active=True,
                active_rank_count=1,
            )
        loss.backward()
        opt_s.step()
        opt_d.step()
        torch.cuda.synchronize()

    batch = move_feature_batch(next(it), device, non_blocking=True)
    model.zero_grad(set_to_none=True)
    with _autocast_context(cfg, device):
        out = model(batch.features, batch.scenario_id)
        loss, _, _ = _loss_terms_from_batch(
            out,
            batch,
            moe_loss_weight=cfg.model.sparse_moe_loss_weight,
            loss_reduction=cfg.training.loss_reduction,
            rank_active=True,
            active_rank_count=1,
        )
    loss.backward()
    torch.cuda.synchronize()
    touched = 0
    rows = 0
    for p in sparse:
        g = p.grad
        if g is None:
            continue
        if not g.is_coalesced():
            g = g.coalesce()
        n = int(g.indices().size(1))
        if n:
            touched += 1
            rows += n
    print("touched_tables", touched, "total_rows", rows)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    opt_s.step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    opt_d.step()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    print(
        "sparse_opt_ms",
        round((t1 - t0) * 1000, 1),
        "dense_opt_ms",
        round((t2 - t1) * 1000, 1),
    )
    close = getattr(it, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
