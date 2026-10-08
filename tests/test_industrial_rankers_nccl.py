"""Opt-in two-GPU NCCL smoke test for the production rankers.

This test is intentionally opt-in because a developer workstation may have
other jobs using its GPUs.  Run with ``MDL_TEST_RANKERS_NCCL=1`` and an
explicit ``CUDA_VISIBLE_DEVICES`` containing exactly two GPUs.
"""

from __future__ import annotations

from datetime import timedelta
from dataclasses import replace
import os
import socket
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from scripts.verify_industrial_rankers import config_for, small_model
from src.benchmark import _synthetic_feature_batch, _synthetic_vocab_maps
from src.modules.gset import iter_gset_tables
from src.optim import ShardedRowWiseAdagrad
from src.train import _classify_model_parameters, _exclude_sparse_parameters_from_ddp


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, port: int) -> None:
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=2,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=90),
    )
    try:
        for name in ("uniformer", "more", "mixformer"):
            torch.manual_seed(1100)
            config = config_for(name, f"cuda:{rank}")
            config = replace(
                config,
                training=replace(config.training, embedding_distribution="sharded"),
                model=replace(config.model, token_dim=32, hidden_dim=64,
                              num_heads=4, num_layers=1),
            )
            base = small_model(config, _synthetic_vocab_maps(config), device)
            groups = _classify_model_parameters(base)
            _exclude_sparse_parameters_from_ddp(
                base, (*groups.sparse_sync, *groups.sharded_ddp_ignore)
            )
            model = DistributedDataParallel(
                base, device_ids=[rank], output_device=rank,
                find_unused_parameters=False, static_graph=name == "mixformer",
            )
            dense = torch.optim.Adam(groups.dense_optimizer, lr=1e-3)
            sparse = ShardedRowWiseAdagrad(groups.sharded_optimizer, lr=1e-3)
            for step in range(3):
                # Distinct local batches exercise the NCCL all-reduce.  On the
                # second step one rank has an entirely empty history.
                batch = _synthetic_feature_batch(
                    config, device, 2, 2, 900 + rank * 17 + step, 2,
                )
                if rank == step % 2:
                    for sequence in config.sequences:
                        batch.features[sequence.name]["lengths"].zero_()
                dense.zero_grad(set_to_none=True)
                sparse.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch.features, batch.scenario_id)["logits"]
                    loss = logits.square().mean()
                loss.backward()
                assert torch.isfinite(logits).all()
                assert all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in groups.dense_optimizer
                )
                dense.step()
                sparse.step()
                # Dense DDP parameters must be identical after every update.
                for parameter in groups.dense_optimizer:
                    reference = parameter.detach().clone()
                    dist.broadcast(reference, src=0)
                    torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
                table = next(iter(iter_gset_tables(base)))
                assert sparse.param_groups[0]["params"][0] is table.weight
                assert all(key % 2 == rank for key in table._key_to_row)
            dist.barrier()
    finally:
        dist.destroy_process_group()


class IndustrialNCCLTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MDL_TEST_RANKERS_NCCL") == "1"
        and torch.cuda.is_available()
        and dist.is_nccl_available(),
        "set MDL_TEST_RANKERS_NCCL=1 with two visible CUDA devices",
    )
    def test_two_gpu_nccl_full_rankers(self):
        visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
        if len(visible) != 2:
            self.skipTest("CUDA_VISIBLE_DEVICES must contain exactly two GPUs")
        with tempfile.TemporaryDirectory(prefix="mdl-ranker-nccl-"):
            mp.spawn(_worker, args=(_free_port(),), nprocs=2, join=True)
