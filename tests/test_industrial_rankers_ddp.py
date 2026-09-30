"""CPU/Gloo regression: dense backbone DDP, including locally empty histories."""
from datetime import timedelta
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from src.modules.more import MORERanker
from src.modules.uniformer import UniFormerRanker
from scripts.verify_industrial_rankers import config_for, small_model
from src.benchmark import _synthetic_feature_batch, _synthetic_vocab_maps
from src.optim import ShardedRowWiseAdagrad
from src.train import _classify_model_parameters, _exclude_sparse_parameters_from_ddp


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=rendezvous, timeout=timedelta(seconds=45))
    try:
        for name in ("uniformer", "more"):
            torch.manual_seed(51)
            if name == "uniformer":
                base = UniFormerRanker(16, 4, 2, 2, 2, num_user_tokens=1,
                                       fim_layers=2, tim_layers=1, attention_backend="sdpa")
            else:
                base = MORERanker(16, 1, 1, 2, num_heads=4, num_blocks=2,
                                  attention_backend="sdpa")
            model = DistributedDataParallel(base, find_unused_parameters=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for step in range(3):
                torch.manual_seed(61 + rank + step)
                sequence = torch.randn(2, 5, 16)
                mask = torch.ones(2, 5, dtype=torch.bool)
                if rank == step % 2:
                    mask[:] = False
                index = torch.tensor([1, 0, 1])
                optimizer.zero_grad(set_to_none=True)
                if name == "uniformer":
                    output = model(torch.randn(3, 2, 16), torch.randn(3, 2, 16),
                                   [sequence, sequence], [mask, mask], request_index=index)
                else:
                    output = model(sequence, torch.randn(3, 1, 16),
                                   sequence_mask=mask, request_index=index)
                output.logits.square().mean().backward()
                for p in model.parameters():
                    assert p.grad is not None and torch.isfinite(p.grad).all()
                optimizer.step()
                for p in model.parameters():
                    reference = p.detach().clone()
                    dist.broadcast(reference, src=0)
                    torch.testing.assert_close(p, reference, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def _full_model_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=rendezvous, timeout=timedelta(seconds=45))
    try:
        for name in ("uniformer", "more", "mixformer"):
            config = config_for(name, "cpu")
            config = replace(config, training=replace(config.training, embedding_distribution="sharded"),
                             model=replace(config.model, token_dim=32, hidden_dim=64,
                                           num_heads=4, num_layers=1))
            base = small_model(config, _synthetic_vocab_maps(config), torch.device("cpu"))
            groups = _classify_model_parameters(base)
            _exclude_sparse_parameters_from_ddp(base, (*groups.sparse_sync, *groups.sharded_ddp_ignore))
            model = DistributedDataParallel(base, find_unused_parameters=False,
                                             static_graph=name == "mixformer")
            dense = torch.optim.Adam(groups.dense_optimizer, lr=1e-3)
            sparse = ShardedRowWiseAdagrad(groups.sharded_optimizer, lr=1e-3)
            for step in range(3):
                batch = _synthetic_feature_batch(config, torch.device("cpu"), 4, 3,
                                                  70 + rank + step, 2)
                if rank == step % 2:
                    for sequence in config.sequences:
                        batch.features[sequence.name]["lengths"].zero_()
                dense.zero_grad(set_to_none=True)
                sparse.zero_grad(set_to_none=True)
                logits = model(batch.features, batch.scenario_id)["logits"]
                logits.square().mean().backward()
                assert torch.isfinite(logits).all()
                assert all(p.grad is not None and torch.isfinite(p.grad).all()
                           for p in groups.dense_optimizer)
                dense.step()
                sparse.step()
                table = base.encoder_bank.gset_table
                assert sparse.param_groups[0]["params"][0] is table.weight
                assert all(key % 2 == rank for key in table._key_to_row)
                for p in groups.dense_optimizer:
                    reference = p.detach().clone()
                    dist.broadcast(reference, src=0)
                    torch.testing.assert_close(p, reference, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


class IndustrialDDPTest(unittest.TestCase):
    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_two_rank_updates_with_empty_local_histories(self):
        with tempfile.TemporaryDirectory(prefix="mdl-ranker-ddp-") as directory:
            rendezvous = "file://" + str(Path(directory) / "rendezvous")
            mp.spawn(_worker, args=(rendezvous,), nprocs=2, join=True)

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_full_models_with_sharded_dynamic_embeddings(self):
        with tempfile.TemporaryDirectory(prefix="mdl-full-ranker-ddp-") as directory:
            rendezvous = "file://" + str(Path(directory) / "rendezvous")
            mp.spawn(_full_model_worker, args=(rendezvous,), nprocs=2, join=True)
