from __future__ import annotations

import os
import socket
import unittest

import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from torch.nn import functional as F
from torch import nn

from src.embeddings import (
    EmbeddingTableSpec,
    ShardedEmbedding,
    embedding_local_bytes,
    grouped_sharded_embedding_lookup,
    plan_embedding_shards,
)
from src.optim import ShardedAdagrad, ShardedRowWiseAdagrad
from src.train import (
    _classify_model_parameters,
    _exclude_sparse_parameters_from_ddp,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _init_gloo(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch_dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _distributed_parity_worker(
    rank: int,
    world_size: int,
    port: int,
    strategy: str,
) -> None:
    _init_gloo(rank, world_size, port)
    try:
        checker = getattr(torch.sparse, "check_sparse_tensor_invariants", None)
        if checker is not None:
            checker.disable()
        table = EmbeddingTableSpec("item", num_embeddings=15, embedding_dim=4)
        plan = plan_embedding_shards(
            [table],
            world_size=world_size,
            strategy=strategy,
            table_wise_max_rows=32,
        )
        embedding = ShardedEmbedding(
            table.num_embeddings,
            table.embedding_dim,
            table_name=table.name,
            shard_spec=plan.tables[table.name],
            padding_idx=0,
        )
        full_weight = torch.arange(
            table.num_embeddings * table.embedding_dim,
            dtype=torch.float32,
        ).view(table.num_embeddings, table.embedding_dim) / 10.0
        full_weight[0].zero_()
        embedding.load_full_weight_(full_weight)
        ids_by_rank = (
            torch.tensor([0, 1, 1, 8, 14], dtype=torch.long),
            torch.tensor([2, 8, 8, 9, 0], dtype=torch.long),
        )
        local_ids = ids_by_rank[rank]

        actual_output = embedding(local_ids)
        expected_output = F.embedding(local_ids, full_weight, padding_idx=0)
        torch.testing.assert_close(actual_output, expected_output)
        actual_output.square().sum().backward()

        optimizer = ShardedAdagrad(
            [embedding.weight],
            lr=0.2,
            lr_decay=0.01,
            initial_accumulator_value=0.1,
            eps=1.0e-10,
        )
        optimizer.step()

        reference_weight = full_weight.clone().requires_grad_(True)
        reference_loss = sum(
            F.embedding(ids, reference_weight, padding_idx=0).square().sum()
            for ids in ids_by_rank
        ) / float(world_size)
        reference_loss.backward()
        reference_optimizer = torch.optim.Adagrad(
            [reference_weight],
            lr=0.2,
            lr_decay=0.01,
            initial_accumulator_value=0.1,
            eps=1.0e-10,
        )
        reference_optimizer.step()

        global_ids = torch.arange(table.num_embeddings)
        owned = plan.tables[table.name].owner(global_ids) == rank
        expected_local_weight = reference_weight.detach()[owned]
        expected_local_state = reference_optimizer.state[reference_weight]["sum"][owned]
        torch.testing.assert_close(
            embedding.weight.detach(), expected_local_weight, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            optimizer.state[embedding.weight]["sum"],
            expected_local_state,
            rtol=1e-6,
            atol=1e-6,
        )
        stats = embedding.consume_communication_stats()
        if stats.local_unique_ids >= stats.active_ids:
            # Both ranks intentionally contain duplicate active IDs.
            raise AssertionError("requester-side duplicate IDs were not merged")
        if stats.backward_sent_bytes <= 0:
            raise AssertionError("backward gradient traffic was not recorded")
    finally:
        torch_dist.destroy_process_group()


class _ToyDistributedModel(nn.Module):
    def __init__(self, embedding: ShardedEmbedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.output = nn.Linear(embedding.embedding_dim, 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.embedding(ids)).sum()


def _ddp_integration_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    try:
        checker = getattr(torch.sparse, "check_sparse_tensor_invariants", None)
        if checker is not None:
            checker.disable()
        table = EmbeddingTableSpec("item", 17, 3)
        plan = plan_embedding_shards(
            [table],
            world_size=world_size,
            strategy="row_wise",
            table_wise_max_rows=4,
        )
        model = _ToyDistributedModel(
            ShardedEmbedding(
                table.num_embeddings,
                table.embedding_dim,
                table_name=table.name,
                shard_spec=plan.tables[table.name],
            )
        )
        groups = _classify_model_parameters(model)
        _exclude_sparse_parameters_from_ddp(model, groups.sharded_ddp_ignore)
        ddp = nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=False,
        )
        dense_optimizer = torch.optim.SGD(groups.dense_optimizer, lr=0.01)
        sparse_optimizer = ShardedAdagrad(
            groups.sharded_optimizer,
            lr=0.1,
            initial_accumulator_value=0.0,
        )
        ids = torch.tensor([1, 3, 3] if rank == 0 else [2, 3, 8])
        loss = ddp(ids)
        loss.backward()
        dense_optimizer.step()
        sparse_optimizer.step()

        for tensor in (model.output.weight.detach(), model.output.bias.detach()):
            rank_zero = tensor.clone()
            torch_dist.broadcast(rank_zero, src=0)
            torch.testing.assert_close(tensor, rank_zero, rtol=0.0, atol=0.0)
        self_state = sparse_optimizer.state[model.embedding.weight]["sum"]
        if tuple(self_state.shape) != tuple(model.embedding.weight.shape):
            raise AssertionError("Adagrad accumulator is not local-shard shaped")
    finally:
        torch_dist.destroy_process_group()


def _grouped_lookup_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    try:
        checker = getattr(torch.sparse, "check_sparse_tensor_invariants", None)
        if checker is not None:
            checker.disable()
        specs = [
            EmbeddingTableSpec("user", 13, 4),
            EmbeddingTableSpec("item", 19, 4),
        ]
        plan = plan_embedding_shards(
            specs,
            world_size=world_size,
            strategy="row_wise",
            table_wise_max_rows=4,
        )
        modules = [
            ShardedEmbedding(
                spec.num_embeddings,
                spec.embedding_dim,
                table_name=spec.name,
                shard_spec=plan.tables[spec.name],
            )
            for spec in specs
        ]
        full_weights = [
            torch.arange(
                spec.num_embeddings * spec.embedding_dim, dtype=torch.float32
            ).view(spec.num_embeddings, spec.embedding_dim)
            / float(10 + index)
            for index, spec in enumerate(specs)
        ]
        for module, weight in zip(modules, full_weights):
            weight[0].zero_()
            module.load_full_weight_(weight)

        ids_by_rank = (
            (
                torch.tensor([1, 1, 4, 0]),
                torch.tensor([2, 8, 8]),
                torch.tensor([4, 7]),
            ),
            (
                torch.tensor([2, 4, 4, 0]),
                torch.tensor([3, 8, 12]),
                torch.tensor([2, 2]),
            ),
        )
        user_ids, item_ids, repeated_user_ids = ids_by_rank[rank]
        actual = grouped_sharded_embedding_lookup(
            [
                (modules[0], user_ids),
                (modules[1], item_ids),
                (modules[0], repeated_user_ids),
            ]
        )
        expected = [
            F.embedding(user_ids, full_weights[0], padding_idx=0),
            F.embedding(item_ids, full_weights[1], padding_idx=0),
            F.embedding(repeated_user_ids, full_weights[0], padding_idx=0),
        ]
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference)
        sum(output.square().sum() for output in actual).backward()

        optimizer = ShardedAdagrad(
            [module.weight for module in modules],
            lr=0.15,
            initial_accumulator_value=0.1,
        )
        optimizer.step()

        reference_weights = [weight.clone().requires_grad_(True) for weight in full_weights]
        reference_loss = 0.0
        for rank_requests in ids_by_rank:
            rank_user, rank_item, rank_user_repeat = rank_requests
            reference_loss = reference_loss + F.embedding(
                rank_user, reference_weights[0], padding_idx=0
            ).square().sum()
            reference_loss = reference_loss + F.embedding(
                rank_item, reference_weights[1], padding_idx=0
            ).square().sum()
            reference_loss = reference_loss + F.embedding(
                rank_user_repeat, reference_weights[0], padding_idx=0
            ).square().sum()
        reference_loss = reference_loss / float(world_size)
        reference_loss.backward()
        reference_optimizer = torch.optim.Adagrad(
            reference_weights,
            lr=0.15,
            initial_accumulator_value=0.1,
        )
        reference_optimizer.step()

        for module, spec, reference_weight in zip(
            modules, specs, reference_weights
        ):
            global_ids = torch.arange(spec.num_embeddings)
            owned = plan.tables[spec.name].owner(global_ids) == rank
            torch.testing.assert_close(
                module.weight.detach(),
                reference_weight.detach()[owned],
                rtol=1e-6,
                atol=1e-6,
            )
            stats = module.consume_communication_stats()
            if stats.backward_sent_bytes <= 0:
                raise AssertionError("grouped backward traffic was not recorded")
    finally:
        torch_dist.destroy_process_group()


class ShardingPlannerTest(unittest.TestCase):
    def test_auto_uses_row_wise_for_large_and_lpt_for_small_tables(self) -> None:
        specs = [
            EmbeddingTableSpec("large", 1000, 8),
            EmbeddingTableSpec("small_a", 20, 8),
            EmbeddingTableSpec("small_b", 10, 8),
        ]
        first = plan_embedding_shards(
            specs,
            world_size=2,
            strategy="auto",
            table_wise_max_rows=32,
        )
        second = plan_embedding_shards(
            reversed(specs),
            world_size=2,
            strategy="auto",
            table_wise_max_rows=32,
        )

        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.tables["large"].strategy, "row_wise")
        self.assertEqual(first.tables["small_a"].strategy, "table_wise")
        self.assertNotEqual(
            first.tables["small_a"].table_owner,
            first.tables["small_b"].table_owner,
        )

    def test_rowwise_cost_model_matches_embedding_local_bytes(self) -> None:
        specs = [
            EmbeddingTableSpec("t0", 1000, 64, element_size=2),
        ]
        plan = plan_embedding_shards(
            specs,
            world_size=2,
            strategy="row_wise",
            table_wise_max_rows=1,
            optimizer_state_layout="rowwise",
        )
        shard = plan.tables["t0"]
        for rank in range(2):
            rows = shard.local_rows(1000, rank)
            self.assertEqual(
                embedding_local_bytes(
                    rows=rows,
                    embedding_dim=64,
                    weight_element_size=2,
                    optimizer_state_layout="rowwise",
                ),
                rows * (64 * 2 + 4),
            )

    def test_rowwise_optimizer_matches_full_table_reference_on_owned_rows(self) -> None:
        torch.manual_seed(0)
        full = nn.Parameter(torch.randn(8, 4, dtype=torch.float32))
        shards = [
            nn.Parameter(full[rank::2].detach().clone())
            for rank in range(2)
        ]
        full_opt = ShardedRowWiseAdagrad(
            [full], lr=0.1, initial_accumulator_value=0.0, eps=1e-8
        )
        shard_opts = [
            ShardedRowWiseAdagrad(
                [shard], lr=0.1, initial_accumulator_value=0.0, eps=1e-8
            )
            for shard in shards
        ]
        # Global sparse rows [0, 3, 5] -> local rows on cyclic owners 0/1/1.
        full.grad = torch.sparse_coo_tensor(
            torch.tensor([[0, 3, 5]]),
            torch.tensor([[1.0, 0.0, -1.0, 0.5], [0.5, 0.5, 0.0, 0.0], [2.0, 0.0, 0.0, 1.0]]),
            size=full.shape,
        ).coalesce()
        full_opt.step()

        shards[0].grad = torch.sparse_coo_tensor(
            torch.tensor([[0]]),
            torch.tensor([[1.0, 0.0, -1.0, 0.5]]),
            size=shards[0].shape,
        ).coalesce()
        shards[1].grad = torch.sparse_coo_tensor(
            torch.tensor([[1, 2]]),
            torch.tensor([[0.5, 0.5, 0.0, 0.0], [2.0, 0.0, 0.0, 1.0]]),
            size=shards[1].shape,
        ).coalesce()
        for opt in shard_opts:
            opt.step()

        self.assertTrue(torch.allclose(shards[0], full[0::2], atol=1e-6))
        self.assertTrue(torch.allclose(shards[1], full[1::2], atol=1e-6))
        self.assertTrue(
            torch.allclose(
                shard_opts[0].state[shards[0]]["sum"],
                full_opt.state[full]["sum"][0::2],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                shard_opts[1].state[shards[1]]["sum"],
                full_opt.state[full]["sum"][1::2],
                atol=1e-6,
            )
        )


class ShardedEmbeddingParityTest(unittest.TestCase):
    def _run_strategy(self, strategy: str) -> None:
        torch_mp.start_processes(
            _distributed_parity_worker,
            args=(2, _free_port(), strategy),
            nprocs=2,
            join=True,
            start_method="spawn",
        )

    def test_row_wise_matches_full_table_adagrad(self) -> None:
        self._run_strategy("row_wise")

    def test_table_wise_matches_full_table_adagrad(self) -> None:
        self._run_strategy("table_wise")

    def test_sharded_parameter_is_excluded_from_dense_ddp(self) -> None:
        torch_mp.start_processes(
            _ddp_integration_worker,
            args=(2, _free_port()),
            nprocs=2,
            join=True,
            start_method="spawn",
        )

    def test_grouped_tables_and_aliases_match_full_table_adagrad(self) -> None:
        torch_mp.start_processes(
            _grouped_lookup_worker,
            args=(2, _free_port()),
            nprocs=2,
            join=True,
            start_method="spawn",
        )


if __name__ == "__main__":
    unittest.main()
