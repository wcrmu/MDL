from __future__ import annotations

import os
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from torch import nn

from src.dataloader import FeatureBatch
from src.train import (
    DistributedContext,
    _NamedSparseParameter,
    _ReplicatedSparseGradientSynchronizer,
    _classify_model_parameters,
    _clip_grad_norm,
    _exclude_sparse_parameters_from_ddp,
    _mark_sparse_invariant_checks_explicitly_disabled,
    _synchronize_sparse_parameter_replicas,
    train_mdl,
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


def _sparse_sync_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    try:
        _mark_sparse_invariant_checks_explicitly_disabled()
        first = nn.Parameter(torch.zeros(6, 2))
        second = nn.Parameter(torch.zeros(5, 3))
        empty = nn.Parameter(torch.zeros(4, 2))
        refs = (
            _NamedSparseParameter("first.weight", first),
            _NamedSparseParameter("second.weight", second),
            _NamedSparseParameter("empty.weight", empty),
        )
        if rank == 0:
            first.grad = torch.sparse_coo_tensor(
                torch.tensor([[1, 3, 3]]),
                torch.tensor([[2.0, 4.0], [1.0, 1.0], [3.0, 5.0]]),
                first.shape,
            )
            second.grad = None
            empty.grad = torch.sparse_coo_tensor(
                torch.empty((1, 0), dtype=torch.long),
                torch.empty((0, 2)),
                empty.shape,
            )
        else:
            first.grad = torch.sparse_coo_tensor(
                torch.tensor([[2, 3]]),
                torch.tensor([[6.0, 8.0], [2.0, 4.0]]),
                first.shape,
            )
            second.grad = torch.sparse_coo_tensor(
                torch.tensor([[4]]),
                torch.tensor([[8.0, 10.0, 12.0]]),
                second.shape,
            )
            empty.grad = None

        context = DistributedContext(
            enabled=True,
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        stats = _ReplicatedSparseGradientSynchronizer(context, refs).synchronize()

        expected_first = torch.zeros_like(first)
        expected_first[1] = torch.tensor([1.0, 2.0])
        expected_first[2] = torch.tensor([3.0, 4.0])
        expected_first[3] = torch.tensor([3.0, 5.0])
        expected_second = torch.zeros_like(second)
        expected_second[4] = torch.tensor([4.0, 5.0, 6.0])
        torch.testing.assert_close(first.grad.to_dense(), expected_first)
        torch.testing.assert_close(second.grad.to_dense(), expected_second)
        if empty.grad is None or empty.grad._nnz() != 0:
            raise AssertionError("globally present empty COO gradient was not preserved")
        if stats.global_rows != 4:
            raise AssertionError(f"expected four global rows, got {stats.global_rows}")

        optimizer = torch.optim.Adagrad(
            [first, second, empty],
            lr=0.1,
            initial_accumulator_value=0.0,
        )
        optimizer.step()
        for tensor in (
            first.detach(),
            second.detach(),
            empty.detach(),
            optimizer.state[first]["sum"],
            optimizer.state[second]["sum"],
            optimizer.state[empty]["sum"],
        ):
            rank_zero = tensor.clone()
            torch_dist.broadcast(rank_zero, src=0)
            torch.testing.assert_close(tensor, rank_zero, rtol=0.0, atol=0.0)
    finally:
        torch_dist.destroy_process_group()


class _ToySparseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(12, 2, sparse=True)
        self.position = nn.Embedding(4, 2)
        self.output = nn.Linear(2, 1)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.position.weight.zero_()
            self.output.weight.fill_(1.0)
            self.output.bias.zero_()

    def forward(
        self,
        features: dict[str, torch.Tensor],
        scenario_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del scenario_id
        values = self.embedding(features["ids"])
        return {"logits": self.output(values)}


def _toy_config() -> SimpleNamespace:
    return SimpleNamespace(
        runtime=SimpleNamespace(device="cpu", precision="fp32", compile=False),
        training=SimpleNamespace(
            sparse_update_mode="ddp_synced_adagrad",
            sparse_parameter_server_adapter=None,
            lr_dense=0.01,
            lr_sparse=0.1,
            lr_schedule="constant",
            lr_warmup_steps=0,
            lr_decay_steps=None,
            lr_min_ratio=0.0,
            rmsprop_alpha=0.9,
            rmsprop_momentum=0.0,
            adagrad_lr_decay=0.0,
            adagrad_weight_decay=0.0,
            adagrad_initial_accumulator_value=0.0,
            adagrad_eps=1.0e-10,
            dense_clip_norm=None,
            sparse_clip_norm=None,
            loss_reduction="sum",
            checkpoint_path=None,
        ),
        model=SimpleNamespace(sparse_moe_loss_weight=0.0),
        task_names=["task"],
    )


def _feature_batch(ids: list[int]) -> FeatureBatch:
    values = torch.tensor(ids, dtype=torch.long)
    return FeatureBatch(
        features={"ids": values},
        labels=torch.zeros(len(ids), 1),
        label_mask=torch.ones(len(ids), 1, dtype=torch.bool),
        scenario_id=torch.zeros(len(ids), dtype=torch.long),
        group_id=[],
    )


def _uneven_train_worker(
    rank: int,
    world_size: int,
    port: int,
    output_queue: object,
) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    batches = (
        [_feature_batch([1, 3]), _feature_batch([5])]
        if rank == 0
        else [_feature_batch([2, 3])]
    )
    model_holder: list[_ToySparseModel] = []

    def build_model(_config: object, _vocabs: object) -> _ToySparseModel:
        model = _ToySparseModel()
        model_holder.append(model)
        return model

    with (
        patch("src.train.load_vocab_maps", return_value={}),
        patch("src.train.build_model", side_effect=build_model),
        patch("src.train.iter_feature_batches", return_value=iter(batches)),
        patch("src.train._non_blocking_transfer", return_value=False),
    ):
        result = train_mdl(
            _toy_config(),
            save_checkpoint=False,
            log_steps=False,
        )

    model = model_holder[0]
    output_queue.put(
        {
            "rank": rank,
            "steps": result.steps,
            "rows": result.rows,
            "embedding": model.embedding.weight.detach().tolist(),
            "output_weight": model.output.weight.detach().tolist(),
        }
    )


def _nccl_sparse_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    torch_dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda", rank)
        context = DistributedContext(
            enabled=True,
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            device=device,
        )
        model = _ToySparseModel().to(device)
        groups = _classify_model_parameters(model)
        _synchronize_sparse_parameter_replicas(context, groups.sparse_sync)
        _exclude_sparse_parameters_from_ddp(model, groups.sparse_sync)
        ddp = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )
        dense_optimizer = torch.optim.RMSprop(groups.dense_optimizer, lr=0.01)
        embedding_optimizer = torch.optim.Adagrad(
            groups.embedding_optimizer,
            lr=0.1,
            initial_accumulator_value=0.0,
        )
        ids = torch.tensor([1, 3] if rank == 0 else [2, 3], device=device)
        scenario_id = torch.zeros(ids.numel(), dtype=torch.long, device=device)
        logits = ddp({"ids": ids}, scenario_id)["logits"]
        logits.sum().backward()
        _ReplicatedSparseGradientSynchronizer(context, groups.sparse_sync).synchronize()
        dense_optimizer.step()
        embedding_optimizer.step()

        for tensor in (
            model.embedding.weight.detach(),
            embedding_optimizer.state[model.embedding.weight]["sum"],
        ):
            rank_zero = tensor.clone()
            torch_dist.broadcast(rank_zero, src=0)
            torch.testing.assert_close(tensor, rank_zero, rtol=0.0, atol=0.0)
    finally:
        torch_dist.destroy_process_group()


class SparseDDPTest(unittest.TestCase):
    def test_clip_grad_norm_handles_dense_and_sparse_values_without_branching(self) -> None:
        _mark_sparse_invariant_checks_explicitly_disabled()
        dense = nn.Parameter(torch.zeros(2))
        sparse = nn.Parameter(torch.zeros(2, 1))
        dense.grad = torch.tensor([3.0, 4.0])
        sparse.grad = torch.sparse_coo_tensor(
            torch.tensor([[1]]),
            torch.tensor([[12.0]]),
            sparse.shape,
        )

        norm = _clip_grad_norm([dense, sparse], 6.5)

        torch.testing.assert_close(norm, torch.tensor(13.0))
        torch.testing.assert_close(
            dense.grad,
            torch.tensor([1.5, 2.0]),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            sparse.grad.coalesce().values(),
            torch.tensor([[6.0]]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_parameter_groups_keep_optimizer_and_sparse_sync_roles_separate(self) -> None:
        model = _ToySparseModel()
        groups = _classify_model_parameters(model)

        self.assertEqual(
            [ref.name for ref in groups.sparse_sync],
            ["embedding.weight"],
        )
        embedding_ids = {id(parameter) for parameter in groups.embedding_optimizer}
        sparse_sync_ids = {id(ref.parameter) for ref in groups.sparse_sync}
        dense_ids = {id(parameter) for parameter in groups.dense_optimizer}
        self.assertIn(id(model.embedding.weight), embedding_ids)
        self.assertIn(id(model.position.weight), embedding_ids)
        self.assertNotIn(id(model.position.weight), sparse_sync_ids)
        self.assertIn(id(model.output.weight), dense_ids)

    def test_sparse_rows_and_adagrad_state_are_identical_across_ranks(self) -> None:
        torch_mp.start_processes(
            _sparse_sync_worker,
            args=(2, _free_port()),
            nprocs=2,
            join=True,
            start_method="spawn",
        )

    def test_training_replays_zero_loss_until_the_longest_shard_finishes(self) -> None:
        context = torch_mp.get_context("spawn")
        output_queue = context.SimpleQueue()
        torch_mp.start_processes(
            _uneven_train_worker,
            args=(2, _free_port(), output_queue),
            nprocs=2,
            join=True,
            start_method="spawn",
        )
        results = sorted(
            [output_queue.get(), output_queue.get()],
            key=lambda item: item["rank"],
        )

        self.assertEqual([item["steps"] for item in results], [2, 2])
        self.assertEqual([item["rows"] for item in results], [5, 5])
        self.assertEqual(results[0]["embedding"], results[1]["embedding"])
        self.assertEqual(results[0]["output_weight"], results[1]["output_weight"])
        self.assertNotEqual(results[0]["embedding"][5], [0.0, 0.0])

    def test_two_gpu_nccl_sparse_smoke(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest(
                "requires a CUDA-capable PyTorch runtime with at least two visible devices"
            )
        torch_mp.start_processes(
            _nccl_sparse_worker,
            args=(2, _free_port()),
            nprocs=2,
            join=True,
            start_method="spawn",
        )


if __name__ == "__main__":
    unittest.main()
