"""CUDA-graph parameter-surface regression tests."""

from __future__ import annotations

from datetime import timedelta
import os
import socket
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from torch import nn

from src.model import (
    _MDLRankMixerGraphedStack,
    _MDLRankMixerSplitGraphedStack,
    _RankMixerGraphedStack,
    _cuda_graph_prewarm_batch_sizes,
    _validate_cuda_graph_module_pool,
)


class _Bucket:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size


class _Config:
    def __init__(self, sizes: list[int], batch_size: int) -> None:
        self.data = SimpleNamespace(
            train=SimpleNamespace(
                reader=SimpleNamespace(
                    length_buckets=[_Bucket(size) for size in sizes]
                )
            )
        )
        self.training = SimpleNamespace(batch_size=batch_size)


class _RankMixerOwner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # This upstream module runs outside the captured dense stack.
        self.encoder = nn.Linear(3, 4)
        self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.logit_layers = nn.ModuleList([nn.Linear(4, 1), nn.Linear(4, 1)])


class _MDLBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.feature = nn.Linear(width, width)
        self.scenario = nn.Linear(width, width)
        self.task = nn.Linear(width, width)

    def forward(
        self,
        feature_tokens: torch.Tensor,
        scenario_tokens: torch.Tensor,
        task_tokens: torch.Tensor,
        scenario_mask: torch.Tensor,
        scenario_prompts: torch.Tensor | None = None,
        task_prompts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del scenario_mask
        scenario_input = (
            scenario_tokens
            if scenario_prompts is None
            else scenario_tokens + scenario_prompts
        )
        task_input = task_tokens if task_prompts is None else task_tokens + task_prompts
        feature_output = self.feature(feature_tokens)
        scenario_output = self.scenario(scenario_input)
        task_output = (
            self.task(task_input)
            + feature_output.mean(dim=1, keepdim=True)
            + scenario_output.mean(dim=1, keepdim=True)
        )
        return feature_output, scenario_output, task_output


class _MDLOwner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model=SimpleNamespace(
                use_scenario_tokens=True,
                use_task_tokens=True,
            )
        )
        # This upstream module must not become an implicit graph input.
        self.encoder = nn.Linear(3, 4)
        self.blocks = nn.ModuleList([_MDLBlock(4)])
        self.logit_layers = nn.ModuleList([nn.Linear(4, 1), nn.Linear(4, 1)])
        self.scenario_tower = None


class _HiddenOwnerWrapper(nn.Module):
    def __init__(self, owner: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "owner", owner)


def _parameter_ids(module: nn.Module) -> set[int]:
    return {id(parameter) for parameter in module.parameters()}


def _expected_dense_parameter_ids(owner: nn.Module) -> set[int]:
    return _parameter_ids(owner.blocks) | _parameter_ids(owner.logit_layers)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _CudaGraphDDPToy(_RankMixerOwner):
    """Small production-shaped graph/eager RankMixer switch."""

    def __init__(self) -> None:
        super().__init__()
        # A plain dict mirrors RankMixerModel and keeps the wrapper out of the
        # owner's registered module tree.
        self.graph_pool: dict[int, nn.Module] = {}

    def prewarm(self, batch_size: int) -> None:
        sample = torch.randn(
            batch_size,
            2,
            4,
            device=self.encoder.weight.device,
            requires_grad=True,
        )
        self.graph_pool[batch_size] = torch.cuda.make_graphed_callables(
            _RankMixerGraphedStack(self),
            (sample,),
            num_warmup_iters=1,
            allow_unused_input=True,
        )
        torch.cuda.synchronize()

    def _dense(self, feature_tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            feature_tokens = block(feature_tokens)
        pooled = feature_tokens.mean(dim=1)
        return torch.cat([head(pooled) for head in self.logit_layers], dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        feature_tokens = self.encoder(inputs).unsqueeze(1).expand(-1, 2, -1)
        graphed = self.graph_pool.get(int(inputs.size(0)))
        if graphed is None:
            return self._dense(feature_tokens)
        return graphed(feature_tokens)


def _cuda_graph_ddp_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
    )
    torch.cuda.set_device(rank)
    torch_dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(2026)
        device = torch.device("cuda", rank)
        model = _CudaGraphDDPToy().to(device)
        model.prewarm(4)
        ddp = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            static_graph=True,
            find_unused_parameters=False,
        )
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)
        initial_dense_weight = model.blocks[0].weight.detach().clone()
        # After a common graph iteration, alternate exact graph hits and ragged
        # eager fallbacks across ranks. The old hidden-parameter wrapper failed
        # on the second iteration because the DDP hook set changed by rank.
        batch_sizes = [4, 4, 3, 4] if rank == 0 else [4, 3, 4, 2]
        for batch_size in batch_sizes:
            optimizer.zero_grad(set_to_none=True)
            inputs = torch.randn(batch_size, 3, device=device)
            ddp(inputs).square().mean().backward()
            optimizer.step()

        if torch.equal(initial_dense_weight, model.blocks[0].weight.detach()):
            raise AssertionError("captured RankMixer dense parameters were not updated")
        for parameter in model.parameters():
            rank_zero = parameter.detach().clone()
            torch_dist.broadcast(rank_zero, src=0)
            torch.testing.assert_close(parameter, rank_zero, rtol=0.0, atol=0.0)
    finally:
        torch_dist.destroy_process_group()


class CudaGraphStaticGraphTest(unittest.TestCase):
    def test_prewarm_defaults_to_all_train_bucket_shapes(self) -> None:
        config = _Config([1280, 800, 512, 384, 640], batch_size=1280)
        self.assertEqual(
            _cuda_graph_prewarm_batch_sizes(config),
            [1280, 800, 640, 512, 384],
        )

    def test_prewarm_env_limit_keeps_largest_shapes(self) -> None:
        config = _Config([1280, 800, 512, 384, 640], batch_size=1280)
        with unittest.mock.patch.dict(os.environ, {"MDL_CUDA_GRAPH_PREWARM_LIMIT": "2"}):
            self.assertEqual(_cuda_graph_prewarm_batch_sizes(config), [1280, 800])

    def test_rankmixer_wrapper_exposes_exact_dense_parameters(self) -> None:
        owner = _RankMixerOwner()
        wrapper = _RankMixerGraphedStack(owner)

        self.assertEqual(
            _parameter_ids(wrapper),
            _expected_dense_parameter_ids(owner),
        )
        self.assertTrue(
            _parameter_ids(wrapper).isdisjoint(_parameter_ids(owner.encoder))
        )

        output = wrapper(torch.randn(3, 2, 4))
        output.sum().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in wrapper.parameters())
        )

    def test_pool_validation_rejects_a_hidden_owner_parameter_surface(self) -> None:
        owner = _RankMixerOwner()
        broken = _HiddenOwnerWrapper(owner)
        with self.assertRaisesRegex(
            RuntimeError,
            "exposes no trainable parameters.*DDP backward hooks will be omitted",
        ):
            _validate_cuda_graph_module_pool(
                {((4, 2, 4),): broken},
                label="RankMixerModel",
            )

    def test_pool_validation_accepts_shared_parameters_across_shapes(self) -> None:
        owner = _RankMixerOwner()
        wrappers = {
            ((4, 2, 4),): _RankMixerGraphedStack(owner),
            ((3, 2, 4),): _RankMixerGraphedStack(owner),
        }
        self.assertEqual(
            _validate_cuda_graph_module_pool(wrappers, label="RankMixerModel"),
            len(_expected_dense_parameter_ids(owner)),
        )

    def test_mdl_wrappers_expose_exact_dense_parameters(self) -> None:
        for wrapper_type in (
            _MDLRankMixerGraphedStack,
            _MDLRankMixerSplitGraphedStack,
        ):
            with self.subTest(wrapper=wrapper_type.__name__):
                owner = _MDLOwner()
                wrapper = wrapper_type(owner)
                self.assertEqual(
                    _parameter_ids(wrapper),
                    _expected_dense_parameter_ids(owner),
                )
                self.assertTrue(
                    _parameter_ids(wrapper).isdisjoint(_parameter_ids(owner.encoder))
                )

                feature = torch.randn(3, 2, 4, requires_grad=True)
                scenario = torch.randn(3, 2, 4, requires_grad=True)
                task = torch.randn(3, 2, 4, requires_grad=True)
                mask = torch.ones(3, 2, dtype=torch.bool)
                if wrapper_type is _MDLRankMixerSplitGraphedStack:
                    output = wrapper(
                        feature,
                        scenario,
                        task,
                        mask,
                        torch.randn_like(scenario, requires_grad=True),
                        torch.randn_like(task, requires_grad=True),
                    )
                else:
                    output = wrapper(feature, scenario, task, mask)
                output.sum().backward()
                self.assertTrue(
                    all(
                        parameter.grad is not None
                        for parameter in wrapper.parameters()
                    )
                )

    def test_two_gpu_static_graph_can_mix_graphed_and_ragged_batches(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest(
                "requires a CUDA-capable PyTorch runtime with two visible devices"
            )
        torch_mp.start_processes(
            _cuda_graph_ddp_worker,
            args=(2, _free_port()),
            nprocs=2,
            join=True,
            start_method="spawn",
        )


if __name__ == "__main__":
    unittest.main()
