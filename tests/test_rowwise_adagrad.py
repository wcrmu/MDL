from __future__ import annotations

from dataclasses import replace
import unittest

import torch
from torch import nn

from src.config import load_app_config
from src.embeddings import embedding_local_bytes, plan_embedding_shards, EmbeddingTableSpec
from src.optim import ShardedRowWiseAdagrad


def _coo_grad(
    rows: list[int],
    values: list[list[float]],
    *,
    size: tuple[int, int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    indices = torch.tensor([rows], dtype=torch.int64)
    value_tensor = torch.tensor(values, dtype=dtype)
    return torch.sparse_coo_tensor(indices, value_tensor, size=size).coalesce()


class ShardedRowWiseAdagradTest(unittest.TestCase):
    def test_state_shape_dtype_and_cpu_step(self) -> None:
        parameter = nn.Parameter(torch.zeros(4, 2, dtype=torch.bfloat16))
        optimizer = ShardedRowWiseAdagrad(
            [parameter],
            lr=0.1,
            initial_accumulator_value=0.1,
        )
        state = optimizer.state[parameter]
        self.assertEqual(tuple(state["sum"].shape), (4,))
        self.assertEqual(state["sum"].dtype, torch.float32)
        self.assertEqual(state["step"].device.type, "cpu")
        self.assertEqual(state["step"].dtype, torch.float64)

    def test_numerical_reference_uses_mean_not_sum(self) -> None:
        parameter = nn.Parameter(torch.zeros(4, 2, dtype=torch.float32))
        optimizer = ShardedRowWiseAdagrad(
            [parameter],
            lr=0.5,
            initial_accumulator_value=0.0,
            eps=1.0e-8,
        )
        parameter.grad = _coo_grad(
            [1, 3],
            [[2.0, 0.0], [1.0, -1.0]],
            size=parameter.shape,
        )
        optimizer.step()

        state = optimizer.state[parameter]["sum"]
        # mean([4, 0]) = 2 ; mean([1, 1]) = 1
        self.assertTrue(torch.allclose(state[1], torch.tensor(2.0)))
        self.assertTrue(torch.allclose(state[3], torch.tensor(1.0)))
        expected_row1 = torch.tensor([2.0, 0.0]) / (2.0**0.5 + 1.0e-8) * 0.5
        expected_row3 = torch.tensor([1.0, -1.0]) / (1.0**0.5 + 1.0e-8) * 0.5
        self.assertTrue(torch.allclose(parameter[1].neg(), expected_row1, atol=1e-6))
        self.assertTrue(torch.allclose(parameter[3].neg(), expected_row3, atol=1e-6))

    def test_repeated_ids_coalesce_once(self) -> None:
        parameter = nn.Parameter(torch.zeros(5, 2, dtype=torch.float32))
        optimizer = ShardedRowWiseAdagrad(
            [parameter],
            lr=1.0,
            initial_accumulator_value=0.0,
            eps=1.0e-8,
        )
        # Three contributions to row 3 that coalesce to [3, 0].
        parameter.grad = torch.sparse_coo_tensor(
            torch.tensor([[3, 3, 3]]),
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
            size=parameter.shape,
        )
        optimizer.step()
        state = optimizer.state[parameter]["sum"]
        self.assertTrue(torch.allclose(state[3], torch.tensor(4.5)))  # mean([9, 0])
        self.assertEqual(int((state != 0).sum().item()), 1)

    def test_empty_sparse_gradient_is_noop(self) -> None:
        parameter = nn.Parameter(torch.ones(3, 2, dtype=torch.float32))
        optimizer = ShardedRowWiseAdagrad(
            [parameter],
            lr=0.1,
            initial_accumulator_value=0.25,
        )
        before_weight = parameter.detach().clone()
        before_state = optimizer.state[parameter]["sum"].clone()
        parameter.grad = torch.sparse_coo_tensor(
            torch.empty((1, 0), dtype=torch.int64),
            torch.empty((0, 2), dtype=torch.float32),
            size=parameter.shape,
        )
        optimizer.step()
        self.assertTrue(torch.equal(parameter, before_weight))
        self.assertTrue(torch.equal(optimizer.state[parameter]["sum"], before_state))
        self.assertEqual(float(optimizer.state[parameter]["step"].item()), 1.0)

    def test_untouched_rows_remain_unchanged_for_bf16_weights(self) -> None:
        parameter = nn.Parameter(torch.arange(8, dtype=torch.bfloat16).reshape(4, 2))
        before = parameter.detach().clone()
        optimizer = ShardedRowWiseAdagrad(
            [parameter],
            lr=0.25,
            initial_accumulator_value=0.0,
            eps=1.0e-8,
        )
        parameter.grad = _coo_grad(
            [1, 1],
            [[1.0, -1.0], [1.0, 1.0]],
            size=parameter.shape,
            dtype=torch.bfloat16,
        )
        optimizer.step()
        # Coalesced row-1 grad is [2, 0]; rows 0/2/3 must stay put.
        self.assertTrue(torch.equal(parameter[0], before[0]))
        self.assertTrue(torch.equal(parameter[2], before[2]))
        self.assertTrue(torch.equal(parameter[3], before[3]))
        self.assertFalse(torch.equal(parameter[1], before[1]))
        state = optimizer.state[parameter]
        self.assertEqual(tuple(state["sum"].shape), (4,))
        self.assertEqual(state["sum"].dtype, torch.float32)
        self.assertEqual(state["step"].device.type, "cpu")
        self.assertTrue(torch.allclose(state["sum"][1], torch.tensor(2.0)))
        self.assertTrue(torch.allclose(state["sum"][[0, 2, 3]], torch.zeros(3)))

    def test_state_dict_round_trip_keeps_fp32_accumulator(self) -> None:
        parameter = nn.Parameter(torch.zeros(4, 2, dtype=torch.bfloat16))
        optimizer = ShardedRowWiseAdagrad(
            [parameter],
            lr=0.1,
            initial_accumulator_value=0.1,
        )
        parameter.grad = _coo_grad(
            [0, 2],
            [[1.0, 2.0], [0.5, 0.5]],
            size=parameter.shape,
            dtype=torch.bfloat16,
        )
        optimizer.step()
        payload = optimizer.state_dict()

        restored_parameter = nn.Parameter(parameter.detach().clone())
        restored = ShardedRowWiseAdagrad(
            [restored_parameter],
            lr=0.1,
            initial_accumulator_value=0.1,
        )
        restored.load_state_dict(payload)
        state = restored.state[restored_parameter]
        self.assertEqual(state["sum"].dtype, torch.float32)
        self.assertEqual(state["step"].device.type, "cpu")
        self.assertTrue(
            torch.allclose(
                state["sum"].float(),
                optimizer.state[parameter]["sum"].float(),
            )
        )

        restored_parameter.grad = _coo_grad(
            [1],
            [[1.0, 0.0]],
            size=restored_parameter.shape,
            dtype=torch.bfloat16,
        )
        restored.step()
        self.assertEqual(state["sum"].dtype, torch.float32)


class PlannerAndConfigTest(unittest.TestCase):
    def test_embedding_local_bytes_formulas(self) -> None:
        self.assertEqual(
            embedding_local_bytes(
                rows=1000,
                embedding_dim=64,
                weight_element_size=2,
                optimizer_state_layout="full",
            ),
            1000 * 64 * (2 + 4),
        )
        self.assertEqual(
            embedding_local_bytes(
                rows=1000,
                embedding_dim=64,
                weight_element_size=2,
                optimizer_state_layout="rowwise",
            ),
            1000 * (64 * 2 + 4),
        )

    def test_plan_default_layout_is_full_and_rowwise_changes_fingerprint(self) -> None:
        specs = [
            EmbeddingTableSpec(name="a", num_embeddings=100, embedding_dim=8, element_size=2),
            EmbeddingTableSpec(name="b", num_embeddings=50, embedding_dim=4, element_size=2),
        ]
        full = plan_embedding_shards(
            specs,
            world_size=2,
            strategy="row_wise",
            table_wise_max_rows=1,
        )
        rowwise = plan_embedding_shards(
            specs,
            world_size=2,
            strategy="row_wise",
            table_wise_max_rows=1,
            optimizer_state_layout="rowwise",
        )
        self.assertNotEqual(full.fingerprint, rowwise.fingerprint)

    def test_config_accepts_rowwise_sharded_and_rejects_invalid(self) -> None:
        root = __import__("pathlib").Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "mdl_rankmixer.yaml")
        self.assertEqual(config.training.sparse_optimizer, "rowwise_adagrad")
        self.assertEqual(config.training.embedding_distribution, "sharded")
        self.assertEqual(config.runtime.attention_backend, "flash")
        self.assertEqual(config.runtime.nproc_per_node, 2)

        bad_decay = replace(
            config.training,
            adagrad_weight_decay=0.01,
        )
        with self.assertRaisesRegex(ValueError, "adagrad_weight_decay"):
            bad_decay.validate()

        bad_dist = replace(
            config.training,
            embedding_distribution="replicated",
        )
        with self.assertRaisesRegex(ValueError, "embedding_distribution=sharded"):
            bad_dist.validate()


if __name__ == "__main__":
    unittest.main()
