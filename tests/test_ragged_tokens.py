"""Ragged token packing keeps real events and matches masked attention."""

from __future__ import annotations

import unittest

import torch

from src.modules.ragged import (
    apply_masked_tokenwise,
    pack_masked_sequences,
    pack_role_streams,
    pack_valid_rows,
    read_ragged_role,
)
from src.modules.ranking_utils import RequestLayout, SequenceMemory, read_sequence
from src.modules.uniformer import QueryCrossAttention


class RaggedTokenTest(unittest.TestCase):
    def test_pack_keeps_real_tokens_and_prefix_lengths(self) -> None:
        role0 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        role1 = torch.tensor([[7.0, 8.0]])
        packed = pack_role_streams(
            [role0, role1],
            [torch.tensor([2, 1]), torch.tensor([0, 1])],
        )

        self.assertEqual(tuple(packed.values.shape), (4, 2))
        torch.testing.assert_close(
            packed.batch_lengths,
            torch.tensor([2, 2], dtype=torch.int32),
        )
        torch.testing.assert_close(
            packed.batch_lengths_added,
            torch.tensor([0, 2, 4], dtype=torch.int32),
        )
        torch.testing.assert_close(packed.role, torch.tensor([0, 0, 0, 1], dtype=torch.int32))
        torch.testing.assert_close(packed.values[0], role0[0])
        torch.testing.assert_close(packed.values[2], role0[2])
        torch.testing.assert_close(packed.values[3], role1[0])
        selected, cumulative, longest = packed.role_values(1)
        self.assertEqual(longest, 1)
        torch.testing.assert_close(cumulative, torch.tensor([0, 0, 1], dtype=torch.int32))
        torch.testing.assert_close(selected, role1)

    def test_masked_pack_drops_padding(self) -> None:
        sequence = torch.arange(12, dtype=torch.float32).view(2, 3, 2)
        mask = torch.tensor([[True, False, True], [False, False, False]])
        packed = pack_masked_sequences([sequence], [mask])
        torch.testing.assert_close(packed.values, sequence[mask])
        torch.testing.assert_close(packed.batch_lengths, torch.tensor([2, 0], dtype=torch.int32))

    def test_varlen_read_matches_masked_sequence(self) -> None:
        torch.manual_seed(0)
        attention = QueryCrossAttention(d_model=8, num_heads=2)
        query = torch.randn(3, 2, 8)
        key = torch.randn(3, 5, 8)
        mask = torch.tensor(
            [
                [True, True, False, False, False],
                [False, False, False, False, False],
                [True, False, True, True, False],
            ]
        )
        dense = attention(query, key, key, mask)
        packed = pack_masked_sequences([key], [mask])
        ragged = attention.read_role(query, packed, 0)
        torch.testing.assert_close(ragged, dense)

        request_index = torch.tensor([0, 0, 1, 2])
        layout = RequestLayout.build(request_index, 3)
        candidate_query = query.index_select(0, request_index)
        dense_grouped = read_sequence(
            attention.q_proj(candidate_query),
            SequenceMemory(key, key, mask),
            heads=2,
            layout=layout,
        )
        ragged_grouped = read_ragged_role(
            attention.q_proj(candidate_query),
            packed,
            0,
            heads=2,
            layout=layout,
        )
        torch.testing.assert_close(ragged_grouped, dense_grouped)

    def test_pack_valid_rows_keeps_mixed_roles_in_time_order(self) -> None:
        tokens = torch.arange(16, dtype=torch.float32).view(2, 4, 2)
        mask = torch.tensor(
            [
                [True, False, True, True],
                [False, True, False, False],
            ]
        )
        role = torch.tensor(
            [
                [0, 1, 2, 0],
                [3, 1, 0, 2],
            ],
            dtype=torch.int32,
        )
        packed = pack_valid_rows(tokens, mask, role, role_count=4)
        torch.testing.assert_close(packed.values, tokens[mask])
        torch.testing.assert_close(
            packed.batch_lengths,
            torch.tensor([3, 1], dtype=torch.int32),
        )
        torch.testing.assert_close(
            packed.batch_lengths_added,
            torch.tensor([0, 3, 4], dtype=torch.int32),
        )
        torch.testing.assert_close(
            packed.role,
            torch.tensor([0, 2, 0, 1], dtype=torch.int32),
        )
        torch.testing.assert_close(
            packed.role_lengths,
            torch.tensor(
                [[2, 0, 1, 0], [0, 1, 0, 0]],
                dtype=torch.int32,
            ),
        )
        gathered = packed.index_rows(torch.tensor([1, 0]))
        torch.testing.assert_close(gathered.values, tokens[mask][torch.tensor([3, 0, 1, 2])])
        torch.testing.assert_close(
            gathered.batch_lengths,
            torch.tensor([1, 3], dtype=torch.int32),
        )

    def test_masked_tokenwise_skips_padding_and_keeps_gradients(self) -> None:
        linear = torch.nn.Linear(2, 2, bias=True)
        tokens = torch.randn(2, 3, 2, requires_grad=True)
        mask = torch.tensor([[True, False, True], [False, False, False]])
        output = apply_masked_tokenwise(linear, tokens, mask)
        self.assertEqual(tuple(output.shape), (2, 3, 2))
        torch.testing.assert_close(output[0, 1], torch.zeros(2))
        torch.testing.assert_close(output[1], torch.zeros(3, 2))
        reference = tokens.new_zeros(2, 3, 2)
        reference[mask] = linear(tokens[mask])
        torch.testing.assert_close(output, reference)
        output.sum().backward()
        self.assertIsNotNone(linear.weight.grad)
        self.assertIsNotNone(tokens.grad)
        torch.testing.assert_close(tokens.grad[0, 1], torch.zeros(2))
        torch.testing.assert_close(tokens.grad[1], torch.zeros(3, 2))


if __name__ == "__main__":
    unittest.main()
