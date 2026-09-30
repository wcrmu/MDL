from __future__ import annotations

import unittest

import torch

from src.modules.attention import RankMixerTokenMixing
from src.modules.more import (
    MORERanker,
    cartesian_sequence_token,
    gated_rankmixer,
    task_boundary_mask,
    weighted_bce_with_logits,
)


class MOREPaperAlignmentTest(unittest.TestCase):
    def test_task_boundary_keeps_private_anchors_isolated(self) -> None:
        mask = task_boundary_mask(num_ns_tokens=2, num_shared_anchors=1, num_private_anchors=2)
        # Layout: F, F, shared, private0, private1.
        self.assertEqual(tuple(mask.shape), (5, 5))
        for row in range(5):
            self.assertTrue(bool(mask[row, 0]))
            self.assertTrue(bool(mask[row, 1]))
            self.assertTrue(bool(mask[row, 2]))
            self.assertEqual(bool(mask[row, 3]), row == 3)
            self.assertEqual(bool(mask[row, 4]), row == 4)

    def test_unit_gate_matches_rankmixer(self) -> None:
        torch.manual_seed(0)
        tokens = torch.randn(2, 4, 8)
        gates = torch.ones(2, 4, 4)
        mask = torch.ones(4, 4, dtype=torch.bool)
        reference = RankMixerTokenMixing(4, 8)(tokens)
        torch.testing.assert_close(gated_rankmixer(tokens, gates, mask), reference)

    def test_fixed_gate_hides_other_private_content(self) -> None:
        torch.manual_seed(1)
        tokens = torch.randn(1, 4, 8)
        gates = torch.ones(1, 4, 4)
        mask = task_boundary_mask(1, 1, 2)
        left = gated_rankmixer(tokens, gates, mask)
        changed = tokens.clone()
        changed[:, 3] += 5
        right = gated_rankmixer(changed, gates, mask)
        torch.testing.assert_close(left[:, :3], right[:, :3])
        self.assertFalse(torch.allclose(left[:, 3], right[:, 3]))

    def test_cartesian_encoding_is_a_sum(self) -> None:
        item = torch.tensor([[1.0, 2.0]])
        action = torch.tensor([[0.5, -1.0]])
        time = torch.tensor([[0.25, 0.25]])
        torch.testing.assert_close(
            cartesian_sequence_token(item, action, time),
            item + action + time,
        )

    def test_forward_shapes_and_request_sharing(self) -> None:
        torch.manual_seed(2)
        model = _small_more()
        sequence = torch.randn(2, 5, 16)
        nonseq = torch.randn(4, 1, 16)
        request_index = torch.tensor([0, 0, 1, 1])
        shared = model(sequence, nonseq, request_index=request_index)
        expanded = model(sequence.index_select(0, request_index), nonseq)

        self.assertEqual(tuple(shared.probabilities.shape), (4, 2))
        self.assertEqual(tuple(shared.sequence.shape), (2, 5, 16))
        self.assertEqual(tuple(shared.private_anchors.shape), (4, 2, 16))
        torch.testing.assert_close(shared.probabilities, expanded.probabilities)
        torch.testing.assert_close(shared.sequence[0], expanded.sequence[0])
        torch.testing.assert_close(shared.sequence[1], expanded.sequence[2])
        self.assertTrue(torch.isfinite(shared.probabilities).all())

    def test_backward(self) -> None:
        torch.manual_seed(3)
        model = _small_more()
        output = model(torch.randn(3, 4, 16), torch.randn(3, 1, 16))
        loss = weighted_bce_with_logits(output.logits, torch.zeros(3, 2))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(torch.isfinite(loss))

    def test_synthetic_labels_are_learnable(self) -> None:
        torch.manual_seed(4)
        model = _small_more()
        sequence = torch.randn(20, 4, 16)
        nonseq = torch.randn(20, 1, 16)
        with torch.no_grad():
            positive = nonseq.mean(dim=(1, 2)) > 0
            labels = torch.stack([positive, ~positive], dim=1).to(dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)
        initial = _more_loss(model, sequence, nonseq, labels)
        for _ in range(25):
            optimizer.zero_grad()
            _more_loss(model, sequence, nonseq, labels).backward()
            optimizer.step()
        final = _more_loss(model, sequence, nonseq, labels)
        self.assertLess(final.detach().item(), initial.detach().item())

    def test_paper_width_divides_hidden_size(self) -> None:
        # Default MORE: M=15, K=8, T=9, d=256.
        self.assertEqual(256 % (15 + 8 + 9), 0)


def _small_more() -> MORERanker:
    return MORERanker(
        d_model=16,
        num_ns_tokens=1,
        num_shared_anchors=1,
        num_private_anchors=2,
        num_heads=4,
        num_blocks=2,
        ffn_expansion=1,
    )


def _more_loss(model, sequence, nonseq, labels) -> torch.Tensor:
    return weighted_bce_with_logits(model(sequence, nonseq).logits, labels)
