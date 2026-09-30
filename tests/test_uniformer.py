from __future__ import annotations

import unittest

import torch

from src.modules.uniformer import (
    GroupedSwiGLUTokenizer,
    MultiHeadSelfAttention,
    QueryCrossAttention,
    TargetAttentionTokenizer,
    UniFormerRanker,
    user_item_allow_mask,
    weighted_bce_with_logits,
)


class UniFormerPaperAlignmentTest(unittest.TestCase):
    def test_user_item_mask_blocks_only_user_to_item(self) -> None:
        mask = user_item_allow_mask(num_ns_tokens=4, num_user_tokens=2)
        self.assertEqual(tuple(mask.shape), (8, 8))
        user_positions = [0, 1, 4, 5]
        item_positions = [2, 3, 6, 7]
        for query in user_positions:
            for key in item_positions:
                self.assertFalse(bool(mask[query, key]))
            for key in user_positions:
                self.assertTrue(bool(mask[query, key]))
        for query in item_positions:
            self.assertTrue(bool(mask[query].all()))

    def test_user_queries_ignore_item_keys(self) -> None:
        torch.manual_seed(0)
        attention = MultiHeadSelfAttention(d_model=8, num_heads=2)
        allow = user_item_allow_mask(4, 2)
        left = torch.randn(2, 8, 8)
        right = left.clone()
        right[:, [2, 3, 6, 7]] += 4.0

        left_out = attention(left, allow)
        right_out = attention(right, allow)

        torch.testing.assert_close(left_out[:, [0, 1, 4, 5]], right_out[:, [0, 1, 4, 5]])
        self.assertFalse(
            torch.allclose(left_out[:, [2, 3, 6, 7]], right_out[:, [2, 3, 6, 7]])
        )

    def test_bool_key_mask_keeps_true_positions(self) -> None:
        torch.manual_seed(1)
        attention = QueryCrossAttention(d_model=8, num_heads=2)
        query = torch.randn(2, 3, 8)
        key = torch.randn(2, 4, 8)
        value = torch.randn(2, 4, 8)
        mask = torch.tensor(
            [[True, False, True, False], [True, True, False, False]]
        )

        masked = attention(query, key, value, mask)
        reference = []
        for row in range(2):
            keep = mask[row]
            reference.append(
                attention(
                    query[row : row + 1],
                    key[row : row + 1, keep],
                    value[row : row + 1, keep],
                )
            )
        torch.testing.assert_close(masked, torch.cat(reference, dim=0))

    def test_grouped_tokenizer_and_target_attention_shapes(self) -> None:
        tokenizer = GroupedSwiGLUTokenizer([3, 5], d_model=8, ffn_expansion=1)
        tokens = tokenizer([torch.randn(2, 3), torch.randn(2, 5)])
        self.assertEqual(tuple(tokens.shape), (2, 2, 8))

        target_attention = TargetAttentionTokenizer(d_model=8, num_heads=2)
        pooled = target_attention(
            torch.randn(2, 8),
            torch.randn(2, 6, 8),
            torch.randn(2, 6, 8),
        )
        self.assertEqual(tuple(pooled.shape), (2, 8))

    def test_forward_shapes_and_lazy_kv(self) -> None:
        torch.manual_seed(2)
        model = _small_uniformer()
        calls = {"n": 0}
        original = model.memory.forward

        def wrapped(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        model.memory.forward = wrapped
        ns = torch.randn(3, 4, 16)
        tasks = torch.randn(3, 2, 16)
        sequences = [torch.randn(3, 5, 16), torch.randn(3, 7, 16)]
        masks = [
            torch.tensor(
                [
                    [True, True, True, False, False],
                    [True, True, True, True, True],
                    [True, False, False, False, False],
                ]
            ),
            None,
        ]
        output = model(ns, tasks, sequences, masks)

        self.assertEqual(calls["n"], 1)
        self.assertEqual(tuple(output.probabilities.shape), (3, 2))
        self.assertEqual(tuple(output.feature_tokens.shape), (3, 8, 16))
        self.assertEqual(tuple(output.task_tokens.shape), (3, 2, 16))
        self.assertTrue(torch.isfinite(output.probabilities).all())
        self.assertTrue(((output.probabilities > 0) & (output.probabilities < 1)).all())

    def test_backward_and_weighted_loss(self) -> None:
        torch.manual_seed(3)
        model = _small_uniformer()
        output = model(
            torch.randn(4, 4, 16),
            torch.randn(4, 2, 16),
            [torch.randn(4, 3, 16), torch.randn(4, 3, 16)],
        )
        labels = torch.zeros(4, 2)
        labels[:, 0] = 1
        loss = weighted_bce_with_logits(output.logits, labels, torch.tensor([0.2, 0.8]))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(torch.isfinite(loss))

    def test_synthetic_labels_are_learnable(self) -> None:
        torch.manual_seed(4)
        model = _small_uniformer()
        ns = torch.randn(24, 4, 16)
        tasks = torch.randn(24, 2, 16)
        sequences = [torch.randn(24, 4, 16), torch.randn(24, 4, 16)]
        with torch.no_grad():
            positive = ns.mean(dim=(1, 2)) > 0
            labels = torch.stack([positive, ~positive], dim=1).to(dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)
        initial = _uniformer_loss(model, ns, tasks, sequences, labels)
        for _ in range(30):
            optimizer.zero_grad()
            _uniformer_loss(model, ns, tasks, sequences, labels).backward()
            optimizer.step()
        final = _uniformer_loss(model, ns, tasks, sequences, labels)
        self.assertLess(final.detach().item(), initial.detach().item())


def _small_uniformer() -> UniFormerRanker:
    return UniFormerRanker(
        d_model=16,
        num_heads=4,
        num_ns_tokens=4,
        num_user_tokens=2,
        num_tasks=2,
        num_sequences=2,
        fim_layers=2,
        tim_layers=1,
        ffn_expansion=1,
    )


def _uniformer_loss(model, ns, tasks, sequences, labels) -> torch.Tensor:
    return weighted_bce_with_logits(model(ns, tasks, sequences).logits, labels)
