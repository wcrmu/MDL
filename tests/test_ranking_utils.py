"""Regression coverage for request sharing, precision dispatch and paper corrections."""
from dataclasses import replace
import unittest

import torch
from torch import nn

from src.config import load_app_config
from src.modules.ranking_utils import RequestLayout, SequenceMemory, read_sequence, token_swiglu
from src.modules.stca import SwiGLUFFN
from src.modules.uniformer import MultiHeadSelfAttention, user_item_allow_mask
from src.modules.more import TokenGate, MORERanker
from src.more_model import MORETokenizer, _align_candidates as align_more
from src.uniformer_model import _align_candidates as align_uniformer
from src.train import _build_dense_optimizer, _requires_varlen_attention, _needs_padded_sdpa_flash
from src.benchmark import _analytical_dense_flops_per_step


class RankingExecutionTest(unittest.TestCase):
    def test_shared_kv_matches_expansion_outputs_and_gradients(self):
        torch.manual_seed(31)
        for same_kv in (False, True):
            for all_empty in (False, True):
                q = torch.randn(5, 3, 16, requires_grad=True)
                k = torch.randn(4, 7, 16, requires_grad=True)
                v = k if same_kv else torch.randn_like(k, requires_grad=True)
                # Unsorted, repeated indices, an unused request, an empty history.
                index = torch.tensor([2, 0, 2, 1, 0])
                mask = torch.rand(4, 7) > .3
                mask[1] = False
                if all_empty:
                    mask[:] = False
                actual = read_sequence(q, SequenceMemory(k, v, mask), 4, "sdpa",
                                       RequestLayout.build(index, 4))
                expected = read_sequence(q, SequenceMemory(k[index], v[index], mask[index]), 4, "sdpa")
                torch.testing.assert_close(actual, expected)
                parameters = (q, k) if same_kv else (q, k, v)
                grads = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
                reference = torch.autograd.grad(expected.square().sum(), parameters)
                for left, right in zip(grads, reference):
                    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
                self.assertTrue(torch.isfinite(actual).all())

    def test_batched_ffns_preserve_outputs_and_gradients(self):
        torch.manual_seed(32)
        ffns = nn.ModuleList([SwiGLUFFN(16, expansion_ratio=2) for _ in range(4)])
        x = torch.randn(3, 4, 16, requires_grad=True)
        actual = token_swiglu(ffns, x)
        expected = torch.stack([f(x[:, i]) for i, f in enumerate(ffns)], 1)
        torch.testing.assert_close(actual, expected)
        params = (x, *ffns.parameters())
        grads = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
        reference = torch.autograd.grad(expected.square().sum(), params)
        for left, right in zip(grads, reference):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)

    def test_ui_attention_split_is_mask_equivalent(self):
        torch.manual_seed(33)
        attn = MultiHeadSelfAttention(16, 4)
        x = torch.randn(3, 8, 16, requires_grad=True)
        mask = user_item_allow_mask(4, 2)
        actual = attn(x, mask, torch.tensor([0, 1, 4, 5]), torch.tensor([2, 3, 6, 7]))
        expected = attn(x, mask)
        torch.testing.assert_close(actual, expected)
        left = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
        right = torch.autograd.grad(expected.sum(), x)[0]
        torch.testing.assert_close(left, right)

    def test_equal_batch_size_still_honors_request_permutation(self):
        x = torch.arange(3.)[:, None, None]
        index = torch.tensor([2, 0, 1])
        for align in (align_more, align_uniformer):
            torch.testing.assert_close(align(x, 3, index), x[index])

    def test_positions_ignore_padding_and_preserve_order(self):
        # Exercise positional logic without allocating production embeddings.
        tokenizer = MORETokenizer.__new__(MORETokenizer)
        nn.Module.__init__(tokenizer)
        tokenizer.position_capacity = 8
        tokenizer.position_embedding = nn.Embedding(9, 4, padding_idx=0)
        x = torch.randn(1, 2, 4)
        a = tokenizer.add_positions(x, torch.ones(1, 2, dtype=torch.bool))
        padded = torch.cat([torch.zeros(1, 2, 4), x], dim=1)
        b = tokenizer.add_positions(padded, torch.tensor([[False, False, True, True]]))
        torch.testing.assert_close(a, b[:, 2:])
        self.assertEqual(b[:, :2].count_nonzero().item(), 0)
        reversed_then_restored = tokenizer.add_positions(x.flip(1), torch.ones(1, 2, dtype=torch.bool)).flip(1)
        self.assertFalse(torch.allclose(a, reversed_then_restored))

    def test_zero_gate_logits_have_unit_gain(self):
        gate = TokenGate(4, 4)
        for parameter in gate.parameters():
            nn.init.zeros_(parameter)
        torch.testing.assert_close(gate(torch.randn(2, 4, 16)), torch.ones(2, 4, 4))

    def test_batched_gate_matches_loop(self):
        gate = TokenGate(4, 4)
        x = torch.randn(3, 4, 16, requires_grad=True)
        heads = x.view(3, 4, 4, 4).permute(0, 2, 1, 3)
        expected = torch.stack([2 * torch.sigmoid(m(heads[:, i])).squeeze(-1)
                                for i, m in enumerate(gate.mlps)], dim=1)
        actual = gate(x)
        torch.testing.assert_close(actual, expected)
        params = (x, *gate.parameters())
        left = torch.autograd.grad(actual.sum(), params, retain_graph=True)
        right = torch.autograd.grad(expected.sum(), params)
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b)

    def test_more_zero_length_history_has_finite_gradients(self):
        model = MORERanker(16, 1, 1, 2, num_heads=4, num_blocks=2)
        output = model(torch.empty(2, 0, 16), torch.randn(3, 1, 16),
                       request_index=torch.tensor([1, 0, 1])).logits
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_strict_flash_does_not_silently_fallback_on_cpu(self):
        x = torch.randn(1, 2, 16)
        with self.assertRaisesRegex(RuntimeError, "strict flash"):
            read_sequence(x, SequenceMemory(x, x), 4, "flash")

    def test_backend_preflight_and_optimizer_selection(self):
        for name in ("uniformer", "more"):
            config = load_app_config(f"configs/{name}.yaml")
            self.assertTrue(_requires_varlen_attention(config))
            self.assertEqual(_needs_padded_sdpa_flash(config), name == "uniformer")
            for kind, cls in (("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW),
                              ("rmsprop", torch.optim.RMSprop)):
                c = replace(config, training=replace(config.training, dense_optimizer=kind))
                p = nn.Parameter(torch.ones(2))
                optimizer = _build_dense_optimizer([p], c, torch.device("cpu"))
                self.assertIsInstance(optimizer, cls)
                p.sum().backward()
                optimizer.step()
                self.assertTrue(torch.isfinite(p).all())

    def test_flops_do_not_saturate_at_512_tokens(self):
        for name in ("uniformer", "more"):
            config = load_app_config(f"configs/{name}.yaml")
            small = _analytical_dense_flops_per_step(config, candidates_per_step=4,
                                                   input_tokens_per_step=4 * 1024)
            large = _analytical_dense_flops_per_step(config, candidates_per_step=4,
                                                   input_tokens_per_step=4 * 8192)
            self.assertGreater(large, small)

    def test_optimizer_ablation_configs_are_isolated(self):
        for name, optimizer in (("uniformer_adamw", "adamw"), ("more_adam", "adam")):
            config = load_app_config(f"configs/{name}.yaml")
            self.assertEqual(config.training.dense_optimizer, optimizer)
            self.assertEqual(config.training.checkpoint.run_name, name)
            self.assertEqual(config.training.checkpoint.resume, "none")
