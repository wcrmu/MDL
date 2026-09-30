"""Correctness regressions for compiled masking, empty histories and dispatch."""
from copy import deepcopy
from dataclasses import replace
import os
import unittest
from unittest.mock import patch

import torch

from src.modules.mixformer import MixFormerBlock, MixFormerCrossAttention
from src.modules.attention import varlen_attention_available
from scripts.verify_industrial_rankers import config_for, small_model
from src.benchmark import _synthetic_feature_batch, _synthetic_vocab_maps


class MixFormerRegressionTest(unittest.TestCase):
    def test_compiled_attention_preserves_mixed_length_mask(self):
        module = MixFormerCrossAttention(2, 4, 8, attention_backend="sdpa")
        compiled = torch.compile(lambda q, h, mask: module._attend_history(q, h, mask),
                                 backend="aot_eager", fullgraph=False)
        for mask in (torch.ones(2, 4, dtype=torch.bool),
                     torch.tensor([[True, True, False, False], [True, True, True, True]]),
                     torch.tensor([[False, False, False, False], [True, False, False, False]])):
            q = torch.ones(2, 2, 4, requires_grad=True)
            history = torch.ones(2, 4, 2, 4) * mask[:, :, None, None]
            history.requires_grad_()
            expected = module._attend_history(q, history, mask)
            actual = compiled(q, history, mask)
            torch.testing.assert_close(actual, expected)
            a = torch.autograd.grad(actual.sum(), (q, history), retain_graph=True)
            b = torch.autograd.grad(expected.sum(), (q, history))
            for left, right in zip(a, b):
                torch.testing.assert_close(left, right)
            self.assertTrue(torch.isfinite(actual).all())

    def test_compiled_blocks_match_eager_with_ragged_requests(self):
        torch.manual_seed(23)
        for ui in (False, True):
            base = MixFormerBlock(2, 4, 8, user_head_count=1 if ui else None,
                                  attention_backend="sdpa")
            candidate = deepcopy(base)
            index = torch.tensor([2, 0, 2, 0, 1])
            sequence = torch.randn(3, 5, 8)
            mask = torch.tensor([[True, True, False, False, False],
                                 [False, False, False, False, False],
                                 [True, True, True, True, True]])
            if ui:
                args = (torch.randn(3, 1, 4), torch.randn(5, 1, 4), sequence, mask, index)
                expected = base.forward_decoupled(*args)
                actual = torch.compile(candidate.forward_decoupled, backend="aot_eager")(*args)
            else:
                args = (torch.randn(5, 2, 4), sequence, mask, index)
                expected = (base(*args),)
                actual = (torch.compile(candidate, backend="aot_eager")(*args),)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)
            sum(t.square().mean() for t in actual).backward()
            sum(t.square().mean() for t in expected).backward()
            for a, b in zip(candidate.parameters(), base.parameters()):
                self.assertIsNotNone(a.grad)
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)

    def test_empty_history_keeps_all_ca_parameters_and_input_in_graph(self):
        for ui in (False, True):
            module = MixFormerCrossAttention(2, 4, 8, user_head_count=1 if ui else None)
            sequence = torch.empty(2, 0, 8, requires_grad=True)
            mask = torch.empty(2, 0, dtype=torch.bool)
            if ui:
                outputs = module.forward_decoupled(torch.randn(2, 1, 4, requires_grad=True),
                    torch.randn(3, 1, 4, requires_grad=True), sequence, mask, torch.tensor([1, 0, 1]))
            else:
                outputs = (module(torch.randn(2, 2, 4, requires_grad=True), sequence, mask),)
            sum(x.sum() for x in outputs).backward()
            self.assertIsNotNone(sequence.grad)
            for name, p in module.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertEqual(p.grad.count_nonzero().item(), 0, name)

    def test_sdpa_does_not_use_varlen_and_flash_rejects_cpu(self):
        module = MixFormerCrossAttention(2, 4, 8, attention_backend="sdpa")
        with patch("src.modules.mixformer.varlen_attention_available", side_effect=AssertionError("must not probe")):
            self.assertFalse(module._can_varlen_history(torch.ones(2, 2, 4)))
        module.attention_backend = "flash"
        with self.assertRaisesRegex(RuntimeError, "strict flash"):
            module(torch.randn(2, 2, 4), torch.randn(2, 3, 8), torch.ones(2, 3, dtype=torch.bool))
        module.sequence_chunk_tokens = 1
        self.assertEqual(module._attention_length_chunk(200), 200)

    def test_full_model_empty_history_checkpoint_and_backend_propagation(self):
        config = config_for("mixformer", "cpu")
        config = replace(config, model=replace(config.model, token_dim=32, hidden_dim=64,
                                               num_layers=2, task_head_hidden_dim=64))
        model = small_model(config, _synthetic_vocab_maps(config), torch.device("cpu"))
        self.assertTrue(all(b.cross_attention.attention_backend == "sdpa" for b in model.blocks))
        batch = _synthetic_feature_batch(config, torch.device("cpu"), 4, 3, 21, 2)
        for sequence in config.sequences:
            batch.features[sequence.name]["lengths"].zero_()
        logits = model(batch.features, batch.scenario_id)["logits"]
        logits.square().mean().backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        self.assertEqual(missing, [])
        restored = small_model(config, _synthetic_vocab_maps(config), torch.device("cpu"))
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(batch.features, batch.scenario_id)["logits"], logits)

    @unittest.skipUnless(os.environ.get("MDL_TEST_MIXFORMER_CUDA") == "1",
                         "explicit GPU opt-in required")
    def test_cuda_bf16_compiled_mixed_masks_match_sdpa(self):
        self.assertTrue(torch.cuda.is_available() and varlen_attention_available())
        torch.manual_seed(71)
        reference = MixFormerBlock(2, 32, 64, user_head_count=1, attention_backend="sdpa").cuda()
        target = deepcopy(reference)
        target.cross_attention.attention_backend = "flash"
        compiled = torch.compile(target.forward_decoupled)
        index = torch.tensor([2, 0, 2, 0, 1], device="cuda")
        for empty in (False, True):
            reference.zero_grad(set_to_none=True)
            target.zero_grad(set_to_none=True)
            args = (torch.randn(3, 1, 32, device="cuda"), torch.randn(5, 1, 32, device="cuda"),
                    torch.randn(3, 7, 64, device="cuda"),
                    torch.tensor([[1, 1, 0, 0, 0, 0, 0], [0] * 7, [1] * 7], device="cuda", dtype=torch.bool), index)
            if empty:
                args = (*args[:2], args[2][:, :0], args[3][:, :0], index)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                expected = reference.forward_decoupled(*args)
                actual = compiled(*args)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=.03, atol=.03)
            sum(t.float().square().mean() for t in actual).backward()
            sum(t.float().square().mean() for t in expected).backward()
            for a, b in zip(target.parameters(), reference.parameters()):
                self.assertIsNotNone(a.grad)
                torch.testing.assert_close(a.grad, b.grad, rtol=.1, atol=.03)
