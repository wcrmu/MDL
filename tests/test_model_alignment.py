from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import Tensor, nn

from src.model import (
    MDLOneTransModel,
    ModelMetadata,
    OneTransBackboneState,
    OneTransBlock,
    RankMixerBlock,
    RankMixerModel,
)
from src.modules.mlp import SparseMoEPerTokenFFN


def _rankmixer_config() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            token_dim=4,
            hidden_dim=8,
            ffn_activation="gelu",
            rankmixer_ffn_type="dense",
            sparse_moe_num_experts=4,
            sparse_moe_use_dtsi=True,
            sparse_moe_inference_threshold=0.0,
            sparse_moe_target_active_ratio=0.25,
            sparse_moe_regularization_initial=1.0e-8,
            sparse_moe_regularization_multiplier=1.2,
        ),
        runtime=SimpleNamespace(
            attention_backend="auto",
            activation_checkpoint=False,
        ),
    )


class RankMixerAlignmentTest(unittest.TestCase):
    def test_block_keeps_second_residual_and_layer_norm(self) -> None:
        block = RankMixerBlock(_rankmixer_config(), feature_token_count=2)
        with torch.no_grad():
            for parameter in block.feature_ffn.parameters():
                parameter.zero_()

        tokens = torch.randn(3, 2, 4, requires_grad=True)
        mixed = block.feature_norm(block.token_mixing(tokens) + tokens)
        expected = block.feature_ffn_norm(mixed)
        actual = block(tokens)

        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        self.assertIsNotNone(tokens.grad)
        self.assertTrue(bool(torch.isfinite(tokens.grad).all()))

    def test_model_uses_mean_pooling_before_task_heads(self) -> None:
        class PassEncoder(nn.Module):
            def forward(
                self,
                features: dict[str, Tensor],
                request_cache: object | None = None,
            ) -> dict[str, Tensor]:
                del request_cache
                return features

        class TokenProjector(nn.Module):
            def forward(self, encoded: dict[str, Tensor]) -> Tensor:
                return encoded["tokens"]

        model = RankMixerModel.__new__(RankMixerModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            runtime=SimpleNamespace(activation_checkpoint=False)
        )
        model.encoder_bank = PassEncoder()
        model.feature_projector = TokenProjector()
        model.blocks = nn.ModuleList()
        head = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            head.weight.fill_(1.0)
        model.logit_layers = nn.ModuleList([head])

        tokens = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]],
                [[2.0, 4.0, 6.0], [6.0, 8.0, 10.0]],
            ]
        )
        output = model(
            {"tokens": tokens},
            scenario_id=torch.zeros(2, dtype=torch.long),
        )
        expected = tokens.mean(dim=1).sum(dim=1, keepdim=True)
        torch.testing.assert_close(output["logits"], expected)


class OneTransCacheAlignmentTest(unittest.TestCase):
    def test_layer_kv_cache_is_equivalent_and_reuses_s_projections(self) -> None:
        config = SimpleNamespace(
            model=SimpleNamespace(token_dim=8, num_heads=2, hidden_dim=16),
            runtime=SimpleNamespace(attention_backend="auto"),
        )
        torch.manual_seed(19)
        block = OneTransBlock(config, ns_token_count=3).eval()
        s_tokens = torch.randn(2, 5, 8)
        ns_tokens = torch.randn(2, 3, 8)
        s_mask = torch.tensor(
            [[False, True, True, True, True], [True, True, True, True, True]]
        )
        valid_mask = torch.cat(
            [s_mask, torch.ones(2, 3, dtype=torch.bool)],
            dim=1,
        )

        full, _full_mask = block(
            torch.cat([s_tokens, ns_tokens], dim=1),
            s_count=5,
            query_s_count=3,
            valid_mask=valid_mask,
        )

        calls = {"key": 0, "value": 0}
        key_hook = block.attention.s_key.register_forward_hook(
            lambda *_args: calls.__setitem__("key", calls["key"] + 1)
        )
        value_hook = block.attention.s_value.register_forward_hook(
            lambda *_args: calls.__setitem__("value", calls["value"] + 1)
        )
        try:
            cache = block.precompute_s(s_tokens, query_s_count=3, valid_mask=s_mask)
            calls_after_precompute = dict(calls)
            cached_ns = block.forward_cached_ns(ns_tokens, cache)
        finally:
            key_hook.remove()
            value_hook.remove()

        torch.testing.assert_close(cached_ns, full[:, -3:, :], rtol=1.0e-5, atol=1.0e-6)
        torch.testing.assert_close(cache.s_output, full[:, :3, :], rtol=1.0e-5, atol=1.0e-6)
        self.assertEqual(calls_after_precompute, {"key": 1, "value": 1})
        self.assertEqual(calls, calls_after_precompute)


class SparseMoEAlignmentTest(unittest.TestCase):
    def test_relu_l1_and_adaptive_coefficient_follow_current_step(self) -> None:
        module = SparseMoEPerTokenFFN(
            num_tokens=1,
            token_dim=3,
            hidden_dim=5,
            num_experts=4,
            target_active_ratio=0.25,
            regularization_initial=1.0e-4,
            regularization_multiplier=2.0,
        ).train()
        with torch.no_grad():
            module.sparse_routers[0].weight.zero_()
            module.sparse_routers[0].bias.fill_(1.0)

        tokens = torch.randn(2, 1, 3, requires_grad=True)
        output = module(tokens)
        regularization = module.regularization_loss(output)

        # Four unit gates per token: lambda_0 * sum_j G_j.
        torch.testing.assert_close(
            regularization,
            regularization.new_tensor(4.0e-4),
        )
        torch.testing.assert_close(
            module.regularization_coefficient,
            module.regularization_coefficient.new_tensor(2.0e-4),
        )
        (output.sum() + regularization).backward()
        self.assertIsNotNone(module.dense_routers[0].weight.grad)
        self.assertIsNotNone(module.sparse_routers[0].weight.grad)


class MDLOneTransLayerwiseAlignmentTest(unittest.TestCase):
    def test_domain_blocks_observe_corresponding_backbone_layer(self) -> None:
        token_dim = 4
        batch_size = 2

        class Encoder(nn.Module):
            def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
                return {"seed": features["seed"]}

        class Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder_bank = Encoder()

            def prepare(
                self,
                features: dict[str, Tensor],
                request_cache: object | None = None,
                encoded_features: dict[str, Tensor] | None = None,
            ) -> OneTransBackboneState:
                del features, request_cache
                assert encoded_features is not None
                tokens = encoded_features["seed"].new_zeros(batch_size, 2, token_dim)
                return OneTransBackboneState(
                    tokens=tokens,
                    valid_mask=torch.ones(batch_size, 2, dtype=torch.bool),
                    s_count=0,
                    ns_count=2,
                    initial_s_count=0,
                    encoded_features=encoded_features,
                )

            def step(
                self,
                state: OneTransBackboneState,
                layer_index: int,
                layer_cache: object | None = None,
            ) -> OneTransBackboneState:
                del layer_cache
                return OneTransBackboneState(
                    tokens=torch.full_like(state.tokens, float(layer_index + 1)),
                    valid_mask=state.valid_mask,
                    s_count=state.s_count,
                    ns_count=state.ns_count,
                    initial_s_count=state.initial_s_count,
                    encoded_features=state.encoded_features,
                )

        class Projector(nn.Module):
            def __init__(self, token_count: int) -> None:
                super().__init__()
                self.token_count = token_count

            def forward(self, encoded: dict[str, Tensor]) -> Tensor:
                return encoded["seed"].new_zeros(batch_size, self.token_count, token_dim)

        class RecordingBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seen: list[float] = []

            def forward(
                self,
                feature_tokens: Tensor,
                scenario_tokens: Tensor,
                task_tokens: Tensor,
                scenario_mask: Tensor,
            ) -> tuple[Tensor, Tensor]:
                del scenario_mask
                self.seen.append(float(feature_tokens.mean().item()))
                return scenario_tokens, task_tokens

        model = MDLOneTransModel.__new__(MDLOneTransModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            model=SimpleNamespace(use_request_cache=False),
            runtime=SimpleNamespace(activation_checkpoint=False),
        )
        model.backbone = Backbone()
        model.metadata = ModelMetadata(feature_token_count=2, scenario_count=1, task_count=1)
        model.scenario_projector = Projector(2)
        model.task_projector = Projector(1)
        blocks = [RecordingBlock(), RecordingBlock()]
        model.blocks = nn.ModuleList(blocks)
        model.logit_layers = nn.ModuleList([nn.Linear(token_dim, 1)])

        output = model(
            {"seed": torch.randn(batch_size, 1)},
            scenario_id=torch.zeros(batch_size, dtype=torch.long),
        )

        self.assertEqual(tuple(output["logits"].shape), (batch_size, 1))
        self.assertEqual(blocks[0].seen, [1.0])
        self.assertEqual(blocks[1].seen, [2.0])


if __name__ == "__main__":
    unittest.main()
