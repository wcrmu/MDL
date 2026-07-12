from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from src.dataloader import FeatureBatch
from src.config import SequenceConfig, SequenceFieldConfig, TokenGroupConfig, load_app_config
from src.model import (
    MDLRankMixerBlock,
    MDLOneTransModel,
    ModelMetadata,
    OneTransBackbone,
    OneTransBackboneState,
    OneTransBlock,
    OneTransRequestCache,
    OneTransTokenizer,
    RankMixerBlock,
    RankMixerModel,
    RankMixerSliceTokenizer,
)
from src.modules.mlp import SparseMoEPerTokenFFN
from src.train import _loss_terms_from_batch, _step_sparse_moe_controllers


def _rankmixer_config() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            token_dim=4,
            hidden_dim=8,
            num_heads=2,
            ffn_activation="gelu",
            mdl_feature_interaction="paper",
            use_task_tokens=True,
            use_scenario_tokens=True,
            use_global_scenario_token=True,
            use_task_feature_interaction=True,
            use_scenario_feature_interaction=True,
            rankmixer_ffn_type="dense",
            sparse_moe_num_experts=4,
            sparse_moe_use_dtsi=True,
            sparse_moe_inference_threshold=0.0,
            sparse_moe_target_active_ratio=0.25,
            sparse_moe_regularization_initial=1.0e-8,
            sparse_moe_regularization_multiplier=1.2,
            sparse_moe_dtsi_training_output=None,
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

    def test_paper_tokenizer_slices_input_width_before_projection(self) -> None:
        tokenizer = RankMixerSliceTokenizer(
            input_names=["all_features"],
            input_dims={"all_features": 6},
            num_tokens=2,
            token_dim=4,
        )
        with torch.no_grad():
            for layer in tokenizer.projection.layers:
                layer.weight.zero_()
                layer.bias.zero_()
                layer.weight[:3, :3] = torch.eye(3)
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

        tokens = tokenizer({"all_features": values})

        expected = torch.tensor(
            [[[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.0]]]
        )
        torch.testing.assert_close(tokens, expected)


class MDLFeatureInteractionAlignmentTest(unittest.TestCase):
    def _block(self, mode: str) -> MDLRankMixerBlock:
        config = _rankmixer_config()
        config.model.mdl_feature_interaction = mode
        block = MDLRankMixerBlock(
            config,
            ModelMetadata(feature_token_count=2, scenario_count=1, task_count=1),
            propagate_scenario_state=False,
        )
        with torch.no_grad():
            for parameter in block.feature_ffn.parameters():
                parameter.zero_()
        return block

    def test_paper_mode_matches_mdl_equation_six_without_second_add_norm(self) -> None:
        block = self._block("paper")
        feature_tokens = torch.randn(2, 2, 4)
        scenario_tokens = torch.randn(2, 2, 4)
        task_tokens = torch.randn(2, 1, 4)
        scenario_mask = torch.ones(2, 1)

        actual, _scenario, _task = block(
            feature_tokens,
            scenario_tokens,
            task_tokens,
            scenario_mask,
        )

        torch.testing.assert_close(actual, torch.zeros_like(actual))
        self.assertIsNone(block.feature_ffn_norm)

    def test_rankmixer_full_mode_remains_an_explicit_compatibility_path(self) -> None:
        block = self._block("rankmixer_full")
        feature_tokens = torch.randn(2, 2, 4)
        mixed = block.feature_norm(block.token_mixing(feature_tokens) + feature_tokens)
        scenario_tokens = torch.randn(2, 2, 4)
        task_tokens = torch.randn(2, 1, 4)

        actual, _scenario, _task = block(
            feature_tokens,
            scenario_tokens,
            task_tokens,
            torch.ones(2, 1),
        )

        assert block.feature_ffn_norm is not None
        torch.testing.assert_close(actual, block.feature_ffn_norm(mixed))

    def test_two_scenario_two_task_domain_fusion_selects_per_instance_tokens(self) -> None:
        config = _rankmixer_config()
        block = MDLRankMixerBlock(
            config,
            ModelMetadata(feature_token_count=2, scenario_count=2, task_count=2),
            propagate_scenario_state=False,
        )
        with torch.no_grad():
            for module in (block.scenario_attention, block.task_attention, block.task_ffn):
                for parameter in module.parameters():
                    parameter.zero_()
        scenario_tokens = torch.tensor(
            [
                [[2.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]],
                [[2.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]],
            ]
        )
        task_tokens = torch.zeros(2, 2, 4)
        scenario_mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        _features, _scenarios, actual_tasks = block(
            torch.randn(2, 2, 4),
            scenario_tokens,
            task_tokens,
            scenario_mask,
        )

        expected_scenario_average = torch.tensor(
            [[2.0, 1.0, 1.0, 1.0], [1.0, 3.0, 1.0, 1.0]]
        )
        expected = expected_scenario_average.unsqueeze(1).expand(-1, 2, -1)
        torch.testing.assert_close(actual_tasks, expected)


class MDLLossAlignmentTest(unittest.TestCase):
    def test_sum_reduction_preserves_masked_sample_and_task_weight(self) -> None:
        logits = torch.zeros(2, 2)
        batch = FeatureBatch(
            features={},
            labels=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            label_mask=torch.tensor([[True, True], [True, False]]),
            scenario_id=torch.zeros(2, dtype=torch.long),
            group_id=[],
        )

        paper_sum, _numerator, _denominator = _loss_terms_from_batch(
            {"logits": logits},
            batch,
            loss_reduction="sum",
        )
        balanced_mean, _numerator, _denominator = _loss_terms_from_batch(
            {"logits": logits},
            batch,
            loss_reduction="mean_per_task",
        )

        unit = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(0.0),
            torch.tensor(0.0),
        )
        torch.testing.assert_close(paper_sum, 3.0 * unit)
        torch.testing.assert_close(balanced_mean, 2.0 * unit)


class OneTransTokenizerAlignmentTest(unittest.TestCase):
    class Encoder(nn.Module):
        def __init__(
            self,
            output_dims: dict[str, int],
            sequence_dims: dict[str, int],
        ) -> None:
            super().__init__()
            self.output_dims = output_dims
            self.sequence_event_input_dims = sequence_dims

        def encode_sequence_event_inputs(
            self,
            sequence_name: str,
            value: dict[str, Tensor | dict[str, Tensor]],
        ) -> tuple[Tensor, Tensor]:
            del sequence_name
            return value["event_inputs"], value["mask"]  # type: ignore[return-value]

        def _align_sequence_inputs(
            self,
            sequence: SequenceConfig,
            inputs: Tensor,
            lengths: Tensor,
        ) -> tuple[Tensor, Tensor]:
            del sequence, lengths
            return inputs, torch.ones(inputs.shape[:2], dtype=torch.bool)

    def _base_config(self):
        root = Path(__file__).resolve().parents[1]
        return load_app_config(root / "configs" / "onetrans.yaml")

    def test_auto_split_and_sequence_tokenizers_each_use_one_mlp(self) -> None:
        config = self._base_config()
        scalar_names = [
            feature.name
            for feature in config.features
            if feature.embedding_scope in {"feature", "shared"}
        ]
        raw_sequence_dim = sum(
            config.resolved.categorical_embedding_dims.get(
                field.qualified_name(config.sequences[0].name),
                field.dimension,
            )
            for field in config.sequences[0].fields
        )
        encoder = self.Encoder(
            {name: config.resolved.encoded_input_dims[name] for name in scalar_names},
            {config.sequences[0].name: raw_sequence_dim},
        )

        tokenizer = OneTransTokenizer(config, encoder)

        self.assertIsInstance(tokenizer.auto_ns_projection, nn.Sequential)
        assert tokenizer.auto_ns_projection is not None
        self.assertEqual(
            sum(isinstance(layer, nn.Linear) for layer in tokenizer.auto_ns_projection),
            2,
        )
        self.assertTrue(
            any(isinstance(layer, nn.GELU) for layer in tokenizer.auto_ns_projection)
        )
        sequence_projection = tokenizer.sequence_projectors[0]
        self.assertIsInstance(sequence_projection, nn.Sequential)
        first_linear = next(
            layer for layer in sequence_projection if isinstance(layer, nn.Linear)
        )
        self.assertEqual(first_linear.in_features, raw_sequence_dim)

    def _fusion_tokenizer(self, fusion: str) -> OneTransTokenizer:
        base = self._base_config()
        sequences = [
            SequenceConfig(
                name=name,
                fields=[
                    SequenceFieldConfig(
                        name="timestamp",
                        kind="dense",
                        source=f"{name}_timestamp",
                    )
                ],
                max_length=2,
                timestamp_field="timestamp",
            )
            for name in ("a", "b")
        ]
        config = replace(
            base,
            sequences=sequences,
            tokenization=replace(
                base.tokenization,
                sequence_tokens=[
                    TokenGroupConfig(name="a", inputs=["a"]),
                    TokenGroupConfig(name="b", inputs=["b"]),
                ],
            ),
            model=replace(
                base.model,
                token_dim=4,
                num_heads=2,
                hidden_dim=8,
                sequence_fusion=fusion,
                num_ns_tokens=1,
            ),
        )
        scalar_names = [
            feature.name
            for feature in config.features
            if feature.embedding_scope in {"feature", "shared"}
        ]
        encoder = self.Encoder(
            {name: 1 for name in scalar_names},
            {"a": 4, "b": 4},
        )
        tokenizer = OneTransTokenizer(config, encoder)
        tokenizer.sequence_projectors = nn.ModuleList([nn.Identity(), nn.Identity()])
        if tokenizer.sequence_type_embeddings is not None:
            with torch.no_grad():
                tokenizer.sequence_type_embeddings.weight.zero_()
        return tokenizer

    def _fusion_features(self) -> dict[str, dict[str, Tensor | dict[str, Tensor]]]:
        return {
            "a": {
                "event_inputs": torch.tensor(
                    [[[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]]
                ),
                "mask": torch.ones(1, 2, dtype=torch.bool),
                "fields": {"timestamp": torch.tensor([[1.0, 3.0]])},
                "lengths": torch.tensor([2]),
            },
            "b": {
                "event_inputs": torch.tensor(
                    [[[2.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]]]
                ),
                "mask": torch.ones(1, 2, dtype=torch.bool),
                "fields": {"timestamp": torch.tensor([[2.0, 4.0]])},
                "lengths": torch.tensor([2]),
            },
        }

    def test_timestamp_aware_fusion_interleaves_events(self) -> None:
        tokenizer = self._fusion_tokenizer("timestamp_aware")

        cache = tokenizer.precompute_request_cache(self._fusion_features())

        torch.testing.assert_close(
            cache.s_tokens[0, :, 0],
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )

    def test_intent_ordered_fusion_inserts_learned_separator(self) -> None:
        tokenizer = self._fusion_tokenizer("intent_ordered")
        with torch.no_grad():
            tokenizer.sep_tokens[0].fill_(9.0)

        cache = tokenizer.precompute_request_cache(self._fusion_features())

        torch.testing.assert_close(
            cache.s_tokens[0, :, 0],
            torch.tensor([1.0, 3.0, 9.0, 2.0, 4.0]),
        )


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

    def _backbone(self, use_pyramid: bool) -> OneTransBackbone:
        class Tokenizer(nn.Module):
            def precompute_request_cache(
                self,
                features: dict[str, Tensor],
            ) -> OneTransRequestCache:
                tokens = features["s_tokens"]
                return OneTransRequestCache(
                    s_tokens=tokens,
                    s_valid_mask=torch.ones(tokens.shape[:2], dtype=torch.bool),
                )

        config = SimpleNamespace(
            model=SimpleNamespace(
                token_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=3,
                use_pyramid=use_pyramid,
                final_s_tokens=2,
                pyramid_round_to=32,
            ),
            runtime=SimpleNamespace(
                attention_backend="auto",
                activation_checkpoint=False,
            ),
        )
        backbone = OneTransBackbone.__new__(OneTransBackbone)
        nn.Module.__init__(backbone)
        backbone.config = config
        backbone.tokenizer = Tokenizer()
        backbone.ns_token_count = 2
        backbone.blocks = nn.ModuleList(
            OneTransBlock(config, ns_token_count=2) for _ in range(3)
        )
        return backbone.eval()

    def test_cross_request_append_cache_matches_full_recompute(self) -> None:
        torch.manual_seed(29)
        backbone = self._backbone(use_pyramid=False)
        old_tokens = torch.randn(1, 5, 8)
        new_tokens = torch.cat([old_tokens, torch.randn(1, 2, 8)], dim=1)

        with torch.no_grad():
            old_cache = backbone.precompute_request_cache({"s_tokens": old_tokens})
            incremental = backbone.update_request_cache(
                {"s_tokens": new_tokens},
                old_cache,
            )
            full = backbone.precompute_request_cache({"s_tokens": new_tokens})

        for incremental_layer, full_layer in zip(incremental.layers, full.layers):
            self.assertEqual(incremental_layer.s_reused_kv_tokens, old_tokens.size(1))
            torch.testing.assert_close(incremental_layer.s_key, full_layer.s_key)
            torch.testing.assert_close(incremental_layer.s_value, full_layer.s_value)
            torch.testing.assert_close(incremental_layer.s_output, full_layer.s_output)

    def test_pyramid_append_cache_rebuilds_only_changed_deeper_windows(self) -> None:
        torch.manual_seed(31)
        backbone = self._backbone(use_pyramid=True)
        old_tokens = torch.randn(1, 5, 8)
        new_tokens = torch.cat([old_tokens, torch.randn(1, 2, 8)], dim=1)

        with torch.no_grad():
            old_cache = backbone.precompute_request_cache({"s_tokens": old_tokens})
            incremental = backbone.update_request_cache(
                {"s_tokens": new_tokens},
                old_cache,
            )
            full = backbone.precompute_request_cache({"s_tokens": new_tokens})

        self.assertEqual(incremental.layers[0].s_reused_kv_tokens, old_tokens.size(1))
        self.assertGreater(incremental.layers[1].s_reused_kv_tokens, 0)
        self.assertEqual(incremental.layers[2].s_reused_kv_tokens, 0)
        for incremental_layer, full_layer in zip(incremental.layers, full.layers):
            torch.testing.assert_close(incremental_layer.s_key, full_layer.s_key)
            torch.testing.assert_close(incremental_layer.s_output, full_layer.s_output)

    def test_cross_request_cache_rejects_non_append_mutation(self) -> None:
        backbone = self._backbone(use_pyramid=False)
        old_tokens = torch.randn(1, 4, 8)
        old_cache = backbone.precompute_request_cache({"s_tokens": old_tokens})
        changed = old_tokens.clone()
        changed[:, 1, :] += 1.0

        with self.assertRaisesRegex(ValueError, "exact prefix"):
            backbone.update_request_cache(
                {"s_tokens": torch.cat([changed, torch.randn(1, 1, 8)], dim=1)},
                old_cache,
            )


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
            dtsi_training_output="dense_router",
        ).train()
        with torch.no_grad():
            module.sparse_routers[0].weight.zero_()
            module.sparse_routers[0].bias.fill_(1.0)

        tokens = torch.randn(2, 1, 3, requires_grad=True)
        output = module(tokens)
        regularization = module.regularization_loss(output)
        module.step_regularization_controller()

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

    def test_dtsi_requires_an_explicit_training_output_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "not specified by RankMixer"):
            SparseMoEPerTokenFFN(
                num_tokens=1,
                token_dim=3,
                hidden_dim=5,
                use_dtsi=True,
            )

    def test_checkpoint_recompute_steps_adaptive_controller_once(self) -> None:
        config = _rankmixer_config()
        config.model.rankmixer_ffn_type = "sparse_moe"
        config.model.sparse_moe_dtsi_training_output = "dense_router"
        config.model.sparse_moe_regularization_initial = 1.0e-4
        config.model.sparse_moe_regularization_multiplier = 2.0
        block = RankMixerBlock(config, feature_token_count=2).train()
        module = block.feature_ffn
        assert isinstance(module, SparseMoEPerTokenFFN)
        with torch.no_grad():
            for router in module.sparse_routers:
                router.weight.zero_()
                router.bias.fill_(1.0)

        tokens = torch.randn(3, 2, 4, requires_grad=True)
        output = checkpoint(block, tokens, use_reentrant=False)
        loss = output.square().mean() + module.regularization_loss(output)
        loss.backward()
        _step_sparse_moe_controllers(block)
        coefficient_after_step = module.regularization_coefficient.clone()
        _step_sparse_moe_controllers(block)

        torch.testing.assert_close(
            coefficient_after_step,
            coefficient_after_step.new_tensor(2.0e-4),
        )
        torch.testing.assert_close(
            module.regularization_coefficient,
            coefficient_after_step,
        )
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
