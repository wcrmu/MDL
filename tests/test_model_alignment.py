from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from src.dataloader import FeatureBatch
from src.embeddings import grouped_sharded_embedding_lookup
from src.config import SequenceConfig, SequenceFieldConfig, TokenGroupConfig, load_app_config
from src.model import (
    FeatureEncoderBank,
    MDLDomainBlock,
    MDLRankMixerBlock,
    MDLRankMixerModel,
    MDLOneTransModel,
    ModelMetadata,
    MixedCausalAttention,
    MixedFFN,
    OneTransBackbone,
    OneTransBackboneState,
    OneTransBlock,
    OneTransOutput,
    OneTransRequestCache,
    OneTransTokenizer,
    RankMixerBlock,
    RankMixerModel,
    RankMixerSliceTokenizer,
    ScenarioTower,
    _call_varlen_attention,
    _embedding_size,
    _forward_domain_interaction,
    _mdl_logits,
    _scenario_mask_from_ids,
    _VarlenPacking,
    varlen_attn,
)
from src.modules.attention import (
    DomainFusedModule,
    RankMixerDomainInteraction,
    VariableLengthDomainAttention,
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
            mdl_feature_interaction="direct_ffn",
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
            tokenizer.projection.weight.zero_()
            tokenizer.projection.bias.zero_()
            tokenizer.projection.weight[:, :3, :3] = torch.eye(3).unsqueeze(0)
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
        )
        with torch.no_grad():
            for parameter in block.feature_ffn.parameters():
                parameter.zero_()
        return block

    def test_direct_ffn_mode_has_no_second_add_norm(self) -> None:
        block = self._block("direct_ffn")
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

    def test_residual_ffn_mode_keeps_second_add_norm(self) -> None:
        block = self._block("residual_ffn")
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


class MDLAblationAlignmentTest(unittest.TestCase):
    @staticmethod
    def _sum_head(scale: float) -> nn.Linear:
        head = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            head.weight.fill_(scale)
        return head

    def test_terminal_model_blocks_keep_scenario_ffn_parameters(self) -> None:
        class Encoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output_dims: dict[str, int] = {}

        class OneTransBackboneStub(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ns_token_count = 3
                self.encoder_bank = Encoder()

        root = Path(__file__).resolve().parents[1]
        rankmixer_config = load_app_config(root / "configs" / "default.yaml")
        with patch("src.model.FeatureEncoderBank", return_value=Encoder()), patch(
            "src.model._build_rankmixer_feature_projector",
            return_value=nn.Identity(),
        ), patch(
            "src.model.DomainTokenProjector",
            side_effect=lambda *_args, **_kwargs: nn.Identity(),
        ):
            rankmixer_model = MDLRankMixerModel(rankmixer_config, {})

        onetrans_config = load_app_config(root / "configs" / "mdl_onetrans.yaml")
        with patch(
            "src.model.OneTransBackbone",
            return_value=OneTransBackboneStub(),
        ), patch(
            "src.model.DomainTokenProjector",
            side_effect=lambda *_args, **_kwargs: nn.Identity(),
        ):
            onetrans_model = MDLOneTransModel(onetrans_config, {})

        for name, model in (
            ("mdl_rankmixer", rankmixer_model),
            ("mdl_onetrans", onetrans_model),
        ):
            with self.subTest(model=name):
                self.assertGreater(len(model.blocks), 0)
                scenario_ffn = model.blocks[-1].scenario_ffn
                self.assertIsNotNone(scenario_ffn)
                assert scenario_ffn is not None
                self.assertGreater(
                    sum(parameter.numel() for parameter in scenario_ffn.parameters()),
                    0,
                )

    def test_every_block_keeps_and_executes_scenario_ffn(self) -> None:
        config = _rankmixer_config()
        block = MDLRankMixerBlock(
            config,
            ModelMetadata(feature_token_count=2, scenario_count=1, task_count=1),
        )

        class AddOne(nn.Module):
            def forward(self, values: Tensor) -> Tensor:
                return torch.ones_like(values)

        with torch.no_grad():
            for module in (block.scenario_attention, block.task_attention, block.task_ffn):
                assert module is not None
                for parameter in module.parameters():
                    parameter.zero_()
        block.scenario_ffn = AddOne()
        scenario_tokens = torch.randn(2, 2, 4)

        _features, actual_scenarios, _tasks = block(
            torch.randn(2, 2, 4),
            scenario_tokens,
            torch.zeros(2, 1, 4),
            torch.ones(2, 1),
        )

        torch.testing.assert_close(actual_scenarios, scenario_tokens + 1.0)

    def test_disabled_domain_attention_uses_feature_dependent_rankmixer(self) -> None:
        config = _rankmixer_config()
        config.model.use_task_feature_interaction = False
        config.model.use_scenario_feature_interaction = False
        block = MDLRankMixerBlock(
            config,
            ModelMetadata(feature_token_count=2, scenario_count=1, task_count=1),
        )

        self.assertIsNone(block.scenario_attention)
        self.assertIsNone(block.task_attention)
        self.assertIsInstance(block.scenario_rankmixer, RankMixerDomainInteraction)
        self.assertIsInstance(block.task_rankmixer, RankMixerDomainInteraction)

        scenario_tokens = torch.zeros(2, 2, 4)
        zero_features = torch.zeros(2, 2, 4)
        informative_features = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[8.0, 7.0, 6.0, 5.0], [4.0, 3.0, 2.0, 1.0]],
            ]
        )
        assert block.scenario_rankmixer is not None
        without_features = block.scenario_rankmixer(scenario_tokens, zero_features)
        with_features = block.scenario_rankmixer(
            scenario_tokens,
            informative_features,
        )

        self.assertFalse(torch.allclose(with_features, without_features))

    def test_disabled_tokens_remove_block_modules_and_scenario_tower_selects(self) -> None:
        config = _rankmixer_config()
        config.model.use_task_tokens = False
        config.model.use_scenario_tokens = False
        block = MDLRankMixerBlock(
            config,
            ModelMetadata(feature_token_count=2, scenario_count=2, task_count=2),
        )

        for module in (
            block.scenario_attention,
            block.scenario_rankmixer,
            block.scenario_ffn,
            block.task_attention,
            block.task_rankmixer,
            block.task_ffn,
            block.domain_fused,
        ):
            self.assertIsNone(module)

        feature_tokens = torch.randn(2, 2, 4)
        empty_tokens = feature_tokens.new_empty(2, 0, 4)
        _features, actual_scenarios, actual_tasks = block(
            feature_tokens,
            empty_tokens,
            empty_tokens,
            torch.eye(2),
        )
        self.assertEqual(tuple(actual_scenarios.shape), (2, 0, 4))
        self.assertEqual(tuple(actual_tasks.shape), (2, 0, 4))

        tower = ScenarioTower(2, token_dim=4, hidden_dim=4, activation="relu")
        tower.networks[0] = nn.Linear(4, 4, bias=False)
        tower.networks[1] = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            tower.networks[0].weight.copy_(torch.eye(4))
            tower.networks[1].weight.copy_(2.0 * torch.eye(4))
        selected = tower(torch.ones(2, 2, 4), torch.eye(2))
        torch.testing.assert_close(
            selected,
            torch.tensor([[1.0] * 4, [2.0] * 4]),
        )
        self.assertFalse(
            any(isinstance(module, DomainFusedModule) for module in tower.modules())
        )

    def test_each_token_switch_removes_only_its_block_path(self) -> None:
        metadata = ModelMetadata(
            feature_token_count=2,
            scenario_count=1,
            task_count=1,
        )
        for use_task_tokens, use_scenario_tokens in (
            (False, True),
            (True, False),
        ):
            with self.subTest(
                use_task_tokens=use_task_tokens,
                use_scenario_tokens=use_scenario_tokens,
            ):
                config = _rankmixer_config()
                config.model.use_task_tokens = use_task_tokens
                config.model.use_scenario_tokens = use_scenario_tokens
                block = MDLRankMixerBlock(config, metadata)

                self.assertEqual(
                    block.scenario_attention is not None,
                    use_scenario_tokens,
                )
                self.assertEqual(
                    block.scenario_ffn is not None,
                    use_scenario_tokens,
                )
                self.assertEqual(
                    block.task_attention is not None,
                    use_task_tokens,
                )
                self.assertEqual(block.task_ffn is not None, use_task_tokens)
                self.assertEqual(
                    block.domain_fused is not None,
                    use_task_tokens and use_scenario_tokens,
                )

                feature_tokens = torch.randn(2, 2, 4)
                scenario_tokens = feature_tokens.new_empty(
                    2,
                    2 if use_scenario_tokens else 0,
                    4,
                )
                task_tokens = feature_tokens.new_empty(
                    2,
                    1 if use_task_tokens else 0,
                    4,
                )
                _features, actual_scenarios, actual_tasks = block(
                    feature_tokens,
                    scenario_tokens,
                    task_tokens,
                    torch.ones(2, 1),
                )
                self.assertEqual(
                    actual_scenarios.size(1),
                    2 if use_scenario_tokens else 0,
                )
                self.assertEqual(
                    actual_tasks.size(1),
                    1 if use_task_tokens else 0,
                )

    def test_both_model_variants_omit_disabled_token_projectors(self) -> None:
        class Encoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output_dims: dict[str, int] = {}

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

        class OneTransBackboneStub(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ns_token_count = 3
                self.encoder_bank = Encoder()

            def prepare(
                self,
                features: dict[str, Tensor],
                request_cache: object | None = None,
                encoded_features: dict[str, Tensor] | None = None,
            ) -> OneTransBackboneState:
                del features, request_cache
                assert encoded_features is not None
                tokens = encoded_features["tokens"]
                return OneTransBackboneState(
                    tokens=tokens,
                    valid_mask=torch.ones(
                        tokens.shape[:2],
                        dtype=torch.bool,
                        device=tokens.device,
                    ),
                    s_count=0,
                    ns_count=tokens.size(1),
                    initial_s_count=0,
                    encoded_features=encoded_features,
                )

            def step(
                self,
                state: OneTransBackboneState,
                layer_index: int,
                layer_cache: object | None = None,
            ) -> OneTransBackboneState:
                del layer_index, layer_cache
                return state

        root = Path(__file__).resolve().parents[1]
        rankmixer_config = load_app_config(root / "configs" / "default.yaml")
        rankmixer_config = replace(
            rankmixer_config,
            model=replace(
                rankmixer_config.model,
                use_task_tokens=False,
                use_scenario_tokens=False,
            ),
        )
        with patch("src.model.FeatureEncoderBank", return_value=Encoder()), patch(
            "src.model._build_rankmixer_feature_projector",
            return_value=TokenProjector(),
        ), patch("src.model.DomainTokenProjector") as domain_projector:
            rankmixer_model = MDLRankMixerModel(rankmixer_config, {})
        domain_projector.assert_not_called()

        onetrans_config = load_app_config(root / "configs" / "mdl_onetrans.yaml")
        onetrans_config = replace(
            onetrans_config,
            model=replace(
                onetrans_config.model,
                use_task_tokens=False,
                use_scenario_tokens=False,
            ),
        )
        with patch(
            "src.model.OneTransBackbone",
            return_value=OneTransBackboneStub(),
        ), patch("src.model.DomainTokenProjector") as domain_projector:
            onetrans_model = MDLOneTransModel(onetrans_config, {})
        domain_projector.assert_not_called()

        for name, model in (
            ("mdl_rankmixer", rankmixer_model),
            ("mdl_onetrans", onetrans_model),
        ):
            with self.subTest(model=name):
                self.assertIsNone(model.scenario_projector)
                self.assertIsNone(model.task_projector)
                self.assertIsInstance(model.scenario_tower, ScenarioTower)
                self.assertFalse(
                    any(
                        isinstance(module, DomainFusedModule)
                        for module in model.modules()
                    )
                )
                for block in model.blocks:
                    self.assertIsNone(block.scenario_attention)
                    self.assertIsNone(block.scenario_ffn)
                    self.assertIsNone(block.task_attention)
                    self.assertIsNone(block.task_ffn)
                    self.assertIsNone(block.domain_fused)

        for name, model, token_count in (
            ("mdl_rankmixer", rankmixer_model, 4),
            ("mdl_onetrans", onetrans_model, 3),
        ):
            with self.subTest(model_forward=name):
                tokens = torch.randn(2, token_count, 32, requires_grad=True)
                output = model(
                    {"tokens": tokens},
                    scenario_id=torch.zeros(2, dtype=torch.long),
                )
                self.assertEqual(tuple(output["logits"].shape), (2, 1))
                output["logits"].sum().backward()
                self.assertIsNotNone(tokens.grad)
                self.assertTrue(bool(torch.isfinite(tokens.grad).all()))

    def test_global_token_switch_removes_global_token_parameters(self) -> None:
        config = _rankmixer_config()
        config.model.use_global_scenario_token = False
        block = MDLRankMixerBlock(
            config,
            ModelMetadata(feature_token_count=2, scenario_count=2, task_count=1),
        )

        assert block.scenario_attention is not None
        assert block.scenario_ffn is not None
        assert block.domain_fused is not None
        self.assertEqual(block.scenario_attention.num_domain_tokens, 2)
        self.assertEqual(len(block.scenario_ffn.networks), 2)
        self.assertFalse(block.domain_fused.has_global_token)

    def test_task_token_ablation_feeds_scenario_context_to_task_towers(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(
                model=SimpleNamespace(
                    use_scenario_tokens=True,
                    use_task_tokens=False,
                    use_global_scenario_token=True,
                )
            ),
            logit_layers=nn.ModuleList(
                [self._sum_head(1.0), self._sum_head(2.0)]
            ),
        )
        feature_tokens = torch.zeros(2, 2, 4)
        scenario_tokens = torch.tensor(
            [
                [[2.0] * 4, [8.0] * 4, [4.0] * 4],
                [[2.0] * 4, [8.0] * 4, [4.0] * 4],
            ]
        )

        logits = _mdl_logits(
            model,
            feature_tokens,
            scenario_tokens,
            feature_tokens.new_empty(2, 0, 4),
            torch.eye(2),
        )

        # Selected scenario + global are mean pooled: [3, 3, 3, 3] and
        # [6, 6, 6, 6], then independent task towers apply different weights.
        torch.testing.assert_close(
            logits,
            torch.tensor([[12.0, 24.0], [24.0, 48.0]]),
        )

    def test_scenario_token_ablation_fuses_scenario_tower_at_output(self) -> None:
        tower = ScenarioTower(2, token_dim=4, hidden_dim=4, activation="relu")
        tower.networks[0] = nn.Linear(4, 4, bias=False)
        tower.networks[1] = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            tower.networks[0].weight.copy_(torch.eye(4))
            tower.networks[1].weight.copy_(2.0 * torch.eye(4))
        model = SimpleNamespace(
            config=SimpleNamespace(
                model=SimpleNamespace(
                    use_scenario_tokens=False,
                    use_task_tokens=True,
                )
            ),
            scenario_tower=tower,
            logit_layers=nn.ModuleList(
                [self._sum_head(1.0), self._sum_head(1.0)]
            ),
        )

        logits = _mdl_logits(
            model,
            torch.ones(2, 2, 4),
            torch.empty(2, 0, 4),
            torch.zeros(2, 2, 4),
            torch.eye(2),
        )

        torch.testing.assert_close(
            logits,
            torch.tensor([[4.0, 4.0], [8.0, 8.0]]),
        )


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


class OneTransStackedProjectionTest(unittest.TestCase):
    def test_ns_attention_projections_match_independent_linears(self) -> None:
        torch.manual_seed(17)
        attention = MixedCausalAttention(
            token_dim=6,
            num_heads=2,
            ns_token_count=3,
        )
        tokens = torch.randn(4, 5, 6, requires_grad=True)

        actual_ns = attention._project_ns_batched(
            tokens[:, 2:, :],
            attention.ns_key,
        )
        expected_ns = torch.cat(
            [
                layer(tokens[:, 2 + index, :]).unsqueeze(1)
                for index, layer in enumerate(attention.ns_key)
            ],
            dim=1,
        )

        torch.testing.assert_close(actual_ns, expected_ns)
        actual_ns.square().sum().backward()
        self.assertTrue(
            all(layer.weight.grad is not None for layer in attention.ns_key)
        )
        self.assertIn("ns_key.0.weight", attention.state_dict())

    def test_ns_ffn_matches_independent_networks(self) -> None:
        torch.manual_seed(19)
        ffn = MixedFFN(token_dim=6, hidden_dim=10, ns_token_count=3)
        tokens = torch.randn(4, 5, 6, requires_grad=True)

        actual = ffn(tokens, query_s_count=2)
        batched_ns = ffn._forward_ns_batched(tokens[:, 2:, :])
        expected = torch.cat(
            [
                ffn.s_ffn(tokens[:, :2, :]),
                *[
                    network(tokens[:, 2 + index, :]).unsqueeze(1)
                    for index, network in enumerate(ffn.ns_ffn)
                ],
            ],
            dim=1,
        )

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(batched_ns, expected[:, 2:, :])
        actual.square().sum().backward()
        self.assertTrue(
            all(network[0].weight.grad is not None for network in ffn.ns_ffn)
        )
        self.assertIn("ns_ffn.0.0.weight", ffn.state_dict())

    def test_stacked_ns_paths_keep_autocast_output_dtype(self) -> None:
        attention = MixedCausalAttention(
            token_dim=8,
            num_heads=2,
            ns_token_count=3,
        )
        ffn = MixedFFN(token_dim=8, hidden_dim=16, ns_token_count=3)
        tokens = torch.randn(2, 3, 8)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            projected = attention._project_all(
                tokens,
                0,
                attention.s_key,
                attention.ns_key,
            )
            transformed = ffn(tokens, query_s_count=0)

        self.assertEqual(projected.dtype, torch.bfloat16)
        self.assertEqual(transformed.dtype, torch.bfloat16)


class VarlenPackingTest(unittest.TestCase):
    def test_reused_indices_preserve_pack_unpack_values_and_gradients(self) -> None:
        mask = torch.tensor(
            [[False, True, True, False], [True, False, True, True]]
        )
        values = torch.randn(2, 4, 3, requires_grad=True)
        reference_values = values.detach().clone().requires_grad_(True)
        packing = _VarlenPacking.from_mask(mask)

        packed = packing.pack(values)
        output = packing.unpack(2.0 * packed, values)
        expected = torch.zeros_like(reference_values)
        expected[mask] = 2.0 * reference_values[mask]

        torch.testing.assert_close(packed[: int(mask.sum())], values[mask])
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(
            packing.lengths,
            torch.tensor([2, 3], dtype=torch.int32),
        )
        torch.testing.assert_close(
            packing.cumulative_lengths,
            torch.tensor([0, 2, 5], dtype=torch.int32),
        )

        output.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(values.grad, reference_values.grad)

    @unittest.skipUnless(
        torch.cuda.is_available() and varlen_attn is not None,
        "fixed-capacity varlen Flash test requires CUDA varlen attention",
    )
    def test_fixed_capacity_flash_matches_exact_dynamic_packing(self) -> None:
        torch.manual_seed(23)
        device = torch.device("cuda")
        mask = torch.tensor(
            [
                [False, False, True, True, True],
                [False, True, True, True, True],
                [True, True, True, True, True],
            ],
            device=device,
        )
        packing = _VarlenPacking.from_mask(mask)
        fixed_inputs = [
            torch.randn(
                3,
                5,
                2,
                16,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for _ in range(3)
        ]
        exact_inputs = [
            value.detach().clone().requires_grad_(True) for value in fixed_inputs
        ]

        fixed_packed = _call_varlen_attention(
            *(packing.pack(value) for value in fixed_inputs),
            packing.cumulative_lengths,
            packing.cumulative_lengths,
            mask.size(1),
            mask.size(1),
            causal=False,
        )
        fixed_output = packing.unpack(fixed_packed, fixed_inputs[0])
        exact_packed = _call_varlen_attention(
            *(value[mask] for value in exact_inputs),
            packing.cumulative_lengths,
            packing.cumulative_lengths,
            mask.size(1),
            mask.size(1),
            causal=False,
        )
        exact_output = torch.zeros_like(exact_inputs[0])
        exact_output[mask] = exact_packed

        torch.testing.assert_close(fixed_output, exact_output)
        fixed_output.float().square().sum().backward()
        exact_output.float().square().sum().backward()
        for fixed, exact in zip(fixed_inputs, exact_inputs):
            torch.testing.assert_close(fixed.grad, exact.grad)


class FeatureEncoderShardedFusionTest(unittest.TestCase):
    @staticmethod
    def _features(config: object, batch_size: int = 2, length: int = 3) -> dict[str, object]:
        values: dict[str, object] = {}
        for feature in config.features:  # type: ignore[attr-defined]
            values[feature.name] = (
                torch.randint(1, 15, (batch_size,))
                if feature.kind == "categorical"
                else torch.randn(batch_size, feature.dimension)
            )
        for sequence in config.sequences:  # type: ignore[attr-defined]
            fields: dict[str, Tensor] = {}
            for field in sequence.fields:
                shape = (
                    (batch_size, length)
                    if field.dimension == 1
                    else (batch_size, length, field.dimension)
                )
                fields[field.name] = (
                    torch.randint(1, 15, shape)
                    if field.kind == "categorical"
                    else torch.randn(shape)
                )
            values[sequence.name] = {
                "fields": fields,
                "lengths": torch.tensor([length, length - 1]),
            }
        return values

    def test_mdl_fuses_scalar_and_all_sequence_lookups_with_output_parity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "mdl_perf.yaml")
        bank = FeatureEncoderBank(
            config,
            {},
            config.model.embedding_dim,
            embedding_size_override=16,
        ).eval()
        features = self._features(config)

        with patch(
            "src.model.grouped_sharded_embedding_lookup",
            wraps=grouped_sharded_embedding_lookup,
        ) as grouped_lookup, torch.no_grad():
            fused = bank(features)

        self.assertEqual(grouped_lookup.call_count, 1)
        with patch.object(bank, "_preencode_sharded_inputs", return_value={}), torch.no_grad():
            unfused = bank(features)
        self.assertEqual(fused.keys(), unfused.keys())
        for name in fused:
            torch.testing.assert_close(fused[name], unfused[name])

    def test_onetrans_fuses_ns_and_sequence_lookups(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "onetrans_perf.yaml")
        bank = FeatureEncoderBank(
            config,
            {},
            config.model.embedding_dim,
            build_sequence_summaries=False,
            embedding_size_override=16,
        ).eval()
        tokenizer = OneTransTokenizer(config, bank).eval()

        with patch(
            "src.model.grouped_sharded_embedding_lookup",
            wraps=grouped_sharded_embedding_lookup,
        ) as grouped_lookup, torch.no_grad():
            output = tokenizer(self._features(config))

        self.assertEqual(grouped_lookup.call_count, 1)
        self.assertEqual(output.feature_tokens.size(0), 2)


class VarlenAttentionCompatibilityTest(unittest.TestCase):
    def _inputs(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        query = torch.randn(3, 2, 4)
        key = torch.randn(5, 2, 4)
        value = torch.randn(5, 2, 4)
        return query, key, value, torch.tensor([0, 3]), torch.tensor([0, 5])

    def test_pytorch_210_is_causal_api(self) -> None:
        observed: list[bool] = []

        def legacy(*args: Tensor | int, is_causal: bool = False) -> Tensor:
            observed.append(is_causal)
            assert isinstance(args[0], Tensor)
            return args[0]

        inputs = self._inputs()
        with patch("src.model.varlen_attn", legacy), patch(
            "src.model._VARLEN_ATTN_USES_WINDOW_SIZE",
            False,
        ):
            output = _call_varlen_attention(
                *inputs,
                3,
                5,
                causal=True,
            )

        self.assertIs(output, inputs[0])
        self.assertEqual(observed, [True])

    def test_pytorch_212_window_size_api(self) -> None:
        observed: list[tuple[int, int]] = []

        def modern(
            *args: Tensor | int,
            window_size: tuple[int, int] = (-1, -1),
        ) -> Tensor:
            observed.append(window_size)
            assert isinstance(args[0], Tensor)
            return args[0]

        inputs = self._inputs()
        with patch("src.model.varlen_attn", modern), patch(
            "src.model._VARLEN_ATTN_USES_WINDOW_SIZE",
            True,
        ):
            output = _call_varlen_attention(
                *inputs,
                3,
                5,
                causal=False,
            )

        self.assertIs(output, inputs[0])
        self.assertEqual(observed, [(-1, -1)])


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
            target_length: int | None = None,
        ) -> tuple[Tensor, Tensor]:
            del sequence_name
            tokens = value["event_inputs"]
            mask = value["mask"]
            assert isinstance(tokens, Tensor)
            assert isinstance(mask, Tensor)
            if target_length is None or target_length == tokens.size(1):
                return tokens, mask
            if target_length < tokens.size(1):
                return tokens[:, -target_length:, :], mask[:, -target_length:]
            padding = target_length - tokens.size(1)
            return (
                torch.cat(
                    [
                        tokens.new_zeros(tokens.size(0), padding, tokens.size(2)),
                        tokens,
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        torch.zeros(
                            mask.size(0),
                            padding,
                            dtype=torch.bool,
                            device=mask.device,
                        ),
                        mask,
                    ],
                    dim=1,
                ),
            )

        def _align_sequence_inputs(
            self,
            sequence: SequenceConfig,
            inputs: Tensor,
            lengths: Tensor,
            target_length: int | None = None,
        ) -> tuple[Tensor, Tensor]:
            del sequence, lengths, target_length
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

    def test_compact_sequence_contract_rejects_global_padding_prefix(self) -> None:
        tokenizer = self._fusion_tokenizer("intent_ordered")
        tokenizer.require_compact_sequence_batches = True
        features = self._fusion_features()
        for value in features.values():
            value["mask"][:, 0] = False

        with self.assertRaisesRegex(ValueError, "padded only to the longest row"):
            tokenizer.precompute_request_cache(features)

    def test_group_sequences_with_different_max_lengths_share_target_length(self) -> None:
        base = self._base_config()
        sequences = [
            SequenceConfig(
                name=name,
                fields=[
                    SequenceFieldConfig(
                        name="value",
                        kind="dense",
                        source=f"{name}_value",
                        dimension=2,
                    )
                ],
                max_length=max_length,
                encoder="raw",
            )
            for name, max_length in (("short", 3), ("long", 5))
        ]
        group = TokenGroupConfig(name="aligned", inputs=["short", "long"])
        config = replace(
            base,
            sequences=sequences,
            tokenization=replace(base.tokenization, sequence_tokens=[group]),
            model=replace(base.model, token_dim=4),
        )
        scalar_names = [
            feature.name
            for feature in config.features
            if feature.embedding_scope in {"feature", "shared"}
        ]
        encoder = self.Encoder(
            {name: 1 for name in scalar_names},
            {"short": 2, "long": 2},
        )
        tokenizer = OneTransTokenizer(config, encoder)
        features = {
            "short": {
                "event_inputs": torch.randn(1, 3, 2),
                "mask": torch.ones(1, 3, dtype=torch.bool),
                "fields": {"value": torch.randn(1, 3, 2)},
                "lengths": torch.tensor([3]),
            },
            "long": {
                "event_inputs": torch.randn(1, 5, 2),
                "mask": torch.tensor([[False, False, True, True, True]]),
                "fields": {"value": torch.randn(1, 5, 2)},
                "lengths": torch.tensor([3]),
            },
        }

        tokens, mask, _timestamps = tokenizer._sequence_group_tokens(
            group,
            nn.Identity(),
            features,
        )

        self.assertEqual(tuple(tokens.shape), (1, 5, 4))
        torch.testing.assert_close(
            mask,
            torch.tensor([[False, False, True, True, True]]),
        )

    def test_group_uses_payload_width_below_configured_capacity(self) -> None:
        base = self._base_config()
        sequence = SequenceConfig(
            name="compact",
            fields=[
                SequenceFieldConfig(
                    name="value",
                    kind="dense",
                    source="compact_value",
                    dimension=4,
                )
            ],
            max_length=100,
            encoder="raw",
        )
        group = TokenGroupConfig(name="compact", inputs=["compact"])
        config = replace(
            base,
            sequences=[sequence],
            tokenization=replace(base.tokenization, sequence_tokens=[group]),
            model=replace(base.model, token_dim=4),
        )
        scalar_names = [
            feature.name
            for feature in config.features
            if feature.embedding_scope in {"feature", "shared"}
        ]
        encoder = self.Encoder(
            {name: 1 for name in scalar_names},
            {"compact": 4},
        )
        tokenizer = OneTransTokenizer(config, encoder)
        features = {
            "compact": {
                "event_inputs": torch.randn(2, 7, 4),
                "mask": torch.tensor(
                    [
                        [False, False, True, True, True, True, True],
                        [True, True, True, True, True, True, True],
                    ]
                ),
                "fields": {"value": torch.randn(2, 7, 4)},
                "lengths": torch.tensor([5, 7]),
            }
        }

        tokens, mask, _timestamps = tokenizer._sequence_group_tokens(
            group,
            nn.Identity(),
            features,
        )

        self.assertEqual(tuple(tokens.shape), (2, 7, 4))
        self.assertEqual(tuple(mask.shape), (2, 7))


class OneTransRuntimeCorrectnessTest(unittest.TestCase):
    def test_shared_vocab_alias_uses_base_vocab_size_for_independent_table(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "mdl_onetrans.yaml")
        vocab_maps = {
            "user_id": {"one": 1, "seven": 7},
            # A stale or partial alias map must not size the independent table.
            "scenario_user_id": {"one": 1},
        }

        self.assertEqual(
            _embedding_size(config, vocab_maps, "scenario_user_id"),
            8,
        )

    def test_fractional_scenario_ids_are_rejected_before_integer_cast(self) -> None:
        for value in (0.9, -0.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "integer-valued ids"):
                    _scenario_mask_from_ids(torch.tensor([value]), scenario_count=2)

        torch.testing.assert_close(
            _scenario_mask_from_ids(torch.tensor([0.0, 1.0]), scenario_count=2),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )


class MDLOneTransSequenceAttentionTest(unittest.TestCase):
    @staticmethod
    def _small_model_and_features(
        batch_size: int,
    ) -> tuple[MDLOneTransModel, dict[str, object]]:
        root = Path(__file__).resolve().parents[1]
        base = load_app_config(root / "configs" / "mdl_onetrans.yaml")
        sequence = replace(base.sequences[0], max_length=6)
        config = replace(
            base,
            sequences=[sequence],
            model=replace(
                base.model,
                embedding_dim=4,
                token_dim=8,
                num_layers=3,
                num_heads=2,
                hidden_dim=16,
                use_pyramid=True,
                pyramid_round_to=1,
                final_s_tokens=2,
                max_position_embeddings=10,
                first_domain_sequence_layer=0,
            ),
        )
        model = MDLOneTransModel(
            config,
            {},
            embedding_size_override=16,
        ).eval()
        features: dict[str, object] = {}
        for feature in config.features:
            if feature.kind == "categorical":
                features[feature.name] = torch.randint(1, 15, (batch_size,))
            else:
                features[feature.name] = torch.randn(batch_size, feature.dimension)
        fields: dict[str, Tensor] = {}
        for field in sequence.fields:
            shape = (batch_size, sequence.max_length)
            fields[field.name] = (
                torch.randint(1, 15, shape)
                if field.kind == "categorical"
                else torch.randn(shape)
            )
        features[sequence.name] = {
            "fields": fields,
            "lengths": torch.full((batch_size,), 5, dtype=torch.long),
        }
        return model, features

    def test_padding_values_do_not_affect_attention_and_empty_rows_are_zero(self) -> None:
        torch.manual_seed(41)
        attention = VariableLengthDomainAttention(8, 2).eval()
        domain_tokens = torch.randn(2, 3, 8)
        sequence_tokens = torch.randn(2, 5, 8)
        sequence_mask = torch.tensor(
            [
                [False, False, True, True, True],
                [False, False, False, False, False],
            ]
        )

        expected = attention(domain_tokens, sequence_tokens, sequence_mask)
        changed = sequence_tokens.clone()
        changed[~sequence_mask] = torch.randn_like(changed[~sequence_mask]) * 1.0e4
        actual = attention(domain_tokens, changed, sequence_mask)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[1], torch.zeros_like(actual[1]))
        self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_attention_accepts_pyramid_sequence_lengths(self) -> None:
        attention = VariableLengthDomainAttention(8, 2).eval()
        domain_tokens = torch.randn(2, 3, 8)

        for layer_index, sequence_length in enumerate((128, 64, 32)):
            with self.subTest(layer_index=layer_index):
                output = attention(
                    domain_tokens,
                    torch.randn(2, sequence_length, 8),
                    torch.ones(2, sequence_length, dtype=torch.bool),
                )
                self.assertEqual(tuple(output.shape), (2, 3, 8))
                self.assertTrue(bool(torch.isfinite(output).all()))

    def test_zero_sequence_gate_exactly_matches_ns_only_domain_path(self) -> None:
        class ZeroGate(nn.Module):
            def forward(self, values: Tensor) -> Tensor:
                return values[..., : values.size(-1) // 3] * 0.0

        config = _rankmixer_config()
        block = MDLDomainBlock(
            config,
            ModelMetadata(feature_token_count=2, scenario_count=2, task_count=2),
            use_sequence_attention=True,
        ).eval()
        self.assertIsNotNone(block.scenario_sequence_gate)
        self.assertIsNotNone(block.task_sequence_gate)
        torch.testing.assert_close(
            block.scenario_sequence_gate[0].bias,
            torch.full_like(block.scenario_sequence_gate[0].bias, -2.0),
        )
        block.scenario_sequence_gate = ZeroGate()
        block.task_sequence_gate = ZeroGate()

        ns_tokens = torch.randn(2, 2, 4)
        s_tokens = torch.randn(2, 5, 4)
        s_mask = torch.tensor(
            [[False, True, True, True, True], [True, True, True, True, True]]
        )
        scenario_tokens = torch.randn(2, 3, 4)
        task_tokens = torch.randn(2, 2, 4)
        scenario_mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        expected = _forward_domain_interaction(
            block,
            ns_tokens,
            scenario_tokens,
            task_tokens,
            scenario_mask,
        )
        actual = block(
            ns_tokens,
            s_tokens,
            s_mask,
            scenario_tokens,
            task_tokens,
            scenario_mask,
        )

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_empty_history_block_output_is_finite_and_matches_ns_only_path(self) -> None:
        block = MDLDomainBlock(
            _rankmixer_config(),
            ModelMetadata(feature_token_count=2, scenario_count=1, task_count=1),
            use_sequence_attention=True,
        ).eval()
        ns_tokens = torch.randn(2, 2, 4)
        scenario_tokens = torch.randn(2, 2, 4)
        task_tokens = torch.randn(2, 1, 4)
        scenario_mask = torch.ones(2, 1)
        expected = _forward_domain_interaction(
            block,
            ns_tokens,
            scenario_tokens,
            task_tokens,
            scenario_mask,
        )

        actual = block(
            ns_tokens,
            torch.randn(2, 5, 4),
            torch.zeros(2, 5, dtype=torch.bool),
            scenario_tokens,
            task_tokens,
            scenario_mask,
        )

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        self.assertTrue(bool(torch.isfinite(actual[0]).all()))
        self.assertTrue(bool(torch.isfinite(actual[1]).all()))

    def test_task_loss_has_gradient_through_sequence_attention_and_s_states(self) -> None:
        torch.manual_seed(43)
        block = MDLDomainBlock(
            _rankmixer_config(),
            ModelMetadata(feature_token_count=2, scenario_count=1, task_count=1),
            use_sequence_attention=True,
        )
        s_tokens = torch.randn(2, 5, 4, requires_grad=True)
        _scenario_tokens, task_tokens = block(
            torch.randn(2, 2, 4),
            s_tokens,
            torch.tensor(
                [[False, True, True, True, True], [True, True, True, True, True]]
            ),
            torch.randn(2, 2, 4),
            torch.randn(2, 1, 4),
            torch.ones(2, 1),
        )

        task_tokens.square().sum().backward()

        assert block.task_sequence_attention is not None
        gradient = block.task_sequence_attention.key_projection.weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None and s_tokens.grad is not None
        self.assertGreater(float(gradient.abs().sum().item()), 0.0)
        self.assertGreater(float(s_tokens.grad.abs().sum().item()), 0.0)
        self.assertTrue(bool(torch.isfinite(s_tokens.grad).all()))

    def test_cached_candidate_fanout_matches_full_recompute_across_pyramid(self) -> None:
        torch.manual_seed(47)
        model, single_features = self._small_model_and_features(batch_size=1)
        candidate_features = {
            name: value.expand(3, *value.shape[1:]).clone()
            for name, value in single_features.items()
            if isinstance(value, Tensor)
        }
        for name in ("item_id", "scenario_item_id", "task_item_id"):
            candidate_features[name] = torch.tensor([1, 2, 3])
        single_history = single_features["hist"]
        assert isinstance(single_history, dict)
        single_fields = single_history["fields"]
        assert isinstance(single_fields, dict)
        candidate_features["hist"] = {
            "fields": {
                name: value.expand(3, -1).clone()
                for name, value in single_fields.items()
            },
            "lengths": single_history["lengths"].expand(3).clone(),
        }
        scenario_id = torch.zeros(3, dtype=torch.long)
        observed_lengths: list[int] = []
        hooks = []
        for block in model.blocks:
            assert block.task_sequence_attention is not None
            hooks.append(
                block.task_sequence_attention.register_forward_pre_hook(
                    lambda _module, args: observed_lengths.append(args[1].size(1))
                )
            )

        try:
            with torch.no_grad():
                request_cache = model.precompute_request_cache(single_features)
                uncached = model(candidate_features, scenario_id)["logits"]
                cached = model(
                    candidate_features,
                    scenario_id,
                    request_cache=request_cache,
                )["logits"]
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(observed_lengths, [4, 3, 2, 4, 3, 2])
        torch.testing.assert_close(cached, uncached, rtol=1.0e-5, atol=1.0e-6)


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
        backbone.unified_position_embeddings = nn.Embedding(32, 8)
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


class OneTransPositionEmbeddingAlignmentTest(unittest.TestCase):
    def test_prepare_adds_logical_positions_to_unified_s_and_ns_tokens(self) -> None:
        class Tokenizer(nn.Module):
            def forward(self, _features: dict[str, Tensor], **_kwargs: object) -> OneTransOutput:
                return OneTransOutput(
                    feature_tokens=torch.zeros(2, 5, 2),
                    encoded_features={},
                    s_token_count=3,
                    ns_token_count=2,
                    s_valid_mask=torch.tensor(
                        [[False, True, True], [True, True, True]]
                    ),
                )

        backbone = OneTransBackbone.__new__(OneTransBackbone)
        nn.Module.__init__(backbone)
        backbone.tokenizer = Tokenizer()
        backbone.unified_position_embeddings = nn.Embedding(5, 2)
        with torch.no_grad():
            backbone.unified_position_embeddings.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 10.0],
                        [2.0, 20.0],
                        [3.0, 30.0],
                        [4.0, 40.0],
                        [5.0, 50.0],
                    ]
                )
            )

        state = backbone.prepare({})

        torch.testing.assert_close(
            state.tokens,
            torch.tensor(
                [
                    [
                        [0.0, 0.0],
                        [1.0, 10.0],
                        [2.0, 20.0],
                        [3.0, 30.0],
                        [4.0, 40.0],
                    ],
                    [
                        [1.0, 10.0],
                        [2.0, 20.0],
                        [3.0, 30.0],
                        [4.0, 40.0],
                        [5.0, 50.0],
                    ],
                ]
            ),
        )

    def test_layer_cache_matches_full_path_with_unified_positions(self) -> None:
        class Tokenizer(nn.Module):
            num_ns_tokens = 2

            def precompute_request_cache(
                self,
                features: dict[str, Tensor],
            ) -> OneTransRequestCache:
                s_tokens = features["s_tokens"]
                return OneTransRequestCache(
                    s_tokens=s_tokens,
                    s_valid_mask=torch.ones(s_tokens.shape[:2], dtype=torch.bool),
                )

            def forward(
                self,
                features: dict[str, Tensor],
                request_cache: OneTransRequestCache | None = None,
                encoded_features: dict[str, Tensor] | None = None,
            ) -> OneTransOutput:
                cache = (
                    self.precompute_request_cache(features)
                    if request_cache is None
                    else request_cache
                )
                ns_tokens = features["ns_tokens"]
                return OneTransOutput(
                    feature_tokens=torch.cat([cache.s_tokens, ns_tokens], dim=1),
                    encoded_features=encoded_features or {},
                    s_token_count=cache.s_tokens.size(1),
                    ns_token_count=ns_tokens.size(1),
                    s_valid_mask=cache.s_valid_mask,
                )

        config = SimpleNamespace(
            model=SimpleNamespace(
                token_dim=4,
                num_heads=2,
                hidden_dim=8,
                num_layers=2,
                use_pyramid=False,
                final_s_tokens=None,
                pyramid_round_to=32,
                use_request_cache=False,
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
        backbone.unified_position_embeddings = nn.Embedding(8, 4)
        backbone.blocks = nn.ModuleList(
            OneTransBlock(config, ns_token_count=2) for _ in range(2)
        )
        backbone.eval()
        features = {
            "s_tokens": torch.randn(1, 3, 4),
            "ns_tokens": torch.randn(1, 2, 4),
        }

        with torch.no_grad():
            full = backbone(features)
            cache = backbone.precompute_request_cache(features)
            cached = backbone(features, request_cache=cache)

        torch.testing.assert_close(cached.feature_tokens, full.feature_tokens)
        torch.testing.assert_close(cache.s_tokens, features["s_tokens"])


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
    def test_backbone_never_builds_a_separate_sequence_summary_encoder(self) -> None:
        encoder_bank = nn.Module()

        class Tokenizer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_ns_tokens = 1

        config = SimpleNamespace(
            model=SimpleNamespace(
                name="mdl_onetrans",
                embedding_dim=4,
                token_dim=4,
                init_std=0.02,
                num_layers=0,
            )
        )
        with patch(
            "src.model.FeatureEncoderBank",
            return_value=encoder_bank,
        ) as encoder_constructor, patch(
            "src.model.OneTransTokenizer",
            return_value=Tokenizer(),
        ), patch(
            "src.model.resolve_onetrans_max_position_embeddings",
            return_value=8,
        ):
            backbone = OneTransBackbone(config, {})

        self.assertIs(backbone.encoder_bank, encoder_bank)
        self.assertFalse(
            encoder_constructor.call_args.kwargs["build_sequence_summaries"]
        )

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
                ns_tokens: Tensor,
                s_tokens: Tensor,
                s_mask: Tensor,
                scenario_tokens: Tensor,
                task_tokens: Tensor,
                scenario_mask: Tensor,
            ) -> tuple[Tensor, Tensor]:
                del s_tokens, s_mask, scenario_mask
                self.seen.append(float(ns_tokens.mean().item()))
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
