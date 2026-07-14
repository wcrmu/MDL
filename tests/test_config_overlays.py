from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from src.config import (
    ResolvedIdentityEncoding,
    load_app_config,
    resolve_categorical_base_input,
    validate_app_config,
)
from src.features import load_vocab_maps, plan_vocab_fit


class ModelConfigOverlayTest(unittest.TestCase):
    def test_all_model_profiles_extend_and_validate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = {
            "default.yaml": "mdl_rankmixer",
            "rankmixer.yaml": "rankmixer",
            "mdl_rankmixer.yaml": "mdl_rankmixer",
            "onetrans.yaml": "onetrans",
            "mdl_onetrans.yaml": "mdl_onetrans",
            "longer.yaml": "longer",
            "rankmixer_paper.yaml": "rankmixer",
            "mdl_rankmixer_paper.yaml": "mdl_rankmixer",
            "onetrans_paper.yaml": "onetrans",
            "longer_paper.yaml": "longer",
            "rankmixer_perf.yaml": "rankmixer",
            "mdl_perf.yaml": "mdl_rankmixer",
            "onetrans_perf.yaml": "onetrans",
            "longer_perf.yaml": "longer",
            "longer_5000_perf.yaml": "longer",
        }
        for filename, model_name in expected.items():
            with self.subTest(filename=filename):
                config = load_app_config(root / "configs" / filename)
                self.assertEqual(config.model.name, model_name)

        experimental = load_app_config(root / "configs" / "mdl_onetrans.yaml")
        self.assertTrue(experimental.model.experimental_model_acknowledged)
        self.assertEqual(experimental.sequences[0].encoder, "raw")
        for token in [
            *experimental.tokenization.scenario_tokens,
            *experimental.tokenization.task_tokens,
        ]:
            self.assertNotIn("hist", token.resolved_inputs())

        rankmixer = load_app_config(root / "configs" / "rankmixer_paper.yaml")
        self.assertEqual(rankmixer.model.token_dim, 768)
        self.assertEqual(rankmixer.training.lr_dense, 0.01)
        self.assertEqual(rankmixer.tokenization.feature_tokenizer, "rankmixer")
        self.assertEqual(rankmixer.resolved.encoded_input_dims["hist"], 33_792)

        mdl = load_app_config(root / "configs" / "mdl_rankmixer_paper.yaml")
        self.assertEqual(len(mdl.scenarios.names), 3)
        self.assertEqual(len(mdl.task_names), 3)
        self.assertEqual(mdl.model.mdl_feature_interaction, "paper")
        self.assertEqual(mdl.training.loss_reduction, "sum")
        scenario_priors = {
            token.name: tuple(token.prior_inputs)
            for token in mdl.resolved.tokenization.scenario_token_specs
            if token.name != "global"
        }
        self.assertEqual(
            scenario_priors,
            {
                "single_column": ("scenario_single_column_history",),
                "double_column": ("scenario_double_column_history",),
                "inner_search": ("scenario_inner_search_history",),
            },
        )
        task_priors = {
            token.name: tuple(token.prior_inputs)
            for token in mdl.resolved.tokenization.task_token_specs
        }
        self.assertEqual(
            task_priors,
            {
                "click": ("task_click_history",),
                "like": ("task_like_history",),
                "favorite": ("task_favorite_history",),
            },
        )
        sequence_scopes = {
            sequence.name: sequence.embedding_scope for sequence in mdl.sequences
        }
        for prior_names in scenario_priors.values():
            self.assertEqual(sequence_scopes[prior_names[0]], "scenario")
        for prior_names in task_priors.values():
            self.assertEqual(sequence_scopes[prior_names[0]], "task")
        domain_prior_names = [
            prior_names[0]
            for prior_names in [*scenario_priors.values(), *task_priors.values()]
        ]
        domain_prior_sources: set[str] = set()
        for prior_name in domain_prior_names:
            categorical_input = mdl.resolved.categorical_input_by_name[
                f"{prior_name}.item_id"
            ]
            domain_prior_sources.add(categorical_input.source)
            self.assertEqual(categorical_input.encoding.encoding, "shared_vocab")
            self.assertFalse(categorical_input.encoding.share_embedding)
        self.assertEqual(len(domain_prior_sources), len(domain_prior_names))

        onetrans = load_app_config(root / "configs" / "onetrans_paper.yaml")
        self.assertEqual(onetrans.model.sequence_fusion, "timestamp_aware")
        self.assertEqual(onetrans.model.num_layers, 6)
        self.assertEqual(onetrans.model.token_dim, 256)
        self.assertEqual(onetrans.model.final_s_tokens, 12)
        self.assertEqual(onetrans.sequences[0].encoder, "raw")

        longer = load_app_config(root / "configs" / "longer_paper.yaml")
        self.assertEqual(longer.sequences[0].sequence_order, "newest_to_oldest")
        self.assertEqual(longer.sequences[0].longer_token_merge, 8)
        self.assertEqual(longer.sequences[0].longer_query_tokens, 100)
        self.assertEqual(longer.resolved.encoded_input_dims["hist"], 26_368)

    def test_performance_profiles_use_direct_bounded_ids(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for filename in (
            "rankmixer_perf.yaml",
            "mdl_perf.yaml",
            "onetrans_perf.yaml",
            "longer_perf.yaml",
            "longer_5000_perf.yaml",
        ):
            with self.subTest(filename=filename):
                config = load_app_config(root / "configs" / filename)
                self.assertEqual(config.training.embedding_distribution, "sharded")
                for categorical_input in config.resolved.categorical_inputs:
                    self.assertIsInstance(
                        categorical_input.encoding,
                        ResolvedIdentityEncoding,
                    )
                    base = resolve_categorical_base_input(
                        config.resolved.categorical_input_by_name,
                        categorical_input.name,
                    )
                    self.assertIsInstance(base.encoding, ResolvedIdentityEncoding)
                    self.assertGreater(categorical_input.encoding.num_buckets, 0)
                self.assertEqual(plan_vocab_fit(config).entries, ())
                self.assertEqual(load_vocab_maps(config), {})

        stress = load_app_config(root / "configs" / "longer_5000_perf.yaml")
        self.assertEqual(stress.sequences[0].max_length, 5000)

    def test_onetrans_rejects_pre_encoded_behavior_sequences(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "onetrans.yaml")
        invalid = replace(
            config,
            sequences=[replace(config.sequences[0], encoder="mean_pool")],
        )

        with self.assertRaisesRegex(ValueError, "requires encoder=raw"):
            validate_app_config(invalid)

    def test_onetrans_rejects_position_capacity_below_unified_token_count(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "onetrans.yaml")
        invalid = replace(
            config,
            model=replace(config.model, max_position_embeddings=103),
        )

        with self.assertRaisesRegex(ValueError, r"\[S; NS\] token maximum"):
            validate_app_config(invalid)

    def test_dynamic_onetrans_length_requires_explicit_position_capacity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "onetrans.yaml")
        invalid = replace(
            config,
            sequences=[replace(config.sequences[0], max_length=None)],
            model=replace(config.model, max_position_embeddings=None),
        )

        with self.assertRaisesRegex(ValueError, "requires model.max_position_embeddings"):
            validate_app_config(invalid)

    def test_raw_sequence_mode_is_reserved_for_onetrans(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "default.yaml")
        invalid = replace(
            config,
            sequences=[
                replace(
                    config.sequences[0],
                    encoder="raw",
                    rankmixer_summary_tokens=1,
                    target_inputs=[],
                    longer_user_global_inputs=[],
                    longer_user_global_tokens=0,
                    longer_cls_tokens=0,
                    longer_candidate_global_tokens=None,
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "only valid"):
            validate_app_config(invalid)

    def test_mdl_onetrans_rejects_s_sequence_domain_priors(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "mdl_onetrans.yaml")
        scenario_tokens = list(config.tokenization.scenario_tokens)
        scenario_tokens[0] = replace(scenario_tokens[0], prior_inputs=["hist"])
        invalid = replace(
            config,
            tokenization=replace(
                config.tokenization,
                scenario_tokens=scenario_tokens,
            ),
        )

        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_app_config(invalid)

    def test_multi_domain_mdl_rejects_generic_or_reused_priors(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "mdl_rankmixer_paper.yaml")

        generic_scenarios = [
            replace(token, prior_inputs=["hist"])
            if token.name != "global"
            else token
            for token in config.tokenization.scenario_tokens
        ]
        invalid_generic = replace(
            config,
            tokenization=replace(
                config.tokenization,
                scenario_tokens=generic_scenarios,
            ),
        )
        with self.assertRaisesRegex(ValueError, "generic shared history"):
            validate_app_config(invalid_generic)

        for section, shared_prior in (
            ("scenario_tokens", "scenario_single_column_history"),
            ("task_tokens", "task_click_history"),
        ):
            with self.subTest(section=section):
                tokens = [
                    replace(token, prior_inputs=[shared_prior])
                    if token.name != "global"
                    else token
                    for token in getattr(config.tokenization, section)
                ]
                invalid_reuse = replace(
                    config,
                    tokenization=replace(
                        config.tokenization,
                        **{section: tokens},
                    ),
                )
                with self.assertRaisesRegex(ValueError, "not reused by another token"):
                    validate_app_config(invalid_reuse)

    def test_sparse_dtsi_requires_explicit_unpublished_output_policy(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "rankmixer.yaml")
        sparse_model = replace(
            config.model,
            rankmixer_ffn_type="sparse_moe",
            sparse_moe_use_dtsi=True,
            sparse_moe_dtsi_training_output=None,
        )

        with self.assertRaisesRegex(ValueError, "does not publish"):
            sparse_model.validate()

    def test_mdl_ablation_switches_are_valid_runtime_paths(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "mdl_rankmixer_paper.yaml")

        for switch in (
            "use_task_tokens",
            "use_scenario_tokens",
            "use_task_feature_interaction",
            "use_scenario_feature_interaction",
            "use_global_scenario_token",
        ):
            with self.subTest(switch=switch):
                ablation = replace(
                    config,
                    model=replace(config.model, **{switch: False}),
                )
                validate_app_config(ablation)


if __name__ == "__main__":
    unittest.main()
