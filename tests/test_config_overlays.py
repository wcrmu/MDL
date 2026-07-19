from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from src.config import (
    ModelConfig,
    ResolvedIdentityEncoding,
    ResolvedPreHashedEncoding,
    RuntimeConfig,
    ScenarioConfig,
    TokenGroupConfig,
    load_app_config,
    resolve_categorical_base_input,
    validate_app_config,
)
from src.features import load_vocab_maps, plan_vocab_fit


class ModelConfigOverlayTest(unittest.TestCase):
    def test_early_adapter_truncation_must_match_sequence_semantics(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "mdl_rankmixer.yaml")
        adapter = config.data.train.adapter
        self.assertIsNotNone(adapter)
        assert adapter is not None
        limits = dict(adapter.options["sequence_max_lengths"])
        limits["impr"] += 1
        bad_adapter = replace(
            adapter,
            options={**adapter.options, "sequence_max_lengths": limits},
        )
        bad_train = replace(config.data.train, adapter=bad_adapter)
        bad_config = replace(config, data=replace(config.data, train=bad_train))

        with self.assertRaisesRegex(ValueError, "must equal sequences.impr.max_length"):
            bad_config.validate()

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
                config = load_app_config(root / "configs" / "reference" / filename)
                self.assertEqual(config.model.name, model_name)

        experimental = load_app_config(
            root / "configs" / "reference" / "mdl_onetrans.yaml"
        )
        self.assertTrue(experimental.model.experimental_model_acknowledged)
        self.assertEqual(experimental.model.first_domain_sequence_layer, 0)

        production = load_app_config(
            root / "configs" / "mdl_onetrans.yaml"
        )
        self.assertTrue(production.model.experimental_model_acknowledged)
        self.assertEqual(production.model.first_domain_sequence_layer, 4)
        self.assertTrue(
            all(sequence.encoder == "raw" for sequence in production.sequences)
        )
        history_names = {sequence.name for sequence in production.sequences}
        for token in [
            *production.tokenization.scenario_tokens,
            *production.tokenization.task_tokens,
        ]:
            self.assertFalse(history_names & set(token.resolved_inputs()))

        rankmixer = load_app_config(
            root / "configs" / "reference" / "rankmixer_paper.yaml"
        )
        self.assertEqual(rankmixer.model.token_dim, 768)
        self.assertEqual(rankmixer.training.lr_dense, 0.01)
        self.assertEqual(rankmixer.tokenization.feature_tokenizer, "rankmixer")
        self.assertEqual(rankmixer.resolved.encoded_input_dims["hist"], 33_792)

        mdl = load_app_config(
            root / "configs" / "reference" / "mdl_rankmixer_paper.yaml"
        )
        self.assertEqual(len(mdl.scenarios.names), 3)
        self.assertEqual(len(mdl.task_names), 3)
        self.assertEqual(mdl.model.mdl_feature_interaction, "direct_ffn")
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

        onetrans = load_app_config(
            root / "configs" / "reference" / "onetrans_paper.yaml"
        )
        self.assertEqual(onetrans.model.sequence_fusion, "timestamp_aware")
        self.assertEqual(onetrans.model.num_layers, 6)
        self.assertEqual(onetrans.model.token_dim, 256)
        self.assertEqual(onetrans.model.final_s_tokens, 12)
        self.assertEqual(onetrans.sequences[0].encoder, "raw")

        longer = load_app_config(
            root / "configs" / "reference" / "longer_paper.yaml"
        )
        self.assertEqual(longer.sequences[0].sequence_order, "newest_to_oldest")
        self.assertEqual(longer.sequences[0].longer_token_merge, 8)
        self.assertEqual(longer.sequences[0].longer_query_tokens, 100)
        self.assertEqual(longer.resolved.encoded_input_dims["hist"], 26_368)

    def test_legacy_feature_interaction_names_are_normalized(self) -> None:
        expected_by_legacy_name = {
            "paper": "direct_ffn",
            "rankmixer_full": "residual_ffn",
        }
        for legacy_name, expected in expected_by_legacy_name.items():
            with self.subTest(legacy_name=legacy_name):
                config = ModelConfig.from_mapping(
                    {
                        "name": "mdl_rankmixer",
                        "mdl_feature_interaction": legacy_name,
                    }
                )

                self.assertEqual(config.mdl_feature_interaction, expected)
                config.validate()

    def test_domain_sequence_attention_layer_is_bounded_and_mdl_onetrans_only(self) -> None:
        for config, expected_error in (
            (
                ModelConfig(
                    name="mdl_onetrans",
                    num_layers=2,
                    first_domain_sequence_layer=2,
                    experimental_model_acknowledged=True,
                ),
                r"first_domain_sequence_layer must be in",
            ),
            (
                ModelConfig(
                    name="onetrans",
                    first_domain_sequence_layer=0,
                ),
                r"only valid for mdl_onetrans",
            ),
        ):
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, expected_error):
                    config.validate()

    def test_yaml_types_are_validated_before_config_coercion(self) -> None:
        root = Path(__file__).resolve().parents[1]
        invalid_overrides = (
            (
                "quoted runtime boolean",
                'runtime:\n  compile: "false"\n',
                r"runtime\.compile must be a boolean",
            ),
            (
                "scalar scenario names",
                "scenarios:\n  names: default\n  source: scenario_id\n",
                r"scenarios\.names must be a list",
            ),
            (
                "scalar token input list",
                "tokenization:\n  feature_token_inputs: user_id\n",
                r"tokenization\.feature_token_inputs must be a list",
            ),
            (
                "boolean integer field",
                "training:\n  batch_size: true\n",
                r"training\.batch_size must be an integer",
            ),
            (
                "quoted model boolean",
                'model:\n  use_pyramid: "false"\n',
                r"model\.use_pyramid must be a boolean",
            ),
        )

        with tempfile.TemporaryDirectory() as temporary:
            for index, (name, override, expected_error) in enumerate(invalid_overrides):
                with self.subTest(name=name):
                    config_path = Path(temporary) / f"invalid_{index}.yaml"
                    config_path.write_text(
                        f'extends: "{(root / "configs" / "reference" / "default.yaml").as_posix()}"\n'
                        + override,
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, expected_error):
                        load_app_config(config_path)

    def test_quick_eval_requires_labels_on_the_selected_split(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        self.assertTrue(config.training.quick_eval.enabled)
        self.assertEqual(config.training.quick_eval.split, "train")
        quick_eval = replace(
            config.training.quick_eval,
            enabled=True,
            split="test",
        )
        configured = replace(
            config,
            training=replace(config.training, quick_eval=quick_eval),
        )

        with self.assertRaisesRegex(
            ValueError,
            r"data\.test\.labels must declare the training tasks",
        ):
            validate_app_config(configured)

    def test_strict_type_validation_preserves_documented_compatibility(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "compatible.yaml"
            config_path.write_text(
                f'extends: "{(root / "configs" / "reference" / "default.yaml").as_posix()}"\n'
                "runtime:\n"
                "  activation_checkpoint: false\n"
                "data:\n"
                "  train:\n"
                "    inputs: /tmp/train.parquet\n",
                encoding="utf-8",
            )

            config = load_app_config(config_path)

        self.assertEqual(config.runtime.activation_checkpoint, "none")
        self.assertEqual(config.data.train.inputs, ("/tmp/train.parquet",))

    def test_dataclass_validation_rejects_programmatic_type_mismatches(self) -> None:
        invalid_configs = (
            (
                ScenarioConfig(names="default", source="scenario_id"),
                r"scenarios\.names must be a list",
            ),
            (
                ScenarioConfig(
                    names=("default",),
                    source="scenario_id",
                    auto_discover=True,
                ),
                r"scenarios\.names must be \[__auto__\]",
            ),
            (
                RuntimeConfig(compile="false"),
                r"runtime\.compile must be a boolean",
            ),
            (
                RuntimeConfig(compile_mode="fast"),
                r"runtime\.compile_mode must be default or reduce-overhead",
            ),
            (
                RuntimeConfig(allow_tf32="false"),
                r"runtime\.allow_tf32 must be a boolean",
            ),
            (
                RuntimeConfig(require_compact_sequence_batches="false"),
                r"runtime\.require_compact_sequence_batches must be a boolean",
            ),
            (
                RuntimeConfig(master_port=True),
                r"runtime\.master_port must be an integer",
            ),
        )
        for config, expected_error in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, expected_error):
                    config.validate()

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
                config = load_app_config(root / "configs" / "reference" / filename)
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

        stress = load_app_config(
            root / "configs" / "reference" / "longer_5000_perf.yaml"
        )
        self.assertEqual(stress.sequences[0].max_length, 5000)

    def test_pre_hashed_strategy_resolves_and_validates_shared_bucket(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            overlay = Path(directory) / "pre_hashed.yaml"
            overlay.write_text(
                "\n".join(
                    [
                        f"extends: {root / 'configs' / 'reference' / 'default.yaml'}",
                        "vocab_strategy:",
                        "  features:",
                        "    shop_id:",
                        "      encoding: pre_hashed",
                        "      source: shop_id",
                        "      num_buckets: 1024",
                        "      salt: null",
                        "    hist.shop_id:",
                        "      encoding: pre_hashed",
                        "      source: hist_shop_id",
                        "      num_buckets: 1024",
                        "      salt: null",
                        "      share_with: shop_id",
                        "      share_embedding: true",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_app_config(overlay)

            root_input = config.resolved.categorical_input_by_name["shop_id"]
            alias_input = config.resolved.categorical_input_by_name["hist.shop_id"]
            self.assertIsInstance(root_input.encoding, ResolvedPreHashedEncoding)
            self.assertIsInstance(alias_input.encoding, ResolvedPreHashedEncoding)
            self.assertEqual(alias_input.encoding.share_with, "shop_id")
            self.assertTrue(alias_input.encoding.share_embedding)
            fitted_names = {entry.feature_name for entry in plan_vocab_fit(config).entries}
            self.assertNotIn("shop_id", fitted_names)
            self.assertNotIn("hist.shop_id", fitted_names)

            mismatched = overlay.read_text(encoding="utf-8").replace(
                "      num_buckets: 1024\n      salt: null\n      share_with: shop_id",
                "      num_buckets: 2048\n      salt: null\n      share_with: shop_id",
            )
            overlay.write_text(mismatched, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match embedding base"):
                load_app_config(overlay)

    def test_gpu_performance_profiles_keep_validated_occupancy_settings(self) -> None:
        root = Path(__file__).resolve().parents[1]
        reference = root / "configs" / "reference"
        mdl = load_app_config(reference / "mdl_perf.yaml")
        onetrans = load_app_config(reference / "onetrans_perf.yaml")
        longer = load_app_config(reference / "longer_perf.yaml")
        longer_stress = load_app_config(reference / "longer_5000_perf.yaml")

        self.assertEqual(mdl.training.batch_size, 4096)
        self.assertTrue(mdl.runtime.compile)
        self.assertEqual(mdl.runtime.compile_mode, "reduce-overhead")
        self.assertEqual(onetrans.training.batch_size, 320)
        self.assertEqual(longer.training.batch_size, 512)
        self.assertTrue(onetrans.runtime.compile)
        self.assertTrue(longer.runtime.compile)
        self.assertEqual(onetrans.runtime.compile_mode, "default")
        self.assertEqual(longer.runtime.compile_mode, "default")
        self.assertTrue(onetrans.runtime.require_compact_sequence_batches)
        self.assertFalse(longer.runtime.require_compact_sequence_batches)
        self.assertEqual(
            [bucket.batch_size for bucket in onetrans.data.train.reader.length_buckets],
            [2048, 1024, 512, 384, 320, 128],
        )
        self.assertEqual(
            [bucket.batch_size for bucket in longer.data.train.reader.length_buckets],
            [3072, 1536, 768, 512, 512, 192],
        )
        for config in (mdl, onetrans, longer):
            with self.subTest(model=config.model.name):
                self.assertTrue(config.training.ddp.static_graph)
                self.assertFalse(config.training.ddp.find_unused_parameters)
                self.assertTrue(config.training.ddp.validated_static_graph)
        self.assertEqual(longer.runtime.activation_checkpoint, "none")
        self.assertEqual(longer_stress.runtime.activation_checkpoint, "selective")

    def test_onetrans_rejects_pre_encoded_behavior_sequences(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(
            root / "configs" / "reference" / "onetrans.yaml"
        )
        invalid = replace(
            config,
            sequences=[
                replace(config.sequences[0], encoder="mean_pool"),
                *config.sequences[1:],
            ],
        )

        with self.assertRaisesRegex(ValueError, "requires encoder=raw"):
            validate_app_config(invalid)

    def test_onetrans_rejects_position_capacity_below_unified_token_count(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(
            root / "configs" / "reference" / "onetrans.yaml"
        )
        invalid = replace(
            config,
            model=replace(
                config.model,
                max_position_embeddings=config.model.max_position_embeddings - 1,
            ),
        )

        with self.assertRaisesRegex(ValueError, r"\[S; NS\] token maximum"):
            validate_app_config(invalid)

    def test_onetrans_ns_tokens_reject_sequences_in_inactive_scopes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(
            root / "configs" / "reference" / "onetrans.yaml"
        )
        hidden_sequence = replace(
            config.sequences[0],
            name="hidden_seq",
            embedding_scope="scenario",
        )
        bad_ns_token = TokenGroupConfig(name="bad_ns", inputs=["hidden_seq"])
        invalid = replace(
            config,
            sequences=[*config.sequences, hidden_sequence],
            tokenization=replace(config.tokenization, ns_tokens=[bad_ns_token]),
        )

        with self.assertRaisesRegex(
            ValueError,
            r"tokenization\.ns_tokens\.bad_ns must not include sequence inputs",
        ):
            validate_app_config(invalid)

    def test_onetrans_sequence_tokens_only_accept_active_sequences(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(
            root / "configs" / "reference" / "onetrans.yaml"
        )
        hidden_sequence = replace(
            config.sequences[0],
            name="hidden_seq",
            embedding_scope="task",
        )

        for invalid_input in (config.features[0].name, "hidden_seq"):
            with self.subTest(invalid_input=invalid_input):
                bad_sequence_token = replace(
                    config.tokenization.sequence_tokens[0],
                    name="bad_sequence",
                    inputs=[config.sequences[0].name, invalid_input],
                )
                invalid = replace(
                    config,
                    sequences=[*config.sequences, hidden_sequence],
                    tokenization=replace(
                        config.tokenization,
                        sequence_tokens=[bad_sequence_token],
                    ),
                )

                with self.assertRaisesRegex(
                    ValueError,
                    r"tokenization\.sequence_tokens\.bad_sequence must only include "
                    r"feature/shared sequence inputs",
                ):
                    validate_app_config(invalid)

    def test_dynamic_onetrans_length_requires_explicit_position_capacity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(
            root / "configs" / "reference" / "onetrans.yaml"
        )
        invalid = replace(
            config,
            sequences=[
                replace(config.sequences[0], max_length=None),
                *config.sequences[1:],
            ],
            model=replace(config.model, max_position_embeddings=None),
        )

        with self.assertRaisesRegex(ValueError, "requires model.max_position_embeddings"):
            validate_app_config(invalid)

    def test_raw_sequence_mode_is_reserved_for_onetrans(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
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
        config = load_app_config(
            root / "configs" / "reference" / "mdl_onetrans.yaml"
        )
        history_name = config.sequences[0].name
        scenario_tokens = list(config.tokenization.scenario_tokens)
        scenario_tokens[0] = replace(
            scenario_tokens[0], prior_inputs=[history_name]
        )
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
        config = load_app_config(
            root / "configs" / "reference" / "mdl_rankmixer_paper.yaml"
        )

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
        config = load_app_config(
            root / "configs" / "reference" / "rankmixer.yaml"
        )
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
        config = load_app_config(
            root / "configs" / "reference" / "mdl_rankmixer_paper.yaml"
        )

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
