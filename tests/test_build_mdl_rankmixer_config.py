from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import yaml

from scripts.build_mdl_rankmixer_config import (
    CONTEXT_SCALAR_FIELDS,
    EXPECTED_LABELS,
    EXPECTED_UPS_TYPES,
    ITEM_BAG_FIELDS,
    ONETRANS_SEQUENCE_LENGTH_CAPS,
    apply_embedding_profile,
    build_config,
    build_name_estimate_report,
    _find_sequence_field,
    _resolve_share_root,
    _categorical_entries_by_name,
    render_config,
)
from scripts.profile_prehashed_parquet import profile_spec_from_mapping
from src.config import ResolvedPreHashedEncoding, load_app_config
from src.dataloader import (
    _load_parquet_adapter,
    _scenario_tensor,
    discover_scenario_values,
    resolve_auto_scenarios,
)
from src.model import build_model
from src.embeddings import ShardedEmbedding


ROOT = Path(__file__).resolve().parents[1]


def _compact_production_config(model_name: str):
    """Keep production wiring while making a CPU forward/backward test cheap."""

    config = load_app_config(
        ROOT / "configs" / f"{model_name}.yaml"
    )
    config = resolve_auto_scenarios(config, [9, 17])
    sequences = tuple(
        replace(
            sequence,
            max_length=2,
            longer_query_tokens=(
                min(sequence.longer_query_tokens, 2)
                if sequence.encoder == "longer"
                else sequence.longer_query_tokens
            ),
        )
        for sequence in config.sequences
    )

    def compact_split(split):
        if split is None or split.adapter is None:
            return split
        limits = {
            name: 2
            for name in split.adapter.options.get("sequence_max_lengths", {})
        }
        adapter = replace(
            split.adapter,
            options={**split.adapter.options, "sequence_max_lengths": limits},
        )
        return replace(split, adapter=adapter)

    onetrans = model_name in {"onetrans", "mdl_onetrans"}
    config = replace(
        config,
        data=replace(
            config.data,
            train=compact_split(config.data.train),
            test=compact_split(config.data.test),
        ),
        sequences=sequences,
        model=replace(
            config.model,
            token_dim=32,
            num_layers=2 if onetrans else 1,
            num_heads=4,
            hidden_dim=64,
            task_head_hidden_dim=64,
            pyramid_round_to=1,
            final_s_tokens=2 if onetrans else config.model.final_s_tokens,
            max_position_embeddings=(64 if onetrans else None),
            first_domain_sequence_layer=(1 if model_name == "mdl_onetrans" else None),
        ),
        runtime=replace(
            config.runtime,
            device="cpu",
            precision="fp32",
            compile=False,
            activation_checkpoint="none",
            attention_backend="auto",
            distributed="none",
            nproc_per_node=None,
        ),
        training=replace(
            config.training,
            batch_size=2,
            embedding_distribution="replicated",
            embedding_weight_dtype="fp32",
            # Toy CPU forwards use replicated tables; Row-Wise is sharded-only.
            sparse_optimizer="adagrad",
        ),
    )
    config.validate()
    return config


def _synthetic_model_features(config, batch_size: int = 2) -> dict[str, object]:
    """Build every configured scalar, bag, and aligned sequence input."""

    result: dict[str, object] = {}
    lengths = torch.tensor(
        [2 if index % 2 == 0 else 1 for index in range(batch_size)],
        dtype=torch.long,
    )
    for feature in config.features:
        if feature.kind == "dense":
            result[feature.name] = torch.randn(batch_size, feature.dimension)
            continue
        if feature.pooling == "mean":
            values = torch.randint(1, 15, (batch_size, 2))
            values[lengths == 1, 1] = 0
            result[feature.name] = {
                "values": values,
                "lengths": lengths.clone(),
            }
            continue
        result[feature.name] = torch.randint(1, 15, (batch_size,))

    for sequence in config.sequences:
        fields: dict[str, torch.Tensor] = {}
        for field in sequence.fields:
            shape = (
                (batch_size, 2)
                if field.dimension == 1
                else (batch_size, 2, field.dimension)
            )
            value = (
                torch.randint(1, 15, shape)
                if field.kind == "categorical"
                else torch.randn(shape)
            )
            value[lengths == 1, 1] = 0
            fields[field.name] = value
        result[sequence.name] = {
            "fields": fields,
            "lengths": lengths.clone(),
        }
    return result


def _synthetic_report(sample: dict) -> dict:
    spec = profile_spec_from_mapping(sample)
    fields = {}
    for source in spec.all_sources:
        fields[source] = {
            "leaf_count": 100,
            "invalid_leaf_count": 0,
            "zero_count": 0,
            "rows_with_empty_list": 0,
            "nulls_by_depth": {},
            "list_lengths_by_depth": {
                "0": {"count": 10, "min": 1, "p50": 2, "p95": 4, "p99": 5, "max": 6},
                "1": {"count": 10, "min": 1, "p50": 2, "p95": 4, "p99": 5, "max": 6},
            },
            "recommended_bucket_size": 1024,
            "suggested_embedding_dim": 8,
        }
    declared_bags = (
        set(spec.context_sources) - CONTEXT_SCALAR_FIELDS
    ) | ITEM_BAG_FIELDS
    for source in (*spec.context_sources, *spec.item_sources):
        if source not in declared_bags:
            fields[source]["list_lengths_by_depth"]["1"] = {
                "count": 10,
                "min": 1,
                "p50": 1,
                "p95": 1,
                "p99": 1,
                "max": 1,
            }
    sequence_sources = {
        source
        for sources in spec.sequence_sources.values()
        for source in sources
    }
    for source in sequence_sources:
        fields[source]["list_lengths_by_depth"]["1"] = {
            "count": 10,
            "min": 1,
            "p50": 1,
            "p95": 1,
            "p99": 1,
            "max": 1,
        }
    shared = {
        root: {
            "sources": list(sources),
            "recommended_bucket_size": 2048,
            "suggested_embedding_dim": 16,
        }
        for root, sources in spec.shared_groups.items()
    }
    sequence_lengths = {
        name: {"count": 10, "min": 0, "p50": 4, "p95": 8, "p99": 10, "max": 12}
        for name in spec.sequence_sources
    }
    return {
        "format_version": 4,
        "rows_scanned": 10,
        "missing_configured_columns_by_input": {"synthetic.parquet": []},
        "fields": fields,
        "shared_embedding_groups": shared,
        "contract": {
            "agg_rows": 10,
            "req_rows": 0,
            "partial_indices_rows": 0,
            "context_outer_mismatches": {},
            "item_outer_mismatches": {},
            "label_length_mismatches": {},
            "invalid_labels": {},
            "sequence_length_mismatches": {},
            "invalid_sequence_membership": {},
            "time_order_violations": {},
            "sequence_lengths_after_request_filter": sequence_lengths,
            "sku_alignment_mismatches": 0,
            "scene_values": [
                {"scene_id": 7, "count": 6},
                {"scene_id": 19, "count": 4},
            ],
        },
    }


class BuildMDLRankMixerConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample = yaml.safe_load((ROOT / "sample.yaml").read_text(encoding="utf-8"))

    def test_builds_valid_report_driven_production_config(self) -> None:
        report = _synthetic_report(self.sample)
        payload, summary = build_config(
            self.sample,
            report,
            train_inputs=["/tmp/train"],
            test_inputs=["/tmp/test"],
        )

        self.assertEqual(payload["runtime"]["nproc_per_node"], 8)
        self.assertEqual(payload["training"]["sparse_optimizer"], "adagrad")
        memory = summary["embedding_memory"]
        self.assertIn("optimizer_state_gib_total", memory)
        self.assertEqual(memory["optimizer_state_layout"], "full")
        self.assertEqual(memory["gpu_count"], 8)

        self.assertEqual([item["name"] for item in payload["features"][:169]], [
            item["name"] for item in self.sample["features"]
        ])
        by_name = {item["name"]: item for item in payload["features"]}
        self.assertEqual(by_name["goods_name_bigram_hn"]["pooling"], "mean")
        self.assertEqual(
            by_name["sku_id_hn"]["pooling_null_policy"],
            "include_as_padding",
        )
        self.assertNotIn("pooling", by_name["sku_spec_vids_hn"])
        self.assertEqual(summary["bag_feature_count"], 50)

        main_sequences = payload["sequences"][:9]
        self.assertEqual([item["name"] for item in main_sequences], [
            item["name"] for item in self.sample["sequences"]
        ])
        for sequence in main_sequences:
            self.assertEqual(sequence["encoder"], "longer")
            self.assertEqual(sequence["longer_output"], "summary")
            self.assertEqual(sequence["longer_token_merge"], 1)
            self.assertEqual(sequence["rankmixer_summary_tokens"], 1)
            self.assertEqual(sequence["target_inputs"], [])
            self.assertEqual(sequence["max_length"], 10)
            self.assertEqual(sequence["sequence_order"], "newest_to_oldest")
            self.assertEqual(sequence["truncation"], "head")
            self.assertEqual(
                sequence["fields"][0]["name"],
                "time_delta_log1p_seconds",
            )
            self.assertEqual(sequence["fields"][0]["kind"], "dense")

        task_priors = {item["name"]: item for item in payload["sequences"][9:]}
        self.assertEqual(task_priors["task_fst_cart_prior"]["encoder"], "mean_pool")
        self.assertTrue(
            any(
                field["source"] == "cart_long_x_goods_id_hn"
                for field in task_priors["task_fst_cart_prior"]["fields"]
            )
        )
        upid_goods = next(
            field
            for field in task_priors["task_upid_pay_prior"]["fields"]
            if field["name"] == "goods_id_hn"
        )
        cateid_goods = next(
            field
            for field in task_priors["task_cateid_filter_prior"]["fields"]
            if field["name"] == "goods_id_hn"
        )
        self.assertNotIn("share_with", upid_goods["encoding"])
        self.assertNotIn("share_with", cateid_goods["encoding"])

        self.assertEqual(
            payload["scenarios"],
            {
                "names": ["7", "19"],
                "source": "scene_id",
                "source_encoding": "index",
            },
        )
        adapter_options = payload["data"]["train"]["adapter"]["options"]
        self.assertEqual(
            payload["data"]["train"]["adapter"]["callable"],
            "src.dataloader:adapt_mdl_rankmixer_parquet",
        )
        adapter_payload = payload["data"]["train"]["adapter"]
        self.assertEqual(len(adapter_payload["input_columns"]), 281)
        self.assertEqual(len(adapter_payload["optional_input_columns"]), 12)
        self.assertEqual(
            len(payload["data"]["test"]["adapter"]["optional_input_columns"]),
            13,
        )
        self.assertIn("impr_x_time", adapter_payload["input_columns"])
        self.assertNotIn("impr_x_indices", adapter_payload["input_columns"])
        self.assertIn("impr_x_indices", adapter_payload["optional_input_columns"])
        self.assertIn(
            "f_goods_view_times_tg_l1_hn",
            adapter_payload["optional_input_columns"],
        )
        self.assertEqual(adapter_options["request_value_maps"]["scene_id"], {7: 0, 19: 1})
        self.assertEqual(len(adapter_options["context_features"]), 47)
        self.assertEqual(len(adapter_options["item_features"]), 122)
        self.assertEqual(
            adapter_options["labels"]["cateid_filter"],
            "cateid_is_fst_scene_sp_filter",
        )
        self.assertEqual(
            payload["data"]["test"]["prediction_keys"]["candidate_position"],
            "candidate_position",
        )
        self.assertNotIn("prediction_keys", payload["data"]["train"])
        self.assertEqual(payload["data"]["test"]["prediction_score_suffix"], "_score")
        self.assertNotIn("label_missing_values", adapter_options)
        self.assertNotIn("label_masks", adapter_options)
        self.assertFalse(payload["data"]["train"].get("label_masks"))
        self.assertEqual(
            adapter_options["sequence_max_lengths"],
            {sequence["name"]: sequence["max_length"] for sequence in main_sequences},
        )
        self.assertNotIn("column_aliases", adapter_options)
        self.assertEqual(adapter_options["time_delta_transform"], "log1p_seconds")

        main_input_width = sum(
            int(by_name[name]["embedding_dim"])
            for name in payload["tokenization"]["feature_token_inputs"][:169]
        ) + 9 * 768
        self.assertEqual(main_input_width % 32, 0)
        self.assertEqual(payload["runtime"]["nproc_per_node"], 8)
        self.assertEqual(payload["training"]["embedding_distribution"], "sharded")
        self.assertEqual(payload["training"]["loss_reduction"], "mean_per_task")
        self.assertEqual(
            payload["training"]["quick_eval"],
            {
                "enabled": True,
                "every_steps": 1000,
                "max_batches": 20,
                "split": "train",
                "auc_bins": 4096,
            },
        )
        self.assertLessEqual(
            summary["embedding_memory"]["planned_weight_plus_state_gib_per_gpu"],
            40.0,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mdl_rankmixer.yaml"
            path.write_text(render_config(payload, summary), encoding="utf-8")
            config = load_app_config(path)
        adapter_name, adapter = _load_parquet_adapter(config.data.train)
        self.assertEqual(adapter_name, "src.dataloader:adapt_mdl_rankmixer_parquet")
        self.assertEqual(adapter.__module__, "src.dataloader")
        goods = config.resolved.categorical_input_by_name["goods_id_hn"]
        history_goods = config.resolved.categorical_input_by_name["impr.goods_id_hn"]
        self.assertIsInstance(goods.encoding, ResolvedPreHashedEncoding)
        self.assertEqual(history_goods.encoding.share_with, "goods_id_hn")
        self.assertTrue(history_goods.encoding.share_embedding)
        self.assertEqual(
            config.resolved.categorical_embedding_dims["goods_id_hn"],
            config.resolved.categorical_embedding_dims["impr.goods_id_hn"],
        )

    def test_builder_cli_options_mutate_payload_and_memory_summary(self) -> None:
        report = _synthetic_report(self.sample)
        payload, summary = build_config(
            self.sample,
            report,
            train_inputs=["/tmp/train"],
            test_inputs=["/tmp/test"],
            gpu_count=2,
            embedding_weight_dtype="bf16",
            sparse_optimizer="rowwise_adagrad",
            embedding_budget_gib_per_gpu=80.0,
        )
        self.assertEqual(payload["runtime"]["nproc_per_node"], 2)
        self.assertEqual(payload["training"]["embedding_weight_dtype"], "bf16")
        self.assertEqual(payload["training"]["sparse_optimizer"], "rowwise_adagrad")
        memory = summary["embedding_memory"]
        self.assertEqual(memory["gpu_count"], 2)
        self.assertEqual(memory["embedding_weight_dtype"], "bf16")
        self.assertEqual(memory["optimizer_state_layout"], "rowwise")
        self.assertIn("optimizer_state_gib_total", memory)
        self.assertNotIn("adagrad_state_gib_total", memory)

    def test_report_rejects_incomplete_or_non_binary_labels(self) -> None:
        report = _synthetic_report(self.sample)
        report["contract"]["label_distribution"] = {
            task: {
                "total": 10,
                "null": 1 if task == "fst_cart" else 0,
                "minus_one": 2 if task == "upid_pay" else 0,
                "zero": 4,
                "one": 3,
                "other": 0,
            }
            for task in EXPECTED_LABELS
        }

        with self.assertRaisesRegex(ValueError, "label_distribution.fst_cart.null"):
            build_config(self.sample, report)

        report = _synthetic_report(self.sample)
        report["contract"]["label_distribution"] = {
            task: {
                "total": 10,
                "null": 0,
                "minus_one": 0,
                "zero": 5,
                "one": 5,
                "other": 0,
            }
            for task in EXPECTED_LABELS
        }
        report["contract"]["label_distribution"]["fst_cart"]["other"] = 1
        with self.assertRaisesRegex(ValueError, "label_distribution.fst_cart.other"):
            build_config(self.sample, report)

    def test_builds_name_estimated_config_with_runtime_scene_discovery(self) -> None:
        report = build_name_estimate_report(self.sample)
        with tempfile.TemporaryDirectory() as directory:
            import pyarrow as pa
            import pyarrow.parquet as pq

            parquet_path = Path(directory) / "scenes.parquet"
            pq.write_table(
                pa.table(
                    {
                        "scene_id": pa.array(
                            [[17, 9], [17]],
                            type=pa.list_(pa.int64()),
                        )
                    }
                ),
                parquet_path,
            )
            payload, summary = build_config(
                self.sample,
                report,
                train_inputs=[str(parquet_path)],
                test_inputs=[str(parquet_path)],
                auto_discover_scenes=True,
            )
            path = Path(directory) / "mdl_rankmixer.yaml"
            path.write_text(render_config(payload, summary), encoding="utf-8")
            config = load_app_config(path)
            discovered = discover_scenario_values(config)

        self.assertTrue(config.scenarios.auto_discover)
        self.assertEqual(config.scenarios.names, ("__auto__",))
        self.assertEqual(config.scenarios.source_encoding, "raw")
        self.assertNotIn(
            "request_value_maps",
            payload["data"]["train"]["adapter"]["options"],
        )
        self.assertIn("scenario_tokens", payload["tokenization"])
        self.assertNotIn("scenario_token_inputs", payload["tokenization"])
        self.assertEqual(discovered, (9, 17))

        resolved = resolve_auto_scenarios(config, [17, 9])
        self.assertFalse(resolved.scenarios.auto_discover)
        self.assertEqual(resolved.scenarios.names, ("9", "17"))
        resolved_feature_names = {feature.name for feature in resolved.features}
        self.assertIn("scenario_9_prior_scene_id_hn", resolved_feature_names)
        self.assertIn("scenario_17_prior_scene_id_hn", resolved_feature_names)
        self.assertNotIn("scenario_prior_scene_id_hn", resolved_feature_names)
        resolved_tokens = {
            token.name: token for token in resolved.tokenization.scenario_tokens
        }
        self.assertIn(
            "scenario_9_prior_scene_id_hn",
            resolved_tokens["9"].prior_inputs,
        )
        self.assertIn(
            "scenario_17_prior_scene_id_hn",
            resolved_tokens["17"].prior_inputs,
        )
        scene_tensor = _scenario_tensor(
            resolved,
            pa.table({"scene_id": [17, 9]}),
            2,
        )
        self.assertEqual(scene_tensor.tolist(), [1, 0])
        with patch(
            "src.dataloader._encode_scenario_item",
            side_effect=AssertionError("trusted scenario path used Python validation"),
        ):
            trusted_scene_tensor = _scenario_tensor(
                resolved,
                pa.table({"scene_id": [17, 9]}),
                2,
                trusted_input=True,
            )
        self.assertEqual(trusted_scene_tensor.tolist(), [1, 0])
        self.assertEqual(resolved.scenarios.source_encoding, "raw")
        with self.assertRaisesRegex(ValueError, "unknown raw scenario id 0"):
            _scenario_tensor(
                resolved,
                pa.table({"scene_id": [0]}),
                1,
            )

        by_name = {item["name"]: item for item in payload["features"]}
        self.assertEqual(
            by_name["goods_id_hn"]["encoding"]["num_buckets"],
            1 << 27,
        )
        self.assertEqual(
            by_name["ups_clkv2_i2i_goods_ids_hit_size"]["encoding"]["num_buckets"],
            1 << 12,
        )
        self.assertEqual(summary["profile"]["settings"]["mode"], "name_heuristic")
        self.assertLess(
            summary["embedding_memory"]["planned_weight_plus_state_gib_per_gpu"],
            40.0,
        )

    def test_scene_discovery_cache_avoids_rescanning_immutable_inputs(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet_path = root / "scenes.parquet"
            cache_path = root / "scene-cache.json"
            pq.write_table(
                pa.table({"scene_id": pa.array([[17, 9], [17]])}),
                parquet_path,
            )
            base = load_app_config(
                ROOT / "configs" / "reference" / "default.yaml"
            )
            config = replace(
                base,
                data=replace(
                    base.data,
                    train=replace(base.data.train, inputs=(str(parquet_path),)),
                ),
                scenarios=replace(
                    base.scenarios,
                    names=("__auto__",),
                    source="scene_id",
                    auto_discover=True,
                    discovery_cache_path=str(cache_path),
                ),
            )

            self.assertEqual(discover_scenario_values(config), (9, 17))
            self.assertTrue(cache_path.is_file())
            parquet_path.unlink()
            self.assertEqual(discover_scenario_values(config), (9, 17))

    def test_builds_architecture_specific_production_variants(self) -> None:
        report = _synthetic_report(self.sample)
        payloads = {
            model_name: build_config(
                self.sample,
                report,
                model_name=model_name,
                train_inputs=["/tmp/train"],
                test_inputs=["/tmp/test"],
            )[0]
            for model_name in ("rankmixer", "onetrans", "mdl_onetrans")
        }

        rankmixer = payloads["rankmixer"]
        self.assertEqual(len(rankmixer["features"]), 169)
        self.assertEqual(len(rankmixer["sequences"]), 9)
        self.assertTrue(
            all(sequence["encoder"] == "longer" for sequence in rankmixer["sequences"])
        )
        self.assertNotIn("scenario_tokens", rankmixer["tokenization"])
        self.assertNotIn("task_tokens", rankmixer["tokenization"])

        onetrans = payloads["onetrans"]
        self.assertEqual(len(onetrans["features"]), 169)
        self.assertEqual(len(onetrans["sequences"]), 9)
        self.assertTrue(
            all(sequence["encoder"] == "raw" for sequence in onetrans["sequences"])
        )
        self.assertTrue(
            all("longer_output" not in sequence for sequence in onetrans["sequences"])
        )
        self.assertEqual(
            [token["name"] for token in onetrans["tokenization"]["sequence_tokens"]],
            list(EXPECTED_UPS_TYPES),
        )
        self.assertEqual(onetrans["model"]["sequence_fusion"], "intent_ordered")
        self.assertEqual(onetrans["model"]["num_ns_tokens"], 32)

        mdl_onetrans = payloads["mdl_onetrans"]
        self.assertEqual(len(mdl_onetrans["features"]), 179)
        self.assertEqual(len(mdl_onetrans["sequences"]), 12)
        self.assertTrue(mdl_onetrans["model"]["experimental_model_acknowledged"])
        self.assertEqual(mdl_onetrans["model"]["first_domain_sequence_layer"], 4)
        prior_names = {
            "task_fst_cart_prior",
            "task_upid_pay_prior",
            "task_cateid_filter_prior",
        }
        self.assertEqual(
            {sequence["name"] for sequence in mdl_onetrans["sequences"][9:]},
            prior_names,
        )
        self.assertEqual(
            [token["name"] for token in mdl_onetrans["tokenization"]["sequence_tokens"]],
            list(EXPECTED_UPS_TYPES),
        )
        task_priors = {
            token["name"]: tuple(token.get("prior_inputs", []))
            for token in mdl_onetrans["tokenization"]["task_tokens"]
        }
        self.assertEqual(
            task_priors,
            {
                "fst_cart": ("task_fst_cart_prior",),
                "upid_pay": ("task_upid_pay_prior",),
                "cateid_filter": ("task_cateid_filter_prior",),
            },
        )
        for token in mdl_onetrans["tokenization"]["scenario_tokens"]:
            self.assertIn("impr", token["prior_inputs"])
            self.assertIn("clk_long", token["prior_inputs"])
            self.assertIn("view_long", token["prior_inputs"])
        adapter_limits = mdl_onetrans["data"]["train"]["adapter"]["options"][
            "sequence_max_lengths"
        ]
        self.assertGreaterEqual(
            adapter_limits["cart_long"],
            next(
                sequence["max_length"]
                for sequence in mdl_onetrans["sequences"]
                if sequence["name"] == "task_fst_cart_prior"
            ),
        )

        self.assertEqual(sum(ONETRANS_SEQUENCE_LENGTH_CAPS.values()), 2048)

    def test_standalone_models_resolve_auto_scenes_without_mdl_templates(self) -> None:
        report = build_name_estimate_report(self.sample)
        payload, summary = build_config(
            self.sample,
            report,
            model_name="rankmixer",
            train_inputs=["/tmp/train"],
            test_inputs=["/tmp/test"],
            auto_discover_scenes=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rankmixer.yaml"
            path.write_text(render_config(payload, summary), encoding="utf-8")
            config = load_app_config(path)

        self.assertFalse(
            any(feature.name == "scenario_prior_scene_id_hn" for feature in config.features)
        )
        resolved = resolve_auto_scenarios(config, [17, 9])
        self.assertEqual(resolved.scenarios.names, ("9", "17"))
        self.assertFalse(resolved.scenarios.auto_discover)
        self.assertEqual(len(resolved.features), 169)

    def test_all_production_configs_complete_forward_and_backward(self) -> None:
        torch.manual_seed(47)
        for model_name in (
            "rankmixer",
            "mdl_rankmixer",
            "onetrans",
            "mdl_onetrans",
        ):
            with self.subTest(model=model_name):
                config = _compact_production_config(model_name)
                model = build_model(
                    config,
                    {},
                    embedding_size_override=16,
                ).train()
                output = model(
                    _synthetic_model_features(config),
                    scenario_id=torch.tensor([0, 1], dtype=torch.long),
                )
                logits = output["logits"]
                self.assertEqual(logits.shape, (2, 3))
                self.assertTrue(bool(torch.isfinite(logits).all()))

                logits.square().mean().backward()
                gradients = []
                for parameter in model.parameters():
                    if parameter.grad is None:
                        continue
                    gradient = parameter.grad
                    gradients.append(
                        gradient._values() if gradient.is_sparse else gradient
                    )
                self.assertTrue(gradients)
                self.assertTrue(
                    all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
                )
                self.assertTrue(
                    any(bool(gradient.ne(0).any()) for gradient in gradients)
                )

    def test_request_indexed_inputs_match_repeated_candidate_inputs(self) -> None:
        torch.manual_seed(53)
        candidate_to_request = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        def expand(value):
            if isinstance(value, torch.Tensor):
                return value.index_select(0, candidate_to_request)
            if isinstance(value, dict):
                return {name: expand(child) for name, child in value.items()}
            return value

        for model_name in (
            "rankmixer",
            "mdl_rankmixer",
            "onetrans",
            "mdl_onetrans",
        ):
            with self.subTest(model=model_name):
                config = _compact_production_config(model_name)
                request_features = _synthetic_model_features(config, batch_size=2)
                expanded = {
                    name: expand(value) for name, value in request_features.items()
                }
                context_sources = set(
                    config.data.train.adapter.options["context_features"]
                )
                indexed = dict(expanded)
                for feature in config.features:
                    if feature.source not in context_sources:
                        continue
                    value = request_features[feature.name]
                    indexed[feature.name] = (
                        {**value, "row_indices": candidate_to_request}
                        if isinstance(value, dict)
                        else {
                            "values": value,
                            "row_indices": candidate_to_request,
                        }
                    )
                for sequence in config.sequences:
                    indexed[sequence.name] = {
                        **request_features[sequence.name],
                        "row_indices": candidate_to_request,
                    }

                model = build_model(config, {}, embedding_size_override=16).eval()
                scenario_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
                with torch.no_grad():
                    expected = model(expanded, scenario_id)["logits"]
                    actual = model(indexed, scenario_id)["logits"]
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_production_profiles_use_expected_runtime(self) -> None:
        expected_runtime = {
            # Current platform: 2×H100 SDPA eager + Row-Wise Adagrad + Phase 2.
            "rankmixer": ("sdpa", False, 2),
            "onetrans": ("sdpa", False, 2),
            "mdl_onetrans": ("sdpa", False, 2),
            "mdl_rankmixer": ("sdpa", False, 2),
        }
        for model_name, (attention_backend, compile_enabled, nproc) in expected_runtime.items():
            with self.subTest(model=model_name):
                config = load_app_config(
                    ROOT / "configs" / f"{model_name}.yaml"
                )
                self.assertEqual(config.runtime.attention_backend, attention_backend)
                self.assertEqual(config.runtime.compile, compile_enabled)
                self.assertEqual(config.runtime.nproc_per_node, nproc)
                self.assertEqual(config.runtime.activation_checkpoint, "none")
                self.assertFalse(config.runtime.trim_all_invalid_sequence_prefix)
                self.assertFalse(config.runtime.validate_scenario_ids)
                self.assertEqual(config.training.embedding_weight_dtype, "bf16")
                self.assertEqual(config.training.sparse_optimizer, "rowwise_adagrad")
                self.assertEqual(
                    config.training.checkpoint_path,
                    f"artifacts/checkpoints/{model_name}_2xh100_phase2_shared_dim",
                )
                self.assertFalse(config.training.embedding_collect_stats)
                self.assertFalse(config.training.embedding_validate_indices)
                self.assertEqual(
                    config.scenarios.discovery_cache_path,
                    "artifacts/scenarios/cvr_allscene.json",
                )
                self.assertEqual(config.scenarios.source_encoding, "raw")
                self.assertTrue(config.training.ddp.static_graph)
                self.assertTrue(
                    config.data.train.reader.deduplicate_request_features
                )
                self.assertFalse(
                    config.data.train.reader.validate_prehashed_nonzero
                )
                self.assertTrue(config.data.train.reader.trusted_input)
                self.assertTrue(config.data.test.reader.trusted_input)
                self.assertFalse(config.data.train.label_masks)
                self.assertFalse(config.data.test.label_masks)
                self.assertEqual(config.training.loss_reduction, "mean_per_task")
                self.assertTrue(config.training.quick_eval.enabled)
                self.assertEqual(config.training.quick_eval.split, "train")
                self.assertEqual(config.data.train.reader.shard_unit, "row_group")
                self.assertEqual(config.data.train.reader.shuffle_buffer_rows, 8192)
                self.assertEqual(config.data.train.reader.shuffle_seed, 2025)
                self.assertFalse(config.data.train.prediction_keys)
                self.assertEqual(
                    config.data.test.prediction_keys["candidate_position"],
                    "candidate_position",
                )
                self.assertTrue(
                    config.data.train.adapter.options["compact_request_lists"]
                )
                main_sequences = {
                    sequence.name: sequence.max_length
                    for sequence in config.sequences
                    if sequence.name
                    in config.data.train.adapter.options["ups_types"]
                }
                adapter_limits = config.data.train.adapter.options[
                    "sequence_max_lengths"
                ]
                self.assertEqual(set(adapter_limits), set(main_sequences))
                for name, limit in adapter_limits.items():
                    self.assertGreaterEqual(limit, main_sequences[name])
                self.assertTrue(config.data.train.reader.coalesce_pinned_tensors)
                self.assertEqual(
                    config.data.train.reader.device_prefetch_batches,
                    2,
                )
                self.assertEqual(config.data.train.reader.length_bucket_metric, "sum")
                self.assertEqual(
                    config.data.train.reader.eager_schema_validation,
                    "sample",
                )
                self.assertEqual(
                    config.data.train.reader.schema_validation_samples,
                    64,
                )
                self.assertEqual(
                    config.resolved.categorical_embedding_dims["goods_id_hn"],
                    48,
                )
                if model_name in {"mdl_rankmixer", "mdl_onetrans"}:
                    prior_goods = config.resolved.categorical_input_by_name[
                        "task_upid_pay_prior.goods_id_hn"
                    ]
                    self.assertTrue(prior_goods.encoding.share_embedding)
                    self.assertEqual(prior_goods.encoding.share_with, "goods_id_hn")
                    self.assertEqual(
                        config.resolved.categorical_embedding_dims[
                            "task_upid_pay_prior.goods_id_hn"
                        ],
                        48,
                    )
                    physical = sum(
                        1
                        for item in config.resolved.categorical_input_by_name.values()
                        if not getattr(item.encoding, "share_embedding", False)
                    )
                    self.assertEqual(physical, 202)
                    if model_name == "mdl_onetrans":
                        self.assertEqual(len(config.sequences), 12)
                        self.assertEqual(
                            {
                                sequence.name
                                for sequence in config.sequences
                                if sequence.name.startswith("task_")
                            },
                            {
                                "task_fst_cart_prior",
                                "task_upid_pay_prior",
                                "task_cateid_filter_prior",
                            },
                        )
                        self.assertGreaterEqual(
                            adapter_limits["cart_long"],
                            next(
                                sequence.max_length
                                for sequence in config.sequences
                                if sequence.name == "task_fst_cart_prior"
                            ),
                        )

    def test_bf16_sharded_embeddings_keep_fp32_dense_parameters(self) -> None:
        config = _compact_production_config("rankmixer")
        config = replace(
            config,
            training=replace(
                config.training,
                embedding_distribution="sharded",
                embedding_weight_dtype="bf16",
                embedding_collect_stats=False,
            ),
        )
        config.validate()
        model = build_model(config, {}, embedding_size_override=16)
        embeddings = [
            module for module in model.modules() if isinstance(module, ShardedEmbedding)
        ]
        self.assertTrue(embeddings)
        self.assertTrue(all(module.weight.dtype == torch.bfloat16 for module in embeddings))
        dense = [
            parameter
            for name, parameter in model.named_parameters()
            if ".embeddings." not in f".{name}"
        ]
        self.assertTrue(dense)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in dense))

    def test_rejects_zero_hash_and_missing_bucket_recommendation(self) -> None:
        report = _synthetic_report(self.sample)
        report["fields"]["goods_id_hn"]["zero_count"] = 1
        with self.assertRaisesRegex(ValueError, "reserved for padding"):
            build_config(
                self.sample,
                report,
                train_inputs=["/tmp/train"],
                test_inputs=["/tmp/test"],
            )

        report = _synthetic_report(self.sample)
        report["shared_embedding_groups"]["goods_id_hn"][
            "recommended_bucket_size"
        ] = None
        with self.assertRaisesRegex(ValueError, "larger --candidate-buckets"):
            build_config(
                self.sample,
                report,
                train_inputs=["/tmp/train"],
                test_inputs=["/tmp/test"],
            )

        report = _synthetic_report(self.sample)
        report["fields"]["sku_spec_vids_hn"]["list_lengths_by_depth"]["1"][
            "max"
        ] = 2
        with self.assertRaisesRegex(ValueError, "configured scalar"):
            build_config(
                self.sample,
                report,
                train_inputs=["/tmp/train"],
                test_inputs=["/tmp/test"],
            )

    def test_embedding_profiles_share_shapes_and_hit_memory_targets(self) -> None:
        report = build_name_estimate_report(self.sample)
        expected = {
            "baseline": (235, 38.285),
            # Important/scenario extras cannot share (MDL invariant); scene prior can.
            "shared": (202, 33.143),
            "shared_dim": (202, 27.643),
            "shared_dim_query_bucket": (202, 26.049),
            "shared_dim_aggressive_bucket": (202, 19.799),
        }
        for profile, (tables, gib) in expected.items():
            with self.subTest(profile=profile):
                payload, summary = build_config(
                    self.sample,
                    report,
                    train_inputs=["/tmp/train"],
                    test_inputs=["/tmp/test"],
                    auto_discover_scenes=True,
                    gpu_count=2,
                    embedding_weight_dtype="bf16",
                    sparse_optimizer="rowwise_adagrad",
                    embedding_budget_gib_per_gpu=80.0,
                    embedding_profile=profile,
                )
                self.assertEqual(summary["embedding_profile"], profile)
                self.assertEqual(summary["physical_embedding_tables"], tables)
                self.assertEqual(
                    summary["embedding_memory"]["unique_tables"],
                    tables,
                )
                planned = summary["embedding_memory"][
                    "planned_weight_plus_state_gib_per_gpu"
                ]
                self.assertAlmostEqual(planned, gib, places=2)
                entries = _categorical_entries_by_name(payload)
                for name, entry in entries.items():
                    encoding = entry["encoding"]
                    if not encoding.get("share_embedding"):
                        continue
                    root = _resolve_share_root(entries, name)
                    physical = entries[root]
                    self.assertEqual(
                        int(entry["embedding_dim"]),
                        int(physical["embedding_dim"]),
                    )
                    self.assertEqual(
                        int(encoding["num_buckets"]),
                        int(physical["encoding"]["num_buckets"]),
                    )
                    self.assertEqual(
                        int(encoding["padding_id"]),
                        int(physical["encoding"]["padding_id"]),
                    )
                if profile != "baseline":
                    self.assertIn("phase2", payload["training"]["checkpoint_path"])
                    spec = _find_sequence_field(payload, "cart_long.spec_hn")
                    sku = _find_sequence_field(payload, "cart_long.sku_ids_hn")
                    self.assertEqual(spec["embedding_dim"], 48)
                    self.assertEqual(spec["encoding"]["num_buckets"], 1 << 23)
                    self.assertEqual(sku["embedding_dim"], 48)
                    self.assertEqual(sku["encoding"]["num_buckets"], 1 << 24)
                    for sequence_name in (
                        "task_fst_cart_prior",
                        "task_upid_pay_prior",
                        "task_cateid_filter_prior",
                    ):
                        goods = _find_sequence_field(
                            payload,
                            f"{sequence_name}.goods_id_hn",
                        )
                        self.assertTrue(goods["encoding"]["share_embedding"])
                        self.assertEqual(
                            goods["encoding"]["share_with"],
                            "goods_id_hn",
                        )
                        expected_dim = 48 if profile != "shared" else 64
                        expected_buckets = (
                            1 << 26
                            if profile == "shared_dim_aggressive_bucket"
                            else 1 << 27
                        )
                        self.assertEqual(goods["embedding_dim"], expected_dim)
                        self.assertEqual(
                            goods["encoding"]["num_buckets"],
                            expected_buckets,
                        )
                        timegap = _find_sequence_field(
                            payload,
                            f"{sequence_name}.timegap_hn",
                        )
                        self.assertFalse(
                            timegap["encoding"].get("share_embedding", False)
                        )
                    for alias, root in (
                        ("buy_long.spec_hn", "cart_long.spec_hn"),
                        ("ups_clk_sku.spec_hn", "cart_long.spec_hn"),
                        ("task_fst_cart_prior.spec_hn", "cart_long.spec_hn"),
                        ("task_upid_pay_prior.spec_hn", "cart_long.spec_hn"),
                        ("task_cateid_filter_prior.spec_hn", "cart_long.spec_hn"),
                        ("buy_long.sku_ids_hn", "cart_long.sku_ids_hn"),
                        ("task_fst_cart_prior.sku_ids_hn", "cart_long.sku_ids_hn"),
                        ("task_upid_pay_prior.sku_ids_hn", "cart_long.sku_ids_hn"),
                        ("task_cateid_filter_prior.sku_ids_hn", "cart_long.sku_ids_hn"),
                    ):
                        field = _find_sequence_field(payload, alias)
                        self.assertTrue(field["encoding"]["share_embedding"])
                        self.assertEqual(field["encoding"]["share_with"], root)
                    # No multi-hop share_with chains remain after Phase 2.
                    for name, entry in entries.items():
                        encoding = entry["encoding"]
                        if not encoding.get("share_embedding"):
                            continue
                        self.assertEqual(
                            encoding["share_with"],
                            _resolve_share_root(entries, name),
                            msg=f"multi-hop share chain for {name}",
                        )

    def test_share_embedding_rejects_cycles(self) -> None:
        report = build_name_estimate_report(self.sample)
        payload, _summary = build_config(
            self.sample,
            report,
            train_inputs=["/tmp/train"],
            test_inputs=["/tmp/test"],
            auto_discover_scenes=True,
            gpu_count=2,
            embedding_weight_dtype="bf16",
            sparse_optimizer="rowwise_adagrad",
            embedding_budget_gib_per_gpu=80.0,
            embedding_profile="baseline",
        )
        goods = next(
            feature
            for feature in payload["features"]
            if feature["name"] == "goods_id_hn"
        )
        goods["encoding"]["share_embedding"] = True
        goods["encoding"]["share_with"] = "impr.goods_id_hn"
        with self.assertRaisesRegex(ValueError, "cycle"):
            apply_embedding_profile(payload, "baseline")


    def test_production_mdl_yamls_flatten_spec_sku_aliases(self) -> None:
        for config_name in ("mdl_rankmixer.yaml", "mdl_onetrans.yaml"):
            with self.subTest(config=config_name):
                config = load_app_config(ROOT / "configs" / config_name)
                for sequence_name in (
                    "buy_long",
                    "task_fst_cart_prior",
                    "task_upid_pay_prior",
                    "task_cateid_filter_prior",
                ):
                    sequence = next(
                        item for item in config.sequences if item.name == sequence_name
                    )
                    for field_name, root in (
                        ("spec_hn", "cart_long.spec_hn"),
                        ("sku_ids_hn", "cart_long.sku_ids_hn"),
                    ):
                        field = next(
                            item for item in sequence.fields if item.name == field_name
                        )
                        self.assertTrue(field.encoding.share_embedding)
                        self.assertEqual(field.encoding.share_with, root)
                ups = next(
                    item for item in config.sequences if item.name == "ups_clk_sku"
                )
                ups_spec = next(item for item in ups.fields if item.name == "spec_hn")
                self.assertEqual(ups_spec.encoding.share_with, "cart_long.spec_hn")

    def test_mdl_models_share_identical_prior_contract(self) -> None:
        report = build_name_estimate_report(self.sample)
        payloads = {}
        for model_name in ("mdl_rankmixer", "mdl_onetrans"):
            payloads[model_name], _summary = build_config(
                self.sample,
                report,
                model_name=model_name,
                train_inputs=["/tmp/train"],
                test_inputs=["/tmp/test"],
                auto_discover_scenes=True,
                gpu_count=2,
                embedding_weight_dtype="bf16",
                sparse_optimizer="rowwise_adagrad",
                embedding_budget_gib_per_gpu=80.0,
                embedding_profile="shared_dim",
            )
        left = payloads["mdl_rankmixer"]
        right = payloads["mdl_onetrans"]
        left_priors = [sequence for sequence in left["sequences"] if sequence["name"].startswith("task_")]
        right_priors = [
            sequence for sequence in right["sequences"] if sequence["name"].startswith("task_")
        ]
        self.assertEqual(
            [sequence["name"] for sequence in left_priors],
            [sequence["name"] for sequence in right_priors],
        )
        for left_seq, right_seq in zip(left_priors, right_priors):
            self.assertEqual(left_seq["max_length"], right_seq["max_length"])
            self.assertEqual(left_seq["encoder"], right_seq["encoder"])
            self.assertEqual(left_seq["embedding_scope"], right_seq["embedding_scope"])
            self.assertEqual(
                [
                    (
                        field["name"],
                        field["kind"],
                        field["source"],
                        field.get("embedding_dim"),
                        field.get("dimension"),
                        field.get("encoding"),
                    )
                    for field in left_seq["fields"]
                ],
                [
                    (
                        field["name"],
                        field["kind"],
                        field["source"],
                        field.get("embedding_dim"),
                        field.get("dimension"),
                        field.get("encoding"),
                    )
                    for field in right_seq["fields"]
                ],
            )
        self.assertEqual(
            left["tokenization"]["task_tokens"],
            right["tokenization"]["task_tokens"],
        )
        self.assertEqual(
            left["tokenization"]["scenario_tokens"],
            right["tokenization"]["scenario_tokens"],
        )
        self.assertEqual(
            [token["name"] for token in right["tokenization"]["sequence_tokens"]],
            list(EXPECTED_UPS_TYPES),
        )
        self.assertNotIn(
            "task_fst_cart_prior",
            [token["name"] for token in right["tokenization"]["sequence_tokens"]],
        )


if __name__ == "__main__":
    unittest.main()
