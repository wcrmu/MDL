from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import yaml

from scripts.build_production_configs import (
    AUTO_SCENARIO_NAME,
    CANDIDATE_ITEM_BAG_FIELDS,
    CANDIDATE_ITEM_SCALAR_FIELDS,
    CONTEXT_FEATURE_COUNT,
    CONTEXT_SCALAR_FIELDS,
    EXPECTED_FEATURE_COUNT,
    EXPECTED_LABELS,
    EXPECTED_UPS_TYPES,
    ITEM_BAG_FIELDS,
    MULTIVALUE_MAX_LENGTHS,
    OBSERVED_MULTIVALUE_MAX_LENGTHS,
    ONETRANS_SEQUENCE_LENGTH_CAPS,
    PACK_MULTIVALUE_MAX_LENGTHS,
    PHASE2_TASK_PRIOR_SEQUENCES,
    PRODUCTION_COARSE_CONFIG_NAMES,
    RANKMIXER_SEMANTIC_FEATURE_GROUPS,
    REQUEST_CONTEXT_BAG_FIELDS,
    REQUEST_CONTEXT_SCALAR_FIELDS,
    SCENARIO_CONDITIONED_HISTORY_PRIOR,
    SCENARIO_IMPORTANT_FIELDS_BY_TOKEN,
    SCENARIO_IMPRESSION_PRIOR_FIELDS,
    SCENARIO_SHARED_PRIOR_UPS,
    TASK_IMPORTANT_FIELDS,
    TASK_IMPORTANT_FIELDS_BY_TASK,
    TASK_IMPORTANT_IDENTITY_SHAPES,
    apply_embedding_profile,
    build_config,
    build_name_estimate_report,
    derive_fine_payload,
    fine_config_name,
    _cap_multivalue_observed_max,
    _find_sequence_field,
    _resolve_share_root,
    _categorical_entries_by_name,
    merge_production_contract,
    render_config,
    task_important_name,
    task_prior_inputs,
    write_fine_siblings,
)
from scripts.profile_prehashed_parquet import profile_spec_from_mapping
from src.config import AppConfig, ResolvedPreHashedEncoding, load_app_config
from src.dataloader import (
    COARSE_SCENE_INDEX_COLUMN,
    COARSE_SCENE_PRIOR_ID_COLUMN,
    RECOMMENDATION_PRIOR_FEATURE,
    SCENARIO_NAMES,
    SEARCH_PRIOR_FEATURE,
    SEARCH_SCENE_IDS,
    _load_parquet_adapter,
    _scenario_tensor,
    discover_scenario_values,
    resolve_auto_scenarios,
)
from src.model import build_model
from src.embeddings import ShardedEmbedding


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FIXTURE = ROOT / "tests" / "fixtures" / "mdl_sample.yaml"


def _compact_production_config(model_name: str):
    """Keep production wiring while making a CPU forward/backward test cheap."""

    config = load_app_config(ROOT / "configs" / f"{model_name}.yaml")
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
            name: 2 for name in split.adapter.options.get("sequence_max_lengths", {})
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


def _compact_generated_stca_config(config: AppConfig) -> AppConfig:
    """Shrink generated nine-stream STCA wiring without changing its topology."""

    sequences = tuple(
        replace(
            sequence,
            max_length=2,
            stca_layers=(2 if sequence.encoder == "stca" else sequence.stca_layers),
            stca_num_heads=(
                4 if sequence.encoder == "stca" else sequence.stca_num_heads
            ),
            stca_expansion_ratio=(
                2 if sequence.encoder == "stca" else sequence.stca_expansion_ratio
            ),
        )
        for sequence in config.sequences
    )

    def compact_split(split):
        if split is None or split.adapter is None:
            return split
        limits = {
            name: 2 for name in split.adapter.options.get("sequence_max_lengths", {})
        }
        return replace(
            split,
            adapter=replace(
                split.adapter,
                options={
                    **split.adapter.options,
                    "sequence_max_lengths": limits,
                },
            ),
        )

    compact = replace(
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
            num_layers=1,
            num_heads=4,
            hidden_dim=64,
            task_head_hidden_dim=64,
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
            sparse_optimizer="adagrad",
        ),
    )
    compact.validate()
    return compact


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
            "empty_lists_by_depth": {},
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
        source for sources in spec.sequence_sources.values() for source in sources
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
            "request_outer_mismatches": {},
            "candidate_outer_mismatches": {},
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
        cls.sample = yaml.safe_load(SAMPLE_FIXTURE.read_text(encoding="utf-8"))

    def test_builds_valid_report_driven_production_config(self) -> None:
        report = _synthetic_report(self.sample)
        payload, summary = build_config(
            self.sample,
            report,
            train_inputs=["/tmp/train"],
            test_inputs=["/tmp/test"],
        )

        self.assertEqual(payload["runtime"]["nproc_per_node"], 2)
        self.assertEqual(payload["training"]["sparse_optimizer"], "rowwise_adagrad")
        memory = summary["embedding_memory"]
        self.assertIn("optimizer_state_gib_total", memory)
        self.assertEqual(memory["optimizer_state_layout"], "rowwise")
        self.assertEqual(memory["gpu_count"], 2)
        self.assertEqual(memory["embedding_weight_dtype"], "bf16")
        self.assertEqual(summary["embedding_profile"], "shared_dim")

        self.assertEqual(
            [item["name"] for item in payload["features"][:EXPECTED_FEATURE_COUNT]],
            [item["name"] for item in self.sample["features"]],
        )
        by_name = {item["name"]: item for item in payload["features"]}
        self.assertEqual(by_name["goods_name_bigram_hn"]["pooling"], "mean")
        self.assertEqual(
            by_name["sku_id_hn"]["pooling_null_policy"],
            "include_as_padding",
        )
        self.assertEqual(by_name["sku_spec_vids_hn"]["pooling"], "mean")
        self.assertEqual(by_name["sku_spec_vids_hn"]["max_length"], 256)
        self.assertEqual(summary["bag_feature_count"], 80)
        self.assertEqual(
            set(MULTIVALUE_MAX_LENGTHS),
            set(OBSERVED_MULTIVALUE_MAX_LENGTHS),
        )
        self.assertTrue(
            all(
                MULTIVALUE_MAX_LENGTHS[name] <= observed
                for name, observed in OBSERVED_MULTIVALUE_MAX_LENGTHS.items()
            )
        )
        self.assertLessEqual(max(MULTIVALUE_MAX_LENGTHS.values()), 512)
        self.assertEqual(sum(PACK_MULTIVALUE_MAX_LENGTHS.values()), 9114)
        self.assertEqual(
            OBSERVED_MULTIVALUE_MAX_LENGTHS["cart_long_spec_vids_hn"],
            10005,
        )
        self.assertEqual(MULTIVALUE_MAX_LENGTHS["cart_long_spec_vids_hn"], 512)
        self.assertEqual(
            [
                _cap_multivalue_observed_max(value)
                for value in (128, 129, 512, 513, 2048, 2049)
            ],
            [128, 128, 128, 256, 256, 512],
        )

        main_sequences = payload["sequences"][:9]
        self.assertEqual(
            [item["name"] for item in main_sequences],
            [item["name"] for item in self.sample["sequences"]],
        )
        for sequence in main_sequences:
            self.assertEqual(sequence["encoder"], "longer")
            self.assertEqual(sequence["longer_output"], "summary")
            self.assertEqual(sequence["longer_token_merge"], 1)
            # Default build_config model is mdl_rankmixer: LONGER keeps scene
            # user-global (scene_id_hn) like standalone RankMixer.
            self.assertEqual(sequence["rankmixer_summary_tokens"], 3)
            self.assertEqual(sequence["longer_dim"], 32)
            self.assertEqual(sequence["longer_num_heads"], 4)
            self.assertEqual(sequence["longer_hidden_dim"], 64)
            self.assertEqual(
                sequence["target_inputs"],
                ["goods_id_hn", "cat1_id_hn", "price_hn"],
            )
            self.assertEqual(
                sequence["longer_user_global_inputs"],
                ["scene_id_hn"],
            )
            self.assertEqual(sequence["longer_user_global_tokens"], 1)
            self.assertEqual(sequence["longer_cls_tokens"], 1)
            self.assertEqual(sequence["longer_candidate_global_tokens"], 1)
            self.assertEqual(sequence["max_length"], 10)
            self.assertEqual(sequence["sequence_order"], "newest_to_oldest")
            self.assertEqual(sequence["truncation"], "head")
            self.assertEqual(
                sequence["fields"][0]["name"],
                "time_delta_log1p_seconds",
            )
            self.assertEqual(sequence["fields"][0]["kind"], "dense")

        by_seq = {item["name"]: item for item in payload["sequences"]}
        self.assertIn(SCENARIO_CONDITIONED_HISTORY_PRIOR, by_seq)
        conditioned = by_seq[SCENARIO_CONDITIONED_HISTORY_PRIOR]
        self.assertEqual(conditioned["encoder"], "attention_pool")
        self.assertEqual(
            conditioned["embedding_scope"], "scenario"
        )
        self.assertEqual(
            conditioned["target_inputs"],
            [
                "scenario_important_scene_id_hn",
                "scenario_important_page_sn_hn",
            ],
        )
        for ups in SCENARIO_SHARED_PRIOR_UPS:
            global_prior = by_seq[f"scenario_global_{ups}_prior"]
            self.assertEqual(global_prior["encoder"], "mean_pool")
            self.assertEqual(global_prior.get("target_inputs", []), [])
        task_priors = {
            name: by_seq[name]
            for name in (
                "task_fst_cart_prior",
                "task_upid_pay_prior",
                "task_cateid_filter_prior",
            )
        }
        self.assertEqual(
            task_priors["task_fst_cart_prior"]["encoder"], "attention_pool"
        )
        self.assertEqual(task_priors["task_fst_cart_prior"]["pool_dim"], 32)
        self.assertEqual(
            TASK_IMPORTANT_FIELDS_BY_TASK["fst_cart"][-4:],
            (
                "cat_id_hn",
                "goods_cluster_id_1w_hn",
                "mall_id_hn",
                "goods_id_hn",
            ),
        )
        # Pay keeps taxonomy identity but drops the two sparsest tables; the
        # order/GMV anchors replace them.
        self.assertEqual(
            TASK_IMPORTANT_FIELDS_BY_TASK["upid_pay"][-5:],
            (
                "idx_c_ordr_cnt_15d_hn",
                "nfk_gmv_14d_hn",
                "u_fst_ordr_cnt_mix_d_hn",
                "cat_id_hn",
                "goods_cluster_id_1w_hn",
            ),
        )
        self.assertNotIn("goods_id_hn", TASK_IMPORTANT_FIELDS_BY_TASK["upid_pay"])
        self.assertNotIn("mall_id_hn", TASK_IMPORTANT_FIELDS_BY_TASK["upid_pay"])
        self.assertEqual(
            TASK_IMPORTANT_FIELDS_BY_TASK["cateid_filter"][-1],
            "origin_query_hash_hn",
        )
        for task, prior in task_priors.items():
            task_name = task.removeprefix("task_").removesuffix("_prior")
            self.assertEqual(
                prior["target_inputs"],
                [
                    task_important_name(task_name, source)
                    for source in TASK_IMPORTANT_FIELDS_BY_TASK[task_name]
                ],
            )
        # Each task owns its important tables, so a source listed by several
        # tasks is no longer one table optimised mostly by the largest weight.
        for task, sources in TASK_IMPORTANT_FIELDS_BY_TASK.items():
            for source in sources:
                shape = TASK_IMPORTANT_IDENTITY_SHAPES.get(source)
                if shape is None:
                    continue
                num_buckets, embedding_dim = shape
                important = by_name[task_important_name(task, source)]
                self.assertEqual(important["source"], source)
                self.assertEqual(important["embedding_scope"], "task")
                self.assertEqual(important["embedding_dim"], embedding_dim)
                self.assertEqual(important["encoding"]["num_buckets"], num_buckets)
                self.assertFalse(important["encoding"].get("share_embedding", False))
                self.assertNotIn("share_with", important["encoding"])
        query_identity = by_name[
            task_important_name("cateid_filter", "origin_query_hash_hn")
        ]
        self.assertEqual(query_identity["pooling"], "mean")
        self.assertEqual(query_identity["max_length"], 46)
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
        self.assertFalse(upid_goods["encoding"].get("share_embedding", False))
        self.assertFalse(cateid_goods["encoding"].get("share_embedding", False))
        self.assertNotIn("share_with", upid_goods["encoding"])
        self.assertNotIn("share_with", cateid_goods["encoding"])
        self.assertEqual(by_name["goods_id_hn"]["embedding_dim"], 32)

        self.assertEqual(
            payload["scenarios"],
            {
                "names": list(SCENARIO_NAMES),
                "source": COARSE_SCENE_INDEX_COLUMN,
                "source_encoding": "index",
                "auto_discover": False,
            },
        )
        adapter_options = payload["data"]["train"]["adapter"]["options"]
        self.assertEqual(
            payload["data"]["train"]["adapter"]["callable"],
            "src.dataloader:adapt_mdl_rankmixer_parquet",
        )
        adapter_payload = payload["data"]["train"]["adapter"]
        self.assertEqual(len(adapter_payload["input_columns"]), 259)
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
        self.assertNotIn("request_value_maps", adapter_options)
        self.assertEqual(
            adapter_options["search_scene_ids"],
            sorted(SEARCH_SCENE_IDS),
        )
        self.assertEqual(
            adapter_options["coarse_scene_index_column"],
            COARSE_SCENE_INDEX_COLUMN,
        )
        self.assertEqual(
            adapter_options["coarse_scene_prior_id_column"],
            COARSE_SCENE_PRIOR_ID_COLUMN,
        )
        self.assertEqual(
            [token["name"] for token in payload["tokenization"]["scenario_tokens"]],
            ["search", "recommendation", "global"],
        )
        self.assertEqual(
            payload["tokenization"]["scenario_tokens"][0]["prior_inputs"][0],
            SEARCH_PRIOR_FEATURE,
        )
        self.assertEqual(
            payload["tokenization"]["scenario_tokens"][1]["prior_inputs"][0],
            RECOMMENDATION_PRIOR_FEATURE,
        )
        for prior_name in (SEARCH_PRIOR_FEATURE, RECOMMENDATION_PRIOR_FEATURE):
            encoding = by_name[prior_name]["encoding"]
            self.assertEqual(
                by_name[prior_name]["source"], COARSE_SCENE_PRIOR_ID_COLUMN
            )
            self.assertEqual(encoding["type"], "identity")
            self.assertEqual(encoding["num_buckets"], 3)
            self.assertEqual(encoding["padding_id"], 0)
            self.assertEqual(encoding["out_of_range"], "error")
            self.assertFalse(encoding.get("share_embedding", False))
            self.assertNotIn("share_with", encoding)
        self.assertNotIn("request_axis_item_features", adapter_options)
        self.assertNotIn("candidate_axis_context_features", adapter_options)
        for name in (
            "offline_outside_goods_id_list_hn_share",
            "buy_long_spec_vids_hn",
            "impr_3h_tg_hn",
            "query_pay_cnt_15d_hn",
            "opt_id_hn",
        ):
            self.assertIn(name, adapter_options["context_features"])
            self.assertNotIn(name, adapter_options["item_features"])
        for name in (
            "clk_cnt_1d_hn",
            "clk_3d_cnt_hn",
            "clk_1d_cat_cnt_hn",
            "cart_cnt_1d_hn",
            "cart_cnt_3d_hn",
        ):
            self.assertIn(name, adapter_options["item_features"])
            self.assertNotIn(name, adapter_options["context_features"])
        for name in (
            "offline_outside_goods_id_list_hn_share",
            "i2i_coclk_hn_share",
            "i2i2cat2_swing_hn",
            "buy_long_spec_vids_hn",
            "impr_3h_tg_hn",
            "cart_long_hit_samestyle_i2i_idx_hn",
        ):
            self.assertIn(name, adapter_options["multivalue_features"])
            self.assertEqual(by_name[name]["pooling"], "mean")
        self.assertEqual(
            by_name["cart_long_hit_samestyle_i2i_idx_hn"]["max_length"],
            16,
        )
        expected_bags = (
            set(adapter_options["context_features"]) - CONTEXT_SCALAR_FIELDS
        ) | ITEM_BAG_FIELDS
        self.assertEqual(set(adapter_options["multivalue_features"]), expected_bags)
        self.assertEqual(summary["bag_feature_count"], len(expected_bags))
        self.assertEqual(
            {
                feature["source"]: feature["max_length"]
                for feature in payload["features"][:EXPECTED_FEATURE_COUNT]
                if feature.get("pooling") == "mean"
            },
            PACK_MULTIVALUE_MAX_LENGTHS,
        )
        for name in (
            "multimodal_i2i_hit_clk_size_hn",
            "multimodal_i2i_hit_cart_size_hn",
            "query_pay_cnt_15d_hn",
            "opt_id_hn",
        ):
            self.assertNotIn(name, adapter_options["multivalue_features"])
            self.assertNotIn("pooling", by_name[name])
        aligned_sku = adapter_options["aligned_multivalue_groups"][0]
        self.assertEqual(len(aligned_sku), 8)
        self.assertNotIn("sku_spec_hn", aligned_sku)
        self.assertIn("sku_id_hn", aligned_sku)
        self.assertIn("sku_spec_hash_hn", aligned_sku)
        self.assertIn("sku_spec_hn", adapter_options["multivalue_features"])
        self.assertEqual(by_name["sku_spec_hn"]["pooling"], "mean")
        self.assertEqual(
            by_name["sku_spec_hn"]["pooling_null_policy"], "include_as_padding"
        )
        self.assertEqual(len(adapter_options["context_features"]), CONTEXT_FEATURE_COUNT)
        self.assertEqual(len(adapter_options["item_features"]), EXPECTED_FEATURE_COUNT - CONTEXT_FEATURE_COUNT)
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

        config = AppConfig.from_mapping(payload)
        self.assertEqual(payload["model"]["mdl_token_state"], "coupled")
        self.assertEqual(config.model.mdl_token_state, "coupled")
        self.assertEqual(
            config.resolved.tokenization.feature_token_count,
            32,
        )
        self.assertEqual(payload["tokenization"]["feature_tokenizer"], "groupwise")
        self.assertEqual(len(payload["tokenization"]["feature_tokens"]), 32)
        self.assertEqual(payload["model"]["token_dim"], 768)
        self.assertEqual(payload["runtime"]["nproc_per_node"], 2)
        self.assertEqual(payload["runtime"]["attention_backend"], "flash")
        self.assertEqual(payload["runtime"]["activation_checkpoint"], "none")
        self.assertEqual(payload["training"]["embedding_distribution"], "sharded")
        self.assertEqual(payload["training"]["embedding_weight_dtype"], "bf16")
        self.assertEqual(payload["training"]["lr_dense"], 1.0e-4)
        self.assertEqual(payload["training"]["lr_sparse"], 1.0e-4)
        self.assertEqual(payload["training"]["lr_schedule"], "constant")
        self.assertEqual(payload["training"]["lr_warmup_steps"], 5000)
        self.assertEqual(payload["training"]["loss_reduction"], "mean_per_task")
        self.assertEqual(
            payload["training"]["task_loss_weights"],
            {
                "fst_cart": 0.5,
                "cateid_filter": 0.01,
                "upid_pay": 0.01,
            },
        )
        self.assertEqual(
            payload["training"]["fixed_test_eval"],
            {
                "enabled": True,
                "every_steps": 5000,
                "files_per_rank": 4,
                "auc_bins": 4096,
            },
        )
        self.assertLessEqual(
            summary["embedding_memory"]["planned_weight_plus_state_gib_per_gpu"],
            80.0,
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

    def test_empty_label_distribution_is_treated_as_absent(self) -> None:
        # Stale profiles that never saw production labels emit {}. That must not
        # be read as "every task entry is missing".
        report = _synthetic_report(self.sample)
        report["contract"]["label_distribution"] = {}
        report["contract"]["invalid_labels"] = {}
        payload, _summary = build_config(
            self.sample,
            report,
            train_inputs=["/tmp/train"],
            test_inputs=["/tmp/test"],
        )
        self.assertGreaterEqual(len(payload["features"]), EXPECTED_FEATURE_COUNT)
        self.assertEqual(
            set(payload["data"]["train"]["labels"]),
            set(EXPECTED_LABELS),
        )

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
            1 << 28,
        )
        self.assertEqual(
            by_name["ups_clkv2_i2i_goods_ids_hit_size"]["encoding"]["num_buckets"],
            256,
        )
        self.assertEqual(summary["profile"]["settings"]["mode"], "name_heuristic")
        self.assertLess(
            summary["embedding_memory"]["planned_weight_plus_state_gib_per_gpu"],
            80.0,
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
            base = load_app_config(ROOT / "configs" / "reference" / "default.yaml")
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
        self.assertEqual(len(rankmixer["features"]), EXPECTED_FEATURE_COUNT)
        self.assertEqual(len(rankmixer["sequences"]), 9)
        self.assertTrue(
            all(sequence["encoder"] == "longer" for sequence in rankmixer["sequences"])
        )
        self.assertNotIn("scenario_tokens", rankmixer["tokenization"])
        self.assertNotIn("task_tokens", rankmixer["tokenization"])

        onetrans = payloads["onetrans"]
        self.assertEqual(len(onetrans["features"]), EXPECTED_FEATURE_COUNT)
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
        self.assertEqual(len(mdl_onetrans["features"]), 185)
        self.assertEqual(len(mdl_onetrans["sequences"]), 17)
        self.assertTrue(mdl_onetrans["model"]["experimental_model_acknowledged"])
        self.assertEqual(mdl_onetrans["model"]["first_domain_sequence_layer"], 0)
        prior_names = {
            SCENARIO_CONDITIONED_HISTORY_PRIOR,
            "scenario_global_impr_prior",
            "scenario_global_clk_long_prior",
            "scenario_global_view_long_prior",
            *PHASE2_TASK_PRIOR_SEQUENCES,
            "task_upid_pay_ups_clk_sku_prior",
        }
        self.assertEqual(
            {sequence["name"] for sequence in mdl_onetrans["sequences"][9:]},
            prior_names,
        )
        self.assertEqual(
            [
                token["name"]
                for token in mdl_onetrans["tokenization"]["sequence_tokens"]
            ],
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
                # Pay carries a second, denser stream: buy_long is an empty
                # list on ~24% of requests.
                "upid_pay": (
                    "task_upid_pay_prior",
                    "task_upid_pay_ups_clk_sku_prior",
                ),
                "cateid_filter": ("task_cateid_filter_prior",),
            },
        )
        for token in mdl_onetrans["tokenization"]["scenario_tokens"]:
            name = token["name"]
            priors = token["prior_inputs"]
            if name == "global":
                self.assertEqual(
                    priors,
                    [
                        f"scenario_global_{ups}_prior"
                        for ups in SCENARIO_SHARED_PRIOR_UPS
                    ],
                )
            else:
                self.assertTrue(priors[0].endswith("_prior_coarse_scene") or priors[0].startswith("scenario_"))
                self.assertTrue(
                    any(p.endswith("_clk_long_prior") for p in priors),
                    msg=priors,
                )
                self.assertNotIn("impr", priors)
                self.assertNotIn("clk_long", priors)
                self.assertNotIn("view_long", priors)
        for token in mdl_onetrans["tokenization"]["task_tokens"]:
            self.assertEqual(
                token["important_inputs"],
                [
                    task_important_name(token["name"], source)
                    for source in TASK_IMPORTANT_FIELDS_BY_TASK[token["name"]]
                ],
            )
            self.assertEqual(
                token["prior_inputs"], task_prior_inputs(token["name"])
            )
        self.assertEqual(
            [
                token["name"]
                for token in mdl_onetrans["tokenization"]["scenario_tokens"]
            ],
            ["search", "recommendation", "global"],
        )
        self.assertEqual(mdl_onetrans["scenarios"]["names"], list(SCENARIO_NAMES))
        self.assertEqual(
            mdl_onetrans["scenarios"]["source"],
            COARSE_SCENE_INDEX_COLUMN,
        )
        self.assertFalse(mdl_onetrans["scenarios"]["auto_discover"])
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

    def test_builds_stca_rankmixer_variants_without_changing_task_priors(
        self,
    ) -> None:
        report = _synthetic_report(self.sample)
        for model_name in ("rankmixer", "mdl_rankmixer"):
            with self.subTest(model=model_name):
                payload, summary = build_config(
                    self.sample,
                    report,
                    model_name=model_name,
                    train_inputs=["/tmp/train"],
                    test_inputs=["/tmp/test"],
                    sequence_encoder="stca",
                )
                main_sequences = payload["sequences"][: len(EXPECTED_UPS_TYPES)]
                self.assertEqual(summary["sequence_encoder"], "stca")
                self.assertEqual(
                    payload["training"]["loss_reduction"],
                    "mean_per_request_per_task",
                )
                self.assertEqual(
                    payload["runtime"]["sequence_projection_chunk_tokens"],
                    65536,
                )
                config = AppConfig.from_mapping(payload)
                self.assertEqual(
                    config.resolved.tokenization.feature_token_inputs[-1],
                    main_sequences[0]["name"],
                )
                regular_inputs = list(
                    config.resolved.tokenization.feature_token_inputs[:-1]
                )
                regular_width = sum(
                    config.resolved.encoded_input_dims[name] for name in regular_inputs
                )
                self.assertEqual(regular_width % 31, 0)
                self.assertEqual(
                    config.resolved.encoded_input_dims[main_sequences[0]["name"]],
                    256,
                )
                self.assertTrue(
                    all(sequence["encoder"] == "stca" for sequence in main_sequences)
                )
                for sequence in main_sequences:
                    self.assertEqual(
                        sequence["target_inputs"],
                        ["goods_id_hn", "cat1_id_hn", "price_hn"],
                    )
                    self.assertEqual(sequence["rankmixer_summary_tokens"], 1)
                    self.assertEqual(sequence["stca_dim"], 256)
                    self.assertEqual(sequence["stca_layers"], 4)
                    self.assertEqual(sequence["stca_num_heads"], 16)
                    self.assertEqual(sequence["stca_expansion_ratio"], 4)
                    self.assertEqual(
                        sequence["stca_parameter_group"],
                        "main_history",
                    )
                    self.assertEqual(
                        sequence["stca_history_group"],
                        "main_history",
                    )

                domain_priors = payload["sequences"][len(EXPECTED_UPS_TYPES) :]
                if model_name == "mdl_rankmixer":
                    self.assertTrue(domain_priors)
                    prior_by_name = {
                        sequence["name"]: sequence for sequence in domain_priors
                    }
                    self.assertEqual(
                        prior_by_name[SCENARIO_CONDITIONED_HISTORY_PRIOR]["encoder"],
                        "attention_pool",
                    )
                    for ups in SCENARIO_SHARED_PRIOR_UPS:
                        self.assertEqual(
                            prior_by_name[f"scenario_global_{ups}_prior"]["encoder"],
                            "mean_pool",
                        )
                    for task in EXPECTED_LABELS:
                        self.assertEqual(
                            prior_by_name[f"task_{task}_prior"]["encoder"],
                            "attention_pool",
                        )
                    main_sequence_names = {
                        sequence["name"] for sequence in main_sequences
                    }
                    for token in payload["tokenization"]["scenario_tokens"]:
                        main_priors = {
                            name
                            for name in token["prior_inputs"]
                            if name in main_sequence_names
                        }
                        # Global uses independent scenario_global_* mean_pool
                        # clones, not backbone LONGER/STCA summaries.
                        self.assertFalse(main_priors)
                else:
                    self.assertFalse(domain_priors)

                compact = _compact_generated_stca_config(config)
                model = build_model(
                    compact,
                    {},
                    embedding_size_override=16,
                ).train()
                self.assertEqual(
                    model.feature_projector.dedicated_token_names,
                    (main_sequences[0]["name"],),
                )
                self.assertEqual(len(model.encoder_bank.sequence_stca_encoders), 1)
                self.assertEqual(
                    len(model.encoder_bank.sequence_query_projectors),
                    6 if model_name == "mdl_rankmixer" else 1,
                )
                output = model(
                    _synthetic_model_features(compact),
                    scenario_id=torch.tensor([0, 1], dtype=torch.long),
                )
                logits = output["logits"]
                self.assertEqual(tuple(logits.shape), (2, 3))
                self.assertTrue(bool(torch.isfinite(logits).all()))
                logits.square().mean().backward()
                action_type_gradients = [
                    parameter.grad
                    for name, parameter in model.named_parameters()
                    if "sequence_stca_type_embeddings" in name
                ]
                self.assertEqual(
                    len(action_type_gradients),
                    len(EXPECTED_UPS_TYPES),
                )
                self.assertTrue(
                    all(
                        gradient is not None and bool(torch.isfinite(gradient).all())
                        for gradient in action_type_gradients
                    )
                )

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
            any(
                feature.name == "scenario_prior_scene_id_hn"
                for feature in config.features
            )
        )
        resolved = resolve_auto_scenarios(config, [17, 9])
        self.assertEqual(resolved.scenarios.names, ("9", "17"))
        self.assertFalse(resolved.scenarios.auto_discover)
        self.assertEqual(len(resolved.features), EXPECTED_FEATURE_COUNT)

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
                if model_name in {"mdl_rankmixer", "mdl_onetrans"}:
                    self.assertEqual(config.model.mdl_token_state, "coupled")
                    self.assertIsNone(model.scenario_readout_seed)
                    self.assertIsNone(model.task_readout_seed)
                    parameter_names = dict(model.named_parameters())
                    self.assertNotIn("scenario_readout_seed", parameter_names)
                    self.assertNotIn("task_readout_seed", parameter_names)
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

    def test_split_mdl_configs_complete_checkpointed_forward_and_backward(self) -> None:
        torch.manual_seed(49)
        for model_name in ("mdl_rankmixer", "mdl_onetrans"):
            with self.subTest(model=model_name):
                config = _compact_production_config(model_name)
                config = replace(
                    config,
                    model=replace(config.model, mdl_token_state="split"),
                    runtime=replace(
                        config.runtime,
                        activation_checkpoint="full",
                        cuda_graph_backbone=False,
                    ),
                )
                config.validate()
                model = build_model(
                    config,
                    {},
                    embedding_size_override=16,
                ).train()

                self.assertIsNotNone(model.scenario_readout_seed)
                self.assertIsNotNone(model.task_readout_seed)
                if model_name == "mdl_rankmixer":
                    self.assertEqual(
                        config.resolved.tokenization.feature_token_count,
                        32,
                    )

                logits = model(
                    _synthetic_model_features(config),
                    scenario_id=torch.tensor([0, 1], dtype=torch.long),
                )["logits"]
                self.assertEqual(tuple(logits.shape), (2, 3))
                self.assertTrue(bool(torch.isfinite(logits).all()))
                logits.square().mean().backward()
                for seed in (
                    model.scenario_readout_seed,
                    model.task_readout_seed,
                ):
                    assert seed is not None
                    self.assertIsNotNone(seed.grad)
                    assert seed.grad is not None
                    self.assertTrue(bool(torch.isfinite(seed.grad).all()))

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
            # checkpoint, graph, packing and fused-dense reflect the latest
            # measured HBM/util winners for each family.
            "rankmixer": ("flash", False, 2, "none", True, "fixed", True),
            "onetrans": ("flash", False, 2, "none", False, "fixed", True),
            "mdl_onetrans": (
                "flash",
                False,
                2,
                "none",
                False,
                "fixed",
                False,
            ),
            "mdl_rankmixer": (
                "flash",
                False,
                2,
                "none",
                True,
                "compact",
                True,
            ),
        }
        for model_name, (
            attention_backend,
            compile_enabled,
            nproc,
            activation_checkpoint,
            cuda_graph_backbone,
            varlen_packing,
            fused_dense_optimizer,
        ) in expected_runtime.items():
            with self.subTest(model=model_name):
                config = load_app_config(ROOT / "configs" / f"{model_name}.yaml")
                self.assertEqual(config.runtime.attention_backend, attention_backend)
                self.assertEqual(config.runtime.compile, compile_enabled)
                self.assertEqual(config.runtime.nproc_per_node, nproc)
                memory_optimized = model_name.startswith("mdl_")
                self.assertEqual(
                    config.runtime.activation_checkpoint, activation_checkpoint
                )
                self.assertEqual(
                    config.runtime.cuda_graph_backbone, cuda_graph_backbone
                )
                self.assertEqual(config.runtime.varlen_packing, varlen_packing)
                self.assertEqual(
                    config.training.fused_dense_optimizer,
                    fused_dense_optimizer,
                )
                self.assertEqual(
                    config.runtime.trim_all_invalid_sequence_prefix,
                    False,
                )
                self.assertFalse(config.runtime.validate_scenario_ids)
                self.assertEqual(config.training.embedding_weight_dtype, "bf16")
                self.assertEqual(config.training.sparse_optimizer, "rowwise_adagrad")
                self.assertEqual(
                    config.training.checkpoint_path,
                    f"artifacts/checkpoints/{model_name}_2xh100_phase2_shared_dim",
                )
                self.assertEqual(
                    config.training.embedding_collect_stats,
                    model_name == "mdl_rankmixer",
                )
                self.assertFalse(config.training.embedding_validate_indices)
                self.assertFalse(config.scenarios.auto_discover)
                self.assertIsNone(config.scenarios.discovery_cache_path)
                self.assertEqual(config.scenarios.names, SCENARIO_NAMES)
                self.assertEqual(
                    config.scenarios.source,
                    COARSE_SCENE_INDEX_COLUMN,
                )
                self.assertEqual(config.scenarios.source_encoding, "index")
                self.assertEqual(
                    list(config.data.train.adapter.options["search_scene_ids"]),
                    sorted(SEARCH_SCENE_IDS),
                )
                self.assertTrue(config.training.ddp.static_graph)
                self.assertTrue(config.data.train.reader.deduplicate_request_features)
                self.assertFalse(config.data.train.reader.validate_prehashed_nonzero)
                self.assertTrue(config.data.train.reader.trusted_input)
                self.assertTrue(config.data.test.reader.trusted_input)
                # Production profiles disable cardinality audit on the GPU rank
                # so startup does not open HDFS before the step watchdog is armed.
                self.assertEqual(
                    config.data.train.reader.effective_cardinality_audit_raw_rows(),
                    0,
                )
                self.assertEqual(
                    config.data.test.reader.effective_cardinality_audit_raw_rows(),
                    0,
                )
                self.assertFalse(config.data.train.label_masks)
                self.assertFalse(config.data.test.label_masks)
                self.assertEqual(config.training.loss_reduction, "mean_per_task")
                self.assertEqual(config.training.lr_dense, 1.0e-4)
                self.assertEqual(config.training.lr_sparse, 1.0e-4)
                self.assertEqual(config.training.lr_schedule, "constant")
                self.assertEqual(config.training.lr_warmup_steps, 5000)
                if memory_optimized:
                    expected_proj_chunk = (
                        131072 if model_name == "mdl_rankmixer" else 81920
                    )
                    expected_batch = 1536 if model_name == "mdl_rankmixer" else 1408
                    expected_buckets = (
                        [1536, 960, 640, 480, 768]
                        if model_name == "mdl_rankmixer"
                        else [1408, 880, 576, 432, 704]
                    )
                    self.assertEqual(
                        config.runtime.sequence_projection_chunk_tokens,
                        expected_proj_chunk,
                    )
                    self.assertEqual(
                        config.runtime.sequence_encoder_chunk_rows,
                        0,
                    )
                    self.assertEqual(
                        config.runtime.sequence_encoder_chunk_tokens,
                        262144 if model_name == "mdl_rankmixer" else 163840,
                    )
                    self.assertTrue(config.runtime.onetrans_batched_ns)
                    self.assertEqual(config.training.batch_size, expected_batch)
                    self.assertEqual(
                        config.training.gradient_accumulation_steps,
                        1,
                    )
                    self.assertEqual(
                        config.training.dense_optimizer_foreach_bucket_mb,
                        128,
                    )
                    self.assertEqual(
                        [
                            bucket.batch_size
                            for bucket in config.data.train.reader.length_buckets
                        ],
                        expected_buckets,
                    )
                self.assertTrue(config.training.fixed_test_eval.enabled)
                self.assertEqual(config.training.fixed_test_eval.every_steps, 5000)
                self.assertEqual(config.training.fixed_test_eval.files_per_rank, 4)
                self.assertEqual(config.data.train.reader.shard_unit, "file")
                self.assertEqual(config.data.train.reader.shuffle_buffer_rows, 512)
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
                    if sequence.name in config.data.train.adapter.options["ups_types"]
                }
                adapter_limits = config.data.train.adapter.options[
                    "sequence_max_lengths"
                ]
                self.assertEqual(set(adapter_limits), set(main_sequences))
                for name, limit in adapter_limits.items():
                    self.assertGreaterEqual(limit, main_sequences[name])
                self.assertTrue(config.data.train.reader.coalesce_pinned_tensors)
                self.assertEqual(config.data.train.reader.num_workers, 2)
                self.assertEqual(config.data.train.reader.adapter_workers, 4)
                self.assertEqual(config.data.train.reader.scanner_batch_rows, 128)
                self.assertEqual(
                    config.data.train.reader.device_prefetch_batches,
                    1,
                )
                self.assertEqual(config.data.train.reader.length_bucket_metric, "sum")
                self.assertEqual(
                    config.data.train.reader.eager_schema_validation,
                    "sample",
                )
                self.assertEqual(
                    config.data.train.reader.schema_validation_samples,
                    1,
                )
                self.assertEqual(
                    config.resolved.categorical_embedding_dims["goods_id_hn"],
                    32,
                )
                if model_name in {"mdl_rankmixer", "mdl_onetrans"}:
                    prior_goods = config.resolved.categorical_input_by_name[
                        "task_upid_pay_prior.goods_id_hn"
                    ]
                    self.assertFalse(
                        getattr(prior_goods.encoding, "share_embedding", False)
                    )
                    self.assertIsNone(
                        getattr(prior_goods.encoding, "share_with", None)
                    )
                    self.assertEqual(
                        config.resolved.categorical_embedding_dims[
                            "task_upid_pay_prior.goods_id_hn"
                        ],
                        32,
                    )
                    physical = sum(
                        1
                        for item in config.resolved.categorical_input_by_name.values()
                        if not getattr(item.encoding, "share_embedding", False)
                    )
                    # Phase-2 keeps task/scenario-history priors independent;
                    # this now includes four candidate/query identity tables
                    # and one important table per task rather than per source.
                    self.assertEqual(physical, 306)
                    if model_name == "mdl_onetrans":
                        self.assertEqual(len(config.sequences), 17)
                        # Every prior a task token reads must exist as a loaded
                        # sequence, which is more than one per task: pay carries
                        # a supplemental stream for the requests where buy_long
                        # is empty.
                        self.assertEqual(
                            {
                                sequence.name
                                for sequence in config.sequences
                                if sequence.name.startswith("task_")
                            },
                            {
                                name
                                for task in EXPECTED_LABELS
                                for name in task_prior_inputs(task)
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
        self.assertTrue(
            all(module.weight.dtype == torch.bfloat16 for module in embeddings)
        )
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
        report["fields"]["goods_id_hn"]["list_lengths_by_depth"]["1"]["max"] = 2
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
            # Growth-aware PROFILE_DRIVEN_EMBEDDING_SHAPES win after every Phase-2
            # tier, so shared/query/aggressive bucket profiles collapse to the same
            # planned memory once those overrides apply.
            # Counts include independent global scenario priors, the per-task
            # important tables, and the pay coverage prior; dead near-constants
            # are removed.
            "baseline": (305, 66.257),
            "shared": (304, 66.257),
            "shared_dim": (304, 66.257),
            "shared_dim_query_bucket": (304, 66.257),
            "shared_dim_aggressive_bucket": (304, 66.257),
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
                    self.assertEqual(spec["embedding_dim"], 32)
                    self.assertEqual(spec["encoding"]["num_buckets"], 1 << 26)
                    # 24-hour growth recommendation (12,000 files), with
                    # width traded for hash capacity under the H100 budget.
                    self.assertEqual(sku["embedding_dim"], 32)
                    self.assertEqual(sku["encoding"]["num_buckets"], 1 << 29)
                    for sequence_name in (
                        "task_fst_cart_prior",
                        "task_upid_pay_prior",
                        "task_cateid_filter_prior",
                    ):
                        goods = _find_sequence_field(
                            payload,
                            f"{sequence_name}.goods_id_hn",
                        )
                        self.assertFalse(
                            goods["encoding"].get("share_embedding", False)
                        )
                        self.assertNotIn("share_with", goods["encoding"])
                        # Independent prior tables stay under PRIOR_INDEPENDENT_BUCKET_CAPS.
                        self.assertEqual(goods["embedding_dim"], 32)
                        self.assertLessEqual(
                            goods["encoding"]["num_buckets"],
                            1 << 24,
                        )
                        timegap = _find_sequence_field(
                            payload,
                            f"{sequence_name}.timegap_hn",
                        )
                        self.assertFalse(
                            timegap["encoding"].get("share_embedding", False)
                        )
                        task_specific_fields = (
                            ()
                            if sequence_name == "task_cateid_filter_prior"
                            else ("spec_hn", "sku_ids_hn")
                        )
                        for field_name in task_specific_fields:
                            field = _find_sequence_field(
                                payload,
                                f"{sequence_name}.{field_name}",
                            )
                            self.assertFalse(
                                field["encoding"].get("share_embedding", False)
                            )
                            self.assertNotIn("share_with", field["encoding"])
                    for alias, root in (
                        ("buy_long.spec_hn", "cart_long.spec_hn"),
                        ("ups_clk_sku.spec_hn", "cart_long.spec_hn"),
                        ("buy_long.sku_ids_hn", "cart_long.sku_ids_hn"),
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

    def test_coarse_scenario_priors_stay_independent_across_phase2_profiles(
        self,
    ) -> None:
        report = build_name_estimate_report(self.sample)
        for profile in (
            "shared",
            "shared_dim",
            "shared_dim_query_bucket",
            "shared_dim_aggressive_bucket",
        ):
            with self.subTest(profile=profile):
                payload, _summary = build_config(
                    self.sample,
                    report,
                    train_inputs=["/tmp/train"],
                    test_inputs=["/tmp/test"],
                    auto_discover_scenes=False,
                    embedding_profile=profile,
                )
                self.assertEqual(payload["scenarios"]["names"], list(SCENARIO_NAMES))
                self.assertEqual(
                    payload["scenarios"]["source"],
                    COARSE_SCENE_INDEX_COLUMN,
                )
                self.assertFalse(payload["scenarios"]["auto_discover"])
                by_name = {item["name"]: item for item in payload["features"]}
                for prior_name in (SEARCH_PRIOR_FEATURE, RECOMMENDATION_PRIOR_FEATURE):
                    encoding = by_name[prior_name]["encoding"]
                    self.assertEqual(encoding["type"], "identity")
                    self.assertEqual(encoding["num_buckets"], 3)
                    self.assertEqual(encoding["padding_id"], 0)
                    self.assertEqual(encoding["out_of_range"], "error")
                    self.assertFalse(encoding.get("share_embedding", False))
                    self.assertNotIn("share_with", encoding)
                tokens = {
                    token["name"]: token
                    for token in payload["tokenization"]["scenario_tokens"]
                }
                self.assertEqual(
                    tokens["search"]["prior_inputs"][0],
                    SEARCH_PRIOR_FEATURE,
                )
                self.assertEqual(
                    tokens["recommendation"]["prior_inputs"][0],
                    RECOMMENDATION_PRIOR_FEATURE,
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

    def test_production_yamls_build_adapter_plans(self) -> None:
        from types import SimpleNamespace

        from src.dataloader import (
            _build_mdl_rankmixer_adapter_plan,
            required_columns_for_split,
        )

        for config_name in (
            "rankmixer.yaml",
            "onetrans.yaml",
            "mdl_rankmixer.yaml",
            "mdl_onetrans.yaml",
        ):
            with self.subTest(config=config_name):
                config = load_app_config(ROOT / "configs" / config_name)
                production_bags = {
                    feature.source: feature.max_length
                    for feature in config.features[:EXPECTED_FEATURE_COUNT]
                    if feature.pooling == "mean"
                }
                self.assertEqual(
                    set(production_bags),
                    set(PACK_MULTIVALUE_MAX_LENGTHS),
                )
                self.assertTrue(
                    all(
                        0 < length <= OBSERVED_MULTIVALUE_MAX_LENGTHS[name]
                        for name, length in production_bags.items()
                    )
                )
                for split_name, split in (
                    ("train", config.data.train),
                    ("test", config.data.test),
                ):
                    with self.subTest(split=split_name):
                        self.assertIsNotNone(split.adapter)
                        options = split.adapter.options
                        self.assertEqual(len(options["context_features"]), CONTEXT_FEATURE_COUNT)
                        self.assertEqual(len(options["item_features"]), EXPECTED_FEATURE_COUNT - CONTEXT_FEATURE_COUNT)
                        self.assertNotIn("request_axis_item_features", options)
                        self.assertNotIn("candidate_axis_context_features", options)
                        self.assertEqual(
                            options.get("unlisted_scene_policy"), "recommendation"
                        )
                        self.assertEqual(
                            len(options.get("search_scene_ids", ())),
                            len(SEARCH_SCENE_IDS),
                        )
                        for name in (
                            "query_pay_cnt_15d_hn",
                            "opt_id_hn",
                            "buy_long_spec_vids_hn",
                            "offline_outside_goods_id_list_hn_share",
                        ):
                            self.assertIn(name, options["context_features"])
                            self.assertNotIn(name, options["item_features"])
                        for name in (
                            "clk_cnt_1d_hn",
                            "cart_cnt_3d_hn",
                        ):
                            self.assertIn(name, options["item_features"])
                            self.assertNotIn(name, options["context_features"])
                            self.assertIn(name, options["multivalue_features"])
                        for name in (
                            REQUEST_CONTEXT_BAG_FIELDS | REQUEST_CONTEXT_SCALAR_FIELDS
                        ):
                            self.assertIn(name, options["context_features"])
                            self.assertNotIn(name, options["item_features"])
                        for name in REQUEST_CONTEXT_BAG_FIELDS:
                            self.assertIn(name, options["multivalue_features"])
                        for name in REQUEST_CONTEXT_SCALAR_FIELDS:
                            self.assertNotIn(name, options["multivalue_features"])
                        for name in (
                            CANDIDATE_ITEM_BAG_FIELDS | CANDIDATE_ITEM_SCALAR_FIELDS
                        ):
                            self.assertIn(name, options["item_features"])
                            self.assertNotIn(name, options["context_features"])
                        for name in CANDIDATE_ITEM_BAG_FIELDS:
                            self.assertIn(name, options["multivalue_features"])
                        for name in CANDIDATE_ITEM_SCALAR_FIELDS:
                            self.assertNotIn(name, options["multivalue_features"])
                        self.assertIn(
                            "cart_long_hit_samestyle_i2i_idx_hn",
                            options["multivalue_features"],
                        )
                        expected_bags = (
                            set(options["context_features"]) - CONTEXT_SCALAR_FIELDS
                        ) | ITEM_BAG_FIELDS
                        self.assertEqual(
                            set(options["multivalue_features"]),
                            expected_bags,
                        )
                        required = required_columns_for_split(config, split)
                        plan = _build_mdl_rankmixer_adapter_plan(
                            SimpleNamespace(
                                options=options,
                                required_columns=tuple(required),
                            )
                        )
                        self.assertEqual(len(plan.context_features), CONTEXT_FEATURE_COUNT)
                        self.assertEqual(len(plan.item_features), EXPECTED_FEATURE_COUNT - CONTEXT_FEATURE_COUNT)

    def test_production_fine_yamls_build_adapter_plans(self) -> None:
        from types import SimpleNamespace

        from src.dataloader import (
            _build_mdl_rankmixer_adapter_plan,
            required_columns_for_split,
        )

        for coarse_name in PRODUCTION_COARSE_CONFIG_NAMES:
            config_name = fine_config_name(coarse_name)
            with self.subTest(config=config_name):
                config = load_app_config(ROOT / "configs" / config_name)
                self.assertTrue(config.scenarios.auto_discover)
                for split_name, split in (
                    ("train", config.data.train),
                    ("test", config.data.test),
                ):
                    with self.subTest(split=split_name):
                        options = split.adapter.options
                        self.assertNotIn("search_scene_ids", options)
                        self.assertNotIn("coarse_scene_index_column", options)
                        self.assertIsNone(options.get("unlisted_scene_policy"))
                        required = required_columns_for_split(config, split)
                        plan = _build_mdl_rankmixer_adapter_plan(
                            SimpleNamespace(
                                options=options,
                                required_columns=tuple(required),
                            )
                        )
                        self.assertIsNone(plan.coarse_scene)
                        self.assertEqual(len(plan.context_features), CONTEXT_FEATURE_COUNT)
                        self.assertEqual(len(plan.item_features), EXPECTED_FEATURE_COUNT - CONTEXT_FEATURE_COUNT)

    def test_production_rankmixer_yamls_use_paper_longer_targets(self) -> None:
        expected_targets = ("goods_id_hn", "cat1_id_hn", "price_hn")
        for config_name in (
            "rankmixer.yaml",
            "mdl_rankmixer.yaml",
            "rankmixer_fine.yaml",
            "mdl_rankmixer_fine.yaml",
        ):
            with self.subTest(config=config_name):
                config = load_app_config(ROOT / "configs" / config_name)
                longer = [
                    sequence
                    for sequence in config.sequences
                    if sequence.encoder == "longer"
                ]
                self.assertTrue(longer)
                for sequence in longer:
                    self.assertEqual(sequence.target_inputs, expected_targets)
                    # RankMixer and MDL-RankMixer both keep scene_id on LONGER
                    # user-global; MDL still also routes scene via scenario tokens.
                    self.assertEqual(
                        sequence.longer_user_global_inputs,
                        ("scene_id_hn",),
                    )
                    self.assertEqual(sequence.longer_user_global_tokens, 1)
                    self.assertEqual(sequence.rankmixer_summary_tokens, 3)
                    self.assertEqual(sequence.longer_cls_tokens, 1)
                    self.assertEqual(sequence.longer_candidate_global_tokens, 1)
                    self.assertEqual(sequence.longer_dim, 32)
                    self.assertEqual(sequence.longer_num_heads, 4)
                    self.assertEqual(sequence.longer_hidden_dim, 64)
                    self.assertEqual(
                        config.resolved.encoded_input_dims[sequence.name],
                        96,
                    )

    def test_production_mdl_yamls_flatten_spec_sku_aliases(self) -> None:
        for config_name in ("mdl_rankmixer.yaml", "mdl_onetrans.yaml"):
            with self.subTest(config=config_name):
                config = load_app_config(ROOT / "configs" / config_name)
                buy = next(item for item in config.sequences if item.name == "buy_long")
                for field_name, root in (
                    ("spec_hn", "cart_long.spec_hn"),
                    ("sku_ids_hn", "cart_long.sku_ids_hn"),
                ):
                    field = next(
                        item for item in buy.fields if item.name == field_name
                    )
                    self.assertTrue(field.encoding.share_embedding)
                    self.assertEqual(field.encoding.share_with, root)
                for sequence_name in (
                    "task_fst_cart_prior",
                    "task_upid_pay_prior",
                    "task_cateid_filter_prior",
                ):
                    sequence = next(
                        item for item in config.sequences if item.name == sequence_name
                    )
                    for field in sequence.fields:
                        if field.kind != "categorical":
                            continue
                        self.assertFalse(
                            getattr(field.encoding, "share_embedding", False),
                            msg=f"{sequence_name}.{field.name}",
                        )
                        self.assertIsNone(
                            getattr(field.encoding, "share_with", None),
                            msg=f"{sequence_name}.{field.name}",
                        )
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
        left_priors = [
            sequence
            for sequence in left["sequences"]
            if sequence["name"].startswith("task_")
        ]
        right_priors = [
            sequence
            for sequence in right["sequences"]
            if sequence["name"].startswith("task_")
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

    def test_derive_fine_payload_preserves_capacity(self) -> None:
        for coarse_name in PRODUCTION_COARSE_CONFIG_NAMES:
            with self.subTest(config=coarse_name):
                coarse_path = ROOT / "configs" / coarse_name
                coarse = yaml.safe_load(coarse_path.read_text(encoding="utf-8"))
                fine = derive_fine_payload(coarse)
                self.assertEqual(
                    fine["scenarios"],
                    {
                        "names": [AUTO_SCENARIO_NAME],
                        "source": "scene_id",
                        "source_encoding": "raw",
                        "auto_discover": True,
                        "max_discovered": 256,
                    },
                )
                self.assertEqual(
                    fine["training"]["batch_size"],
                    coarse["training"]["batch_size"],
                )
                for split_name in ("train", "test"):
                    options = fine["data"][split_name]["adapter"]["options"]
                    self.assertNotIn("search_scene_ids", options)
                    self.assertNotIn("coarse_scene_index_column", options)
                    self.assertNotIn("unlisted_scene_policy", options)
                feature_names = {item["name"] for item in fine["features"]}
                if coarse_name.startswith("mdl_"):
                    self.assertIn("scenario_prior_scene_id_hn", feature_names)
                    self.assertNotIn(SEARCH_PRIOR_FEATURE, feature_names)
                    self.assertNotIn(RECOMMENDATION_PRIOR_FEATURE, feature_names)
                    tokens = {
                        token["name"]: token
                        for token in fine["tokenization"]["scenario_tokens"]
                    }
                    self.assertEqual(
                        set(tokens),
                        {AUTO_SCENARIO_NAME, "global"},
                    )
                    self.assertEqual(
                        tokens[AUTO_SCENARIO_NAME]["prior_inputs"][0],
                        "scenario_prior_scene_id_hn",
                    )
                else:
                    self.assertNotIn("scenario_prior_scene_id_hn", feature_names)

    def test_merge_production_contract_keeps_ops_not_stale_model_contract(
        self,
    ) -> None:
        generated, _summary = build_config(
            self.sample,
            _synthetic_report(self.sample),
            model_name="mdl_rankmixer",
            train_inputs=["/generated/train"],
            test_inputs=["/generated/test"],
        )
        current = yaml.safe_load(yaml.safe_dump(generated, sort_keys=False))
        current["runtime"]["master_port"] = 29999
        current["training"]["batch_size"] = 777
        current["data"]["train"]["inputs"] = ["/production/train"]
        current["data"]["train"]["reader"]["num_workers"] = 7
        current["model"]["mdl_feature_interaction"] = "direct_ffn"
        current["features"].append(
            {
                "name": "uid_or_bg_hn",
                "kind": "categorical",
                "source": "uid_or_bg_hn",
                "embedding_scope": "feature",
                "embedding_dim": 32,
                "encoding": {
                    "type": "pre_hashed",
                    "num_buckets": 1024,
                    "padding_id": 0,
                },
            }
        )

        merged = merge_production_contract(generated, current)

        self.assertEqual(merged["runtime"]["master_port"], 29999)
        self.assertEqual(merged["training"]["batch_size"], 777)
        self.assertEqual(merged["data"]["train"]["inputs"], ["/production/train"])
        self.assertEqual(merged["data"]["train"]["reader"]["num_workers"], 7)
        self.assertNotIn(
            "uid_or_bg_hn",
            {feature["name"] for feature in merged["features"]},
        )
        self.assertEqual(merged["tokenization"]["feature_tokenizer"], "groupwise")
        self.assertEqual(len(merged["tokenization"]["feature_tokens"]), 32)
        self.assertEqual(
            merged["model"]["mdl_feature_interaction"],
            "residual_ffn",
        )
        self.assertEqual(merged["model"]["mdl_token_state"], "coupled")

    def test_production_fine_yamls_match_derived_siblings(self) -> None:
        for coarse_name in PRODUCTION_COARSE_CONFIG_NAMES:
            with self.subTest(config=coarse_name):
                coarse = yaml.safe_load(
                    (ROOT / "configs" / coarse_name).read_text(encoding="utf-8")
                )
                fine_name = fine_config_name(coarse_name)
                fine_path = ROOT / "configs" / fine_name
                self.assertTrue(fine_path.is_file(), fine_name)
                written = yaml.safe_load(fine_path.read_text(encoding="utf-8"))
                derived = derive_fine_payload(coarse)
                self.assertEqual(written, derived)
                config = load_app_config(fine_path)
                self.assertTrue(config.scenarios.auto_discover)
                self.assertEqual(config.scenarios.names, (AUTO_SCENARIO_NAME,))
                self.assertEqual(config.scenarios.source, "scene_id")
                self.assertEqual(config.scenarios.source_encoding, "raw")
                for split in (config.data.train, config.data.test):
                    self.assertNotIn("search_scene_ids", split.adapter.options)

    def test_write_fine_siblings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configs_dir = Path(directory)
            for coarse_name in PRODUCTION_COARSE_CONFIG_NAMES:
                source = ROOT / "configs" / coarse_name
                (configs_dir / coarse_name).write_text(
                    source.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            written = write_fine_siblings(configs_dir)
            self.assertEqual(
                [path.name for path in written],
                [fine_config_name(name) for name in PRODUCTION_COARSE_CONFIG_NAMES],
            )
            for coarse_name, output in zip(PRODUCTION_COARSE_CONFIG_NAMES, written):
                expected = ROOT / "configs" / fine_config_name(coarse_name)
                # Compare parsed payloads: safe_dump drops hand-written comments
                # and may render null / inline lists differently than checked-in
                # fine YAMLs.
                generated = yaml.safe_load(output.read_text(encoding="utf-8"))
                checked_in = yaml.safe_load(expected.read_text(encoding="utf-8"))
                self.assertEqual(generated, checked_in)


if __name__ == "__main__":
    unittest.main()
