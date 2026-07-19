from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from scripts.profile_prehashed_parquet import (
    _bucket_report,
    ContractProfile,
    FieldProfile,
    ProfileSpec,
    load_profile_spec,
    profile_paths,
)


class PreHashedParquetProfileTest(unittest.TestCase):
    def test_bucket_recommendation_projects_full_cardinality_collisions(self) -> None:
        report, recommendation = _bucket_report(
            100_000,
            tuple(range(1, 4097)),
            (1 << 18, 1 << 23),
            collision_target=0.01,
            cardinality_headroom=1.5,
        )

        self.assertEqual(report[0]["sample_collision_rate"], 0.0)
        self.assertGreater(report[0]["projected_uniform_collision_rate"], 0.01)
        self.assertLessEqual(report[1]["projected_uniform_collision_rate"], 0.01)
        self.assertEqual(recommendation, 1 << 23)

    def test_null_label_is_reported_as_an_explicit_missing_category(self) -> None:
        contract = ContractProfile(
            ProfileSpec(
                all_sources=(),
                categorical_sources=(),
                time_sources=(),
                context_sources=(),
                item_sources=(),
                sequence_sources={},
                sequence_time_sources={},
                label_sources={"task": "label"},
                shared_groups={},
                sku_fields=(),
                scene_source="scene_id",
            )
        )

        contract.observe({"label": [0, None, 1], "scene_id": 7})

        report = contract.as_dict()
        self.assertEqual(report["invalid_labels"], {"task": 1})
        self.assertEqual(
            report["label_distribution"]["task"],
            {
                "examples": 2,
                "positives": 1,
                "negatives": 1,
                "invalid": 1,
                "total": 3,
                "null": 1,
                "minus_one": 0,
                "zero": 1,
                "one": 1,
                "other": 0,
            },
        )

    def test_unlabeled_req_schema_does_not_report_missing_training_labels(self) -> None:
        spec = ProfileSpec(
            all_sources=("ctx_hn", "item_hn"),
            categorical_sources=("ctx_hn", "item_hn"),
            time_sources=(),
            context_sources=("ctx_hn",),
            item_sources=("item_hn",),
            sequence_sources={},
            sequence_time_sources={},
            label_sources={"task": "label"},
            shared_groups={},
            sku_fields=(),
            scene_source="scene_id",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "req.parquet"
            pq.write_table(
                pa.table(
                    {
                        "ctx_hn": [[11]],
                        "item_hn": [[[21], [22]]],
                        "scene_id": [7],
                        "impr_time": [5000],
                    }
                ),
                path,
            )

            report = profile_paths(
                [str(path)],
                spec,
                candidate_buckets=(16,),
                collision_target=1.0,
                cardinality_headroom=1.0,
                sample_size=16,
                hll_precision=10,
                progress=False,
            )

        self.assertEqual(report["missing_configured_columns_by_input"][str(path)], [])
        self.assertEqual(report["contract"]["req_rows"], 1)
        self.assertEqual(report["contract"]["label_distribution"]["task"]["total"], 0)

    def test_configured_field_name_is_profiled_exactly(self) -> None:
        canonical = "f_goods_view_times_tg_l1_hn"
        spec = ProfileSpec(
            all_sources=(canonical,),
            categorical_sources=(canonical,),
            time_sources=(),
            context_sources=(),
            item_sources=(canonical,),
            sequence_sources={},
            sequence_time_sources={},
            label_sources={},
            shared_groups={},
            sku_fields=(),
            scene_source="scene_id",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "canonical.parquet"
            pq.write_table(
                pa.table(
                    {
                        canonical: [[[-7]]],
                        "scene_id": [3],
                        "impr_time": [5000],
                    }
                ),
                path,
            )
            report = profile_paths(
                [str(path)],
                spec,
                candidate_buckets=(16,),
                collision_target=1.0,
                cardinality_headroom=1.0,
                sample_size=16,
                hll_precision=10,
                progress=False,
            )

            legacy_typo = canonical.replace("l1", "1" * 2)
            typo_path = root / "typo.parquet"
            pq.write_table(
                pa.table(
                    {
                        legacy_typo: [[[-7]]],
                        "scene_id": [3],
                        "impr_time": [5000],
                    }
                ),
                typo_path,
            )
            typo_report = profile_paths(
                [str(typo_path)],
                spec,
                candidate_buckets=(16,),
                collision_target=1.0,
                cardinality_headroom=1.0,
                sample_size=16,
                hll_precision=10,
                progress=False,
            )

        self.assertEqual(
            report["resolved_column_aliases_by_input"][str(path)],
            {},
        )
        self.assertEqual(report["missing_configured_columns_by_input"][str(path)], [])
        self.assertEqual(report["fields"][canonical]["negative_count"], 1)
        self.assertEqual(
            typo_report["missing_configured_columns_by_input"][str(typo_path)],
            [canonical],
        )
        self.assertEqual(typo_report["fields"][canonical]["leaf_count"], 0)

    def test_nested_null_sign_and_power_of_two_collision_stats(self) -> None:
        profile = FieldProfile(sample_size=32, hll_precision=10)
        profile.observe(None)
        profile.observe([[None, -(1 << 63)], [], [1, -1, 0]])
        report = profile.as_dict(
            candidate_buckets=(4, 8),
            collision_target=1.0,
            cardinality_headroom=1.0,
        )

        self.assertEqual(report["nulls_by_depth"], {"0": 1, "2": 1})
        self.assertEqual(report["empty_lists_by_depth"], {"1": 1})
        self.assertEqual(report["negative_count"], 2)
        self.assertEqual(report["positive_count"], 1)
        self.assertEqual(report["zero_count"], 1)
        self.assertEqual(report["signed_min"], -(1 << 63))
        self.assertEqual(report["unsigned_max"], (1 << 64) - 1)
        self.assertGreater(report["bucket_candidates"][0]["sample_collisions"], 0)

    def test_profiles_synthetic_agg_contract_without_hdfs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "sample.yaml"
            parquet_path = root / "part.parquet"
            config = {
                "data": {
                    "train": {
                        "agg_layout": {
                            "labels": {"a": "label_a", "b": "label_b", "c": "label_c"}
                        }
                    }
                },
                "features": [
                    {"name": "ctx_hn", "source": "ctx_hn"},
                    {"name": "goods_id_hn", "source": "goods_id_hn"},
                    {"name": "sku_id_hn", "source": "sku_id_hn"},
                    {"name": "sku_price_v2_hn", "source": "sku_price_v2_hn"},
                ],
                "sequences": [
                    {
                        "name": "impr",
                        "fields": [
                            {"name": "time", "source": "impr_x_time"},
                            {"name": "goods_id_hn", "source": "impr_x_goods_id_hn"},
                        ],
                    }
                ],
                "vocab_strategy": {
                    "features": {
                        "ctx_hn": {"encoding": "pre_hashed"},
                        "goods_id_hn": {"encoding": "pre_hashed"},
                        "sku_id_hn": {"encoding": "pre_hashed"},
                        "sku_price_v2_hn": {"encoding": "pre_hashed"},
                        "impr.time": {"encoding": "identity"},
                        "impr.goods_id_hn": {"encoding": "pre_hashed"},
                    }
                },
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            table = pa.table(
                {
                    "context_indices": pa.array([[0, 1]], type=pa.list_(pa.int64())),
                    "target_indices": pa.array([[0, 0]], type=pa.list_(pa.int64())),
                    "ctx_hn": pa.array(
                        [[[11], None]], type=pa.list_(pa.list_(pa.int64()))
                    ),
                    "goods_id_hn": pa.array(
                        [[[-10], [-99]]], type=pa.list_(pa.list_(pa.int64()))
                    ),
                    "sku_id_hn": pa.array(
                        [[[21, 22], [23]]], type=pa.list_(pa.list_(pa.int64()))
                    ),
                    "sku_price_v2_hn": pa.array(
                        [[[31, None], [32]]], type=pa.list_(pa.list_(pa.int64()))
                    ),
                    "impr_x_time": pa.array(
                        [[[3000], [2000], [1000]]],
                        type=pa.list_(pa.list_(pa.int64())),
                    ),
                    "impr_x_goods_id_hn": pa.array(
                        [[[-10], [-11], [-12]]],
                        type=pa.list_(pa.list_(pa.int64())),
                    ),
                    "impr_x_indices": pa.array(
                        [[[0, 1], [0], [0]]], type=pa.list_(pa.list_(pa.int64()))
                    ),
                    "scene_id": pa.array([[3, 4]], type=pa.list_(pa.int64())),
                    "impr_time": pa.array([[4000, 4000]], type=pa.list_(pa.int64())),
                    "label_a": pa.array([[0, 1]], type=pa.list_(pa.int64())),
                    "label_b": pa.array([[1, 0]], type=pa.list_(pa.int64())),
                    "label_c": pa.array([[0, 0]], type=pa.list_(pa.int64())),
                }
            )
            pq.write_table(table, parquet_path)

            spec = load_profile_spec(config_path, context_feature_count=1)
            report = profile_paths(
                [str(parquet_path)],
                spec,
                candidate_buckets=(16, 64),
                collision_target=1.0,
                cardinality_headroom=1.0,
                sample_size=64,
                hll_precision=10,
                progress=False,
            )

        self.assertEqual(report["rows_scanned"], 1)
        self.assertEqual(report["contract"]["agg_rows"], 1)
        self.assertEqual(report["contract"]["partial_indices_rows"], 0)
        self.assertEqual(report["contract"]["time_order_violations"], {})
        self.assertEqual(report["contract"]["event_after_request_time"], {})
        self.assertEqual(report["contract"]["invalid_request_time_layout"], 0)
        self.assertEqual(report["contract"]["sku_alignment_mismatches"], 0)
        self.assertEqual(report["contract"]["invalid_labels"], {})
        self.assertEqual(
            report["contract"]["sequence_lengths_after_request_filter"]["impr"],
            {"count": 2, "min": 1, "p50": 1, "p95": 3, "p99": 3, "max": 3},
        )
        self.assertEqual(
            report["contract"]["candidate_scene_values"],
            [{"scene_id": 3, "count": 2}],
        )
        self.assertEqual(
            report["contract"]["label_distribution"]["a"],
            {
                "examples": 2,
                "positives": 1,
                "negatives": 1,
                "invalid": 0,
                "total": 2,
                "null": 0,
                "minus_one": 0,
                "zero": 1,
                "one": 1,
                "other": 0,
            },
        )
        self.assertEqual(report["contract"]["scene_values"], [
            {"scene_id": 3, "count": 1},
            {"scene_id": 4, "count": 1},
        ])
        self.assertEqual(report["fields"]["ctx_hn"]["nulls_by_depth"], {"1": 1})
        self.assertEqual(report["fields"]["impr_x_goods_id_hn"]["negative_count"], 3)
        self.assertEqual(
            report["shared_embedding_groups"]["goods_id_hn"]["sources"],
            ["goods_id_hn", "impr_x_goods_id_hn"],
        )
        overlap = report["shared_embedding_groups"]["goods_id_hn"][
            "bottom_k_pairwise_overlap"
        ]
        self.assertEqual(overlap[0]["sample_intersection"], 1)
        self.assertNotIn("impr_x_time", report["shared_embedding_groups"])


if __name__ == "__main__":
    unittest.main()
