from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.profile_prehashed_parquet import MASK64, ProfileSpec, _mix64
from scripts.recommend_categorical_encodings import (
    StrategyFieldProfile,
    _bucket_id,
    _collision_stats,
    load_current_encodings,
    recommend_strategy,
    scan_for_recommendations,
)


class RecommendCategoricalEncodingsTest(unittest.TestCase):
    def test_remix_changes_low_bits_for_aligned_ids(self) -> None:
        values = tuple(i * 4096 for i in range(1, 65))
        raw = _collision_stats(values, 4096, remix=False)
        remixed = _collision_stats(values, 4096, remix=True, seed=123)
        self.assertEqual(raw["occupied"], 1)
        self.assertEqual(raw["collision_rate"], 1.0 - 1.0 / len(values))
        self.assertGreater(remixed["occupied"], 1)

    def test_identity_recommendation_for_dense_small_codes(self) -> None:
        profile = StrategyFieldProfile(sample_size=64, hll_precision=10)
        for value in range(1, 21):
            for _ in range(3):
                profile.observe(value)

        result = recommend_strategy(
            source="clk_cnt_1d_hn",
            profile=profile,
            current={"encoding_type": "pre_hashed", "num_buckets": 4096},
            candidate_buckets=(1024, 4096),
            input_count=3,
        )
        self.assertEqual(result["strategy"], "identity")
        self.assertEqual(result["suggested_num_buckets"], 21)

    def test_enlarge_when_current_buckets_below_distinct(self) -> None:
        profile = StrategyFieldProfile(sample_size=256, hll_precision=12)
        # Spread across uint64 so identity/vocab heuristics do not trigger.
        for index in range(1, 801):
            value = (index * 0x9E3779B97F4A7C15) & MASK64
            if value == 0:
                value = 1
            # Force into signed int64 range used by Arrow int64 leaves.
            if value >= 1 << 63:
                value -= 1 << 64
            profile.observe(value)

        result = recommend_strategy(
            source="i2i_list_swingv3gmv_hn_share",
            profile=profile,
            current={"encoding_type": "pre_hashed", "num_buckets": 256},
            candidate_buckets=(256, 1024, 4096, 8192),
            input_count=2,
        )
        self.assertEqual(result["strategy"], "enlarge_pre_hashed")
        self.assertGreaterEqual(result["suggested_num_buckets"], 1024)

    def test_within_row_collision_tracks_bag_collapse(self) -> None:
        profile = StrategyFieldProfile(
            sample_size=32,
            hll_precision=10,
            within_row_buckets=(4,),
        )
        # Four distinct values that all share the same low 2 bits under raw mask.
        profile.observe([4, 8, 12, 16])
        self.assertEqual(profile.within_row_groups, 1)
        self.assertEqual(profile.within_row_raw_mapped[4], 1)
        remixed_mapped = profile.within_row_remix_mapped[4]
        self.assertGreaterEqual(remixed_mapped, 1)

    def test_scan_end_to_end_writes_strategies(self) -> None:
        spec = ProfileSpec(
            all_sources=("bucket_hn", "aligned_hn"),
            categorical_sources=("bucket_hn", "aligned_hn"),
            time_sources=(),
            context_sources=("bucket_hn",),
            item_sources=("aligned_hn",),
            sequence_sources={},
            sequence_time_sources={},
            label_sources={},
            shared_groups={},
            sku_fields=(),
            scene_source="scene_id",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part.parquet"
            pq.write_table(
                pa.table(
                    {
                        "bucket_hn": [[1], [2], [3], [1]],
                        "aligned_hn": [
                            [(1 << 12) + 1],
                            [(2 << 12) + 1],
                            [(3 << 12) + 1],
                            [(4 << 12) + 1],
                        ],
                        "scene_id": [7, 7, 8, 8],
                    }
                ),
                path,
            )
            report = scan_for_recommendations(
                [str(path)],
                spec,
                {
                    "bucket_hn": {
                        "encoding_type": "pre_hashed",
                        "num_buckets": 4096,
                        "embedding_dim": 16,
                    },
                    "aligned_hn": {
                        "encoding_type": "pre_hashed",
                        "num_buckets": 4,
                        "embedding_dim": 8,
                    },
                },
                candidate_buckets=(4, 16, 64, 256, 1024, 4096),
                sample_size=32,
                hll_precision=10,
                progress=False,
            )

        by_source = {item["source"]: item for item in report["recommendations"]}
        self.assertEqual(by_source["bucket_hn"]["strategy"], "identity")
        self.assertIn(by_source["aligned_hn"]["strategy"], {
            "enlarge_pre_hashed",
            "remixed_pre_hashed",
            "keep_pre_hashed",
            "head_tail",
            "vocab_candidate",
        })
        self.assertEqual(report["rows_scanned"], 4)
        # Aligned IDs all map to bucket 1 under raw mask size 4.
        aligned_cmp = {
            row["bucket_size"]: row
            for row in by_source["aligned_hn"]["bucket_comparisons"]
        }
        self.assertEqual(aligned_cmp[4]["raw"]["occupied"], 1)
        self.assertGreater(aligned_cmp[4]["remixed"]["occupied"], 1)

    def test_load_current_encodings_reads_features_and_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cfg.yaml"
            path.write_text(
                "\n".join(
                    [
                        "features:",
                        "  - name: sku_price_v2_hn",
                        "    source: sku_price_v2_hn",
                        "    embedding_dim: 16",
                        "    encoding:",
                        "      type: pre_hashed",
                        "      num_buckets: 4096",
                        "sequences:",
                        "  - name: impr",
                        "    fields:",
                        "      - name: goods_id_hn",
                        "        source: goods_id_hn",
                        "        embedding_dim: 48",
                        "        encoding:",
                        "          type: pre_hashed",
                        "          num_buckets: 134217728",
                        "          share_embedding: true",
                        "          share_with: goods_id_hn",
                    ]
                ),
                encoding="utf-8",
            )
            encodings = load_current_encodings(path)
        self.assertEqual(encodings["sku_price_v2_hn"]["num_buckets"], 4096)
        self.assertEqual(encodings["goods_id_hn"]["num_buckets"], 134217728)
        self.assertTrue(encodings["goods_id_hn"]["share_embedding"])

    def test_bucket_id_remix_is_seeded(self) -> None:
        value = 123456789
        left = _bucket_id(value, 1024, remix=True, seed=1)
        right = _bucket_id(value, 1024, remix=True, seed=2)
        self.assertNotEqual(left, right)
        self.assertEqual(
            _bucket_id(value, 1024, remix=False),
            value & 1023,
        )
        # Stable against the shared mixer.
        self.assertEqual(
            _bucket_id(value, 1024, remix=True, seed=7),
            _mix64((value & MASK64) ^ 7) & 1023,
        )


if __name__ == "__main__":
    unittest.main()
