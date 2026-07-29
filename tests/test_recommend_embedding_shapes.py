from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.recommend_embedding_shapes import (
    build_recommendations,
    estimate_growth,
    load_json_report,
)
from scripts.build_mdl_rankmixer_config import (
    PROFILE_DRIVEN_EMBEDDING_SHAPES,
    _BOOTSTRAP_PROFILE_DRIVEN_EMBEDDING_SHAPES,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RecommendEmbeddingShapesTest(unittest.TestCase):
    def test_checked_report_and_bootstrap_shapes_stay_in_sync(self) -> None:
        self.assertEqual(
            PROFILE_DRIVEN_EMBEDDING_SHAPES,
            _BOOTSTRAP_PROFILE_DRIVEN_EMBEDDING_SHAPES,
        )

    def test_load_json_report_accepts_platform_log_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(
                'time="2026-07-27" level=info msg="launch"\n'
                'profiled file: 10 rows\n'
                '{"rows_scanned": 10, "fields": {}}\n',
                encoding="utf-8",
            )
            self.assertEqual(load_json_report(path)["rows_scanned"], 10)

    def test_growth_alpha_is_clamped_for_projection(self) -> None:
        estimate = estimate_growth(
            small_distinct=10,
            large_distinct=100,
            small_files=10,
            large_files=100,
            target_files=500,
            stress_files=12_000,
        )
        self.assertEqual(estimate.tier, "climbing")
        self.assertAlmostEqual(estimate.raw_alpha, 1.0)
        self.assertAlmostEqual(estimate.alpha, 0.95)
        self.assertEqual(
            estimate.projected_distinct,
            462,
        )

    def test_real_profiles_use_24_hours_budget_and_shared_union(self) -> None:
        small_path = REPOSITORY_ROOT / "docs/profile.json"
        large_path = REPOSITORY_ROOT / "docs/new_profile.json"
        config_path = REPOSITORY_ROOT / "configs/mdl_rankmixer.yaml"
        report = build_recommendations(
            load_json_report(small_path),
            load_json_report(large_path),
            config_path=config_path,
            small_path="docs/profile.json",
            large_path="docs/new_profile.json",
            files_per_hour=500,
            target_hours=24.0,
            stress_hours=168.0,
        )

        self.assertEqual(report["method"]["primary_horizon"]["files"], 12_000)
        self.assertEqual(
            report["method"]["retention_floor_horizon"]["files"],
            500,
        )
        self.assertEqual(report["method"]["stress_horizon"]["files"], 84_000)
        self.assertEqual(report["summary"]["profiled_fields"], 277)
        self.assertEqual(report["summary"]["shared_groups"], 10)

        shapes = report["recommended_shapes"]
        self.assertEqual(
            shapes["goods_id_hn"],
            {
                "num_buckets": 1 << 28,
                "embedding_dim": 32,
                "ideal_num_buckets": 1 << 30,
                "ideal_embedding_dim": 64,
                "basis": "shared_group_union",
                "projected_distinct": 315_767_932,
                "projected_load": 1.176327,
                "projected_uniform_collision": 0.412076,
            },
        )
        self.assertEqual(shapes["page_sn_hn"]["num_buckets"], 1 << 12)
        self.assertEqual(
            shapes["cart_long_x_sku_ids_hn"]["num_buckets"],
            1 << 29,
        )
        self.assertEqual(
            shapes["cart_long_x_sku_ids_hn"]["embedding_dim"],
            32,
        )
        self.assertEqual(
            report["summary"]["configured_sources_excluded"],
            [
                "buy_long_x_time_delta_log1p_seconds",
                "cart_long_x_time_delta_log1p_seconds",
                "clk_long_x_time_delta_log1p_seconds",
                "coarse_scene_prior_id",
                "flatten_query_hash_x_time_delta_log1p_seconds",
                "impr_x_time_delta_log1p_seconds",
                "semi_clk_x_time_delta_log1p_seconds",
                "srch_q2i_x_time_delta_log1p_seconds",
                "ups_clk_sku_x_time_delta_log1p_seconds",
                "view_long_x_time_delta_log1p_seconds",
            ],
        )


if __name__ == "__main__":
    unittest.main()
