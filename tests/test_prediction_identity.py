from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from src.config import LengthBucketConfig, ParquetAdapterConfig, load_app_config
from src.dataloader import (
    FeatureBatch,
    adapt_mdl_rankmixer_parquet,
    iter_flat_tables,
)
from src.train import _iter_batch_tables, predict_mdl


ROOT = Path(__file__).resolve().parents[1]


class _IdentityScoreModel(nn.Module):
    def forward(
        self,
        features: dict[str, torch.Tensor],
        scenario_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del scenario_id
        return {"logits": features["logits"]}


class PredictionIdentityTest(unittest.TestCase):
    def test_unlabeled_req_file_scans_through_the_builtin_adapter(self) -> None:
        base = load_app_config(ROOT / "configs" / "default.yaml")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "req.parquet"
            pq.write_table(
                pa.table(
                    {
                        "goods_id_hn": [[[10], [20]]],
                        "search_id": ["r0"],
                        "impr_time": [5000],
                        "example_ids": [["A", "B"]],
                    }
                ),
                path,
            )
            split = replace(
                base.data.test,
                format="adapter_parquet",
                inputs=(str(path),),
                request_id="search_id",
                group_id="search_id",
                labels={"click": "label_click"},
                label_masks={"click": "label_click__valid"},
                prediction_keys={
                    "search_id": "search_id",
                    "candidate_position": "candidate_position",
                    "example_id": "example_ids",
                    "goods_id_hn": "goods_id_hn",
                },
                adapter=ParquetAdapterConfig(
                    callable="src.dataloader:adapt_mdl_rankmixer_parquet",
                    input_columns=(
                        "goods_id_hn",
                        "search_id",
                        "impr_time",
                        "label_click",
                    ),
                    optional_input_columns=("example_ids",),
                    options={
                        "context_features": [],
                        "item_features": ["goods_id_hn"],
                        "multivalue_features": [],
                        "ups_types": [],
                        "request_columns": ["search_id", "impr_time"],
                        "integer_request_columns": ["impr_time"],
                        "labels": {"click": "label_click"},
                        "label_masks": {"click": "label_click__valid"},
                        "label_missing_values": [None],
                        "candidate_position_column": "candidate_position",
                        "candidate_metadata_columns": ["example_ids"],
                    },
                ),
            )
            feature = replace(
                base.features[0],
                name="goods_id_hn",
                source="goods_id_hn",
            )
            config = replace(
                base,
                data=replace(base.data, test=split),
                features=(feature,),
                sequences=(),
            )

            flat = next(iter_flat_tables(config, "test", require_labels=False))

        self.assertEqual(flat["search_id"].to_pylist(), ["r0", "r0"])
        self.assertEqual(flat["candidate_position"].to_pylist(), [0, 1])
        self.assertEqual(flat["example_ids"].to_pylist(), ["A", "B"])
        self.assertNotIn("label_click", flat.column_names)

    def test_adapter_bucket_and_predict_preserve_candidate_join_keys(self) -> None:
        raw = pa.table(
            {
                "context_indices": [[0, 1]],
                # Physical candidate order is A, C, B.
                "target_indices": [[0, 1, 0]],
                "ctx_hn": [[[101], [102]]],
                "goods_id_hn": [[[10], [30], [20]]],
                "impr_x_goods_id_hn": [[100, 101, 102, 103, 104]],
                "impr_x_indices": [[[0, 1], [1], [1], [1], [1]]],
                "scene_id": [[7, 7]],
                "search_id": [[100, 200]],
                "impr_time": [[10_000, 10_000]],
                "example_ids": [["A", "C", "B"]],
            }
        )
        required = (
            "ctx_hn",
            "goods_id_hn",
            "impr_x_goods_id_hn",
            "scene_id",
            "search_id",
            "candidate_position",
            "example_ids",
        )
        context = SimpleNamespace(
            required_columns=required,
            options={
                "context_features": ["ctx_hn"],
                "item_features": ["goods_id_hn"],
                "multivalue_features": [],
                "ups_types": ["impr"],
                "request_columns": ["scene_id", "search_id", "impr_time"],
                "integer_request_columns": ["scene_id", "impr_time"],
                # The split supports evaluation labels, but this req prediction
                # table intentionally omits them.
                "labels": {"click": "label_click"},
                "label_masks": {"click": "label_click__valid"},
                "label_missing_values": [None],
                "candidate_position_column": "candidate_position",
                "candidate_metadata_columns": ["example_ids"],
            },
        )
        flat = adapt_mdl_rankmixer_parquet(raw, context=context)
        self.assertEqual(flat["candidate_position"].to_pylist(), [0, 0, 1])

        base = load_app_config(ROOT / "configs" / "default.yaml")
        sequence = replace(
            base.sequences[0],
            max_length=10,
            fields=(
                replace(
                    base.sequences[0].fields[0],
                    source="impr_x_goods_id_hn",
                ),
            ),
        )
        test_split = replace(
            base.data.test,
            labels={},
            label_masks={},
            prediction_keys={
                "search_id": "search_id",
                "candidate_position": "candidate_position",
                "example_id": "example_ids",
                "goods_id_hn": "goods_id_hn",
            },
            prediction_score_suffix="_score",
            reader=replace(
                base.data.test.reader,
                length_buckets=(
                    LengthBucketConfig(max_length=2, batch_size=4),
                    LengthBucketConfig(max_length=None, batch_size=4),
                ),
            ),
        )
        config = replace(
            base,
            runtime=replace(
                base.runtime,
                device="cpu",
                precision="fp32",
                compile=False,
            ),
            data=replace(base.data, test=test_split),
            sequences=(sequence,),
            training=replace(base.training, checkpoint_path=None),
        )

        with patch("src.train.iter_candidate_tables", return_value=iter([flat])):
            bucketed = list(
                _iter_batch_tables(
                    config,
                    "test",
                    0,
                    1,
                    require_labels=False,
                )
            )
        self.assertEqual(
            [
                value
                for table in bucketed
                for value in table["example_ids"].to_pylist()
            ],
            ["A", "B", "C"],
        )

        logit_by_goods = {10: -1.0, 20: 0.0, 30: 1.0}
        batches: list[FeatureBatch] = []
        for table in bucketed:
            goods = table["goods_id_hn"].to_pylist()
            batches.append(
                FeatureBatch(
                    features={
                        "logits": torch.tensor(
                            [[logit_by_goods[value]] for value in goods],
                            dtype=torch.float32,
                        )
                    },
                    labels=None,
                    label_mask=None,
                    scenario_id=torch.zeros(table.num_rows, dtype=torch.long),
                    group_id=[str(value) for value in table["search_id"].to_pylist()],
                    prediction_keys={
                        output: table[source].to_pylist()
                        for output, source in test_split.prediction_keys.items()
                    },
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "predictions.parquet"
            with (
                patch("src.train.load_vocab_maps", return_value={}),
                patch("src.train.build_model", return_value=_IdentityScoreModel()),
                patch("src.train.iter_feature_batches", return_value=iter(batches)),
            ):
                result = predict_mdl(
                    config,
                    output_path=str(output_path),
                    allow_random_init=True,
                )
            output = pq.read_table(output_path).to_pylist()

        self.assertEqual(result.rows, 3)
        restored = {
            (row["search_id"], row["candidate_position"]): row
            for row in output
        }
        self.assertEqual(restored[(100, 0)]["example_id"], "A")
        self.assertEqual(restored[(100, 1)]["example_id"], "B")
        self.assertEqual(restored[(200, 0)]["example_id"], "C")
        self.assertAlmostEqual(
            restored[(100, 0)]["click_score"],
            1.0 / (1.0 + math.exp(1.0)),
        )
        self.assertEqual(restored[(100, 1)]["goods_id_hn"], 20)


if __name__ == "__main__":
    unittest.main()
