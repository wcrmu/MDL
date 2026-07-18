from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import pyarrow as pa

from src.dataloader import (
    _column_array,
    _normalized_list_array,
    _select_sequence,
    _sequence_membership_positions,
    _time_deltas,
    adapt_mdl_rankmixer_parquet,
)
from src.train import _table_sequence_lengths


adapt = adapt_mdl_rankmixer_parquet


def _context(required: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        required_columns=tuple(required),
        options={
            "context_features": ["ctx_scalar_hn", "ctx_bag_hn"],
            "item_features": ["item_scalar_hn", "sku_a_hn", "sku_b_hn"],
            "multivalue_features": ["ctx_bag_hn", "sku_a_hn", "sku_b_hn"],
            "aligned_multivalue_groups": [["sku_a_hn", "sku_b_hn"]],
            "ups_types": ["impr"],
            "request_columns": ["scene_id", "search_id", "impr_time"],
            "integer_request_columns": ["scene_id", "impr_time"],
            "labels": {
                "a": "label_a",
                "b": "label_b",
                "c": "label_c",
            },
            "request_time_column": "impr_time",
            "time_delta_outputs": {"impr": "impr_x_time_delta_ms"},
        },
    )


REQUIRED = [
    "ctx_scalar_hn",
    "ctx_bag_hn",
    "item_scalar_hn",
    "sku_a_hn",
    "sku_b_hn",
    "impr_x_goods_id_hn",
    "impr_x_time_delta_ms",
    "scene_id",
    "search_id",
    "label_a",
    "label_b",
    "label_c",
]


class MDLRankMixerParquetAdapterTest(unittest.TestCase):
    def test_rejects_unknown_membership_and_non_monotonic_event_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "requests without context"):
            _sequence_membership_positions(
                [[0], [2]],
                known_requests={0, 1},
                index_column="impr_x_indices",
                raw_row=3,
            )
        with self.assertRaisesRegex(ValueError, "newest-to-oldest"):
            _time_deltas(
                [4000, 4500],
                5000,
                sequence="impr",
                raw_row=3,
                transform="raw_ms",
            )
        with self.assertRaisesRegex(ValueError, "empty sequence"):
            _select_sequence(
                [],
                None,
                expected_length=None,
                column="impr_x_goods_id_hn",
                raw_row=3,
            )
        with self.assertRaisesRegex(ValueError, "only the complete sequence may be null"):
            _select_sequence(
                [[None]],
                None,
                expected_length=None,
                column="impr_x_goods_id_hn",
                raw_row=3,
            )

    def test_vectorized_long_time_delta_path_preserves_semantics(self) -> None:
        events = [10_000 - index * 100 for index in range(128)]

        actual = _time_deltas(
            events,
            11_000,
            sequence="impr",
            raw_row=0,
            transform="log1p_seconds",
        )

        self.assertEqual(len(actual), 128)
        self.assertAlmostEqual(actual[0], math.log1p(1.0))
        self.assertAlmostEqual(actual[-1], math.log1p(13.7))
        broken = list(events)
        broken[80] = broken[79] + 1
        with self.assertRaisesRegex(ValueError, "newest-to-oldest"):
            _time_deltas(
                broken,
                11_000,
                sequence="impr",
                raw_row=0,
                transform="seconds",
            )

    def test_agg_expands_candidates_and_filters_shared_ups_membership(self) -> None:
        table = pa.table(
            {
                "context_indices": pa.array([[0, 1]], type=pa.list_(pa.int64())),
                "target_indices": pa.array([[0, 1, 1]], type=pa.list_(pa.int64())),
                "ctx_scalar_hn": pa.array(
                    [[[101], [102]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "ctx_bag_hn": pa.array(
                    [[[1, 2], None]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "item_scalar_hn": pa.array(
                    [[[201], [202], [203]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "sku_a_hn": pa.array(
                    [[[11, 12], [13], [14, 15]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "sku_b_hn": pa.array(
                    [[[21, None], [22], [23, 24]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "impr_x_goods_id_hn": pa.array(
                    [[-1, -2, -3]], type=pa.list_(pa.int64())
                ),
                "impr_x_time": pa.array(
                    [[4900, 4500, 3000]], type=pa.list_(pa.int64())
                ),
                "impr_x_indices": pa.array(
                    [[[0, 1], [1], [0]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "scene_id": pa.array([[7, 8]], type=pa.list_(pa.int64())),
                "search_id": pa.array([["r0", "r1"]], type=pa.list_(pa.string())),
                "impr_time": pa.array([[5000, 6000]], type=pa.list_(pa.int64())),
                "label_a": pa.array([[0, 1, 0]], type=pa.list_(pa.int64())),
                "label_b": pa.array([[1, 0, 1]], type=pa.list_(pa.int64())),
                "label_c": pa.array([[0, 0, 1]], type=pa.list_(pa.int64())),
            }
        )

        actual = adapt(table, context=_context(REQUIRED)).to_pydict()

        self.assertEqual(actual["ctx_scalar_hn"], [101, 102, 102])
        self.assertEqual(actual["ctx_bag_hn"], [[1, 2], None, None])
        self.assertEqual(actual["item_scalar_hn"], [201, 202, 203])
        self.assertEqual(actual["sku_b_hn"], [[21, None], [22], [23, 24]])
        self.assertEqual(
            actual["impr_x_goods_id_hn"],
            [[-1, -3], [-1, -2], [-1, -2]],
        )
        self.assertEqual(
            actual["impr_x_time_delta_ms"],
            [[100.0, 2000.0], [1100.0, 1500.0], [1100.0, 1500.0]],
        )
        self.assertEqual(actual["scene_id"], [7, 8, 8])
        self.assertEqual(actual["search_id"], ["r0", "r1", "r1"])
        self.assertEqual(actual["label_c"], [0, 0, 1])

    def test_compact_request_lists_survive_concat_and_decode(self) -> None:
        table = pa.table(
            {
                "context_indices": [[0, 1]],
                "target_indices": [[0, 1, 1]],
                "ctx_scalar_hn": [[[101], [102]]],
                "ctx_bag_hn": [[[1, 2], [3]]],
                "item_scalar_hn": [[[201], [202], [203]]],
                "sku_a_hn": [[[11], [12], [13]]],
                "sku_b_hn": [[[21], [22], [23]]],
                "impr_x_goods_id_hn": [[-1, -2]],
                "impr_x_time": [[4900, 4000]],
                "impr_x_indices": [[[0, 1], [1]]],
                "scene_id": [[7, 8]],
                "search_id": [["r0", "r1"]],
                "impr_time": [[5000, 6000]],
                "label_a": [[0, 1, 0]],
                "label_b": [[1, 0, 1]],
                "label_c": [[0, 0, 1]],
            }
        )
        context = _context(REQUIRED)
        context.options["compact_request_lists"] = True

        compact = adapt(table, context=context)

        self.assertTrue(pa.types.is_dictionary(compact.schema.field("ctx_bag_hn").type))
        self.assertTrue(
            pa.types.is_dictionary(compact.schema.field("impr_x_goods_id_hn").type)
        )
        self.assertEqual(
            compact["impr_x_goods_id_hn"].chunk(0).dictionary.to_pylist(),
            [[-1], [-1, -2]],
        )
        combined = pa.concat_tables([compact, compact])
        unified = _column_array(combined, "impr_x_goods_id_hn")
        self.assertTrue(pa.types.is_dictionary(unified.type))
        decoded = _normalized_list_array(combined, "impr_x_goods_id_hn")
        self.assertEqual(
            decoded.to_pylist(),
            [[-1], [-1, -2], [-1, -2]] * 2,
        )
        sequence = SimpleNamespace(
            fields=[SimpleNamespace(source="impr_x_goods_id_hn")],
            max_length=None,
        )
        self.assertEqual(
            _table_sequence_lengths(None, sequence, combined).tolist(),
            [1, 2, 2, 1, 2, 2],
        )

    def test_req_uses_all_ups_tokens_for_each_candidate(self) -> None:
        table = pa.table(
            {
                "ctx_scalar_hn": pa.array([[101]], type=pa.list_(pa.int64())),
                "ctx_bag_hn": pa.array([[1, 2]], type=pa.list_(pa.int64())),
                "item_scalar_hn": pa.array(
                    [[[201], [202]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "sku_a_hn": pa.array(
                    [[[11, 12], [13]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "sku_b_hn": pa.array(
                    [[[21, 22], [23]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "impr_x_goods_id_hn": pa.array([[-1, -2]], type=pa.list_(pa.int64())),
                "impr_x_time": pa.array([[4900, 4000]], type=pa.list_(pa.int64())),
                "scene_id": pa.array([7], type=pa.int64()),
                "search_id": pa.array(["r0"], type=pa.string()),
                "impr_time": pa.array([5000], type=pa.int64()),
                "label_a": pa.array([[0, 1]], type=pa.list_(pa.int64())),
                "label_b": pa.array([[1, 0]], type=pa.list_(pa.int64())),
                "label_c": pa.array([[0, 0]], type=pa.list_(pa.int64())),
            }
        )

        actual = adapt(table, context=_context(REQUIRED)).to_pydict()

        self.assertEqual(actual["ctx_scalar_hn"], [101, 101])
        self.assertEqual(actual["ctx_bag_hn"], [[1, 2], [1, 2]])
        self.assertEqual(actual["impr_x_goods_id_hn"], [[-1, -2], [-1, -2]])
        self.assertEqual(actual["impr_x_time_delta_ms"], [[100.0, 1000.0]] * 2)

        context = _context(REQUIRED)
        context.options["time_delta_transform"] = "log1p_seconds"
        transformed = adapt(table, context=context).to_pydict()[
            "impr_x_time_delta_ms"
        ]
        for row in transformed:
            self.assertAlmostEqual(row[0], math.log1p(0.1))
            self.assertAlmostEqual(row[1], math.log1p(1.0))

    def test_req_accepts_optional_single_request_axis_on_context(self) -> None:
        table = pa.table(
            {
                "ctx_scalar_hn": pa.array(
                    [[[101]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "ctx_bag_hn": pa.array(
                    [[[1, 2]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "item_scalar_hn": [[[201]]],
                "sku_a_hn": [[[11]]],
                "sku_b_hn": [[[21]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "scene_id": [7],
                "search_id": ["r0"],
                "impr_time": [5000],
                "label_a": [[0]],
                "label_b": [[1]],
                "label_c": [[0]],
            }
        )

        actual = adapt(table, context=_context(REQUIRED)).to_pydict()

        self.assertEqual(actual["ctx_scalar_hn"], [101])
        self.assertEqual(actual["ctx_bag_hn"], [[1, 2]])

    def test_req_flattens_nested_user_bag_and_singleton_s_tokens(self) -> None:
        table = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": pa.array(
                    [[[1], [2]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "item_scalar_hn": [[[201]]],
                "sku_a_hn": [[[11]]],
                "sku_b_hn": [[[21]]],
                "impr_x_goods_id_hn": pa.array(
                    [[[-1], [-2]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "impr_x_time": pa.array(
                    [[[4900], [4000]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "scene_id": [7],
                "search_id": ["r0"],
                "impr_time": [5000],
                "label_a": [[0]],
                "label_b": [[1]],
                "label_c": [[0]],
            }
        )

        actual = adapt(table, context=_context(REQUIRED)).to_pydict()

        self.assertEqual(actual["ctx_bag_hn"], [[1, 2]])
        self.assertEqual(actual["impr_x_goods_id_hn"], [[-1, -2]])
        self.assertEqual(actual["impr_x_time_delta_ms"], [[100.0, 1000.0]])

    def test_maps_raw_scene_ids_to_contiguous_model_indices(self) -> None:
        table = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[1]],
                "item_scalar_hn": [[[201]]],
                "sku_a_hn": [[[11]]],
                "sku_b_hn": [[[21]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "scene_id": [17],
                "search_id": ["r0"],
                "impr_time": [5000],
                "label_a": [[0]],
                "label_b": [[1]],
                "label_c": [[0]],
            }
        )
        context = _context(REQUIRED)
        context.options["request_value_maps"] = {"scene_id": {"17": 1, "9": 0}}

        actual = adapt(table, context=context).to_pydict()

        self.assertEqual(actual["scene_id"], [1])

        context.options["request_value_maps"] = {"scene_id": {9: 0}}
        with self.assertRaisesRegex(ValueError, "unmapped value 17"):
            adapt(table, context=context)

    def test_rejects_partial_indices_and_misaligned_sku_fields(self) -> None:
        partial = pa.table(
            {
                "context_indices": pa.array([[0]], type=pa.list_(pa.int64())),
                "ctx_scalar_hn": pa.array([[[1]]], type=pa.list_(pa.list_(pa.int64()))),
            }
        )
        with self.assertRaisesRegex(ValueError, "both context_indices and target_indices"):
            adapt(partial, context=_context(REQUIRED))

        req = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[1]],
                "item_scalar_hn": [[[201]]],
                "sku_a_hn": [[[11, 12]]],
                "sku_b_hn": [[[21]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "scene_id": [7],
                "search_id": ["r0"],
                "impr_time": [5000],
                "label_a": [[0]],
                "label_b": [[1]],
                "label_c": [[0]],
            }
        )
        with self.assertRaisesRegex(ValueError, "aligned multivalue group mismatch"):
            adapt(req, context=_context(REQUIRED))


if __name__ == "__main__":
    unittest.main()
