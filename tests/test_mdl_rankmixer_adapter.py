from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa

from src.dataloader import (
    COARSE_SCENE_INDEX_COLUMN,
    COARSE_SCENE_PRIOR_ID_COLUMN,
    FeatureCardinalityAuditor,
    _adapter_table_to_python,
    _arrow_array_to_pylist,
    _column_array,
    _normalize_optional_outer_list,
    _normalized_list_array,
    _scalarize,
    _select_sequence,
    _sequence_membership_positions,
    _time_deltas,
    _validate_complete_label_contract,
    adapt_mdl_rankmixer_parquet,
    coarse_scene_ids,
)
from src.train import _table_sequence_lengths


adapt = adapt_mdl_rankmixer_parquet

SEARCH_IDS = frozenset({2, 21, 23, 1137})


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
    def test_nested_numeric_arrow_fast_path_preserves_nulls_and_empty_lists(self) -> None:
        array = pa.array(
            [[[1, 2], None, []], None, [[3]]],
            type=pa.list_(pa.list_(pa.int64())),
        )

        self.assertEqual(_arrow_array_to_pylist(pa, array), array.to_pylist())

    def test_trusted_adapter_validates_one_raw_row_then_uses_fast_path(self) -> None:
        one_row = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[1, 2]],
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
        table = pa.concat_tables([one_row, one_row])
        context = _context(REQUIRED)
        context.trusted_input = True
        context._runtime_cache = {}

        with patch(
            "src.dataloader._adapter_table_to_python",
            wraps=_adapter_table_to_python,
        ) as convert:
            adapt(table, context=context)
            self.assertEqual(
                [
                    (call.args[0].num_rows, call.kwargs["validate_contract"])
                    for call in convert.call_args_list
                ],
                [(1, True), (2, False)],
            )

            convert.reset_mock()
            adapt(table, context=context)
            self.assertEqual(
                [
                    (call.args[0].num_rows, call.kwargs["validate_contract"])
                    for call in convert.call_args_list
                ],
                [(2, False)],
            )

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
        with self.assertRaisesRegex(ValueError, "empty membership"):
            _sequence_membership_positions(
                [[], [0]],
                known_requests={0, 1},
                index_column="impr_x_indices",
                raw_row=3,
            )
        self.assertEqual(
            _select_sequence(
                [],
                None,
                expected_length=None,
                column="impr_x_goods_id_hn",
                raw_row=3,
            ),
            [],
        )
        self.assertEqual(
            _select_sequence(
                None,
                None,
                expected_length=None,
                column="impr_x_goods_id_hn",
                raw_row=3,
            ),
            [],
        )
        self.assertEqual(
            _select_sequence(
                [[None]],
                None,
                expected_length=None,
                column="impr_x_goods_id_hn",
                raw_row=3,
            ),
            [None],
        )
        self.assertEqual(
            _normalize_optional_outer_list(None),
            [],
        )
        self.assertEqual(
            _normalize_optional_outer_list([]),
            [],
        )
        # Helper must not recurse into memberships.
        nested = [[], [0]]
        self.assertEqual(_normalize_optional_outer_list(nested), [[], [0]])
        self.assertIs(_normalize_optional_outer_list(nested)[0], nested[0])

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
                "example_ids": pa.array(
                    [["e0", "e1", "e2"]], type=pa.list_(pa.string())
                ),
            }
        )

        context = _context([*REQUIRED, "candidate_position", "example_ids"])
        context.options["candidate_position_column"] = "candidate_position"
        context.options["candidate_metadata_columns"] = ["example_ids"]
        actual = adapt(table, context=context).to_pydict()

        self.assertEqual(actual["ctx_scalar_hn"], [101, 102, 102])
        self.assertEqual(actual["ctx_bag_hn"], [[1, 2], [], []])
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
        self.assertEqual(actual["candidate_position"], [0, 0, 1])
        self.assertEqual(actual["example_ids"], ["e0", "e1", "e2"])

    def test_zero_payloads_do_not_compact_complete_agg_axes(self) -> None:
        table = pa.table(
            {
                "context_indices": [[0, 1, 2, 3, 4]],
                "target_indices": [[0, 1, 2, 3, 4, 0, 1]],
                "ctx_scalar_hn": [[[101], [102], [0], [0], [0]]],
                "ctx_bag_hn": [[[1], [2], [0], [0], [0]]],
                "item_scalar_hn": [
                    [[201], [202], [0], [0], [0], [0], [0]]
                ],
                "sku_a_hn": [[[11], [12], [0], [0], [0], [0], [0]]],
                "sku_b_hn": [[[21], [22], [0], [0], [0], [0], [0]]],
                "impr_x_goods_id_hn": [[-1, -2, 0, 0]],
                "impr_x_time": [[4900, 4800, 4700, 4600]],
                "impr_x_indices": [[[0], [1], [2], [3]]],
                "scene_id": [[7, 8, 0, 0, 0]],
                "search_id": [["r0", "r1", "r2", "r3", "r4"]],
                "impr_time": [[5000, 5000, 5000, 5000, 5000]],
                "label_a": [[0, 0, 0, 0, 0, 0, 0]],
                "label_b": [[0, 0, 0, 0, 0, 0, 0]],
                "label_c": [[0, 0, 0, 0, 0, 0, 0]],
            }
        )

        actual = adapt(table, context=_context(REQUIRED)).to_pydict()

        self.assertEqual(len(actual["search_id"]), 7)
        self.assertEqual(actual["ctx_scalar_hn"], [101, 102, 0, 0, 0, 101, 102])
        self.assertEqual(actual["item_scalar_hn"], [201, 202, 0, 0, 0, 0, 0])
        self.assertEqual(actual["impr_x_goods_id_hn"][2], [0])
        self.assertEqual(actual["label_a"], [0, 0, 0, 0, 0, 0, 0])

    def test_missing_labels_emit_independent_masks_and_aliases_are_exact(self) -> None:
        table = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[1, 2]],
                "item_scalar_11_hn": [[[201], [202]]],
                "sku_a_hn": [[[11], [12]]],
                "sku_b_hn": [[[21], [22]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "scene_id": [7],
                "search_id": ["r0"],
                "impr_time": [5000],
                "label_a": pa.array([[0.0, None]], type=pa.list_(pa.float64())),
                "label_b": [[-1, 1]],
                "label_c": [[1, 0]],
            }
        )
        masks = {task: f"label_{task}_valid" for task in ("a", "b", "c")}
        context = _context([*REQUIRED, *masks.values()])
        context.options["label_masks"] = masks
        context.options["label_missing_values"] = {
            "a": [None],
            "b": [-1],
            "c": [],
        }
        context.options["column_aliases"] = {
            "item_scalar_hn": ["item_scalar_11_hn"]
        }

        actual = adapt(table, context=context).to_pydict()

        self.assertEqual(actual["item_scalar_hn"], [201, 202])
        self.assertEqual(actual["label_a"], [0, None])
        self.assertEqual(actual["label_b"], [None, 1])
        self.assertEqual(actual["label_c"], [1, 0])
        self.assertEqual(actual["label_a_valid"], [1, 0])
        self.assertEqual(actual["label_b_valid"], [0, 1])
        self.assertEqual(actual["label_c_valid"], [1, 1])

        ambiguous = table.append_column("item_scalar_hn", table["item_scalar_11_hn"])
        with self.assertRaisesRegex(ValueError, "multiple aliases"):
            adapt(ambiguous, context=context)

        null_outer = table.set_column(
            table.schema.get_field_index("label_b"),
            "label_b",
            pa.array([None], type=pa.list_(pa.int64())),
        )
        context.options["label_missing_values"]["b"] = [None]
        null_result = adapt(null_outer, context=context).to_pydict()
        self.assertEqual(null_result["label_b"], [None, None])
        self.assertEqual(null_result["label_b_valid"], [0, 0])

    def test_all_null_candidate_metadata_keeps_its_physical_scalar_type(self) -> None:
        table = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[1, 2]],
                "item_scalar_hn": [[[201], [202]]],
                "sku_a_hn": [[[11], [12]]],
                "sku_b_hn": [[[21], [22]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "scene_id": [7],
                "search_id": ["r0"],
                "impr_time": [5000],
                "label_a": [[0, 1]],
                "label_b": [[1, 0]],
                "label_c": [[0, 1]],
                "example_ids": pa.array(
                    [[None, None]],
                    type=pa.list_(pa.string()),
                ),
            }
        )
        context = _context([*REQUIRED, "example_ids"])
        context.options["candidate_metadata_columns"] = ["example_ids"]

        actual = adapt(table, context=context)

        self.assertEqual(actual["example_ids"].to_pylist(), [None, None])
        self.assertTrue(pa.types.is_string(actual.schema.field("example_ids").type))
        combined = pa.concat_tables([actual, actual])
        self.assertEqual(combined.num_rows, 4)

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

    def test_req_truncates_before_time_delta_and_ignores_discarded_history(self) -> None:
        table = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[1, 2]],
                "item_scalar_hn": [[[201]]],
                "sku_a_hn": [[[11]]],
                "sku_b_hn": [[[21]]],
                "impr_x_goods_id_hn": [[-1, -2, -3, -4]],
                # The discarded suffix violates both temporal contracts. It
                # must never be inspected or transformed after head truncation.
                "impr_x_time": [[4900, 4000, 4500, 6000]],
                "scene_id": [7],
                "search_id": ["r0"],
                "impr_time": [5000],
                "label_a": [[0]],
                "label_b": [[1]],
                "label_c": [[0]],
            }
        )
        context = _context(REQUIRED)
        context.options["sequence_max_lengths"] = {"impr": 2}

        actual = adapt(table, context=context).to_pydict()

        self.assertEqual(actual["impr_x_goods_id_hn"], [[-1, -2]])
        self.assertEqual(actual["impr_x_time_delta_ms"], [[100.0, 1000.0]])

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

    def test_coarse_scene_dual_columns_and_rejects_invalid_raw_ids(self) -> None:
        self.assertEqual(coarse_scene_ids(2, SEARCH_IDS), (0, 1))
        self.assertEqual(coarse_scene_ids(21, SEARCH_IDS), (0, 1))
        self.assertEqual(coarse_scene_ids(1137, SEARCH_IDS), (0, 1))
        self.assertEqual(coarse_scene_ids(0, SEARCH_IDS), (1, 2))
        self.assertEqual(coarse_scene_ids(7, SEARCH_IDS), (1, 2))
        self.assertEqual(coarse_scene_ids(1138, SEARCH_IDS), (1, 2))
        self.assertNotIn(0, {1, 2})
        for invalid in (None, True, "abc"):
            with self.assertRaisesRegex(ValueError, "scene_id must be an integer"):
                coarse_scene_ids(invalid, SEARCH_IDS)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            coarse_scene_ids(-1, SEARCH_IDS)
        with self.assertRaisesRegex(ValueError, "not in the configured search allowlist"):
            coarse_scene_ids(7, SEARCH_IDS, unlisted_policy="error")
        self.assertEqual(
            coarse_scene_ids(21, SEARCH_IDS, unlisted_policy="error"),
            (0, 1),
        )

        required = [
            *REQUIRED,
            COARSE_SCENE_INDEX_COLUMN,
            COARSE_SCENE_PRIOR_ID_COLUMN,
        ]
        table = pa.table(
            {
                "ctx_scalar_hn": [[101], [102]],
                "ctx_bag_hn": [[1], [2]],
                "item_scalar_hn": [[[201]], [[202]]],
                "sku_a_hn": [[[11]], [[12]]],
                "sku_b_hn": [[[21]], [[22]]],
                "impr_x_goods_id_hn": [[-1], [-2]],
                "impr_x_time": [[4900], [4800]],
                "scene_id": [21, 7],
                "search_id": ["r0", "r1"],
                "impr_time": [5000, 5001],
                "label_a": [[0], [1]],
                "label_b": [[1], [0]],
                "label_c": [[0], [0]],
            }
        )
        context = _context(required)
        context.options["search_scene_ids"] = sorted(SEARCH_IDS)

        actual = adapt(table, context=context).to_pydict()
        self.assertEqual(actual["scene_id"], [21, 7])
        self.assertEqual(actual[COARSE_SCENE_INDEX_COLUMN], [0, 1])
        self.assertEqual(actual[COARSE_SCENE_PRIOR_ID_COLUMN], [1, 2])

        bad = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[1]],
                "item_scalar_hn": [[[201]]],
                "sku_a_hn": [[[11]]],
                "sku_b_hn": [[[21]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "scene_id": [None],
                "search_id": ["r0"],
                "impr_time": [5000],
                "label_a": [[0]],
                "label_b": [[1]],
                "label_c": [[0]],
            }
        )
        with self.assertRaisesRegex(ValueError, "scene_id must be an integer"):
            adapt(bad, context=context)

    def test_coarse_scene_mapping_runs_once_per_request_not_per_candidate(self) -> None:
        required = [
            *REQUIRED,
            COARSE_SCENE_INDEX_COLUMN,
            COARSE_SCENE_PRIOR_ID_COLUMN,
        ]
        table = pa.table(
            {
                "context_indices": [[0, 1]],
                "target_indices": [[0, 1, 1]],
                "ctx_scalar_hn": [[[101], [102]]],
                "ctx_bag_hn": [[[1], [2]]],
                "item_scalar_hn": [[[201], [202], [203]]],
                "sku_a_hn": [[[11], [12], [13]]],
                "sku_b_hn": [[[21], [22], [23]]],
                "impr_x_goods_id_hn": [[-1, -2]],
                "impr_x_time": [[4900, 4800]],
                "impr_x_indices": [[[0, 1], [1]]],
                "scene_id": [[21, 7]],
                "search_id": [["r0", "r1"]],
                "impr_time": [[5000, 5001]],
                "label_a": [[0, 1, 0]],
                "label_b": [[1, 0, 1]],
                "label_c": [[0, 0, 1]],
            }
        )
        context = _context(required)
        context.options["search_scene_ids"] = sorted(SEARCH_IDS)

        with patch(
            "src.dataloader.coarse_scene_ids",
            wraps=coarse_scene_ids,
        ) as mapped:
            actual = adapt(table, context=context).to_pydict()

        self.assertEqual(actual[COARSE_SCENE_INDEX_COLUMN], [0, 1, 1])
        self.assertEqual(actual[COARSE_SCENE_PRIOR_ID_COLUMN], [1, 2, 2])
        self.assertEqual(mapped.call_count, 2)

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

    def test_empty_outer_bag_and_ups_are_zero_length(self) -> None:
        table = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[]],
                "item_scalar_hn": [[[201]]],
                "sku_a_hn": [[[11]]],
                "sku_b_hn": [[[21]]],
                "impr_x_goods_id_hn": [[]],
                "impr_x_time": [[]],
                "scene_id": [7],
                "search_id": ["r0"],
                "impr_time": [5000],
                "label_a": [[0]],
                "label_b": [[1]],
                "label_c": [[0]],
            }
        )
        actual = adapt(table, context=_context(REQUIRED)).to_pydict()
        self.assertEqual(actual["ctx_bag_hn"], [[]])
        self.assertEqual(actual["impr_x_goods_id_hn"], [[]])
        self.assertEqual(actual["impr_x_time_delta_ms"], [[]])

        null_outer = table.set_column(
            table.schema.get_field_index("ctx_bag_hn"),
            "ctx_bag_hn",
            pa.array([None], type=pa.list_(pa.int64())),
        )
        null_outer = null_outer.set_column(
            null_outer.schema.get_field_index("impr_x_goods_id_hn"),
            "impr_x_goods_id_hn",
            pa.array([None], type=pa.list_(pa.int64())),
        )
        null_outer = null_outer.set_column(
            null_outer.schema.get_field_index("impr_x_time"),
            "impr_x_time",
            pa.array([None], type=pa.list_(pa.int64())),
        )
        null_actual = adapt(null_outer, context=_context(REQUIRED)).to_pydict()
        self.assertEqual(null_actual["ctx_bag_hn"], [[]])
        self.assertEqual(null_actual["impr_x_goods_id_hn"], [[]])

    def test_scalarize_empty_list_is_missing_but_longer_lists_fail(self) -> None:
        self.assertIsNone(
            _scalarize([], column="search_method_hn", raw_row=0, logical_row=0)
        )
        self.assertIsNone(
            _scalarize(None, column="search_method_hn", raw_row=0, logical_row=0)
        )
        self.assertEqual(
            _scalarize([7], column="search_method_hn", raw_row=0, logical_row=0),
            7,
        )
        with self.assertRaisesRegex(ValueError, "inner length 5"):
            _scalarize(
                [1, 2, 3, 4, 5],
                column="goods_id_hn",
                raw_row=0,
                logical_row=0,
            )
        # Trusted hot path must never silently take value[0].
        with self.assertRaisesRegex(ValueError, "inner length 2"):
            _scalarize(
                [9, 8],
                column="price_hn",
                raw_row=1,
                logical_row=0,
                validate_contract=False,
            )

    def test_soft_cardinality_auditor_collects_all_scalar_multis(self) -> None:
        auditor = FeatureCardinalityAuditor(soft=True)
        self.assertIsNone(
            _scalarize(
                [],
                column="search_method_hn",
                raw_row=0,
                logical_row=0,
                auditor=auditor,
            )
        )
        self.assertIsNone(
            _scalarize(
                [1, 2, 3, 4, 5],
                column="goods_id_hn",
                raw_row=0,
                logical_row=0,
                auditor=auditor,
            )
        )
        self.assertIsNone(
            _scalarize(
                [11, 12, 13],
                column="price_hn",
                raw_row=0,
                logical_row=1,
                auditor=auditor,
            )
        )
        self.assertEqual(
            _scalarize(
                [7],
                column="search_method_hn",
                raw_row=1,
                logical_row=0,
                auditor=auditor,
            ),
            7,
        )
        self.assertTrue(auditor.has_scalar_multis())
        self.assertEqual(auditor.scalar_stats["search_method_hn"].empty_count, 1)
        self.assertEqual(auditor.scalar_stats["search_method_hn"].singleton_count, 1)
        self.assertEqual(auditor.scalar_stats["search_method_hn"].multi_count, 0)
        self.assertEqual(auditor.scalar_stats["goods_id_hn"].multi_count, 1)
        self.assertEqual(auditor.scalar_stats["price_hn"].multi_count, 1)
        report = auditor.format_report()
        self.assertIn("goods_id_hn", report)
        self.assertIn("price_hn", report)

        peer = FeatureCardinalityAuditor(soft=False)
        peer.observe_scalar("cat1_id_hn", [4, 5])
        peer.note_raw_rows(3)
        auditor.merge_payload(peer.to_payload())
        self.assertEqual(auditor.raw_rows_seen, 3)
        self.assertEqual(auditor.scalar_stats["cat1_id_hn"].multi_count, 1)

    def test_soft_auditor_on_adapter_does_not_fail_fast_on_first_multi(self) -> None:
        table = pa.table(
            {
                "context_indices": pa.array([[0]], type=pa.list_(pa.int64())),
                "target_indices": pa.array([[0]], type=pa.list_(pa.int64())),
                "ctx_scalar_hn": pa.array(
                    [[[101, 102]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "ctx_bag_hn": pa.array(
                    [[[1, 2]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "item_scalar_hn": pa.array(
                    [[[201, 202, 203]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "sku_a_hn": pa.array([[[11]]], type=pa.list_(pa.list_(pa.int64()))),
                "sku_b_hn": pa.array([[[21]]], type=pa.list_(pa.list_(pa.int64()))),
                "impr_x_goods_id_hn": pa.array([[-1]], type=pa.list_(pa.int64())),
                "impr_x_time": pa.array([[4900]], type=pa.list_(pa.int64())),
                "impr_x_indices": pa.array(
                    [[[0]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "scene_id": pa.array([[7]], type=pa.list_(pa.int64())),
                "search_id": pa.array([["r0"]], type=pa.list_(pa.string())),
                "impr_time": pa.array([[5000]], type=pa.list_(pa.int64())),
                "label_a": pa.array([[0]], type=pa.list_(pa.int64())),
                "label_b": pa.array([[1]], type=pa.list_(pa.int64())),
                "label_c": pa.array([[0]], type=pa.list_(pa.int64())),
            }
        )
        context = _context(REQUIRED)
        auditor = FeatureCardinalityAuditor(soft=True)
        context._runtime_cache = {"cardinality_auditor": auditor}
        flat = adapt(table, context=context)
        self.assertEqual(flat.num_rows, 1)
        self.assertTrue(auditor.has_scalar_multis())
        self.assertEqual(auditor.scalar_stats["ctx_scalar_hn"].multi_count, 1)
        self.assertEqual(auditor.scalar_stats["item_scalar_hn"].multi_count, 1)
        # Without soft mode, the same row still fails hard.
        hard_context = _context(REQUIRED)
        hard_context._runtime_cache = {}
        with self.assertRaisesRegex(ValueError, "inner length"):
            adapt(table, context=hard_context)

    def test_trusted_structure_rejects_values_indices_mismatch_on_row1(self) -> None:
        good = {
            "context_indices": [[0]],
            "target_indices": [[0]],
            "ctx_scalar_hn": [[[101]]],
            "ctx_bag_hn": [[[1]]],
            "item_scalar_hn": [[[201]]],
            "sku_a_hn": [[[11]]],
            "sku_b_hn": [[[21]]],
            "impr_x_goods_id_hn": [[-1]],
            "impr_x_time": [[4900]],
            "impr_x_indices": [[[0]]],
            "scene_id": [[7]],
            "search_id": [["r0"]],
            "impr_time": [[5000]],
            "label_a": [[0]],
            "label_b": [[1]],
            "label_c": [[0]],
        }
        bad = dict(good)
        bad["impr_x_goods_id_hn"] = [[-1, -2]]  # length 2 vs indices length 1
        table = pa.table(
            {
                key: pa.array([good[key][0], bad[key][0]])
                for key in good
            }
        )
        context = _context(REQUIRED)
        context.trusted_input = True
        context._runtime_cache = {}
        with self.assertRaisesRegex(ValueError, "does not match its indices"):
            adapt(table, context=context)

        # After warm-up on a clean first batch, a later batch still validates structure.
        warm = pa.table({key: pa.array([good[key][0]]) for key in good})
        adapt(warm, context=context)
        self.assertTrue(
            context._runtime_cache.get("mdl_rankmixer_raw_sample_validated")
        )
        with self.assertRaisesRegex(ValueError, "does not match its indices"):
            adapt(table.slice(1, 1), context=context)

    def test_complete_label_contract_rejects_non_first_row_and_batch(self) -> None:
        split = SimpleNamespace(labels={"a": "label_a"}, label_masks={})
        good = pa.table({"label_a": pa.array([0, 1], type=pa.int64())})
        _validate_complete_label_contract(split, good, ["label_a"])

        bad_null = pa.table(
            {"label_a": pa.array([0, None], type=pa.int64())}
        )
        with self.assertRaisesRegex(ValueError, "contains null"):
            _validate_complete_label_contract(split, bad_null, ["label_a"])

        bad_value = pa.table({"label_a": pa.array([0, 2], type=pa.int64())})
        with self.assertRaisesRegex(ValueError, "only 0/1"):
            _validate_complete_label_contract(split, bad_value, ["label_a"])

    def test_request_context_bags_expand_by_target_indices(self) -> None:
        required = [
            *REQUIRED,
            "offline_outside_goods_id_list_hn_share",
            "i2i_coclk_hn_share",
        ]
        table = pa.table(
            {
                "context_indices": [[11, 22]],
                "target_indices": [[11, 22, 11]],
                "ctx_scalar_hn": [[[101], [102]]],
                "ctx_bag_hn": [[[1], [2]]],
                "offline_outside_goods_id_list_hn_share": [[[1, 2], [3]]],
                "item_scalar_hn": [[[201], [202], [203]]],
                "sku_a_hn": [[[11], [12], [13]]],
                "sku_b_hn": [[[21], [22], [23]]],
                "i2i_coclk_hn_share": [[[10], [], [20, 21]]],
                "impr_x_goods_id_hn": [[-1, -2]],
                "impr_x_time": [[4900, 4800]],
                "impr_x_indices": [[[11, 22], [22]]],
                "scene_id": [[21, 7]],
                "search_id": [["r0", "r1"]],
                "impr_time": [[5000, 5001]],
                "label_a": [[0, 1, 0]],
                "label_b": [[1, 0, 1]],
                "label_c": [[0, 0, 1]],
            }
        )
        context = _context(required)
        context.options["context_features"] = [
            *context.options["context_features"],
            "offline_outside_goods_id_list_hn_share",
        ]
        context.options["item_features"] = [
            *context.options["item_features"],
            "i2i_coclk_hn_share",
        ]
        context.options["multivalue_features"] = [
            *context.options["multivalue_features"],
            "offline_outside_goods_id_list_hn_share",
            "i2i_coclk_hn_share",
        ]

        actual = adapt(table, context=context).to_pydict()
        self.assertEqual(
            actual["offline_outside_goods_id_list_hn_share"],
            [[1, 2], [3], [1, 2]],
        )
        self.assertEqual(actual["i2i_coclk_hn_share"], [[10], [], [20, 21]])

    def test_request_context_bag_accepts_request_count_not_candidate_count(
        self,
    ) -> None:
        required = [*REQUIRED, "offline_outside_goods_id_list_hn_share"]
        table = pa.table(
            {
                "context_indices": [[0, 1]],
                "target_indices": [[0, 0, 1]],
                "ctx_scalar_hn": [[[101], [102]]],
                "ctx_bag_hn": [[[1], [2]]],
                "offline_outside_goods_id_list_hn_share": [[[1], [2]]],
                "item_scalar_hn": [[[201], [202], [203]]],
                "sku_a_hn": [[[11], [12], [13]]],
                "sku_b_hn": [[[21], [22], [23]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "impr_x_indices": [[[0, 1]]],
                "scene_id": [[7, 8]],
                "search_id": [["r0", "r1"]],
                "impr_time": [[5000, 6000]],
                "label_a": [[0, 1, 0]],
                "label_b": [[1, 0, 1]],
                "label_c": [[0, 0, 1]],
            }
        )
        context = _context(required)
        context.options["context_features"] = [
            *context.options["context_features"],
            "offline_outside_goods_id_list_hn_share",
        ]
        context.options["multivalue_features"] = [
            *context.options["multivalue_features"],
            "offline_outside_goods_id_list_hn_share",
        ]
        actual = adapt(table, context=context).to_pydict()
        self.assertEqual(len(actual["offline_outside_goods_id_list_hn_share"]), 3)
        self.assertEqual(
            actual["offline_outside_goods_id_list_hn_share"],
            [[1], [1], [2]],
        )

    def test_request_axis_and_candidate_axis_length_errors(self) -> None:
        required = [
            *REQUIRED,
            "offline_outside_goods_id_list_hn_share",
            "i2i_coclk_hn_share",
        ]
        base = {
            "context_indices": [[0, 1]],
            "target_indices": [[0, 1, 1]],
            "ctx_scalar_hn": [[[101], [102]]],
            "ctx_bag_hn": [[[1], [2]]],
            "offline_outside_goods_id_list_hn_share": [[[1], [2]]],
            "item_scalar_hn": [[[201], [202], [203]]],
            "sku_a_hn": [[[11], [12], [13]]],
            "sku_b_hn": [[[21], [22], [23]]],
            "i2i_coclk_hn_share": [[[10], [11], [12]]],
            "impr_x_goods_id_hn": [[-1]],
            "impr_x_time": [[4900]],
            "impr_x_indices": [[[0, 1]]],
            "scene_id": [[7, 8]],
            "search_id": [["r0", "r1"]],
            "impr_time": [[5000, 6000]],
            "label_a": [[0, 1, 0]],
            "label_b": [[1, 0, 1]],
            "label_c": [[0, 0, 1]],
        }
        context = _context(required)
        context.options["context_features"] = [
            *context.options["context_features"],
            "offline_outside_goods_id_list_hn_share",
        ]
        context.options["item_features"] = [
            *context.options["item_features"],
            "i2i_coclk_hn_share",
        ]
        context.options["multivalue_features"] = [
            *context.options["multivalue_features"],
            "offline_outside_goods_id_list_hn_share",
            "i2i_coclk_hn_share",
        ]

        bad_request = dict(base)
        bad_request["offline_outside_goods_id_list_hn_share"] = [[[1]]]
        with self.assertRaisesRegex(
            ValueError,
            r"request-axis feature 'offline_outside_goods_id_list_hn_share' "
            r"length 1 != request count 2",
        ):
            adapt(pa.table(bad_request), context=context)

        bad_candidate = dict(base)
        bad_candidate["i2i_coclk_hn_share"] = [[[10], [11]]]
        with self.assertRaisesRegex(
            ValueError,
            r"candidate-axis feature 'i2i_coclk_hn_share' "
            r"length 2 != candidate count 3",
        ):
            adapt(pa.table(bad_candidate), context=context)

    def test_request_context_scalar_expands_empty_as_missing(self) -> None:
        required = [*REQUIRED, "query_pay_cnt_15d_hn"]
        table = pa.table(
            {
                "context_indices": [[10, 20]],
                "target_indices": [[10, 10, 20]],
                "ctx_scalar_hn": [[[101], [102]]],
                "ctx_bag_hn": [[[1], [2]]],
                "query_pay_cnt_15d_hn": [[[7], []]],
                "item_scalar_hn": [[[201], [202], [203]]],
                "sku_a_hn": [[[11], [12], [13]]],
                "sku_b_hn": [[[21], [22], [23]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "impr_x_indices": [[[10, 20]]],
                "scene_id": [[7, 8]],
                "search_id": [["r0", "r1"]],
                "impr_time": [[5000, 6000]],
                "label_a": [[0, 1, 0]],
                "label_b": [[1, 0, 1]],
                "label_c": [[0, 0, 1]],
            }
        )
        context = _context(required)
        context.options["context_features"] = [
            *context.options["context_features"],
            "query_pay_cnt_15d_hn",
        ]
        actual = adapt(table, context=context).to_pydict()
        self.assertEqual(actual["query_pay_cnt_15d_hn"], [7, 7, None])

    def test_request_context_bag_expands_by_target_indices(self) -> None:
        required = [*REQUIRED, "buy_long_spec_vids_hn"]
        table = pa.table(
            {
                "context_indices": [[10, 20]],
                "target_indices": [[10, 10, 20]],
                "ctx_scalar_hn": [[[101], [102]]],
                "ctx_bag_hn": [[[1], [2]]],
                "buy_long_spec_vids_hn": [[[101, 102], [201]]],
                "item_scalar_hn": [[[201], [202], [203]]],
                "sku_a_hn": [[[11], [12], [13]]],
                "sku_b_hn": [[[21], [22], [23]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "impr_x_indices": [[[10, 20]]],
                "scene_id": [[7, 8]],
                "search_id": [["r0", "r1"]],
                "impr_time": [[5000, 6000]],
                "label_a": [[0, 1, 0]],
                "label_b": [[1, 0, 1]],
                "label_c": [[0, 0, 1]],
            }
        )
        context = _context(required)
        context.options["context_features"] = [
            *context.options["context_features"],
            "buy_long_spec_vids_hn",
        ]
        context.options["multivalue_features"] = [
            *context.options["multivalue_features"],
            "buy_long_spec_vids_hn",
        ]
        actual = adapt(table, context=context).to_pydict()
        self.assertEqual(
            actual["buy_long_spec_vids_hn"],
            [[101, 102], [101, 102], [201]],
        )

    def test_candidate_item_bag_not_request_cached(self) -> None:
        required = [*REQUIRED, "clk_cnt_1d_hn"]
        table = pa.table(
            {
                "context_indices": [[10, 20]],
                "target_indices": [[10, 10, 20]],
                "ctx_scalar_hn": [[[101], [102]]],
                "ctx_bag_hn": [[[1], [2]]],
                "item_scalar_hn": [[[201], [202], [203]]],
                "sku_a_hn": [[[11], [12], [13]]],
                "sku_b_hn": [[[21], [22], [23]]],
                "clk_cnt_1d_hn": [[[3], [], [8]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "impr_x_indices": [[[10, 20]]],
                "scene_id": [[7, 8]],
                "search_id": [["r0", "r1"]],
                "impr_time": [[5000, 6000]],
                "label_a": [[0, 1, 0]],
                "label_b": [[1, 0, 1]],
                "label_c": [[0, 0, 1]],
            }
        )
        context = _context(required)
        context.options["item_features"] = [
            *context.options["item_features"],
            "clk_cnt_1d_hn",
        ]
        context.options["multivalue_features"] = [
            *context.options["multivalue_features"],
            "clk_cnt_1d_hn",
        ]
        actual = adapt(table, context=context).to_pydict()
        self.assertEqual(actual["clk_cnt_1d_hn"], [[3], [], [8]])
        self.assertEqual(actual["ctx_scalar_hn"], [101, 101, 102])

    def test_all_request_context_scalars_expand(self) -> None:
        from scripts.build_mdl_rankmixer_config import REQUEST_CONTEXT_SCALAR_FIELDS

        for field in sorted(REQUEST_CONTEXT_SCALAR_FIELDS):
            with self.subTest(field=field):
                required = [*REQUIRED, field]
                table = pa.table(
                    {
                        "context_indices": [[10, 20]],
                        "target_indices": [[10, 10, 20]],
                        "ctx_scalar_hn": [[[101], [102]]],
                        "ctx_bag_hn": [[[1], [2]]],
                        field: [[[7], []]],
                        "item_scalar_hn": [[[201], [202], [203]]],
                        "sku_a_hn": [[[11], [12], [13]]],
                        "sku_b_hn": [[[21], [22], [23]]],
                        "impr_x_goods_id_hn": [[-1]],
                        "impr_x_time": [[4900]],
                        "impr_x_indices": [[[10, 20]]],
                        "scene_id": [[7, 8]],
                        "search_id": [["r0", "r1"]],
                        "impr_time": [[5000, 6000]],
                        "label_a": [[0, 1, 0]],
                        "label_b": [[1, 0, 1]],
                        "label_c": [[0, 0, 1]],
                    }
                )
                context = _context(required)
                context.options["context_features"] = [
                    *context.options["context_features"],
                    field,
                ]
                actual = adapt(table, context=context).to_pydict()
                self.assertEqual(actual[field], [7, 7, None])

    def test_all_request_context_bags_expand(self) -> None:
        from scripts.build_mdl_rankmixer_config import REQUEST_CONTEXT_BAG_FIELDS

        for field in sorted(REQUEST_CONTEXT_BAG_FIELDS):
            with self.subTest(field=field):
                required = [*REQUIRED, field]
                table = pa.table(
                    {
                        "context_indices": [[10, 20]],
                        "target_indices": [[10, 10, 20]],
                        "ctx_scalar_hn": [[[101], [102]]],
                        "ctx_bag_hn": [[[1], [2]]],
                        field: [[[101, 102], [201]]],
                        "item_scalar_hn": [[[201], [202], [203]]],
                        "sku_a_hn": [[[11], [12], [13]]],
                        "sku_b_hn": [[[21], [22], [23]]],
                        "impr_x_goods_id_hn": [[-1]],
                        "impr_x_time": [[4900]],
                        "impr_x_indices": [[[10, 20]]],
                        "scene_id": [[7, 8]],
                        "search_id": [["r0", "r1"]],
                        "impr_time": [[5000, 6000]],
                        "label_a": [[0, 1, 0]],
                        "label_b": [[1, 0, 1]],
                        "label_c": [[0, 0, 1]],
                    }
                )
                context = _context(required)
                context.options["context_features"] = [
                    *context.options["context_features"],
                    field,
                ]
                context.options["multivalue_features"] = [
                    *context.options["multivalue_features"],
                    field,
                ]
                actual = adapt(table, context=context).to_pydict()
                self.assertEqual(actual[field], [[101, 102], [101, 102], [201]])

    def test_all_candidate_item_clk_cart_bags_stay_candidate_aligned(self) -> None:
        candidate_item_bags = {
            "clk_cnt_1d_hn",
            "clk_3d_cnt_hn",
            "clk_1d_cat_cnt_hn",
            "cart_cnt_1d_hn",
            "cart_cnt_3d_hn",
        }

        for field in sorted(candidate_item_bags):
            with self.subTest(field=field):
                required = [*REQUIRED, field]
                table = pa.table(
                    {
                        "context_indices": [[10, 20]],
                        "target_indices": [[10, 10, 20]],
                        "ctx_scalar_hn": [[[101], [102]]],
                        "ctx_bag_hn": [[[1], [2]]],
                        "item_scalar_hn": [[[201], [202], [203]]],
                        "sku_a_hn": [[[11], [12], [13]]],
                        "sku_b_hn": [[[21], [22], [23]]],
                        field: [[[3], [], [8]]],
                        "impr_x_goods_id_hn": [[-1]],
                        "impr_x_time": [[4900]],
                        "impr_x_indices": [[[10, 20]]],
                        "scene_id": [[7, 8]],
                        "search_id": [["r0", "r1"]],
                        "impr_time": [[5000, 6000]],
                        "label_a": [[0, 1, 0]],
                        "label_b": [[1, 0, 1]],
                        "label_c": [[0, 0, 1]],
                    }
                )
                context = _context(required)
                context.options["item_features"] = [
                    *context.options["item_features"],
                    field,
                ]
                context.options["multivalue_features"] = [
                    *context.options["multivalue_features"],
                    field,
                ]
                actual = adapt(table, context=context).to_pydict()
                self.assertEqual(actual[field], [[3], [], [8]])
                self.assertEqual(actual["ctx_scalar_hn"], [101, 101, 102])

    def test_obsolete_axis_override_options_rejected(self) -> None:
        context = _context(REQUIRED)
        context.options["request_axis_item_features"] = ["item_scalar_hn"]
        with self.assertRaisesRegex(ValueError, "request_axis_item_features.*removed"):
            adapt(
                pa.table(
                    {
                        "ctx_scalar_hn": [[101]],
                        "ctx_bag_hn": [[1]],
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
                ),
                context=context,
            )

    def test_coarse_scene_unlisted_policy_error_rejects_during_adapt(self) -> None:
        required = [
            *REQUIRED,
            COARSE_SCENE_INDEX_COLUMN,
            COARSE_SCENE_PRIOR_ID_COLUMN,
        ]
        table = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[1]],
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
        context = _context(required)
        context.options["search_scene_ids"] = sorted(SEARCH_IDS)
        context.options["unlisted_scene_policy"] = "error"
        with self.assertRaisesRegex(ValueError, "not in the configured search allowlist"):
            adapt(table, context=context)

    def test_agg_rejects_length_one_request_metadata_broadcast(self) -> None:
        table = pa.table(
            {
                "context_indices": [[0, 1]],
                "target_indices": [[0, 1]],
                "ctx_scalar_hn": [[[101], [102]]],
                "ctx_bag_hn": [[[1], [2]]],
                "item_scalar_hn": [[[201], [202]]],
                "sku_a_hn": [[[11], [12]]],
                "sku_b_hn": [[[21], [22]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "impr_x_indices": [[[0, 1]]],
                "scene_id": [[7]],
                "search_id": [["r0", "r1"]],
                "impr_time": [[5000, 6000]],
                "label_a": [[0, 1]],
                "label_b": [[1, 0]],
                "label_c": [[0, 0]],
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            r"agg request-level column 'scene_id' has length 1, expected 2",
        ):
            adapt(table, context=_context(REQUIRED))

    def test_agg_rejects_bare_scalar_request_metadata_broadcast(self) -> None:
        table = pa.table(
            {
                "context_indices": [[0, 1]],
                "target_indices": [[0, 1]],
                "ctx_scalar_hn": [[[101], [102]]],
                "ctx_bag_hn": [[[1], [2]]],
                "item_scalar_hn": [[[201], [202]]],
                "sku_a_hn": [[[11], [12]]],
                "sku_b_hn": [[[21], [22]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "impr_x_indices": [[[0, 1]]],
                "scene_id": [7],
                "search_id": [["r0", "r1"]],
                "impr_time": [[5000, 6000]],
                "label_a": [[0, 1]],
                "label_b": [[1, 0]],
                "label_c": [[0, 0]],
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            r"agg request-level column 'scene_id' is scalar but request_count=2",
        ):
            adapt(table, context=_context(REQUIRED))

    def test_request_level_scalarizes_selected_singleton_wrapper(self) -> None:
        from src.dataloader import _request_level_value

        self.assertEqual(
            _request_level_value(
                [[7], [8]],
                request_position=1,
                request_count=2,
                column="scene_id",
                raw_row=0,
                agg=True,
            ),
            8,
        )

    def test_complete_label_fast_path_scalarizes_singleton_wrappers(self) -> None:
        table = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[1]],
                "item_scalar_hn": [[[201], [202]]],
                "sku_a_hn": [[[11], [12]]],
                "sku_b_hn": [[[21], [22]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "scene_id": [7],
                "search_id": ["r0"],
                "impr_time": [5000],
                "label_a": [[[0], [1]]],
                "label_b": [[[1], [0]]],
                "label_c": [[[0], [1]]],
            }
        )
        actual = adapt(table, context=_context(REQUIRED)).to_pydict()
        self.assertEqual(actual["label_a"], [0, 1])
        self.assertEqual(actual["label_b"], [1, 0])
        self.assertEqual(actual["label_c"], [0, 1])

    def test_masked_label_path_scalarizes_empty_as_missing(self) -> None:
        table = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[1]],
                "item_scalar_hn": [[[201], [202]]],
                "sku_a_hn": [[[11], [12]]],
                "sku_b_hn": [[[21], [22]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "scene_id": [7],
                "search_id": ["r0"],
                "impr_time": [5000],
                "label_a": [[[0], []]],
                "label_b": [[[1], [0]]],
                "label_c": [[[0], [1]]],
            }
        )
        masks = {task: f"label_{task}_valid" for task in ("a", "b", "c")}
        context = _context([*REQUIRED, *masks.values()])
        context.options["label_masks"] = masks
        context.options["label_missing_values"] = {"a": [None], "b": [], "c": []}
        actual = adapt(table, context=context).to_pydict()
        self.assertEqual(actual["label_a"], [0, None])
        self.assertEqual(actual["label_a_valid"], [1, 0])

    def test_multi_value_label_cell_raises(self) -> None:
        table = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[1]],
                "item_scalar_hn": [[[201]]],
                "sku_a_hn": [[[11]]],
                "sku_b_hn": [[[21]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "scene_id": [7],
                "search_id": ["r0"],
                "impr_time": [5000],
                "label_a": [[[0, 1]]],
                "label_b": [[[1]]],
                "label_c": [[[0]]],
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            r"single-valued feature 'label_a' has inner length 2",
        ):
            adapt(table, context=_context(REQUIRED))

    def test_illegal_label_values_raise_on_masked_path(self) -> None:
        table = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[1]],
                "item_scalar_hn": [[[201]]],
                "sku_a_hn": [[[11]]],
                "sku_b_hn": [[[21]]],
                "impr_x_goods_id_hn": [[-1]],
                "impr_x_time": [[4900]],
                "scene_id": [7],
                "search_id": ["r0"],
                "impr_time": [5000],
                "label_a": [[2]],
                "label_b": [[1]],
                "label_c": [[0]],
            }
        )
        masks = {"a": "label_a_valid", "b": "label_b_valid", "c": "label_c_valid"}
        context = _context([*REQUIRED, *masks.values()])
        context.options["label_masks"] = masks
        context.options["label_missing_values"] = {"a": [None], "b": [], "c": []}
        with self.assertRaisesRegex(ValueError, r"label 'label_a' must be numeric 0/1"):
            adapt(table, context=context)

    def test_sku_spec_hn_null_outside_aligned_group_is_allowed(self) -> None:
        from scripts.profile_prehashed_parquet import ALIGNED_SKU_FIELDS

        aligned = list(ALIGNED_SKU_FIELDS)
        required = [
            "ctx_scalar_hn",
            "ctx_bag_hn",
            "item_scalar_hn",
            *aligned,
            "sku_spec_hn",
            "impr_x_goods_id_hn",
            "impr_x_time_delta_ms",
            "scene_id",
            "search_id",
            "label_a",
            "label_b",
            "label_c",
        ]
        sku_bags = {name: [[[10, 11]]] for name in aligned}
        table = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[1]],
                "item_scalar_hn": [[[201]]],
                **sku_bags,
                "sku_spec_hn": [[None]],
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
        context = _context(required)
        context.options["item_features"] = ["item_scalar_hn", *aligned, "sku_spec_hn"]
        context.options["multivalue_features"] = [
            "ctx_bag_hn",
            *aligned,
            "sku_spec_hn",
        ]
        context.options["aligned_multivalue_groups"] = [aligned]
        actual = adapt(table, context=context).to_pydict()
        self.assertEqual(actual["sku_id_hn"], [[10, 11]])
        self.assertEqual(actual["sku_spec_hn"], [[]])

    def test_aligned_sku_group_still_rejects_inner_length_mismatch(self) -> None:
        from scripts.profile_prehashed_parquet import ALIGNED_SKU_FIELDS

        aligned = list(ALIGNED_SKU_FIELDS)
        required = [
            "ctx_scalar_hn",
            "ctx_bag_hn",
            "item_scalar_hn",
            *aligned,
            "impr_x_goods_id_hn",
            "impr_x_time_delta_ms",
            "scene_id",
            "search_id",
            "label_a",
            "label_b",
            "label_c",
        ]
        sku_bags = {name: [[[10, 11]]] for name in aligned}
        sku_bags["sku_price_v2_hn"] = [[[10]]]  # length 1 vs 2
        table = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[1]],
                "item_scalar_hn": [[[201]]],
                **sku_bags,
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
        context = _context(required)
        context.options["item_features"] = ["item_scalar_hn", *aligned]
        context.options["multivalue_features"] = ["ctx_bag_hn", *aligned]
        context.options["aligned_multivalue_groups"] = [aligned]
        with self.assertRaisesRegex(ValueError, "aligned multivalue group mismatch"):
            adapt(table, context=context)

    def test_candidate_scalar_rejects_multi_value_slots(self) -> None:
        required = [*REQUIRED, "multimodal_i2i_hit_clk_size_hn"]
        table = pa.table(
            {
                "ctx_scalar_hn": [[101]],
                "ctx_bag_hn": [[1]],
                "item_scalar_hn": [[[201]]],
                "sku_a_hn": [[[11]]],
                "sku_b_hn": [[[21]]],
                "multimodal_i2i_hit_clk_size_hn": [[[2, 3]]],
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
        context = _context(required)
        context.options["item_features"] = [
            *context.options["item_features"],
            "multimodal_i2i_hit_clk_size_hn",
        ]
        with self.assertRaisesRegex(ValueError, "inner length 2"):
            adapt(table, context=context)

if __name__ == "__main__":
    unittest.main()
