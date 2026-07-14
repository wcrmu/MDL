from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa
import torch

from src.config import (
    CategoricalEncodingConfig,
    FeatureConfig,
    ResolvedCategoricalInput,
    ResolvedIdentityEncoding,
    ResolvedSharedVocabEncoding,
    SequenceConfig,
    SequenceFieldConfig,
)
from src.dataloader import _tensorize_categorical, _tensorize_multi_field_sequence


def _identity_input(
    name: str,
    source: str,
    *,
    num_buckets: int = 8,
    out_of_range: str = "error",
    sequence_name: str | None = None,
    field_name: str | None = None,
) -> ResolvedCategoricalInput:
    return ResolvedCategoricalInput(
        name=name,
        source=source,
        location="sequence_field" if sequence_name else "feature",
        sequence_name=sequence_name,
        field_name=field_name,
        encoding=ResolvedIdentityEncoding(
            num_buckets=num_buckets,
            padding_id=0,
            out_of_range=out_of_range,
        ),
    )


def _config(inputs: list[ResolvedCategoricalInput]) -> SimpleNamespace:
    return SimpleNamespace(
        resolved=SimpleNamespace(
            categorical_input_by_name={item.name: item for item in inputs}
        ),
        vocab_strategy=SimpleNamespace(
            defaults=SimpleNamespace(unseen_policy="error")
        ),
    )


class IdentityConfigTest(unittest.TestCase):
    def test_num_buckets_is_an_exclusive_bound(self) -> None:
        encoding = CategoricalEncodingConfig.from_mapping(
            {
                "type": "identity",
                "num_buckets": 8,
                "padding_id": 0,
                "out_of_range": "error",
            }
        )
        assert encoding is not None
        encoding.validate("feature.encoding")

    def test_rejects_ambiguous_legacy_and_new_bounds(self) -> None:
        encoding = CategoricalEncodingConfig.from_mapping(
            {"type": "identity", "num_buckets": 8, "max_id": 7}
        )
        assert encoding is not None
        with self.assertRaisesRegex(ValueError, "not both"):
            encoding.validate("feature.encoding")


class ScalarDirectIdTest(unittest.TestCase):
    def test_arrow_integer_column_bypasses_python_encoder(self) -> None:
        categorical = _identity_input("item_id", "item")
        config = _config([categorical])
        feature = FeatureConfig(name="item_id", kind="categorical", source="item")
        table = pa.table({"item": pa.array([0, 1, 7], type=pa.int64())})

        with patch(
            "src.dataloader.encode_categorical_values",
            side_effect=AssertionError("identity path called Python encoder"),
        ):
            actual = _tensorize_categorical(config, feature, table, {})

        torch.testing.assert_close(actual, torch.tensor([0, 1, 7]))

    def test_out_of_range_error_reports_batch_bounds(self) -> None:
        categorical = _identity_input("item_id", "item")
        config = _config([categorical])
        feature = FeatureConfig(name="item_id", kind="categorical", source="item")
        table = pa.table({"item": [1, 8]})

        with self.assertRaisesRegex(ValueError, r"outside \[0, 8\)"):
            _tensorize_categorical(config, feature, table, {})

    def test_out_of_range_padding_is_vectorized(self) -> None:
        categorical = _identity_input(
            "item_id", "item", out_of_range="padding"
        )
        config = _config([categorical])
        feature = FeatureConfig(name="item_id", kind="categorical", source="item")
        table = pa.table({"item": [-1, 2, 9]})

        actual = _tensorize_categorical(config, feature, table, {})

        torch.testing.assert_close(actual, torch.tensor([0, 2, 0]))

    def test_shared_namespace_with_identity_root_stays_vectorized(self) -> None:
        root = _identity_input("item_id", "item", num_buckets=16)
        alias = ResolvedCategoricalInput(
            name="scenario_item_id",
            source="scenario_item",
            location="feature",
            sequence_name=None,
            field_name=None,
            encoding=ResolvedSharedVocabEncoding(
                share_with="item_id",
                share_embedding=False,
            ),
        )
        config = _config([root, alias])
        feature = FeatureConfig(
            name="scenario_item_id",
            kind="categorical",
            source="scenario_item",
        )
        table = pa.table({"scenario_item": [0, 7, 15]})

        with patch(
            "src.dataloader.encode_categorical_values",
            side_effect=AssertionError("identity alias called Python encoder"),
        ):
            actual = _tensorize_categorical(config, feature, table, {})

        torch.testing.assert_close(actual, torch.tensor([0, 7, 15]))


class SequenceDirectIdTest(unittest.TestCase):
    def test_uses_offsets_and_vectorized_tail_padding(self) -> None:
        categorical = _identity_input(
            "hist.item_id",
            "hist_item",
            sequence_name="hist",
            field_name="item_id",
        )
        config = _config([categorical])
        sequence = SequenceConfig(
            name="hist",
            fields=[
                SequenceFieldConfig(
                    name="item_id", kind="categorical", source="hist_item"
                ),
                SequenceFieldConfig(
                    name="age", kind="dense", source="hist_age"
                ),
            ],
            max_length=2,
            truncation="tail",
        )
        table = pa.table(
            {
                "hist_item": [[1, 2, 3], [4]],
                "hist_age": [[0.1, 0.2, 0.3], [0.4]],
            }
        )

        with patch(
            "src.dataloader._sequence_rows",
            side_effect=AssertionError("identity sequence used Python rows"),
        ):
            actual = _tensorize_multi_field_sequence(config, sequence, table, {})

        torch.testing.assert_close(actual["lengths"], torch.tensor([2, 1]))
        torch.testing.assert_close(
            actual["fields"]["item_id"],
            torch.tensor([[2, 3], [4, 0]]),
        )
        torch.testing.assert_close(
            actual["fields"]["age"],
            torch.tensor([[0.2, 0.3], [0.4, 0.0]]),
        )

    def test_rejects_misaligned_field_offsets(self) -> None:
        categorical = _identity_input(
            "hist.item_id",
            "hist_item",
            sequence_name="hist",
            field_name="item_id",
        )
        config = _config([categorical])
        sequence = SequenceConfig(
            name="hist",
            fields=[
                SequenceFieldConfig(
                    name="item_id", kind="categorical", source="hist_item"
                ),
                SequenceFieldConfig(name="age", kind="dense", source="hist_age"),
            ],
        )
        table = pa.table(
            {
                "hist_item": [[1, 2], [3]],
                "hist_age": [[0.1], [0.2, 0.3]],
            }
        )

        with self.assertRaisesRegex(ValueError, "offsets"):
            _tensorize_multi_field_sequence(config, sequence, table, {})

    def test_shared_sequence_namespace_uses_identity_offsets_path(self) -> None:
        root = _identity_input("item_id", "item", num_buckets=16)
        alias = ResolvedCategoricalInput(
            name="hist.item_id",
            source="hist_item",
            location="sequence_field",
            sequence_name="hist",
            field_name="item_id",
            encoding=ResolvedSharedVocabEncoding(
                share_with="item_id",
                share_embedding=True,
            ),
        )
        config = _config([root, alias])
        sequence = SequenceConfig(
            name="hist",
            fields=[
                SequenceFieldConfig(
                    name="item_id", kind="categorical", source="hist_item"
                )
            ],
            max_length=3,
        )
        table = pa.table({"hist_item": [[1, 2], [3]]})

        with patch(
            "src.dataloader._sequence_rows",
            side_effect=AssertionError("identity alias used Python rows"),
        ):
            actual = _tensorize_multi_field_sequence(config, sequence, table, {})

        torch.testing.assert_close(
            actual["fields"]["item_id"],
            torch.tensor([[1, 2], [3, 0]]),
        )


if __name__ == "__main__":
    unittest.main()
