from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa
import torch

from src.config import (
    FeatureConfig,
    ResolvedCategoricalInput,
    ResolvedIdentityEncoding,
    SequenceConfig,
    SequenceFieldConfig,
)
from src.dataloader import (
    _sequence_rows,
    _tensorize_dense_column,
    _tensorize_multi_field_sequence,
)
from src.model import FeatureEncoderBank


def _identity_input(
    name: str,
    source: str,
    *,
    sequence_name: str,
    field_name: str,
) -> ResolvedCategoricalInput:
    return ResolvedCategoricalInput(
        name=name,
        source=source,
        location="sequence_field",
        sequence_name=sequence_name,
        field_name=field_name,
        encoding=ResolvedIdentityEncoding(
            num_buckets=16,
            padding_id=0,
            out_of_range="error",
        ),
    )


def _config(categoricals: list[ResolvedCategoricalInput]) -> SimpleNamespace:
    return SimpleNamespace(
        resolved=SimpleNamespace(
            categorical_input_by_name={item.name: item for item in categoricals}
        ),
        vocab_strategy=SimpleNamespace(
            defaults=SimpleNamespace(unseen_policy="oov")
        ),
    )


class NullAnchorCompressTest(unittest.TestCase):
    def test_python_path_drops_anchor_null_steps_synchronously(self) -> None:
        sequence = SequenceConfig(
            name="hist",
            fields=(
                SequenceFieldConfig(
                    name="goods_id_hn", kind="categorical", source="goods"
                ),
                SequenceFieldConfig(name="age", kind="dense", source="age"),
            ),
            null_anchor_field="goods_id_hn",
        )
        table = pa.table(
            {
                "goods": [[101, None, 103], [None]],
                "age": [[0.1, 0.2, 0.3], [0.9]],
            }
        )
        rows, lengths = _sequence_rows(table, sequence)
        self.assertEqual(lengths, [2, 0])
        self.assertEqual(rows["goods_id_hn"], [[101, 103], []])
        self.assertEqual(rows["age"], [[0.1, 0.3], []])

    def test_direct_path_compacts_anchor_nulls_and_sets_has_sequence(self) -> None:
        categorical = _identity_input(
            "hist.goods_id_hn",
            "goods",
            sequence_name="hist",
            field_name="goods_id_hn",
        )
        config = _config([categorical])
        sequence = SequenceConfig(
            name="hist",
            fields=(
                SequenceFieldConfig(
                    name="goods_id_hn", kind="categorical", source="goods"
                ),
                SequenceFieldConfig(name="age", kind="dense", source="age"),
            ),
            null_anchor_field="goods_id_hn",
            max_length=4,
            truncation="tail",
        )
        table = pa.table(
            {
                "goods": pa.array([[1, None, 3], []], type=pa.list_(pa.int64())),
                "age": pa.array([[0.1, 0.2, 0.3], []], type=pa.list_(pa.float32())),
            }
        )
        with patch(
            "src.dataloader._sequence_rows",
            side_effect=AssertionError("should use direct path"),
        ):
            actual = _tensorize_multi_field_sequence(config, sequence, table, {})
        torch.testing.assert_close(actual["lengths"], torch.tensor([2, 0]))
        self.assertTrue(
            torch.equal(actual["has_sequence"], torch.tensor([True, False]))
        )
        torch.testing.assert_close(
            actual["fields"]["goods_id_hn"],
            torch.tensor([[1, 3], [0, 0]]),
        )
        torch.testing.assert_close(
            actual["fields"]["age"],
            torch.tensor([[0.1, 0.3], [0.0, 0.0]]),
        )


class DensePresenceTest(unittest.TestCase):
    def test_null_and_zero_are_distinguished(self) -> None:
        feature = FeatureConfig(
            name="price", kind="dense", source="price", presence=True
        )
        table = pa.table({"price": pa.array([None, 0.0, 1.5], type=pa.float32())})
        payload = _tensorize_dense_column(feature, table)
        assert isinstance(payload, dict)
        torch.testing.assert_close(payload["values"], torch.tensor([0.0, 0.0, 1.5]))
        torch.testing.assert_close(
            payload["presence"],
            torch.tensor([[0.0], [1.0], [1.0]]),
        )

    def test_encoder_concatenates_presence(self) -> None:
        feature = FeatureConfig(
            name="price", kind="dense", source="price", presence=True
        )
        bank = FeatureEncoderBank.__new__(FeatureEncoderBank)
        bank.embedding_weight_dtype = torch.float32
        encoded = FeatureEncoderBank._encode_scalar_feature(
            bank,
            feature,
            {
                "values": torch.tensor([0.0, 0.0, 1.5]),
                "presence": torch.tensor([[0.0], [1.0], [1.0]]),
            },
        )
        torch.testing.assert_close(
            encoded,
            torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.5, 1.0]]),
        )


if __name__ == "__main__":
    unittest.main()
