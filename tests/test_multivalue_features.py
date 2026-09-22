from __future__ import annotations

from types import SimpleNamespace
import unittest

import pyarrow as pa
import torch

from src.config import (
    FeatureConfig,
    ResolvedCategoricalInput,
    ResolvedPreHashedEncoding,
)
from src.dataloader import _tensorize_categorical_bag
from src.model import _mean_pool_categorical_bag, _pool_categorical_bag


class MultiValueCategoricalFeatureTest(unittest.TestCase):
    @staticmethod
    def _config() -> SimpleNamespace:
        categorical = ResolvedCategoricalInput(
            name="tokens",
            source="tokens",
            location="feature",
            sequence_name=None,
            field_name=None,
            encoding=ResolvedPreHashedEncoding(num_buckets=8, padding_id=0),
        )
        return SimpleNamespace(
            resolved=SimpleNamespace(
                categorical_input_by_name={"tokens": categorical}
            ),
            vocab_strategy=SimpleNamespace(
                defaults=SimpleNamespace(unseen_policy="error")
            ),
        )

    def test_pre_hashed_bag_preserves_null_slots_and_top_null(self) -> None:
        feature = FeatureConfig(
            name="tokens",
            kind="categorical",
            source="tokens",
            pooling="mean",
            pooling_null_policy="include_as_padding",
            max_length=3,
            truncation="head",
        )
        table = pa.table(
            {
                "tokens": pa.array(
                    [[1, None, -1, 4], None, [-8]],
                    type=pa.list_(pa.int64()),
                )
            }
        )

        actual = _tensorize_categorical_bag(self._config(), feature, table, {})

        torch.testing.assert_close(actual["lengths"], torch.tensor([3, 0, 1]))
        # Flat CSR: row0 keeps 3 tokens, row1 empty, row2 keeps 1.
        torch.testing.assert_close(
            actual["values"],
            torch.tensor([1, -1, 7, 0]),
        )
        self.assertEqual(actual["values"].ndim, 1)

    def test_pre_hashed_bag_tail_truncation(self) -> None:
        feature = FeatureConfig(
            name="tokens",
            kind="categorical",
            source="tokens",
            pooling="mean",
            pooling_null_policy="include_as_padding",
            max_length=3,
            truncation="tail",
        )
        table = pa.table(
            {
                "tokens": pa.array(
                    [[1, None, -1, 4], [7], [-8, 3, 5, 9]],
                    type=pa.list_(pa.int64()),
                )
            }
        )

        actual = _tensorize_categorical_bag(self._config(), feature, table, {})

        torch.testing.assert_close(actual["lengths"], torch.tensor([3, 1, 3]))
        # Tail keeps last three of row0 ([None,-1,4]) and row2 ([3,5,9]).
        torch.testing.assert_close(
            actual["values"],
            torch.tensor([-1, 7, 4, 7, 3, 5, 1]),
        )

    def test_pooling_null_policies_have_distinct_denominators(self) -> None:
        indices = torch.tensor([2, -1, 8])
        lengths = torch.tensor([3, 0])
        embedded = torch.tensor([[2.0], [0.0], [8.0]])

        excluded = _mean_pool_categorical_bag(
            embedded, indices, lengths, "exclude"
        )
        preserved = _mean_pool_categorical_bag(
            embedded, indices, lengths, "include_as_padding"
        )

        torch.testing.assert_close(excluded, torch.tensor([[5.0], [0.0]]))
        torch.testing.assert_close(
            preserved,
            torch.tensor([[(2.0 + 8.0) / 3.0], [0.0]]),
        )

    def test_padded_bag_pooling_path_still_supported(self) -> None:
        indices = torch.tensor([[2, -1, 8], [-1, -1, -1]])
        lengths = torch.tensor([3, 0])
        embedded = torch.tensor(
            [[[2.0], [0.0], [8.0]], [[0.0], [0.0], [0.0]]]
        )
        excluded = _mean_pool_categorical_bag(
            embedded, indices, lengths, "exclude"
        )
        torch.testing.assert_close(excluded, torch.tensor([[5.0], [0.0]]))

    def test_sum_pooling_keeps_valid_ids_and_zeros_empty_bags(self) -> None:
        indices = torch.tensor([2, -1, 8, 0])
        lengths = torch.tensor([3, 1])
        embedded = torch.tensor([[2.0], [0.0], [8.0], [4.0]])

        summed = _pool_categorical_bag(
            embedded, indices, lengths, "exclude", pooling="sum"
        )
        meaned = _pool_categorical_bag(
            embedded, indices, lengths, "exclude", pooling="mean"
        )

        torch.testing.assert_close(summed, torch.tensor([[10.0], [4.0]]))
        torch.testing.assert_close(meaned, torch.tensor([[5.0], [4.0]]))

    def test_batch_sum_pooling_includes_id_zero(self) -> None:
        from src.model import _batch_pool_flat_bags

        embedded = torch.tensor([[1.0], [3.0], [0.0], [5.0]])
        indices = torch.tensor([0, 2, -1, 4])
        lengths = torch.tensor([2, 2])
        actual = _batch_pool_flat_bags(
            [
                ("a", embedded, indices, lengths, "exclude", "sum"),
                ("b", embedded, indices, lengths, "exclude", "sum"),
            ]
        )
        torch.testing.assert_close(actual["a"], torch.tensor([[4.0], [5.0]]))
        torch.testing.assert_close(actual["b"], torch.tensor([[4.0], [5.0]]))

    def test_dense_feature_cannot_enable_categorical_pooling(self) -> None:
        feature = FeatureConfig(
            name="dense",
            kind="dense",
            source="dense",
            pooling="mean",
        )
        with self.assertRaisesRegex(ValueError, "only supported for categorical"):
            feature.validate()


if __name__ == "__main__":
    unittest.main()
