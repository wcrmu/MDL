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
from src.model import _mean_pool_categorical_bag


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
        torch.testing.assert_close(
            actual["values"],
            torch.tensor([[2, 0, 8], [0, 0, 0], [1, 0, 0]]),
        )

    def test_pooling_null_policies_have_distinct_denominators(self) -> None:
        indices = torch.tensor([[2, 0, 8], [0, 0, 0]])
        lengths = torch.tensor([3, 0])
        embedded = indices.float().unsqueeze(-1)

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
