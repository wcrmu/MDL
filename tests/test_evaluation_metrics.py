from __future__ import annotations

import unittest

import torch

from src.train import _binary_auc, _group_auc


class EvaluationMetricTest(unittest.TestCase):
    def test_binary_auc_handles_ordering_and_ties_exactly(self) -> None:
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])

        self.assertEqual(
            _binary_auc(torch.tensor([0.1, 0.2, 0.8, 0.9]), labels),
            1.0,
        )
        self.assertEqual(
            _binary_auc(torch.tensor([0.9, 0.8, 0.2, 0.1]), labels),
            0.0,
        )
        self.assertEqual(
            _binary_auc(torch.ones(4), labels),
            0.5,
        )

    def test_binary_auc_returns_none_for_single_class(self) -> None:
        self.assertIsNone(
            _binary_auc(torch.tensor([0.1, 0.2]), torch.tensor([1.0, 1.0]))
        )

    def test_group_auc_is_unweighted_and_skips_single_class_groups(self) -> None:
        scores = torch.tensor(
            [
                0.1, 0.9,  # group a: AUC 1
                0.8, 0.2,  # group b: AUC 0
                0.5, 0.6,  # group c: only positives, excluded
            ]
        )
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        groups = ["a", "a", "b", "b", "c", "c"]

        self.assertEqual(_group_auc(scores, labels, groups), 0.5)


if __name__ == "__main__":
    unittest.main()
