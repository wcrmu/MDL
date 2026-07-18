from __future__ import annotations

import unittest

import torch

from src.train import (
    _DiskBackedGroupAUC,
    _StreamingHistogramAUC,
    _binary_auc,
    _group_auc,
)


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

    def test_streaming_histogram_matches_separated_exact_scores(self) -> None:
        scores = torch.tensor([0.1, 0.9, 0.8, 0.2])
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        accumulator = _StreamingHistogramAUC(1024)
        accumulator.update(scores[:2], labels[:2])
        accumulator.update(scores[2:], labels[2:])
        self.assertEqual(accumulator.compute(), _binary_auc(scores, labels))
        self.assertEqual(accumulator.counts(), (4, 2, 2))

    def test_disk_group_histogram_aggregates_across_batches(self) -> None:
        accumulator = _DiskBackedGroupAUC(1024)
        try:
            accumulator.add(
                0,
                ["a", "b"],
                torch.tensor([0.1, 0.8]),
                torch.tensor([0.0, 0.0]),
                torch.tensor([[True], [True]]),
            )
            accumulator.add(
                0,
                ["a", "b", "c"],
                torch.tensor([0.9, 0.2, 0.7]),
                torch.tensor([1.0, 1.0, 1.0]),
                torch.tensor([[True], [True], [True]]),
            )
            self.assertEqual(accumulator.compute(0, -1), 0.5)
            self.assertEqual(accumulator.compute(0, 0), 0.5)
        finally:
            accumulator.close()


if __name__ == "__main__":
    unittest.main()
