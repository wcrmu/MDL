from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.config import QuickEvalConfig
from src.dataloader import FeatureBatch
from src.train import (
    DistributedContext,
    _DiskBackedGroupAUC,
    _StreamingHistogramAUC,
    _binary_auc,
    _group_auc,
    _reduce_evaluation_histograms,
    _run_training_quick_eval,
)


class _ModeTrackingEvaluationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.training_modes: list[bool] = []

    def forward(
        self,
        features: dict[str, torch.Tensor],
        scenario_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del scenario_id
        self.training_modes.append(self.training)
        return {"logits": features["logits"]}


def _quick_eval_batch(logits: list[float], labels: list[float]) -> FeatureBatch:
    return FeatureBatch(
        features={"logits": torch.tensor(logits).unsqueeze(1)},
        labels=torch.tensor(labels).unsqueeze(1),
        label_mask=None,
        scenario_id=torch.zeros(len(labels), dtype=torch.long),
        group_id=[],
    )


class EvaluationMetricTest(unittest.TestCase):
    def test_training_quick_eval_stages_exact_batches_and_restores_training(self) -> None:
        model = _ModeTrackingEvaluationModel().train()
        config = SimpleNamespace(
            runtime=SimpleNamespace(precision="fp32"),
            data=SimpleNamespace(train=object(), test=object()),
            task_names=["click"],
        )
        context = DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
        )
        batches = [
            _quick_eval_batch([-2.0, 2.0], [0.0, 1.0]),
            _quick_eval_batch([-1.0, 1.0], [0.0, 1.0]),
        ]

        with patch("src.train.iter_feature_batches") as separate_reader:
            result, staged_batches = _run_training_quick_eval(
                config,
                model,
                {},
                context,
                QuickEvalConfig(
                    enabled=True,
                    max_batches=2,
                    split="train",
                    auc_bins=128,
                ),
                fallback_batch=None,
                training_batch_iterator=iter(batches),
            )

        separate_reader.assert_not_called()
        self.assertTrue(model.training)
        self.assertEqual(model.training_modes, [False, False])
        self.assertEqual(len(staged_batches), 2)
        self.assertIs(staged_batches[0], batches[0])
        self.assertIs(staged_batches[1], batches[1])
        self.assertEqual(result.rows, 4)
        self.assertEqual(result.metrics["click"]["auc"], 1.0)
        self.assertEqual(result.metrics["click"]["examples"], 4)
        self.assertEqual(result.metrics["click"]["positives"], 2)
        self.assertEqual(result.metrics["click"]["negatives"], 2)

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

    def test_distributed_histogram_reduction_sums_counts_and_rows(self) -> None:
        accumulator = _StreamingHistogramAUC(16)
        accumulator.update(torch.tensor([0.1, 0.9]), torch.tensor([0.0, 1.0]))
        context = DistributedContext(
            enabled=True,
            rank=0,
            local_rank=0,
            world_size=2,
            device=torch.device("cpu"),
        )

        def double(value: torch.Tensor, **_kwargs: object) -> None:
            value.mul_(2)

        with patch("src.train.torch_dist.all_reduce", side_effect=double):
            rows = _reduce_evaluation_histograms(context, [[accumulator]], 2)

        self.assertEqual(rows, 4)
        self.assertEqual(accumulator.counts(), (4, 2, 2))
        self.assertEqual(accumulator.compute(), 1.0)

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
