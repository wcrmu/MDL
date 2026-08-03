from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.config import FixedTestEvalConfig
from src.dataloader import FeatureBatch
from src.train import (
    DistributedContext,
    _DiskBackedGroupAUC,
    _StreamingHistogramAUC,
    _StreamingTaskMonitor,
    _binary_auc,
    _group_auc,
    _print_fixed_test_eval,
    _reduce_evaluation_histograms,
    _run_fixed_test_eval,
    _task_monitor_stats_from_batch,
    _task_monitor_warning_parts,
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


def _evaluation_batch(logits: list[float], labels: list[float]) -> FeatureBatch:
    return FeatureBatch(
        features={"logits": torch.tensor(logits).unsqueeze(1)},
        labels=torch.tensor(labels).unsqueeze(1),
        label_mask=None,
        scenario_id=torch.zeros(len(labels), dtype=torch.long),
        group_id=[],
    )


class EvaluationMetricTest(unittest.TestCase):
    def test_fixed_test_eval_reads_full_manifest_and_restores_training(self) -> None:
        model = _ModeTrackingEvaluationModel().train()
        config = SimpleNamespace(
            runtime=SimpleNamespace(precision="fp32"),
            data=SimpleNamespace(
                train=object(),
                test=SimpleNamespace(
                    inputs=("part-0.parquet", "part-1.parquet"),
                    reader=SimpleNamespace(device_prefetch_batches=0),
                ),
            ),
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
            _evaluation_batch([-2.0, 2.0], [0.0, 1.0]),
            _evaluation_batch([-1.0, 1.0], [0.0, 1.0]),
        ]

        with patch(
            "src.train.iter_feature_batches",
            return_value=iter(batches),
        ) as test_reader, patch(
            "src.train._non_blocking_transfer",
            return_value=False,
        ):
            result = _run_fixed_test_eval(
                config,
                model,
                {},
                context,
                FixedTestEvalConfig(
                    enabled=True,
                    auc_bins=128,
                ),
                fallback_batch=None,
            )

        test_reader.assert_called_once()
        self.assertTrue(model.training)
        self.assertEqual(model.training_modes, [False, False])
        self.assertEqual(result.rows, 4)
        self.assertEqual(result.files, 2)
        self.assertEqual(result.metrics["click"]["auc"], 1.0)
        self.assertNotIn("copc", result.metrics["click"])
        self.assertEqual(result.metrics["click"]["examples"], 4)
        self.assertEqual(result.metrics["click"]["positives"], 2)
        self.assertEqual(result.metrics["click"]["negatives"], 2)
        self.assertIsNotNone(result.metrics["click"]["loss"])
        self.assertAlmostEqual(
            float(result.metrics["click"]["prob_mean"]),
            float(torch.sigmoid(torch.tensor([-2.0, 2.0, -1.0, 1.0])).mean()),
            places=5,
        )
        self.assertAlmostEqual(
            float(result.metrics["click"]["logit_mean"]),
            0.0,
            places=5,
        )
        self.assertGreater(float(result.metrics["click"]["logit_std"]), 0.0)

    def test_fixed_test_eval_closes_its_reader(self) -> None:
        model = _ModeTrackingEvaluationModel().train()
        config = SimpleNamespace(
            runtime=SimpleNamespace(precision="fp32"),
            data=SimpleNamespace(
                train=object(),
                test=SimpleNamespace(
                    inputs=("part-0.parquet",),
                    reader=SimpleNamespace(device_prefetch_batches=0),
                ),
            ),
            task_names=["click"],
        )
        context = DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
        )

        class _CloseableIterator:
            def __init__(self) -> None:
                self._iterator = iter(
                    [_evaluation_batch([-2.0, 2.0], [0.0, 1.0])]
                )
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._iterator)

            def close(self) -> None:
                self.closed = True

        reader = _CloseableIterator()
        with patch(
            "src.train.iter_feature_batches",
            return_value=reader,
        ), patch("src.train._non_blocking_transfer", return_value=False):
            _run_fixed_test_eval(
                config,
                model,
                {},
                context,
                FixedTestEvalConfig(enabled=True, auc_bins=128),
                fallback_batch=None,
            )

        self.assertTrue(reader.closed)

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

    def test_task_monitor_detects_probability_collapse(self) -> None:
        monitor = _StreamingTaskMonitor()
        logits = torch.full((8,), 8.0)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        monitor.update(logits, labels)
        stats = monitor.compute()
        self.assertGreater(float(stats["prob_mean"]), 0.99)
        self.assertLess(float(stats["logit_std"]), 1.0e-6)
        self.assertIn(
            "prob_mean=",
            " ".join(_task_monitor_warning_parts(prob_mean=float(stats["prob_mean"]))),
        )

    def test_task_monitor_stats_respect_label_mask(self) -> None:
        logits = torch.tensor([[8.0, -8.0], [8.0, -8.0]])
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        mask = torch.tensor([[True, False], [True, False]])
        stats = _task_monitor_stats_from_batch(logits, labels, mask, task_count=2)
        self.assertGreater(float(stats[0]["prob_mean"]), 0.99)
        self.assertIsNone(stats[1]["prob_mean"])

    def test_fixed_test_print_emits_metrics_and_warnings(self) -> None:
        result = type(
            "FixedTestEvalResult",
            (),
            {
                "rows": 10,
                "files": 4,
                "elapsed_seconds": 0.01,
                "metrics": {
                    "fst_cart": {
                        "auc": 0.43,
                        "loss": 0.9,
                        "prob_mean": 0.97,
                        "logit_mean": 3.5,
                        "logit_std": 0.1,
                        "examples": 10,
                        "positives": 2,
                        "negatives": 8,
                    }
                },
            },
        )()
        lines: list[str] = []
        with patch("builtins.print", side_effect=lambda *args, **_kwargs: lines.append(" ".join(str(a) for a in args))):
            _print_fixed_test_eval(
                3000,
                FixedTestEvalConfig(enabled=True, files_per_rank=4),
                result,
            )
        task_line = next(
            line for line in lines if line.startswith("Fixed test eval task")
        )
        self.assertIn("prob_mean=0.970000", task_line)
        self.assertIn("logit_mean=3.500000", task_line)
        self.assertIn("logloss=0.900000", task_line)
        self.assertNotIn("copc=", task_line)
        warning_line = next(
            line for line in lines if line.startswith("Fixed test eval warning")
        )
        self.assertIn("prob_mean=", warning_line)
        self.assertNotIn("copc=", warning_line)
        self.assertIn("auc=", warning_line)


if __name__ == "__main__":
    unittest.main()
