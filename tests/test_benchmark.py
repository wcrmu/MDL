from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.benchmark import (
    BenchmarkOptions,
    LocalBenchmarkSummary,
    ProfilerSummary,
    _TraceCollector,
    _build_report,
    _id_embedding_modules,
    _nvidia_smi_device_selector,
    _percentile,
    _replace_id_embeddings_with_synthetic,
    _synthetic_vocab_maps,
    _trace,
)
from src.config import load_app_config
from src.model import build_model
from src.train import DistributedContext


class BenchmarkOptionsTest(unittest.TestCase):
    def test_rejects_invalid_measurement_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "measured_steps"):
            BenchmarkOptions(mode="data", measured_steps=0).validate()

    def test_percentile_uses_nearest_rank(self) -> None:
        self.assertEqual(_percentile([4.0, 1.0, 3.0, 2.0], 0.95), 4.0)
        self.assertEqual(_percentile([], 0.95), 0.0)

    def test_batch_override_is_compute_only(self) -> None:
        BenchmarkOptions(mode="compute", batch_size=8).validate()
        with self.assertRaisesRegex(ValueError, "compute mode"):
            BenchmarkOptions(mode="data", batch_size=8).validate()


class TraceCollectorTest(unittest.TestCase):
    def test_excludes_warmup_and_keeps_requested_steps(self) -> None:
        collector = _TraceCollector(
            warmup_steps=1,
            measured_steps=2,
            device=torch.device("cpu"),
        )
        for step in range(1, 5):
            collector.observe(
                _trace(
                    step=step,
                    rows=2,
                    input_tokens=4,
                    step_seconds=float(step),
                ),
                host_batch_bytes=step * 10,
            )
        summary = collector.finish(0, ProfilerSummary())
        self.assertEqual([trace.step for trace in summary.traces], [2, 3])
        self.assertEqual(summary.host_batch_bytes_peak, 30)

    def test_report_uses_rank_max_step_and_global_samples(self) -> None:
        traces = (
            _trace(step=1, rows=3, input_tokens=6, step_seconds=2.0),
            _trace(step=2, rows=3, input_tokens=6, step_seconds=4.0),
        )
        local = LocalBenchmarkSummary(
            rank=0,
            traces=traces,
            peak_hbm_allocated_bytes=0,
            peak_hbm_reserved_bytes=0,
            process_peak_rss_bytes=100,
            cpu_utilization_percent=50.0,
            gpu_utilization_percent=None,
        )
        context = DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
        )
        with patch("src.benchmark._environment", return_value={"device": "cpu"}):
            report = _build_report(
                SimpleNamespace(),
                context,
                BenchmarkOptions(mode="compute", warmup_steps=0, measured_steps=2),
                local,
            )
        self.assertEqual(report.samples, 6)
        self.assertEqual(report.input_tokens, 12)
        self.assertEqual(report.elapsed_seconds, 6.0)
        self.assertEqual(report.samples_per_second, 1.0)
        self.assertEqual(report.p95_step_seconds, 4.0)
        self.assertEqual(report.benchmark_options["measured_steps"], 2)


class GpuUtilizationSamplerTest(unittest.TestCase):
    def test_visible_device_index_maps_to_physical_nvidia_smi_selector(self) -> None:
        with patch.dict(
            "src.benchmark.os.environ",
            {"CUDA_VISIBLE_DEVICES": "2, 5, GPU-example"},
            clear=True,
        ):
            self.assertEqual(
                _nvidia_smi_device_selector(torch.device("cuda", 0)),
                "2",
            )
            self.assertEqual(
                _nvidia_smi_device_selector(torch.device("cuda", 1)),
                "5",
            )
            self.assertEqual(
                _nvidia_smi_device_selector(torch.device("cuda", 2)),
                "GPU-example",
            )

    def test_unrestricted_cuda_index_is_already_physical(self) -> None:
        with patch.dict("src.benchmark.os.environ", {}, clear=True):
            self.assertEqual(
                _nvidia_smi_device_selector(torch.device("cuda", 3)),
                "3",
            )


class SyntheticEmbeddingReplacementTest(unittest.TestCase):
    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.embeddings = nn.ModuleDict(
                {"item": nn.Embedding(16, 4, sparse=True)}
            )
            self.encoder.sequence_position_embeddings = nn.ModuleDict(
                {"hist": nn.Embedding(8, 4)}
            )

    def test_replaces_only_industrial_id_embeddings(self) -> None:
        model = self.ToyModel()
        original_position = model.encoder.sequence_position_embeddings["hist"]
        self.assertEqual(len(_id_embedding_modules(model)), 1)

        count = _replace_id_embeddings_with_synthetic(model)

        self.assertEqual(count, 1)
        self.assertIs(model.encoder.sequence_position_embeddings["hist"], original_position)
        output = model.encoder.embeddings["item"](torch.tensor([[1, 2]]))
        self.assertEqual(tuple(output.shape), (1, 2, 4))
        self.assertEqual(len(list(model.encoder.embeddings["item"].parameters())), 0)

    def test_model_build_can_cap_id_tables_before_compute_replacement(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "default.yaml")
        model = build_model(
            config,
            _synthetic_vocab_maps(config),
            embedding_size_override=2,
        )

        id_embeddings = _id_embedding_modules(model)
        self.assertGreater(len(id_embeddings), 0)
        for _name, embedding in id_embeddings:
            self.assertEqual(embedding.num_embeddings, 2)

    def test_identity_alias_reuses_embedding_without_hiding_its_bound(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "rankmixer_perf.yaml")
        model = build_model(config, {}, embedding_size_override=2)

        self.assertIs(
            model.encoder_bank.embeddings["item_id"],
            model.encoder_bank.embeddings["hist__item_id"],
        )


if __name__ == "__main__":
    unittest.main()
