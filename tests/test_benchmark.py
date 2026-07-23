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
    _resolve_benchmark_scenarios,
    _synthetic_feature_batch,
    _synthetic_vocab_maps,
    _trace,
)
from src.config import load_app_config
from src.dataloader import FeatureBatch
from src.model import build_model
from src.train import (
    DistributedContext,
    _batch_input_token_count,
    _batch_padded_token_slots,
)


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

    def test_agg_compute_controls_are_compute_only(self) -> None:
        BenchmarkOptions(
            mode="compute",
            reserve_hbm_gib=24.0,
            candidates_per_request=4,
        ).validate()
        with self.assertRaisesRegex(ValueError, "reserve_hbm_gib"):
            BenchmarkOptions(mode="data", reserve_hbm_gib=1.0).validate()
        with self.assertRaisesRegex(ValueError, "candidates_per_request"):
            BenchmarkOptions(mode="embedding", candidates_per_request=2).validate()

    def test_named_sequence_lengths_require_positive_integers(self) -> None:
        BenchmarkOptions(
            mode="compute",
            sequence_lengths={"hist": 32},
        ).validate()
        with self.assertRaisesRegex(ValueError, "sequence_lengths"):
            BenchmarkOptions(
                mode="compute",
                sequence_lengths={"hist": 0},
            ).validate()

    def test_synthetic_scenario_count_must_be_positive(self) -> None:
        BenchmarkOptions(mode="compute", synthetic_scenario_count=1).validate()
        with self.assertRaisesRegex(ValueError, "synthetic_scenario_count"):
            BenchmarkOptions(mode="compute", synthetic_scenario_count=0).validate()

    def test_gpu_utilization_slo_is_bounded_and_compute_only(self) -> None:
        BenchmarkOptions(
            mode="compute",
            min_gpu_utilization_percent=60.0,
        ).validate()
        with self.assertRaisesRegex(ValueError, r"\[0, 100\]"):
            BenchmarkOptions(
                mode="compute",
                min_gpu_utilization_percent=101.0,
            ).validate()
        with self.assertRaisesRegex(ValueError, "compute or end-to-end"):
            BenchmarkOptions(
                mode="data",
                min_gpu_utilization_percent=60.0,
            ).validate()


class BenchmarkScenarioResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.config = load_app_config(cls.root / "configs" / "mdl_rankmixer.yaml")

    def _fake_context(self) -> DistributedContext:
        return DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
        )

    def test_compute_keeps_fixed_coarse_scenarios_without_data_scan(self) -> None:
        self.assertEqual(self.config.data.train.inputs, ())
        self.assertFalse(self.config.scenarios.auto_discover)
        self.assertEqual(self.config.scenarios.names, ("search", "recommendation"))
        options = BenchmarkOptions(
            mode="compute",
            synthetic_scenario_count=4,
            warmup_steps=0,
            measured_steps=1,
            profile_steps=0,
        )
        with patch(
            "src.benchmark._resolve_distributed_auto_scenarios"
        ) as real_discovery:
            resolved = _resolve_benchmark_scenarios(
                self.config,
                self._fake_context(),
                options,
            )
        real_discovery.assert_not_called()
        self.assertEqual(resolved.scenarios.names, ("search", "recommendation"))
        self.assertFalse(resolved.scenarios.auto_discover)
        self.assertEqual(
            [token.name for token in resolved.tokenization.scenario_tokens],
            ["search", "recommendation", "global"],
        )

    def test_embedding_keeps_fixed_coarse_scenarios_without_data_scan(self) -> None:
        options = BenchmarkOptions(
            mode="embedding",
            synthetic_scenario_count=3,
            warmup_steps=0,
            measured_steps=1,
            profile_steps=0,
        )
        with patch(
            "src.benchmark._resolve_distributed_auto_scenarios"
        ) as real_discovery:
            resolved = _resolve_benchmark_scenarios(
                self.config,
                self._fake_context(),
                options,
            )
        real_discovery.assert_not_called()
        self.assertEqual(resolved.scenarios.names, ("search", "recommendation"))

    def test_compute_uses_synthetic_scenarios_when_auto_discover_enabled(self) -> None:
        from dataclasses import replace

        # Non-MDL configs resolve synthetic scenes without the MDL prior
        # template feature that production mdl_* YAMLs no longer ship.
        base = load_app_config(self.root / "configs" / "rankmixer.yaml")
        auto_config = replace(
            base,
            scenarios=replace(
                base.scenarios,
                auto_discover=True,
                names=("__auto__",),
                source="scene_id",
                source_encoding="raw",
            ),
        )
        options = BenchmarkOptions(
            mode="compute",
            synthetic_scenario_count=4,
            warmup_steps=0,
            measured_steps=1,
            profile_steps=0,
        )
        with patch(
            "src.benchmark._resolve_distributed_auto_scenarios"
        ) as real_discovery:
            resolved = _resolve_benchmark_scenarios(
                auto_config,
                self._fake_context(),
                options,
            )
        real_discovery.assert_not_called()
        self.assertEqual(resolved.scenarios.names, ("0", "1", "2", "3"))
        self.assertFalse(resolved.scenarios.auto_discover)

    def test_end_to_end_still_uses_real_discovery(self) -> None:
        options = BenchmarkOptions(
            mode="end-to-end",
            warmup_steps=0,
            measured_steps=1,
            profile_steps=0,
        )
        sentinel = object()
        with patch(
            "src.benchmark._resolve_distributed_auto_scenarios",
            return_value=sentinel,
        ) as discovery:
            resolved = _resolve_benchmark_scenarios(
                self.config,
                self._fake_context(),
                options,
            )
        discovery.assert_called_once()
        self.assertIs(resolved, sentinel)

    def test_data_mode_still_uses_real_discovery(self) -> None:
        options = BenchmarkOptions(
            mode="data",
            warmup_steps=0,
            measured_steps=1,
            profile_steps=0,
        )
        with patch(
            "src.benchmark._resolve_distributed_auto_scenarios",
            return_value=self.config,
        ) as discovery:
            _resolve_benchmark_scenarios(
                self.config,
                self._fake_context(),
                options,
            )
        discovery.assert_called_once()


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
        self.assertTrue(collector._measurement_stopped)

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

    def test_report_prefers_synchronized_measurement_wall_time(self) -> None:
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
            measurement_elapsed_seconds=8.0,
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

        self.assertEqual(report.elapsed_seconds, 8.0)
        self.assertEqual(report.mean_step_seconds, 4.0)
        self.assertEqual(report.samples_per_second, 0.75)

    def test_sequence_token_metrics_exclude_categorical_bags(self) -> None:
        row_indices = torch.tensor([0, 0, 1, 1])
        batch = FeatureBatch(
            features={
                "bag": {
                    "values": torch.ones(8, dtype=torch.long),
                    "lengths": torch.full((4,), 2, dtype=torch.long),
                },
                "history": {
                    "fields": {"item": torch.ones(2, 3, dtype=torch.long)},
                    "lengths": torch.tensor([2, 1]),
                    "row_indices": row_indices,
                },
            },
            labels=None,
            label_mask=None,
            scenario_id=torch.zeros(4, dtype=torch.long),
            group_id=[],
        )

        self.assertEqual(_batch_input_token_count(batch), 6)
        self.assertEqual(_batch_padded_token_slots(batch), 12)


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
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
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
        config = load_app_config(
            root / "configs" / "reference" / "rankmixer_perf.yaml"
        )
        model = build_model(config, {}, embedding_size_override=2)

        self.assertIs(
            model.encoder_bank.embeddings["item_id"],
            model.encoder_bank.embeddings["hist__item_id"],
        )

    def test_synthetic_agg_batch_keeps_sequences_at_request_granularity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")

        batch = _synthetic_feature_batch(
            config,
            torch.device("cpu"),
            batch_size=6,
            sequence_length=2,
            seed=7,
            candidates_per_request=3,
        )

        sequence = batch.features[config.sequences[0].name]
        self.assertEqual(sequence["lengths"].tolist(), [2, 2])
        self.assertEqual(sequence["row_indices"].tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(batch.scenario_id.numel(), 6)

    def test_synthetic_batch_uses_named_sequence_length(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        sequence_name = config.sequences[0].name

        batch = _synthetic_feature_batch(
            config,
            torch.device("cpu"),
            batch_size=2,
            sequence_length=None,
            seed=7,
            sequence_lengths={sequence_name: 3},
        )

        self.assertEqual(
            batch.features[sequence_name]["lengths"].tolist(),
            [3, 3],
        )


if __name__ == "__main__":
    unittest.main()
