from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

import torch

from src.benchmark import BenchmarkOptions, run_benchmark
from src.config import load_app_config
from src.train import (
    _attention_runtime_description,
    _requires_varlen_attention,
    _varlen_attention_reasons,
    train_mdl,
)


ROOT = Path(__file__).resolve().parents[1]


class AttentionPreflightTest(unittest.TestCase):
    def test_mdl_rankmixer_longer_requires_varlen_for_flash(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        self.assertEqual(config.model.name, "mdl_rankmixer")
        self.assertTrue(
            any(sequence.encoder == "longer" for sequence in config.sequences)
        )
        self.assertTrue(_requires_varlen_attention(config))
        reasons = _varlen_attention_reasons(config)
        self.assertTrue(any(item.startswith("LONGER sequences=") for item in reasons))

        flash_config = replace(
            config,
            runtime=replace(config.runtime, attention_backend="flash"),
        )
        with (
            patch("src.train.varlen_attention_available", return_value=False),
            patch.object(
                torch.backends.cuda,
                "is_flash_attention_available",
                return_value=True,
                create=True,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, r"varlen.*sdpa.*2\.10"):
                _attention_runtime_description(flash_config, torch.device("cuda"))

    def test_sdpa_does_not_require_varlen_api(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        self.assertEqual(config.runtime.attention_backend, "sdpa")
        self.assertFalse(config.runtime.compile)
        self.assertEqual(config.runtime.nproc_per_node, 2)
        with patch("src.train.varlen_attention_available", return_value=False):
            description = _attention_runtime_description(config, torch.device("cuda"))
        self.assertIn("resolved=padded_sdpa", description)
        self.assertIn("requires_varlen=True", description)
        self.assertIn("varlen_api_available=False", description)

    def test_train_preflight_runs_before_scenario_discovery(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        fake_context = MagicMock()
        fake_context.device = torch.device("cpu")
        fake_context.rank = 0
        fake_context.world_size = 1
        fake_context.enabled = False
        with (
            patch("src.train._setup_distributed", return_value=fake_context),
            patch(
                "src.train._attention_runtime_description",
                side_effect=RuntimeError("preflight failed"),
            ),
            patch("src.train._resolve_distributed_auto_scenarios") as resolve_scenarios,
            patch("src.train.load_vocab_maps") as load_vocab,
            patch("src.train.build_model") as build_model,
        ):
            with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                train_mdl(config, max_steps=1, log_steps=False)
        resolve_scenarios.assert_not_called()
        load_vocab.assert_not_called()
        build_model.assert_not_called()

    def test_benchmark_compute_preflight_before_compute_body(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        options = BenchmarkOptions(
            mode="compute",
            warmup_steps=0,
            measured_steps=1,
            profile_steps=0,
        )
        fake_context = MagicMock()
        fake_context.device = torch.device("cpu")
        fake_context.rank = 0
        fake_context.world_size = 1
        fake_context.enabled = False
        fake_context.local_rank = 0
        with (
            patch("src.benchmark._setup_distributed", return_value=fake_context),
            patch(
                "src.benchmark._attention_runtime_description",
                side_effect=RuntimeError("varlen missing"),
            ),
            patch("src.benchmark._resolve_distributed_auto_scenarios") as resolve_scenarios,
            patch("src.benchmark._benchmark_compute") as benchmark_compute,
        ):
            with self.assertRaisesRegex(RuntimeError, "varlen missing"):
                run_benchmark(config, options)
        resolve_scenarios.assert_not_called()
        benchmark_compute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
