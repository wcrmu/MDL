from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import unittest

import torch

from src.benchmark import BenchmarkOptions, run_benchmark
from src.config import load_app_config
from src.modules.attention import validate_varlen_inputs
from src.train import (
    _attention_runtime_description,
    _requires_varlen_attention,
    attention_runtime_description,
    evaluate_mdl,
    needs_padded_sdpa_flash,
    needs_varlen_flash,
    predict_mdl,
    train_mdl,
    varlen_attention_reasons,
)


ROOT = Path(__file__).resolve().parents[1]


def _flash_config(config):
    return replace(
        config,
        runtime=replace(config.runtime, attention_backend="flash"),
    )


class AttentionCapabilityHelperTest(unittest.TestCase):
    def test_mdl_rankmixer_longer_requires_varlen_for_flash(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        self.assertEqual(config.model.name, "mdl_rankmixer")
        self.assertTrue(
            any(sequence.encoder == "longer" for sequence in config.sequences)
        )
        self.assertTrue(needs_varlen_flash(config))
        self.assertTrue(_requires_varlen_attention(config))
        reasons = varlen_attention_reasons(config)
        self.assertTrue(any(item.startswith("LONGER sequences=") for item in reasons))

        with (
            patch(
                "src.train.varlen_attention_available",
                return_value=False,
            ),
            patch.object(
                torch.backends.cuda,
                "is_flash_attention_available",
                return_value=True,
                create=True,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, r"varlen\.varlen_attn.*sdpa"):
                attention_runtime_description(
                    _flash_config(config),
                    torch.device("cuda"),
                )

    def test_sdpa_log_uses_flash_path_requires_fields(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        self.assertEqual(config.runtime.attention_backend, "sdpa")
        with patch(
            "src.train.varlen_attention_available",
            return_value=False,
        ):
            description = attention_runtime_description(config, torch.device("cuda"))
            # train.py re-exports the same helper under a private alias.
            self.assertEqual(
                _attention_runtime_description(config, torch.device("cuda")),
                description,
            )
        self.assertIn("resolved=padded_sdpa", description)
        self.assertIn("flash_path_requires_varlen=True", description)
        self.assertIn("flash_path_requires_padded_sdpa=True", description)
        self.assertIn("varlen_api_available=False", description)
        self.assertNotRegex(description, r"(?<![_\w])requires_varlen=")

    def test_mdl_rankmixer_targets_2xh100_sdpa(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        self.assertEqual(config.runtime.nproc_per_node, 2)
        self.assertEqual(config.runtime.attention_backend, "sdpa")
        self.assertFalse(config.runtime.compile)
        self.assertEqual(config.runtime.precision, "bf16")
        self.assertEqual(config.training.sparse_optimizer, "rowwise_adagrad")
        self.assertEqual(config.training.sparse_update_mode, "ddp_synced_adagrad")

    def test_mixed_attention_needs_both_flash_capabilities(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        self.assertTrue(
            any(sequence.encoder == "longer" for sequence in config.sequences)
        )
        self.assertTrue(config.model.use_task_tokens)
        self.assertTrue(config.model.use_task_feature_interaction)
        self.assertTrue(config.model.use_scenario_tokens)
        self.assertTrue(config.model.use_scenario_feature_interaction)
        self.assertTrue(needs_varlen_flash(config))
        self.assertTrue(needs_padded_sdpa_flash(config))

        flash_config = _flash_config(config)
        with (
            patch(
                "src.train.varlen_attention_available",
                return_value=True,
            ),
            patch(
                "src.train._ordinary_sdpa_flash_available",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "ordinary SDPA FlashAttention"):
                attention_runtime_description(flash_config, torch.device("cuda"))

        with (
            patch(
                "src.train.varlen_attention_available",
                return_value=True,
            ),
            patch(
                "src.train._ordinary_sdpa_flash_available",
                return_value=True,
            ),
        ):
            description = attention_runtime_description(
                flash_config,
                torch.device("cuda"),
            )
        self.assertIn("resolved=varlen_flash+padded_sdpa_flash", description)
        self.assertIn("flash_path_requires_varlen=True", description)
        self.assertIn("flash_path_requires_padded_sdpa=True", description)

    def test_padded_flash_capability_matrix(self) -> None:
        """Padded Flash is only required when MDL DomainAwareAttention is built."""

        cases = (
            ("rankmixer", True, False, None),
            ("onetrans", True, False, None),
            ("mdl_rankmixer", True, True, None),
            (
                "mdl_rankmixer",
                True,
                False,
                {
                    "use_task_feature_interaction": False,
                    "use_scenario_feature_interaction": False,
                },
            ),
            ("mdl_onetrans", True, True, None),
        )
        for model_name, expect_varlen, expect_padded, model_overrides in cases:
            with self.subTest(model=model_name, overrides=model_overrides):
                config = load_app_config(ROOT / "configs" / f"{model_name}.yaml")
                if model_overrides is not None:
                    config = replace(
                        config,
                        model=replace(config.model, **model_overrides),
                    )
                self.assertEqual(needs_varlen_flash(config), expect_varlen)
                self.assertEqual(needs_padded_sdpa_flash(config), expect_padded)

                if expect_varlen and not expect_padded:
                    flash_config = _flash_config(config)
                    with (
                        patch(
                            "src.train.varlen_attention_available",
                            return_value=True,
                        ),
                        patch(
                            "src.train._ordinary_sdpa_flash_available",
                            return_value=False,
                        ),
                    ):
                        description = attention_runtime_description(
                            flash_config,
                            torch.device("cuda"),
                        )
                    self.assertIn("resolved=varlen_flash ", description)
                    self.assertIn("flash_path_requires_padded_sdpa=False", description)

    def test_validate_varlen_inputs_rejects_unsupported_tensors(self) -> None:
        query = torch.zeros(2, 2, 4, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "CUDA tensors"):
            validate_varlen_inputs(
                query,
                query,
                query,
                dropout_p=0.0,
                training=False,
            )


class AttentionEntryPreflightOrderTest(unittest.TestCase):
    def _fake_context(self) -> MagicMock:
        fake_context = MagicMock()
        fake_context.device = torch.device("cpu")
        fake_context.rank = 0
        fake_context.world_size = 1
        fake_context.enabled = False
        fake_context.local_rank = 0
        return fake_context

    def test_train_preflight_runs_before_scenario_discovery(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        with (
            patch("src.train._setup_distributed", return_value=self._fake_context()),
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

    def test_evaluate_preflight_runs_before_scenario_discovery(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        with (
            patch("src.train._setup_distributed", return_value=self._fake_context()),
            patch(
                "src.train._attention_runtime_description",
                side_effect=RuntimeError("preflight failed"),
            ),
            patch("src.train._resolve_distributed_auto_scenarios") as resolve_scenarios,
            patch("src.train._load_inference_model") as load_inference,
            patch("src.train._cleanup_distributed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                evaluate_mdl(config, allow_random_init=True)
        resolve_scenarios.assert_not_called()
        load_inference.assert_not_called()

    def test_predict_preflight_runs_before_scenario_and_pyarrow(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        with (
            patch("src.train._select_device", return_value=torch.device("cpu")),
            patch(
                "src.train._attention_runtime_description",
                side_effect=RuntimeError("preflight failed"),
            ),
            patch("src.train.resolve_auto_scenarios") as resolve_scenarios,
            patch("src.train._require_pyarrow") as require_pyarrow,
            patch("src.train.build_model") as build_model,
        ):
            with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                predict_mdl(config, allow_random_init=True)
        resolve_scenarios.assert_not_called()
        require_pyarrow.assert_not_called()
        build_model.assert_not_called()

    def test_benchmark_compute_preflight_before_compute_body(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        options = BenchmarkOptions(
            mode="compute",
            warmup_steps=0,
            measured_steps=1,
            profile_steps=0,
        )
        with (
            patch(
                "src.benchmark._setup_distributed",
                return_value=self._fake_context(),
            ),
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


class AttentionEntryMatrixTest(unittest.TestCase):
    """Control-flow matrix: SDPA passes; flash without varlen fails early."""

    def _assert_sdpa_ok(self, config) -> None:
        with patch(
            "src.train.varlen_attention_available",
            return_value=False,
        ):
            description = attention_runtime_description(config, torch.device("cuda"))
        self.assertIn("requested=sdpa", description)
        self.assertIn("strict=false", description)

    def _assert_flash_without_varlen_fails(self, config) -> None:
        flash_config = _flash_config(config)
        with (
            patch(
                "src.train.varlen_attention_available",
                return_value=False,
            ),
            patch(
                "src.train._ordinary_sdpa_flash_available",
                return_value=True,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "varlen"):
                attention_runtime_description(flash_config, torch.device("cuda"))

    def _assert_flash_with_varlen_ok(self, config) -> None:
        flash_config = _flash_config(config)
        with (
            patch(
                "src.train.varlen_attention_available",
                return_value=True,
            ),
            patch(
                "src.train._ordinary_sdpa_flash_available",
                return_value=True,
            ),
        ):
            description = attention_runtime_description(
                flash_config,
                torch.device("cuda"),
            )
        self.assertIn("requested=flash", description)
        self.assertIn("strict=true", description)

    def test_capability_matrix_for_mdl_rankmixer(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        self._assert_sdpa_ok(config)
        self._assert_flash_without_varlen_fails(config)
        self._assert_flash_with_varlen_ok(config)


class TunerDefaultsAndPreflightTest(unittest.TestCase):
    def test_nproc_defaults_from_yaml_and_cli_overrides(self) -> None:
        from scripts import tune_a100_batch_size as tuner

        config_path = ROOT / "configs" / "mdl_rankmixer.yaml"
        with (
            patch.object(
                tuner,
                "_run_attention_preflight",
                return_value="ok",
            ),
            patch.object(tuner, "_execute_tuning", return_value=0) as execute,
            patch.object(
                tuner,
                "generate_synthetic_agg_dataset",
                side_effect=AssertionError("should not generate"),
            ),
        ):
            with patch(
                "sys.argv",
                [
                    "tune_a100_batch_size.py",
                    "--config",
                    str(config_path),
                    "--compute-only",
                    "--candidate-batches",
                    "8",
                    "--steps",
                    "1",
                    "--warmup-steps",
                    "0",
                ],
            ):
                self.assertEqual(tuner.main(), 0)
            self.assertEqual(execute.call_args.args[0].nproc_per_node, 2)

        with (
            patch.object(tuner, "_run_attention_preflight", return_value="ok"),
            patch.object(tuner, "_execute_tuning", return_value=0) as execute,
        ):
            with patch(
                "sys.argv",
                [
                    "tune_a100_batch_size.py",
                    "--config",
                    str(config_path),
                    "--compute-only",
                    "--nproc-per-node",
                    "1",
                    "--candidate-batches",
                    "8",
                    "--steps",
                    "1",
                    "--warmup-steps",
                    "0",
                ],
            ):
                self.assertEqual(tuner.main(), 0)
            self.assertEqual(execute.call_args.args[0].nproc_per_node, 1)

    def test_tuner_preflight_blocks_synthetic_generation(self) -> None:
        from scripts import tune_a100_batch_size as tuner

        config_path = ROOT / "configs" / "mdl_rankmixer.yaml"
        with (
            patch.object(
                tuner,
                "_run_attention_preflight",
                side_effect=RuntimeError("preflight failed"),
            ),
            patch.object(
                tuner,
                "generate_synthetic_agg_dataset",
            ) as generate,
            patch.object(tuner, "_run_reader_trial") as reader,
            patch.object(tuner, "_run_candidate") as compute,
            patch.object(tuner, "_execute_tuning") as execute,
        ):
            with patch(
                "sys.argv",
                [
                    "tune_a100_batch_size.py",
                    "--config",
                    str(config_path),
                    "--candidate-batches",
                    "8",
                    "--steps",
                    "1",
                    "--warmup-steps",
                    "0",
                ],
            ):
                with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                    tuner.main()
        generate.assert_not_called()
        reader.assert_not_called()
        compute.assert_not_called()
        execute.assert_not_called()

    def test_peak_tflops_defaults_to_none(self) -> None:
        from scripts import tune_a100_batch_size as tuner

        config_path = ROOT / "configs" / "mdl_rankmixer.yaml"
        with (
            patch.object(tuner, "_run_attention_preflight", return_value="ok"),
            patch.object(tuner, "_execute_tuning", return_value=0) as execute,
        ):
            with patch(
                "sys.argv",
                [
                    "tune_a100_batch_size.py",
                    "--config",
                    str(config_path),
                    "--compute-only",
                    "--candidate-batches",
                    "8",
                    "--steps",
                    "1",
                    "--warmup-steps",
                    "0",
                ],
            ):
                self.assertEqual(tuner.main(), 0)
            self.assertIsNone(execute.call_args.args[0].peak_tflops)

    def test_compute_only_forwards_synthetic_scenario_count(self) -> None:
        from scripts import tune_a100_batch_size as tuner

        args = SimpleNamespace(
            config=ROOT / "configs" / "mdl_rankmixer.yaml",
            compute_only=True,
            warmup_steps=0,
            steps=1,
            profile_steps=0,
            nproc_per_node=2,
            peak_tflops=None,
            reserve_hbm_gib=32.0,
            candidates_per_request=4,
            sequence_lengths={"impr": 8},
            scenario_count=32,
            omp_num_threads=4,
        )
        with (
            patch.object(tuner, "_free_port", return_value=29501),
            patch.object(tuner.subprocess, "run") as run,
        ):
            run.return_value = SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="CUDA out of memory",
            )
            result = tuner._run_candidate(
                args,
                batch_size=8,
                workspace=ROOT / "artifacts",
                parquet_dir=None,
            )
        self.assertEqual(result["status"], "oom")
        command = run.call_args.args[0]
        self.assertIn("--synthetic-scenario-count", command)
        self.assertEqual(
            command[command.index("--synthetic-scenario-count") + 1],
            "32",
        )


if __name__ == "__main__":
    unittest.main()
