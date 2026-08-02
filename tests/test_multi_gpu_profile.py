"""World-size-aware batch derate and NCCL multi-GPU defaults."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from src.config import load_app_config
from src.train import (
    _apply_local_rank_cpu_affinity,
    _apply_world_size_training_profile,
    _configure_nccl_runtime_env,
    _local_batch_scale_for_world_size,
)


class MultiGpuProfileTest(unittest.TestCase):
    def test_local_batch_scale_defaults(self) -> None:
        os.environ.pop("MDL_LOCAL_BATCH_SCALE", None)
        self.assertEqual(_local_batch_scale_for_world_size(4), 1.0)
        self.assertEqual(_local_batch_scale_for_world_size(6), 1.0)
        self.assertEqual(_local_batch_scale_for_world_size(8), 0.75)
        self.assertEqual(
            _local_batch_scale_for_world_size(8, rankmixer_family=True), 1.0
        )
        self.assertLess(_local_batch_scale_for_world_size(16), 0.75)

    def test_local_batch_scale_env_override(self) -> None:
        with mock.patch.dict("os.environ", {"MDL_LOCAL_BATCH_SCALE": "0.5"}):
            self.assertEqual(_local_batch_scale_for_world_size(8), 0.5)

    def test_apply_profile_derates_eight_gpu_batches(self) -> None:
        os.environ.pop("MDL_LOCAL_BATCH_SCALE", None)
        os.environ.pop("MDL_GROUPED_EMB_MAX_OUTPUT_MIB", None)
        config = load_app_config("configs/onetrans.yaml")
        updated = _apply_world_size_training_profile(config, world_size=8)
        self.assertEqual(updated.training.batch_size, 960)  # 1280 * 0.75
        train_buckets = updated.data.train.reader.length_buckets
        self.assertEqual(train_buckets[0].batch_size, 960)
        self.assertEqual(updated.data.train.reader.device_prefetch_batches, 0)
        self.assertGreaterEqual(
            updated.data.train.reader.host_prepare_prefetch, 4
        )
        self.assertGreaterEqual(updated.data.train.reader.prefetch_batches, 3)
        self.assertGreaterEqual(updated.training.ddp.bucket_cap_mb, 125.0)
        self.assertEqual(os.environ.get("MDL_GROUPED_EMB_MAX_OUTPUT_MIB"), "384")

    def test_apply_profile_keeps_six_gpu_batches(self) -> None:
        os.environ.pop("MDL_LOCAL_BATCH_SCALE", None)
        os.environ.pop("MDL_GROUPED_EMB_MAX_OUTPUT_MIB", None)
        config = load_app_config("configs/onetrans.yaml")
        updated = _apply_world_size_training_profile(config, world_size=6)
        self.assertEqual(updated.training.batch_size, config.training.batch_size)
        self.assertEqual(
            updated.data.train.reader.device_prefetch_batches,
            config.data.train.reader.device_prefetch_batches,
        )
        self.assertGreaterEqual(
            updated.data.train.reader.host_prepare_prefetch, 4
        )
        self.assertGreaterEqual(updated.training.ddp.bucket_cap_mb, 100.0)
        self.assertEqual(os.environ.get("MDL_GROUPED_EMB_MAX_OUTPUT_MIB"), "512")

    def test_rankmixer_profile_deepens_prefetch_pipeline(self) -> None:
        os.environ.pop("MDL_LOCAL_BATCH_SCALE", None)
        os.environ.pop("MDL_GROUPED_EMB_MAX_OUTPUT_MIB", None)
        config = load_app_config("configs/rankmixer.yaml")
        self.assertEqual(config.runtime.activation_checkpoint, "none")
        self.assertTrue(config.runtime.cuda_graph_backbone)
        with mock.patch(
            "src.train._local_cuda_p2p_accessible", return_value=True
        ), mock.patch("src.train._small_hbm_cuda_device", return_value=False):
            updated = _apply_world_size_training_profile(config, world_size=8)
        # Large-HBM 8-GPU: keep full batch (no 0.9 derate) for util protect.
        self.assertEqual(updated.training.batch_size, 1280)
        self.assertEqual(updated.data.train.reader.device_prefetch_batches, 2)
        self.assertGreaterEqual(
            updated.data.train.reader.host_prepare_prefetch, 10
        )
        self.assertGreaterEqual(updated.data.train.reader.prefetch_batches, 6)
        self.assertGreaterEqual(updated.training.ddp.bucket_cap_mb, 250.0)
        self.assertEqual(os.environ.get("MDL_GROUPED_EMB_MAX_OUTPUT_MIB"), "1024")

    def test_rankmixer_eight_gpu_small_hbm_still_mild_derate(self) -> None:
        os.environ.pop("MDL_LOCAL_BATCH_SCALE", None)
        config = load_app_config("configs/rankmixer.yaml")
        with mock.patch(
            "src.train._local_cuda_p2p_accessible", return_value=True
        ), mock.patch("src.train._small_hbm_cuda_device", return_value=True):
            updated = _apply_world_size_training_profile(config, world_size=8)
        self.assertEqual(updated.training.batch_size, 1152)  # 1280 * 0.9

    def test_mdl_rankmixer_six_gpu_emb_cap(self) -> None:
        os.environ.pop("MDL_LOCAL_BATCH_SCALE", None)
        os.environ.pop("MDL_GROUPED_EMB_MAX_OUTPUT_MIB", None)
        config = load_app_config("configs/mdl_rankmixer.yaml")
        self.assertEqual(config.runtime.activation_checkpoint, "none")
        self.assertTrue(config.runtime.cuda_graph_backbone)
        self.assertTrue(config.training.fused_dense_optimizer)
        with mock.patch(
            "src.train._local_cuda_p2p_accessible", return_value=True
        ), mock.patch("src.train._small_hbm_cuda_device", return_value=False):
            updated = _apply_world_size_training_profile(config, world_size=6)
        self.assertEqual(updated.training.batch_size, config.training.batch_size)
        self.assertEqual(updated.data.train.reader.device_prefetch_batches, 2)
        self.assertEqual(os.environ.get("MDL_GROUPED_EMB_MAX_OUTPUT_MIB"), "1024")
        self.assertGreaterEqual(updated.training.ddp.bucket_cap_mb, 250.0)
        self.assertGreaterEqual(
            updated.data.train.reader.host_prepare_prefetch, 10
        )

    def test_rankmixer_no_p2p_auto_upscales_batch(self) -> None:
        os.environ.pop("MDL_LOCAL_BATCH_SCALE", None)
        os.environ.pop("MDL_GROUPED_EMB_MAX_OUTPUT_MIB", None)
        config = load_app_config("configs/rankmixer.yaml")
        with mock.patch(
            "src.train._local_cuda_p2p_accessible", return_value=False
        ), mock.patch("src.train._small_hbm_cuda_device", return_value=True), mock.patch.dict(
            "os.environ", {"NCCL_P2P_DISABLE": "1"}, clear=False
        ):
            updated = _apply_world_size_training_profile(config, world_size=4)
            self.assertEqual(os.environ.get("MDL_GROUPED_EMB_MAX_OUTPUT_MIB"), "512")
        self.assertEqual(updated.training.batch_size, 1818)  # 1280 * 1.42
        self.assertGreaterEqual(
            updated.data.train.reader.host_prepare_prefetch, 10
        )
        self.assertGreaterEqual(updated.data.train.reader.prefetch_batches, 6)
        self.assertLessEqual(updated.data.train.reader.hdfs_op_timeout, 15.0)

    def test_nccl_env_sets_buffer_caps(self) -> None:
        env: dict[str, str] = {"WORLD_SIZE": "8"}
        with mock.patch("src.train._local_cuda_p2p_accessible", return_value=True):
            _configure_nccl_runtime_env(env)
        self.assertEqual(env.get("TORCH_NCCL_ASYNC_ERROR_HANDLING"), "1")
        self.assertEqual(env.get("NCCL_BUFFSIZE"), str(2 * 1024 * 1024))
        self.assertEqual(env.get("NCCL_CUMEM_ENABLE"), "0")
        self.assertEqual(env.get("NCCL_MAX_NCHANNELS"), "4")

    def test_nccl_env_channels_at_six_gpu(self) -> None:
        env: dict[str, str] = {"WORLD_SIZE": "6"}
        with mock.patch("src.train._local_cuda_p2p_accessible", return_value=True):
            _configure_nccl_runtime_env(env)
        self.assertEqual(env.get("NCCL_MAX_NCHANNELS"), "4")

    def test_nccl_prefer_collective_bw_for_rankmixer(self) -> None:
        env: dict[str, str] = {"WORLD_SIZE": "8"}
        with mock.patch("src.train._local_cuda_p2p_accessible", return_value=True):
            _configure_nccl_runtime_env(env, prefer_collective_bw=True)
        self.assertEqual(env.get("NCCL_BUFFSIZE"), str(8 * 1024 * 1024))
        self.assertEqual(env.get("NCCL_CUMEM_ENABLE"), "0")
        self.assertNotIn("NCCL_MAX_NCHANNELS", env)

    def test_nccl_prefer_collective_bw_at_four_gpu(self) -> None:
        env: dict[str, str] = {"WORLD_SIZE": "4"}
        with mock.patch("src.train._local_cuda_p2p_accessible", return_value=True):
            _configure_nccl_runtime_env(env, prefer_collective_bw=True)
        self.assertEqual(env.get("NCCL_BUFFSIZE"), str(8 * 1024 * 1024))
        self.assertEqual(env.get("CUDA_DEVICE_MAX_CONNECTIONS"), "8")

    def test_local_rank_cpu_affinity_partitions_cores(self) -> None:
        assigned: dict[str, list[int]] = {}

        class _FakeProc:
            def cpu_affinity(self, cores=None):
                if cores is None:
                    return assigned.get("cores", [])
                assigned["cores"] = list(cores)
                return None

        with mock.patch.dict(
            "os.environ",
            {"LOCAL_RANK": "1", "LOCAL_WORLD_SIZE": "4", "WORLD_SIZE": "4"},
            clear=False,
        ), mock.patch("psutil.cpu_count", return_value=32), mock.patch(
            "psutil.Process", return_value=_FakeProc()
        ):
            prep = _apply_local_rank_cpu_affinity("host_prepare")
            train = _apply_local_rank_cpu_affinity("train")
        self.assertTrue(prep)
        self.assertTrue(train)
        self.assertTrue(set(prep).isdisjoint(set(train)))
        self.assertTrue(all(8 <= c < 16 for c in prep + train))


if __name__ == "__main__":
    unittest.main()
