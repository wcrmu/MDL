"""Adaptive NCCL P2P / fallback environment configuration."""

from __future__ import annotations

import unittest
from unittest import mock

from src.train import _configure_nccl_runtime_env, _local_cuda_p2p_accessible


class NcclP2PEnvTest(unittest.TestCase):
    def test_p2p_ok_leaves_ignore_unset(self) -> None:
        env: dict[str, str] = {}
        with mock.patch("src.train._local_cuda_p2p_accessible", return_value=True):
            _configure_nccl_runtime_env(env)
        self.assertNotIn("NCCL_IGNORE_DISABLED_P2P", env)
        self.assertNotIn("NCCL_P2P_DISABLE", env)

    def test_p2p_broken_enables_fallback(self) -> None:
        env: dict[str, str] = {}
        with mock.patch("src.train._local_cuda_p2p_accessible", return_value=False):
            _configure_nccl_runtime_env(env)
        self.assertEqual(env.get("NCCL_IGNORE_DISABLED_P2P"), "1")
        self.assertEqual(env.get("NCCL_P2P_DISABLE"), "1")

    def test_explicit_override_wins(self) -> None:
        env = {"NCCL_IGNORE_DISABLED_P2P": "0"}
        with mock.patch("src.train._local_cuda_p2p_accessible", return_value=False):
            _configure_nccl_runtime_env(env)
        self.assertEqual(env.get("NCCL_IGNORE_DISABLED_P2P"), "0")
        self.assertNotIn("NCCL_P2P_DISABLE", env)

    def test_inconclusive_probe_defaults_ignore(self) -> None:
        env: dict[str, str] = {}
        with mock.patch("src.train._local_cuda_p2p_accessible", return_value=None):
            _configure_nccl_runtime_env(env)
        self.assertEqual(env.get("NCCL_IGNORE_DISABLED_P2P"), "1")

    def test_single_visible_gpu_solo_job_reports_accessible(self) -> None:
        with mock.patch("src.train.torch.cuda.is_available", return_value=True), mock.patch(
            "src.train.torch.cuda.device_count", return_value=1
        ), mock.patch("src.train._env_int", return_value=1):
            self.assertTrue(_local_cuda_p2p_accessible())

    def test_single_visible_gpu_multirank_is_inconclusive(self) -> None:
        with mock.patch("src.train.torch.cuda.is_available", return_value=True), mock.patch(
            "src.train.torch.cuda.device_count", return_value=1
        ), mock.patch("src.train._env_int", side_effect=lambda name, default=0: 4):
            self.assertIsNone(_local_cuda_p2p_accessible())


if __name__ == "__main__":
    unittest.main()
