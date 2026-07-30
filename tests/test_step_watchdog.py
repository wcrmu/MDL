"""Step watchdog phase tracking and stall dump."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.train import _StepWatchdog, _abort_rank_for_cuda_oom, _step_watchdog_beat


class StepWatchdogTest(unittest.TestCase):
    def test_beat_records_phase_and_memory(self) -> None:
        watchdog = _StepWatchdog(timeout_sec=600.0, rank=2)
        watchdog.beat(
            "backward",
            "steps=18600 local_rows=800",
            allocated_bytes=64 * 1024**3,
            reserved_bytes=70 * 1024**3,
        )
        stalled, phase, detail, allocated, reserved = watchdog._snapshot()
        self.assertLess(stalled, 1.0)
        self.assertEqual(phase, "backward")
        self.assertIn("local_rows=800", detail)
        self.assertEqual(allocated, 64 * 1024**3)
        self.assertEqual(reserved, 70 * 1024**3)

    def test_helper_beat_noop_without_watchdog(self) -> None:
        _step_watchdog_beat(None, "forward", detail="steps=1")

    def test_stall_message_includes_phase(self) -> None:
        watchdog = _StepWatchdog(timeout_sec=600.0, rank=1)
        watchdog.beat("sparse_sync", "steps=7 local_rows=128")
        printed: list[str] = []

        def _fake_exit(code: int) -> None:
            raise SystemExit(code)

        with patch.object(
            watchdog._stop, "wait", side_effect=[False, True]
        ), patch(
            "src.train.os._exit", side_effect=_fake_exit
        ), patch(
            "builtins.print", side_effect=lambda *a, **k: printed.append(str(a[0]))
        ), patch.object(
            watchdog,
            "_snapshot",
            return_value=(700.0, "sparse_sync", "steps=7", 1024**3, 2 * 1024**3),
        ):
            with self.assertRaises(SystemExit) as raised:
                watchdog._run()
        self.assertEqual(raised.exception.code, 70)
        self.assertTrue(any("last_phase=sparse_sync" in line for line in printed))
        self.assertTrue(any("cuda_allocated_mib=1024.0" in line for line in printed))
        self.assertTrue(any("TORCH_NCCL_ASYNC_ERROR_HANDLING" in line for line in printed))

    def test_cuda_oom_abort_exits_70(self) -> None:
        printed: list[str] = []

        def _fake_exit(code: int) -> None:
            raise SystemExit(code)

        with patch("src.train.os._exit", side_effect=_fake_exit), patch(
            "builtins.print", side_effect=lambda *a, **k: printed.append(str(a[0]))
        ):
            with self.assertRaises(SystemExit) as raised:
                _abort_rank_for_cuda_oom(
                    RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB"),
                    rank=0,
                    steps=18600,
                    detail="local_rows=800",
                )
        self.assertEqual(raised.exception.code, 70)
        self.assertTrue(any("CUDA_OOM" in line for line in printed))
        self.assertTrue(any("step=18600" in line for line in printed))


if __name__ == "__main__":
    unittest.main()
