"""Async active-rank allreduce overlaps with host work."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from src.train import _ActiveRankCountHandle, _active_rank_count, _start_active_rank_count


class ActiveRankCountAsyncTest(unittest.TestCase):
    def test_disabled_context_is_local(self) -> None:
        context = SimpleNamespace(enabled=False, control_group=None, device=torch.device("cpu"))
        self.assertEqual(_active_rank_count(context, True), 1)
        self.assertEqual(_start_active_rank_count(context, False).wait(), 0)

    def test_handle_waits_for_async_work(self) -> None:
        value = torch.tensor(3, dtype=torch.long)
        work = MagicMock()
        handle = _ActiveRankCountHandle(value, work)
        self.assertEqual(handle.wait(), 3)
        work.wait.assert_called_once()

    def test_control_group_uses_async_all_reduce(self) -> None:
        work = MagicMock()
        context = SimpleNamespace(
            enabled=True,
            control_group=object(),
            device=torch.device("cpu"),
        )
        with patch("src.train.torch_dist.all_reduce", return_value=work) as all_reduce:
            handle = _start_active_rank_count(context, True)
        all_reduce.assert_called_once()
        kwargs = all_reduce.call_args.kwargs
        self.assertTrue(kwargs.get("async_op"))
        self.assertIs(kwargs.get("group"), context.control_group)
        work.wait.assert_not_called()
        self.assertEqual(handle.wait(), 1)
        work.wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
