"""Async rank-supply allreduce overlaps with host work."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from src.train import (
    _RankSupplyHandle,
    _active_rank_count,
    _start_rank_supply_count,
)


class ActiveRankCountAsyncTest(unittest.TestCase):
    def test_disabled_context_is_local(self) -> None:
        context = SimpleNamespace(enabled=False, control_group=None, device=torch.device("cpu"))
        self.assertEqual(_active_rank_count(context, True), 1)
        supply = _start_rank_supply_count(
            context,
            rank_active=False,
            rank_exhausted=True,
        ).wait()
        self.assertEqual(supply.active, 0)
        self.assertEqual(supply.exhausted, 1)

    def test_handle_waits_for_async_work(self) -> None:
        value = torch.tensor([3, 1], dtype=torch.long)
        work = MagicMock()
        handle = _RankSupplyHandle(value, work)
        supply = handle.wait()
        self.assertEqual((supply.active, supply.exhausted), (3, 1))
        work.wait.assert_called_once()

    def test_control_group_uses_async_all_reduce(self) -> None:
        work = MagicMock()
        context = SimpleNamespace(
            enabled=True,
            control_group=object(),
            device=torch.device("cpu"),
        )
        with patch("src.train.torch_dist.all_reduce", return_value=work) as all_reduce:
            handle = _start_rank_supply_count(
                context,
                rank_active=True,
                rank_exhausted=False,
            )
        all_reduce.assert_called_once()
        kwargs = all_reduce.call_args.kwargs
        self.assertTrue(kwargs.get("async_op"))
        self.assertIs(kwargs.get("group"), context.control_group)
        work.wait.assert_not_called()
        self.assertEqual(handle.wait().active, 1)
        work.wait.assert_called_once()

    def test_starved_rank_is_neither_active_nor_exhausted(self) -> None:
        """The whole point: starved must not be reduced to end-of-epoch."""

        context = SimpleNamespace(enabled=False, control_group=None, device=torch.device("cpu"))
        supply = _start_rank_supply_count(
            context,
            rank_active=False,
            rank_exhausted=False,
        ).wait()
        self.assertEqual((supply.active, supply.exhausted), (0, 0))


if __name__ == "__main__":
    unittest.main()
