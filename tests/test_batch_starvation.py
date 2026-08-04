"""A slow reader degrades throughput instead of killing the world.

The failure this guards against: one rank's host-prepare child stops delivering
batches, the rank blocks forever in ``next()``, never reaches the active-rank
vote, and the step watchdog eventually exits every rank with code 70.
"""

from __future__ import annotations

import queue
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from src.config import LengthBucketConfig, ReaderConfig
from src.dataloader import FeatureBatch, RemoteIoStallError
from src.train import (
    BATCH_NOT_READY,
    _next_batch_within,
    _ProcessHostPrepareIterator,
    _RankSupply,
    _start_rank_supply_count,
    _supply_verdict,
)


def _idle_iterator(
    *,
    idle_timeout_sec: float | None,
    heartbeat: float | None,
) -> _ProcessHostPrepareIterator:
    """A live child that has delivered before but is producing nothing now."""

    iterator = _ProcessHostPrepareIterator.__new__(_ProcessHostPrepareIterator)
    iterator._pin_memory = False
    iterator._ipc_mode = "memfd"
    iterator._fd_recv = None
    iterator._closed = False
    iterator._startup_timeout_sec = None
    iterator._idle_timeout_sec = idle_timeout_sec
    iterator._started_at = 0.0
    iterator._last_progress_at = 0.0
    iterator._received_item = True
    if heartbeat is None:
        iterator._progress_mtime = None
    else:
        progress = MagicMock()
        progress.value = heartbeat
        iterator._progress_mtime = progress
    iterator._queue = MagicMock()
    iterator._queue.get.side_effect = queue.Empty
    iterator._queue.empty.return_value = False
    iterator._process = MagicMock()
    iterator._process.is_alive.return_value = True
    iterator._process.pid = 4242
    return iterator


class StarvedReadTest(unittest.TestCase):
    def test_budget_expiry_reports_not_ready_rather_than_blocking(self) -> None:
        iterator = _idle_iterator(idle_timeout_sec=None, heartbeat=None)
        self.addCleanup(setattr, iterator, "_closed", True)

        got = iterator.next_within(0.05)

        self.assertIs(got, BATCH_NOT_READY)
        # Starvation must not tear the child down: the batch is still coming.
        self.assertFalse(iterator._closed)
        iterator._process.kill.assert_not_called()

    def test_a_ready_batch_is_returned_within_budget(self) -> None:
        iterator = _idle_iterator(idle_timeout_sec=None, heartbeat=None)
        self.addCleanup(setattr, iterator, "_closed", True)
        batch = MagicMock(spec=FeatureBatch)
        iterator._queue.get.side_effect = [queue.Empty, batch]

        with patch(
            "src.train.privatize_shared_feature_batch",
            side_effect=lambda item: item,
        ):
            got = iterator.next_within(30.0)

        self.assertIs(got, batch)

    def test_unbudgeted_read_keeps_the_blocking_contract(self) -> None:
        """next() must never surface the sentinel to legacy callers."""

        iterator = _idle_iterator(idle_timeout_sec=None, heartbeat=None)
        self.addCleanup(setattr, iterator, "_closed", True)
        batch = MagicMock(spec=FeatureBatch)
        iterator._queue.get.side_effect = [queue.Empty, queue.Empty, batch]

        with patch(
            "src.train.privatize_shared_feature_batch",
            side_effect=lambda item: item,
        ):
            got = next(iterator)

        self.assertIs(got, batch)

    def test_fallback_blocks_for_iterators_without_a_budgeted_read(self) -> None:
        plain = iter(["batch"])
        self.assertEqual(_next_batch_within(plain, 5.0), "batch")

    def test_no_budget_uses_plain_next(self) -> None:
        reader = MagicMock()
        reader.next_within.side_effect = AssertionError("should not be consulted")
        with patch("builtins.next", return_value="batch") as plain_next:
            self.assertEqual(_next_batch_within(reader, None), "batch")
        plain_next.assert_called_once_with(reader)


class IdleCeilingTest(unittest.TestCase):
    """The ceiling that makes starvation tolerance safe rather than infinite."""

    def test_idle_timeout_ignores_a_fresh_heartbeat(self) -> None:
        # The regression: the child beats per record batch and every 15s from
        # io_progress_pulses, so folding heartbeats into the idle measure made
        # this timer unreachable and left the 600s step watchdog as the only
        # backstop -- which kills every rank, not just the starving one.
        iterator = _idle_iterator(idle_timeout_sec=300.0, heartbeat=1_000.0)
        aborted: list[BaseException] = []

        def _abort(error: BaseException) -> None:
            aborted.append(error)
            raise SystemExit(70)

        with patch("src.train.perf_counter", side_effect=[0.0, 400.0, 400.0]), patch(
            "src.train.time", return_value=1_000.05
        ), patch(
            "src.train.abort_rank_for_remote_io_stall", side_effect=_abort
        ), patch("builtins.print"), patch("src.train._close_process_queue"), patch(
            "src.train.os.killpg"
        ):
            with self.assertRaises(SystemExit) as raised:
                iterator.next_within(600.0)

        self.assertEqual(raised.exception.code, 70)
        self.assertIsInstance(aborted[0], RemoteIoStallError)
        self.assertIn("delivered no batch", str(aborted[0]))

    def test_startup_timeout_still_honours_heartbeats(self) -> None:
        # Long HDFS list/footer/adapt work before the first batch is healthy,
        # so startup must stay heartbeat-aware even though idle no longer is.
        iterator = _idle_iterator(idle_timeout_sec=None, heartbeat=1_000.0)
        self.addCleanup(setattr, iterator, "_closed", True)
        iterator._received_item = False
        iterator._startup_timeout_sec = 300.0
        batch = MagicMock(spec=FeatureBatch)
        iterator._queue.get.side_effect = [queue.Empty, batch]

        with patch(
            "src.train.perf_counter", side_effect=[0.0, 400.0, 400.0, 400.0]
        ), patch("src.train.time", return_value=1_000.05), patch(
            "src.train.privatize_shared_feature_batch",
            side_effect=lambda item: item,
        ):
            got = iterator.next_within(600.0)

        self.assertIs(got, batch)

    def test_config_rejects_a_ceiling_below_the_step_budget(self) -> None:
        reader = ReaderConfig(
            length_buckets=(LengthBucketConfig(max_length=None, batch_size=8),),
            host_prepare_prefetch=1,
            step_batch_budget_sec=30.0,
            host_prepare_idle_timeout_sec=20.0,
        )
        with self.assertRaisesRegex(ValueError, "must exceed"):
            reader.validate()

    def test_config_defaults_leave_room_for_starved_steps(self) -> None:
        reader = ReaderConfig(
            length_buckets=(LengthBucketConfig(max_length=None, batch_size=8),),
        )
        reader.validate()
        self.assertGreater(
            reader.host_prepare_idle_timeout_sec,
            reader.step_batch_budget_sec,
        )


class SupplyVerdictTest(unittest.TestCase):
    """Starved and exhausted must never collapse into the same decision."""

    def test_every_rank_exhausted_ends_the_epoch(self) -> None:
        supply = _RankSupply(active=0, exhausted=4)
        self.assertEqual(_supply_verdict(supply, 4), "stop")

    def test_world_wide_starvation_retries_instead_of_ending(self) -> None:
        supply = _RankSupply(active=0, exhausted=0)
        self.assertEqual(_supply_verdict(supply, 4), "retry")

    def test_partial_exhaustion_with_starvation_still_retries(self) -> None:
        supply = _RankSupply(active=0, exhausted=3)
        self.assertEqual(_supply_verdict(supply, 4), "retry")

    def test_a_starved_rank_replays_while_peers_train(self) -> None:
        supply = _RankSupply(active=3, exhausted=0)
        self.assertEqual(_supply_verdict(supply, 4), "replay")

    def test_an_exhausted_rank_replays_while_peers_train(self) -> None:
        supply = _RankSupply(active=3, exhausted=1)
        self.assertEqual(_supply_verdict(supply, 4), "replay")

    def test_single_rank_exhaustion_ends_the_epoch(self) -> None:
        supply = _RankSupply(active=0, exhausted=1)
        self.assertEqual(_supply_verdict(supply, 1), "stop")

    def test_single_rank_starvation_retries(self) -> None:
        supply = _RankSupply(active=0, exhausted=0)
        self.assertEqual(_supply_verdict(supply, 1), "retry")


class SupplyVoteTest(unittest.TestCase):
    def test_starved_and_exhausted_reduce_to_different_counters(self) -> None:
        context = SimpleNamespace(
            enabled=False, control_group=None, device=torch.device("cpu")
        )
        starved = _start_rank_supply_count(
            context, rank_active=False, rank_exhausted=False
        ).wait()
        exhausted = _start_rank_supply_count(
            context, rank_active=False, rank_exhausted=True
        ).wait()
        self.assertEqual((starved.active, starved.exhausted), (0, 0))
        self.assertEqual((exhausted.active, exhausted.exhausted), (0, 1))

    def test_vote_reduces_both_counters_in_one_collective(self) -> None:
        work = MagicMock()
        context = SimpleNamespace(
            enabled=True, control_group=object(), device=torch.device("cpu")
        )
        with patch("src.train.torch_dist.all_reduce", return_value=work) as all_reduce:
            _start_rank_supply_count(context, rank_active=True, rank_exhausted=False)
        all_reduce.assert_called_once()
        reduced = all_reduce.call_args.args[0]
        self.assertEqual(reduced.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
