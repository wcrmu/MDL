"""Host memory telemetry on the periodic training log.

Two rounds of host-RSS fixes shipped with nothing on the training path
recording host RSS, so a multi-hour climb could only be seen from an external
dashboard and never attributed to a layer. These counters close that gap, and
must never be able to break training themselves.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.dataloader import (
    arrow_pool_bytes,
    process_peak_resident_bytes,
    process_resident_bytes,
)
from src.train import _host_memory_report, _MIB


class ProcessMemoryProbeTest(unittest.TestCase):
    def test_own_resident_bytes_is_plausible(self) -> None:
        resident = process_resident_bytes()
        self.assertIsNotNone(resident)
        # A torch-importing interpreter is comfortably above 1 MiB.
        self.assertGreater(resident, _MIB)

    def test_peak_is_at_least_current(self) -> None:
        self.assertGreaterEqual(
            process_peak_resident_bytes(),
            process_resident_bytes(),
        )

    def test_unreadable_pid_returns_none_rather_than_raising(self) -> None:
        self.assertIsNone(process_resident_bytes(99_999_999))

    def test_arrow_pool_bytes_is_non_negative(self) -> None:
        self.assertGreaterEqual(arrow_pool_bytes(), 0)


class HostMemoryReportTest(unittest.TestCase):
    def test_report_covers_parent_counters_without_a_reader(self) -> None:
        report = _host_memory_report(None)
        self.assertIn("rank_rss_mib=", report)
        self.assertIn("rank_peak_rss_mib=", report)
        self.assertIn("rank_arrow_mib=", report)

    def test_reader_fields_are_appended(self) -> None:
        reader = MagicMock()
        reader.memory_report.return_value = "child_rss_mib=12.5 pinned_idle_mib=3.0"
        report = _host_memory_report(reader)
        self.assertIn("child_rss_mib=12.5", report)
        self.assertIn("pinned_idle_mib=3.0", report)
        self.assertIn("rank_rss_mib=", report)

    def test_a_broken_reader_probe_never_breaks_the_log(self) -> None:
        reader = MagicMock()
        reader.memory_report.side_effect = RuntimeError("probe exploded")
        report = _host_memory_report(reader)
        self.assertIn("rank_rss_mib=", report)
        self.assertNotIn("child_rss_mib", report)

    def test_unreadable_counters_are_dropped_not_reported_as_zero(self) -> None:
        with patch("src.train.process_resident_bytes", return_value=None), patch(
            "src.train.process_peak_resident_bytes", return_value=None
        ), patch("src.train.arrow_pool_bytes", return_value=None):
            self.assertEqual(_host_memory_report(None), "")


if __name__ == "__main__":
    unittest.main()
