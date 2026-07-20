from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.config import load_app_config
from src.dataloader import (
    ParquetScanner,
    _add_unique_scenario_values,
    _scenario_discovery_split,
    discover_scenario_values,
)
from src.train import DistributedContext, _resolve_distributed_auto_scenarios


ROOT = Path(__file__).resolve().parents[1]


class ScenarioDiscoveryScannerTest(unittest.TestCase):
    def test_discovery_split_overrides_training_scanner_knobs(self) -> None:
        base = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        discovery = _scenario_discovery_split(base.data.train)
        self.assertEqual(discovery.reader.scanner_batch_rows, 262_144)
        self.assertEqual(discovery.reader.shard_unit, "file")
        self.assertEqual(discovery.reader.schema_validation_samples, 1)
        self.assertEqual(discovery.reader.eager_schema_validation, "sample")
        self.assertEqual(base.data.train.reader.scanner_batch_rows, 64)
        self.assertEqual(base.data.train.reader.shard_unit, "row_group")

    def test_arrow_unique_handles_list_and_scalar_scene_ids(self) -> None:
        values: set[int] = set()
        _add_unique_scenario_values(
            values,
            pa.array([[17, 9], [17], [3]], type=pa.list_(pa.int64())),
            source="scene_id",
            max_discovered=256,
        )
        self.assertEqual(values, {3, 9, 17})

        scalar_values: set[int] = set()
        _add_unique_scenario_values(
            scalar_values,
            pa.array([9, 17, 9], type=pa.int64()),
            source="scene_id",
            max_discovered=256,
        )
        self.assertEqual(scalar_values, {9, 17})

        with self.assertRaisesRegex(ValueError, "empty list"):
            _add_unique_scenario_values(
                set(),
                pa.array([[17], []], type=pa.list_(pa.int64())),
                source="scene_id",
                max_discovered=256,
            )
        with self.assertRaisesRegex(ValueError, "contains null"):
            _add_unique_scenario_values(
                set(),
                pa.array([[17], None], type=pa.list_(pa.int64())),
                source="scene_id",
                max_discovered=256,
            )

    def test_discover_uses_discovery_split_and_avoids_row_group_lpt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenes.parquet"
            pq.write_table(
                pa.table(
                    {
                        "scene_id": pa.array(
                            [[17, 9], [17]],
                            type=pa.list_(pa.int64()),
                        )
                    }
                ),
                path,
            )
            base = load_app_config(ROOT / "configs" / "reference" / "default.yaml")
            config = replace(
                base,
                data=replace(
                    base.data,
                    train=replace(
                        base.data.train,
                        inputs=(str(path),),
                        reader=replace(
                            base.data.train.reader,
                            scanner_batch_rows=64,
                            shard_unit="row_group",
                            schema_validation_samples=64,
                        ),
                    ),
                ),
                scenarios=replace(
                    base.scenarios,
                    auto_discover=True,
                    source="scene_id",
                    names=("__auto__",),
                    discovery_cache_path=None,
                ),
            )

            captured: dict[str, object] = {}
            real_init = ParquetScanner.__init__

            def _capture(self, split, columns, *args, **kwargs):
                captured["split"] = split
                captured["columns"] = list(columns)
                return real_init(self, split, columns, *args, **kwargs)

            with patch.object(ParquetScanner, "__init__", _capture):
                discovered = discover_scenario_values(config)

            self.assertEqual(discovered, (9, 17))
            split = captured["split"]
            self.assertEqual(split.reader.scanner_batch_rows, 262_144)
            self.assertEqual(split.reader.shard_unit, "file")
            self.assertEqual(captured["columns"], ["scene_id"])


class ScenarioBroadcastGroupTest(unittest.TestCase):
    def test_broadcast_uses_control_group_when_available(self) -> None:
        base = load_app_config(ROOT / "configs" / "reference" / "default.yaml")
        config = replace(
            base,
            scenarios=replace(
                base.scenarios,
                auto_discover=True,
                source="scene_id",
                names=("__auto__",),
            ),
        )
        control = object()
        context = DistributedContext(
            enabled=True,
            rank=0,
            local_rank=0,
            world_size=2,
            device=Mock(spec=torch.device, type="cuda"),
            control_group=control,  # type: ignore[arg-type]
        )
        with (
            patch(
                "src.train.discover_scenario_values",
                return_value=(9, 17),
            ),
            patch("src.train.resolve_auto_scenarios", side_effect=lambda c, v: c),
            patch("src.train.torch_dist.broadcast_object_list") as broadcast,
        ):
            _resolve_distributed_auto_scenarios(config, context)
        broadcast.assert_called_once()
        self.assertIs(broadcast.call_args.kwargs.get("group"), control)


if __name__ == "__main__":
    unittest.main()
