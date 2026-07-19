from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
import pickle
import unittest

from src.config import (
    FeatureConfig,
    ParquetAdapterConfig,
    ParquetSplitConfig,
    load_app_config,
)


class ConfigImmutabilityTest(unittest.TestCase):
    def _assert_no_mutable_containers(self, value: object) -> None:
        if is_dataclass(value):
            for config_field in fields(value):
                self._assert_no_mutable_containers(
                    getattr(value, config_field.name)
                )
            return
        if isinstance(value, Mapping):
            self.assertNotIsInstance(value, dict)
            for key, item in value.items():
                self._assert_no_mutable_containers(key)
                self._assert_no_mutable_containers(item)
            return
        if isinstance(value, (tuple, frozenset)):
            for item in value:
                self._assert_no_mutable_containers(item)
            return
        self.assertNotIsInstance(value, (list, dict, set))

    def test_loaded_and_resolved_configs_are_deeply_immutable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        resolved = config.resolved

        self._assert_no_mutable_containers(config)
        self._assert_no_mutable_containers(resolved)

        with self.assertRaises(AttributeError):
            config.features.append(
                FeatureConfig(name="z", kind="dense", source="z")
            )
        with self.assertRaises(AttributeError):
            config.scenarios.names.append("other")
        with self.assertRaises(TypeError):
            config.data.train.labels["other"] = "other_label"
        with self.assertRaises(TypeError):
            resolved.encoded_input_dims["z"] = 1
        with self.assertRaises(AttributeError):
            resolved.categorical_input_names.add("z")

        self.assertIs(config.resolved, resolved)
        self.assertNotIn("z", config.resolved.encoded_input_dims)

    def test_constructor_inputs_are_copied_before_freezing(self) -> None:
        inputs = ["train.parquet"]
        labels = {"click": "click_label"}
        split = ParquetSplitConfig(
            format="flat_parquet",
            inputs=inputs,
            labels=labels,
        )

        inputs.append("later.parquet")
        labels["purchase"] = "purchase_label"

        self.assertEqual(split.inputs, ("train.parquet",))
        self.assertEqual(dict(split.labels), {"click": "click_label"})

    def test_arbitrary_adapter_options_are_frozen_recursively(self) -> None:
        options = {"schema": {"columns": ["item_id"]}}
        adapter = ParquetAdapterConfig(
            callable="examples.parquet_identity_adapter:adapt",
            input_columns=["item_id"],
            optional_input_columns=["context_indices"],
            options=options,
        )

        options["schema"]["columns"].append("later")
        options["new"] = True

        self.assertEqual(adapter.input_columns, ("item_id",))
        self.assertEqual(adapter.optional_input_columns, ("context_indices",))
        self.assertEqual(adapter.options["schema"]["columns"], ("item_id",))
        self.assertNotIn("new", adapter.options)
        with self.assertRaises(TypeError):
            adapter.options["schema"]["other"] = True

    def test_immutable_mappings_remain_pickleable(self) -> None:
        split = ParquetSplitConfig(
            format="flat_parquet",
            inputs=["train.parquet"],
            labels={"click": "click_label"},
        )

        restored = pickle.loads(pickle.dumps(split))

        self.assertEqual(restored, split)
        with self.assertRaises(TypeError):
            restored.labels["purchase"] = "purchase_label"


if __name__ == "__main__":
    unittest.main()
