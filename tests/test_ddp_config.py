from __future__ import annotations

import unittest

from src.config import DDPConfig


class DDPConfigTest(unittest.TestCase):
    def test_safe_default_is_dynamic_with_unused_detection(self) -> None:
        config = DDPConfig()
        config.validate()
        self.assertFalse(config.static_graph)
        self.assertTrue(config.find_unused_parameters)

    def test_static_graph_cannot_contradict_unused_detection(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires find_unused_parameters=false"):
            DDPConfig(
                static_graph=True,
                find_unused_parameters=True,
                validated_static_graph=True,
            ).validate()

    def test_optimized_modes_require_recorded_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "validated_no_unused_parameters"):
            DDPConfig(find_unused_parameters=False).validate()
        with self.assertRaisesRegex(ValueError, "validated_static_graph"):
            DDPConfig(
                static_graph=True,
                find_unused_parameters=False,
            ).validate()
        DDPConfig(
            find_unused_parameters=False,
            validated_no_unused_parameters=True,
        ).validate()
        DDPConfig(
            static_graph=True,
            find_unused_parameters=False,
            validated_static_graph=True,
        ).validate()


if __name__ == "__main__":
    unittest.main()
