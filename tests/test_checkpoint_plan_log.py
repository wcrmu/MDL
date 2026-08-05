"""Checkpoint arming must be visible before any HDFS open.

Trainjob logs often keep only the training-step tail. A banner that printed
only after ``open_run_store`` made a hang or a silent disable look identical
to "checkpoint code never ran".
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config import CheckpointConfig, load_app_config
from src.train import _CheckpointCoordinator, _checkpoint_plan_message


class CheckpointPlanMessageTest(unittest.TestCase):
    def test_enabled_config_announces_dir_and_cadence(self) -> None:
        config = load_app_config("configs/mdl_onetrans.yaml")
        message = _checkpoint_plan_message(config)
        self.assertTrue(message.startswith("Checkpointing | enabled "))
        self.assertIn("every_steps=2000", message)
        self.assertIn(str(config.training.checkpoint.dir), message)

    def test_missing_dir_announces_disabled(self) -> None:
        config = load_app_config("configs/mdl_onetrans.yaml")
        config = SimpleNamespace(
            model=config.model,
            training=SimpleNamespace(checkpoint=CheckpointConfig()),
        )
        message = _checkpoint_plan_message(config)
        self.assertTrue(message.startswith("Checkpointing | disabled "))


class CheckpointCreateBannerTest(unittest.TestCase):
    def test_enabled_banner_prints_before_opening_the_store(self) -> None:
        config = load_app_config("configs/mdl_onetrans.yaml")
        context = SimpleNamespace(rank=0, world_size=1, enabled=False, device="cpu")
        order: list[str] = []

        def _print(*args: object, **_kwargs: object) -> None:
            order.append(" ".join(str(arg) for arg in args))

        def _open(*_args: object, **_kwargs: object) -> MagicMock:
            order.append("open_run_store")
            store = MagicMock()
            store.root_uri = "file:///tmp/ckpt"
            store.is_remote = False
            return store

        with patch("src.train.open_run_store", side_effect=_open), patch(
            "src.train.CheckpointUploader"
        ), patch("builtins.print", side_effect=_print):
            coordinator = _CheckpointCoordinator.create(config, context, log_steps=True)

        self.assertIsNotNone(coordinator)
        self.assertTrue(order[0].startswith("Checkpointing | enabled "))
        self.assertEqual(order[1], "open_run_store")
        self.assertTrue(order[2].startswith("Checkpointing | ready "))

    def test_disabled_banner_skips_store_open(self) -> None:
        config = load_app_config("configs/mdl_onetrans.yaml")
        config = SimpleNamespace(
            model=config.model,
            training=SimpleNamespace(checkpoint=CheckpointConfig()),
        )
        context = SimpleNamespace(rank=0, world_size=1, enabled=False, device="cpu")
        with patch("src.train.open_run_store") as opened, patch("builtins.print") as printed:
            coordinator = _CheckpointCoordinator.create(config, context, log_steps=True)
        self.assertIsNone(coordinator)
        opened.assert_not_called()
        texts = [" ".join(str(a) for a in call.args) for call in printed.call_args_list]
        self.assertTrue(any(t.startswith("Checkpointing | disabled ") for t in texts))


if __name__ == "__main__":
    unittest.main()
