from __future__ import annotations

from pathlib import Path
import unittest

from src.config import load_app_config


class ModelConfigOverlayTest(unittest.TestCase):
    def test_all_model_profiles_extend_and_validate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = {
            "default.yaml": "mdl_rankmixer",
            "rankmixer.yaml": "rankmixer",
            "mdl_rankmixer.yaml": "mdl_rankmixer",
            "onetrans.yaml": "onetrans",
            "mdl_onetrans.yaml": "mdl_onetrans",
            "longer.yaml": "longer",
        }
        for filename, model_name in expected.items():
            with self.subTest(filename=filename):
                config = load_app_config(root / "configs" / filename)
                self.assertEqual(config.model.name, model_name)

        experimental = load_app_config(root / "configs" / "mdl_onetrans.yaml")
        self.assertTrue(experimental.model.experimental_model_acknowledged)


if __name__ == "__main__":
    unittest.main()
