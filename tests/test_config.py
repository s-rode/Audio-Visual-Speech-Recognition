from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from team1_asr.config import load_config


class ConfigTest(unittest.TestCase):
    def test_loads_wandb_section(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text(
                """
experiment_name = "wandb-demo"

[wandb]
enabled = true
project = "team1-asr"
run_name = "demo-run"
mode = "offline"
log_artifacts = false
transcript_table_limit = 10
""",
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.experiment_name, "wandb-demo")
        self.assertTrue(config.wandb.enabled)
        self.assertEqual(config.wandb.project, "team1-asr")
        self.assertEqual(config.wandb.run_name, "demo-run")
        self.assertEqual(config.wandb.mode, "offline")
        self.assertFalse(config.wandb.log_artifacts)
        self.assertEqual(config.wandb.transcript_table_limit, 10)


if __name__ == "__main__":
    unittest.main()
