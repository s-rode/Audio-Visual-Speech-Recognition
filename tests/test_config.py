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

    def test_loads_advanced_noise_section(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text(
                """
[noise]
seed = 7

[noise.demand]
enabled = true
snr_db = 10.0
noise_path = "DEMAND/16k"

[noise.reverb]
enabled = true
rt60_range = [0.2, 0.8]

[noise.bandwidth]
enabled = true
available_filters = ["butter"]
available_cutoffs_hz = [4000]
""",
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.noise.seed, 7)
        self.assertTrue(config.noise.advanced_enabled)
        self.assertTrue(config.noise.demand.enabled)
        self.assertEqual(config.noise.demand.snr_db, 10.0)
        self.assertEqual(config.noise.demand.noise_path, "DEMAND/16k")
        self.assertTrue(config.noise.reverb.enabled)
        self.assertEqual(config.noise.reverb.rt60_range, [0.2, 0.8])
        self.assertTrue(config.noise.bandwidth.enabled)
        self.assertEqual(config.noise.bandwidth.available_filters, ["butter"])

    def test_legacy_white_noise_section_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text(
                """
[noise]
enabled = true
kind = "white"
snr_db = 5.0
seed = 3
""",
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertTrue(config.noise.enabled)
        self.assertEqual(config.noise.kind, "white")
        self.assertEqual(config.noise.snr_db, 5.0)
        self.assertFalse(config.noise.advanced_enabled)


if __name__ == "__main__":
    unittest.main()
