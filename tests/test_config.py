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

    def test_loads_mms_llama_backend_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text(
                """
[backend]
kind = "mms-llama"
repo_path = "external/MMS-LLaMA"
checkpoint_path = "models/checkpoint.pt"
noise_wav = "noise/babble.wav"
modalities = ["video", "audio"]
results_dir = "outputs/mms"
python_executable = "python"
""",
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.backend.kind, "mms-llama")
        self.assertEqual(config.backend.checkpoint_path, "models/checkpoint.pt")
        self.assertEqual(config.backend.noise_wav, "noise/babble.wav")
        self.assertEqual(config.backend.modalities, ["video", "audio"])
        self.assertEqual(config.backend.results_dir, "outputs/mms")

    def test_simple_toml_parser_supports_string_arrays(self) -> None:
        from team1_asr.config import _parse_simple_value

        self.assertEqual(_parse_simple_value('["video", "audio"]'), ["video", "audio"])


if __name__ == "__main__":
    unittest.main()
