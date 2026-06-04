from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from team1_asr.config import ExperimentConfig
from team1_asr.tracking import WandbTracker


class WandbTrackerTest(unittest.TestCase):
    def test_disabled_tracker_does_not_import_wandb(self) -> None:
        config = ExperimentConfig()
        config.wandb.enabled = False

        tracker = WandbTracker(config)
        tracker.start()
        tracker.log_summary({}, Path("outputs"))
        tracker.finish()

        self.assertFalse(tracker.enabled)

    def test_enabled_tracker_logs_metrics_table_and_artifact(self) -> None:
        fake_wandb = _FakeWandb()
        config = ExperimentConfig(experiment_name="demo run")
        config.wandb.enabled = True
        config.wandb.project = "team1-asr-test"
        config.wandb.mode = "offline"
        config.wandb.log_artifacts = True

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            transcript_path = output_dir / "transcripts.jsonl"
            transcript_path.write_text(
                json.dumps(
                    {
                        "utterance_id": "utt-1",
                        "reference": "HELLO",
                        "prediction": "hello",
                        "used_audio_path": "audio/utt-1.wav",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = {
                "experiment_name": "demo run",
                "wer": 0.0,
                "cer": 0.0,
                "raw_wer": 1.0,
                "transcript_path": str(transcript_path),
            }

            with patch.dict(sys.modules, {"wandb": fake_wandb}):
                tracker = WandbTracker(config)
                tracker.start()
                tracker.log_summary(summary, output_dir)
                tracker.finish()

        self.assertEqual(fake_wandb.init_kwargs["project"], "team1-asr-test")
        self.assertEqual(fake_wandb.init_kwargs["name"], "demo run")
        self.assertEqual(fake_wandb.init_kwargs["mode"], "offline")
        self.assertIn({"wer": 0.0, "cer": 0.0, "raw_wer": 1.0}, fake_wandb.logs)
        self.assertEqual(fake_wandb.tables[0].rows[0][0], "utt-1")
        self.assertEqual(fake_wandb.artifacts[0].name, "demo-run-outputs")
        self.assertTrue(fake_wandb.finished)


class _FakeWandb(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("wandb")
        self.init_kwargs = {}
        self.logs = []
        self.tables = []
        self.artifacts = []
        self.finished = False

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return object()

    def log(self, data):
        self.logs.append(data)

    def finish(self):
        self.finished = True

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def Table(self, columns):
        table = _FakeTable(columns)
        self.tables.append(table)
        return table

    def Artifact(self, name, type):
        return _FakeArtifact(name, type)


class _FakeTable:
    def __init__(self, columns) -> None:
        self.columns = columns
        self.rows = []

    def add_data(self, *values) -> None:
        self.rows.append(values)


class _FakeArtifact:
    def __init__(self, name, type) -> None:
        self.name = name
        self.type = type
        self.paths = []

    def add_dir(self, path) -> None:
        self.paths.append(path)


if __name__ == "__main__":
    unittest.main()
