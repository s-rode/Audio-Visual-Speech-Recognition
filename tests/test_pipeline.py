from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from team1_asr.config import BackendConfig, ExperimentConfig, InputConfig, OutputConfig
from team1_asr.pipeline import run_experiment


class FailingBackend:
    def transcribe(self, request):
        raise RuntimeError("backend failed")


class PipelineTests(unittest.TestCase):
    def test_failed_run_preserves_existing_transcript_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "manifest.csv"
            output_dir = root / "outputs"
            output_dir.mkdir()
            transcript_path = output_dir / "transcripts.jsonl"
            transcript_path.write_text("previous successful output\n", encoding="utf-8")
            manifest_path.write_text(
                "utterance_id,audio_path,reference,language\n"
                "utt1,audio.wav,hello,en\n",
                encoding="utf-8",
            )
            config = ExperimentConfig(
                backend=BackendConfig(kind="noop"),
                input=InputConfig(manifest_path=str(manifest_path), video_column=None),
                output=OutputConfig(output_dir=str(output_dir)),
            )

            with patch("team1_asr.pipeline.create_backend", return_value=FailingBackend()), patch(
                "team1_asr.pipeline.apply_noise", return_value="audio.wav"
            ):
                with self.assertRaisesRegex(RuntimeError, "backend failed"):
                    run_experiment(config)

            self.assertEqual(
                transcript_path.read_text(encoding="utf-8"),
                "previous successful output\n",
            )


if __name__ == "__main__":
    unittest.main()
