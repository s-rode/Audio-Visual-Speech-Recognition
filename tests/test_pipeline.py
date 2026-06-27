from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from team1_asr.config import BackendConfig, ExperimentConfig, InputConfig, OutputConfig
from team1_asr.pipeline import run_experiment


class PipelineTest(unittest.TestCase):
    def test_transcript_output_includes_video_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_dir = root / "audio"
            video_dir = root / "video"
            output_dir = root / "outputs"
            audio_dir.mkdir()
            video_dir.mkdir()
            (audio_dir / "utt.wav").write_bytes(b"not-a-real-wav")
            (video_dir / "utt.mp4").write_bytes(b"not-a-real-video")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "utterance_id,audio_path,video_path,reference,language\n"
                "utt,audio/utt.wav,video/utt.mp4,hello,en\n",
                encoding="utf-8",
            )

            run_experiment(
                ExperimentConfig(
                    backend=BackendConfig(kind="noop", model="smoke-test"),
                    input=InputConfig(manifest_path=str(manifest)),
                    output=OutputConfig(output_dir=str(output_dir)),
                )
            )

            lines = (output_dir / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
            record = json.loads(lines[0])

        self.assertTrue(record["video_path"].endswith("video/utt.mp4"))


if __name__ == "__main__":
    unittest.main()
