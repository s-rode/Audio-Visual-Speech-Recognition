from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from team1_asr.config import BackendConfig, ExperimentConfig, InputConfig, OutputConfig
from team1_asr.pipeline import run_experiment
from team1_asr.types import Segment, Transcript


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

    def test_backend_error_fails_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = _write_manifest(root)
            output_dir = root / "outputs"

            with patch("team1_asr.pipeline.create_backend", return_value=_FailingBackend()):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    run_experiment(
                        ExperimentConfig(
                            backend=BackendConfig(kind="noop", model="smoke-test"),
                            input=InputConfig(manifest_path=str(manifest)),
                            output=OutputConfig(output_dir=str(output_dir)),
                        )
                    )

    def test_skip_transcription_errors_writes_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = _write_manifest(root)
            output_dir = root / "outputs"

            with patch("team1_asr.pipeline.create_backend", return_value=_SometimesFailingBackend()):
                summary = run_experiment(
                    ExperimentConfig(
                        backend=BackendConfig(
                            kind="noop",
                            model="smoke-test",
                            skip_transcription_errors=True,
                        ),
                        input=InputConfig(manifest_path=str(manifest)),
                        output=OutputConfig(output_dir=str(output_dir)),
                    )
                )

            transcript_lines = (output_dir / "transcripts.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            failure_lines = (output_dir / "failed_utterances.csv").read_text(
                encoding="utf-8"
            ).splitlines()

        self.assertEqual(summary["utterances_scored"], 1)
        self.assertEqual(summary["utterances_failed"], 1)
        self.assertEqual(len(transcript_lines), 1)
        self.assertIn("fail", failure_lines[1])
        self.assertIn("synthetic failure", failure_lines[1])


class _FailingBackend:
    def transcribe(self, request):
        raise RuntimeError("synthetic failure")


class _SometimesFailingBackend:
    def transcribe(self, request):
        if request.utterance_id == "fail":
            raise RuntimeError("synthetic failure")
        return Transcript(
            text=request.reference or "",
            language=request.language,
            duration=1.0,
            segments=[Segment(start=0.0, end=1.0, text=request.reference or "")],
            metadata={"backend": "test"},
        )


def _write_manifest(root: Path) -> Path:
    audio_dir = root / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "ok.wav")
    _write_wav(audio_dir / "fail.wav")
    manifest = root / "manifest.csv"
    manifest.write_text(
        "utterance_id,audio_path,reference,language\n"
        "ok,audio/ok.wav,hello,en\n"
        "fail,audio/fail.wav,goodbye,en\n",
        encoding="utf-8",
    )
    return manifest


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 16000)


if __name__ == "__main__":
    unittest.main()
