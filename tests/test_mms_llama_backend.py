from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from team1_asr.backends.base import TranscriptionRequest
from team1_asr.backends.mms_llama_backend import (
    MmsLlamaBackend,
    _format_eval_error,
    _resolve_executable,
)
from team1_asr.config import BackendConfig


class MmsLlamaBackendTest(unittest.TestCase):
    def test_writes_manifest_runs_eval_and_reads_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / "MMS-LLaMA"
            (repo / "src" / "conf").mkdir(parents=True)
            (repo / "fairseq").mkdir()
            (repo / "src" / "eval.py").write_text("", encoding="utf-8")
            audio_path = root / "sample.wav"
            video_path = root / "sample.mp4"
            _write_wav(audio_path, frames=16000)
            video_path.write_bytes(b"mp4")

            commands: list[list[str]] = []

            def fake_run(command, **kwargs):
                commands.append(command)
                if command[0] == "ffprobe":
                    return SimpleNamespace(returncode=0, stdout="25\n", stderr="")
                results_arg = next(arg for arg in command if arg.startswith("common_eval.results_path="))
                results_dir = Path(results_arg.split("=", 1)[1])
                results_dir.mkdir(parents=True, exist_ok=True)
                (results_dir / "hypo-123.json").write_text(
                    json.dumps({"utt_id": ["utt-1"], "hypo": ["hello visual world"]}),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            config = BackendConfig(
                kind="mms-llama",
                repo_path=str(repo),
                checkpoint_path=str(root / "checkpoint.pt"),
                noise_wav=str(root / "noise.wav"),
                results_dir=str(root / "runs"),
                python_executable="python",
            )
            backend = MmsLlamaBackend(config)

            with patch.object(subprocess, "run", side_effect=fake_run):
                transcript = backend.transcribe(
                    TranscriptionRequest(
                        audio_path=str(audio_path),
                        video_path=str(video_path),
                        reference="hello world",
                        language="en",
                        utterance_id="utt-1",
                    )
                )

            self.assertEqual(transcript.text, "hello visual world")
            manifest_dir = root / "runs" / "utt-1" / "manifest"
            self.assertEqual((manifest_dir / "test.wrd").read_text(encoding="utf-8"), "hello world\n")
            tsv_lines = (manifest_dir / "test.tsv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(tsv_lines[0], "/")
            self.assertIn(str(video_path), tsv_lines[1])
            self.assertIn(str(audio_path), tsv_lines[1])
            self.assertIn("override.noise_prob=0", commands[-1])
            self.assertTrue(any(arg.startswith("override.modalities=") for arg in commands[-1]))

    def test_requires_video_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "MMS-LLaMA"
            (repo / "src").mkdir(parents=True)
            (repo / "src" / "eval.py").write_text("", encoding="utf-8")
            backend = MmsLlamaBackend(BackendConfig(kind="mms-llama", repo_path=str(repo)))
            with self.assertRaises(ValueError):
                backend.transcribe(
                    TranscriptionRequest(
                        audio_path="audio.wav",
                        language="en",
                        reference="hello",
                    )
                )

    def test_resolves_path_like_python_executable(self) -> None:
        self.assertEqual(_resolve_executable("python"), "python")
        self.assertTrue(_resolve_executable("external/MMS-LLaMA/.venv/bin/python").startswith("/"))

    def test_formats_missing_dependency_error(self) -> None:
        message = _format_eval_error(
            "Traceback...\nModuleNotFoundError: No module named 'editdistance'",
            "python",
        )

        self.assertIn("missing Python module 'editdistance'", message)
        self.assertIn("python_executable='python'", message)


    def test_formats_gated_huggingface_error(self) -> None:
        message = _format_eval_error(
            "huggingface_hub.errors.GatedRepoError: Cannot access gated repo",
            "python",
        )

        self.assertIn("could not access the gated LLaMA model", message)
        self.assertIn("huggingface-cli login", message)



def _write_wav(path: Path, frames: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * frames)


if __name__ == "__main__":
    unittest.main()
