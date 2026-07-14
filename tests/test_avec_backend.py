from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import patch

from team1_asr.backends.avec_backend import AvecBackend, _bridge_command
from team1_asr.backends.base import TranscriptionRequest
from team1_asr.config import BackendConfig


class AvecBackendTest(unittest.TestCase):
    def test_transcribes_through_mocked_runner(self) -> None:
        process = _FakeProcess(
            "\n".join(
                [
                    '{"type":"ready","model":"avec-audio-visual"}',
                    '{"type":"transcript","text":"hello world","duration":1.25,'
                    '"language":"en","metadata":{"runtime":"mock"}}',
                ]
            )
            + "\n"
        )
        config = BackendConfig(kind="avec", model="avec-audio-visual")

        with patch("team1_asr.backends.avec_backend._start_bridge", return_value=process):
            backend = AvecBackend(config)
            transcript = backend.transcribe(
                TranscriptionRequest(
                    audio_path="audio/utt.wav",
                    video_path="video/utt.mp4",
                    language="en",
                    utterance_id="utt",
                )
            )
            backend.close()

        self.assertEqual(transcript.text, "hello world")
        self.assertEqual(transcript.duration, 1.25)
        self.assertEqual(transcript.metadata["backend"], "avec")
        self.assertEqual(transcript.metadata["runtime"], "mock")
        self.assertIn('"type": "shutdown"', process.stdin.getvalue())
        self.assertIn('"utterance_id": "utt"', process.stdin.getvalue())

    def test_bridge_command_uses_default_runner_file(self) -> None:
        config = BackendConfig(
            kind="avec",
            model="avec-visual",
            repo_path="external/AVEC",
            config_file="configs/LRS23/VO/EffConfInterCTC.py",
            checkpoint="checkpoints_swa-equal-90-100.ckpt",
            callback_path="callbacks/custom/visual",
            modality="visual",
            device="cpu",
            extra_args=["--mock-transcript", "ok"],
        )

        command = _bridge_command(config)

        self.assertEqual(command[0], "python")
        self.assertEqual(Path(command[1]).name, "run_avec_manifest.py")
        self.assertIn("--cpu", command)
        self.assertIn("--callback-path", command)
        self.assertIn("callbacks/custom/visual", command)
        self.assertEqual(command[-2:], ["--mock-transcript", "ok"])


class _FakeProcess:
    def __init__(self, stdout: str) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(stdout)
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 143

    def kill(self) -> None:
        self.killed = True
        self.returncode = 9


if __name__ == "__main__":
    unittest.main()
