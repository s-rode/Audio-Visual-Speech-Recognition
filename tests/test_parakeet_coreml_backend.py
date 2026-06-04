from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from team1_asr.backends.parakeet_coreml_backend import (
    ParakeetCoreMLBackend,
    _bridge_command,
)
from team1_asr.backends.base import TranscriptionRequest
from team1_asr.config import BackendConfig


class ParakeetCoreMLBackendTest(unittest.TestCase):
    def test_transcribes_through_mocked_bridge(self) -> None:
        process = _FakeProcess(
            "\n".join(
                [
                    '{"type":"ready","model":"FluidInference/parakeet-tdt-0.6b-v3-coreml"}',
                    '{"type":"transcript","text":"hello world","duration":1.5,'
                    '"confidence":0.95,"processing_time":0.2,"rtfx":7.5}',
                ]
            )
            + "\n"
        )
        config = BackendConfig(
            kind="parakeet-coreml",
            model="FluidInference/parakeet-tdt-0.6b-v3-coreml",
            device="ane",
        )

        with patch(
            "team1_asr.backends.parakeet_coreml_backend._start_bridge",
            return_value=process,
        ):
            backend = ParakeetCoreMLBackend(config)
            transcript = backend.transcribe(
                TranscriptionRequest(audio_path="audio/utt.wav", language="en")
            )
            backend.close()

        self.assertEqual(transcript.text, "hello world")
        self.assertEqual(transcript.duration, 1.5)
        self.assertEqual(transcript.metadata["backend"], "parakeet-coreml")
        self.assertEqual(transcript.metadata["runtime"], "FluidAudio CoreML")
        self.assertIn('"audio_path": "audio/utt.wav"', process.stdin.getvalue())
        self.assertIn('"type": "shutdown"', process.stdin.getvalue())

    def test_bridge_command_passes_model_and_device(self) -> None:
        config = BackendConfig(
            kind="parakeet-coreml",
            model="FluidInference/parakeet-tdt-0.6b-v3-coreml",
            device="cpu",
        )

        command = _bridge_command(config)

        self.assertEqual(command[-4:], ["--model", config.model, "--device", "cpu"])


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
