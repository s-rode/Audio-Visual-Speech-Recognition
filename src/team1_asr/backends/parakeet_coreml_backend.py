from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, TextIO

from team1_asr.backends.base import ASRBackend, TranscriptionRequest
from team1_asr.config import BackendConfig
from team1_asr.types import Segment, Transcript


class ParakeetCoreMLBackend(ASRBackend):
    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        self._process = _start_bridge(config)
        self._stdout = self._require_stream(self._process.stdout, "stdout")
        self._stdin = self._require_stream(self._process.stdin, "stdin")
        ready = self._read_message()
        if ready.get("type") != "ready":
            raise RuntimeError(f"Parakeet CoreML bridge did not become ready: {ready}")

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        self._write_message(
            {
                "type": "transcribe",
                "audio_path": request.audio_path,
                "language": request.language or self._config.language,
            }
        )
        response = self._read_message()
        if response.get("type") == "error":
            raise RuntimeError(f"Parakeet CoreML bridge error: {response.get('message')}")
        if response.get("type") != "transcript":
            raise RuntimeError(f"Unexpected Parakeet CoreML bridge response: {response}")

        text = str(response.get("text", "")).strip()
        duration = _optional_float(response.get("duration"))
        return Transcript(
            text=text,
            language=request.language or self._config.language,
            duration=duration,
            segments=[Segment(start=0.0, end=duration or 0.0, text=text)],
            metadata={
                "backend": "parakeet-coreml",
                "model": self._config.model,
                "runtime": "FluidAudio CoreML",
                "device": self._config.device,
                "confidence": _optional_float(response.get("confidence")),
                "processing_time": _optional_float(response.get("processing_time")),
                "rtfx": _optional_float(response.get("rtfx")),
            },
        )

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            self._write_message({"type": "shutdown"})
            self._process.wait(timeout=5)
        except Exception:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    @staticmethod
    def _require_stream(stream: TextIO | None, name: str) -> TextIO:
        if stream is None:
            raise RuntimeError(f"Parakeet CoreML bridge has no {name} stream.")
        return stream

    def _write_message(self, message: dict[str, object]) -> None:
        if self._process.poll() is not None:
            raise RuntimeError("Parakeet CoreML bridge process exited before request.")
        self._stdin.write(json.dumps(message) + "\n")
        self._stdin.flush()

    def _read_message(self) -> dict[str, Any]:
        line = self._stdout.readline()
        if not line:
            exit_code = self._process.poll()
            raise RuntimeError(
                "Parakeet CoreML bridge exited without a response"
                + (f" (exit code {exit_code})." if exit_code is not None else ".")
            )
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Parakeet CoreML bridge wrote non-JSON output: {line.strip()}"
            ) from exc
        return message


def _start_bridge(config: BackendConfig) -> subprocess.Popen[str]:
    command = _bridge_command(config)
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not start the Parakeet CoreML bridge. Install Xcode command line "
            "tools and run 'swift build -c release --package-path "
            "tools/parakeet_coreml_bridge'."
        ) from exc


def _bridge_command(config: BackendConfig) -> list[str]:
    package_dir = _repo_root() / "tools" / "parakeet_coreml_bridge"
    env_binary = os.environ.get("TEAM1_PARAKEET_COREML_BRIDGE")
    release_binary = package_dir / ".build" / "release" / "parakeet-coreml-bridge"
    if env_binary:
        command = [env_binary]
    elif release_binary.exists():
        command = [str(release_binary)]
    else:
        command = [
            "swift",
            "run",
            "-c",
            "release",
            "--package-path",
            str(package_dir),
            "parakeet-coreml-bridge",
        ]

    command.extend(["--model", config.model, "--device", config.device])
    return command


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
