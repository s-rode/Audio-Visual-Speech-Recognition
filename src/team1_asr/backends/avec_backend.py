from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, TextIO

from team1_asr.backends.base import ASRBackend, TranscriptionRequest
from team1_asr.config import BackendConfig
from team1_asr.types import Segment, Transcript


class AvecBackend(ASRBackend):
    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        self._process = _start_bridge(config)
        self._stdout = self._require_stream(self._process.stdout, "stdout")
        self._stdin = self._require_stream(self._process.stdin, "stdin")
        ready = self._read_message()
        if ready.get("type") == "error":
            raise RuntimeError(f"AVEC runner failed to start: {ready.get('message')}")
        if ready.get("type") != "ready":
            raise RuntimeError(f"AVEC runner did not become ready: {ready}")

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        return self.transcribe_with_events(request)

    def transcribe_with_events(
        self,
        request: TranscriptionRequest,
        emit: Callable[[dict[str, object]], None] | None = None,
    ) -> Transcript:
        self._write_message(
            {
                "type": "transcribe",
                "audio_path": str(Path(request.audio_path).resolve()),
                "video_path": (
                    str(Path(request.video_path).resolve())
                    if request.video_path
                    else None
                ),
                "language": request.language or self._config.language,
                "utterance_id": request.utterance_id,
                "reference": request.reference,
                "video_preprocessing": request.video_preprocessing,
            }
        )
        while True:
            response = self._read_message()
            response_type = response.get("type")
            if response_type == "progress":
                event = response.get("event")
                if emit is not None and isinstance(event, dict):
                    emit({str(key): value for key, value in event.items()})
                continue
            if response_type == "error":
                raise RuntimeError(f"AVEC runner error: {response.get('message')}")
            if response_type != "transcript":
                raise RuntimeError(f"Unexpected AVEC runner response: {response}")
            return self._transcript_from_response(request, response)

    def _transcript_from_response(
        self,
        request: TranscriptionRequest,
        response: dict[str, Any],
    ) -> Transcript:
        text = str(response.get("text", "")).strip()
        duration = _optional_float(response.get("duration"))
        return Transcript(
            text=text,
            language=str(
                response.get("language")
                or request.language
                or self._config.language
            ),
            duration=duration,
            segments=[Segment(start=0.0, end=duration or 0.0, text=text)],
            metadata={
                "backend": "avec",
                "model": self._config.model,
                "modality": self._config.modality,
                "repo_path": self._config.repo_path,
                "config_file": self._config.config_file,
                "checkpoint": self._config.checkpoint,
                **_metadata_dict(response.get("metadata")),
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
            raise RuntimeError(f"AVEC runner has no {name} stream.")
        return stream

    def _write_message(self, message: dict[str, object]) -> None:
        if self._process.poll() is not None:
            raise RuntimeError("AVEC runner exited before request.")
        self._stdin.write(json.dumps(message) + "\n")
        self._stdin.flush()

    def _read_message(self) -> dict[str, Any]:
        line = self._stdout.readline()
        if not line:
            exit_code = self._process.poll()
            raise RuntimeError(
                "AVEC runner exited without a response"
                + (f" (exit code {exit_code})." if exit_code is not None else ".")
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"AVEC runner wrote non-JSON output: {line.strip()}") from exc


def _start_bridge(config: BackendConfig) -> subprocess.Popen[str]:
    project_root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment.setdefault("HF_HOME", str(project_root / ".tmp/hf"))
    matplotlib_cache = project_root / ".tmp" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    environment.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    return subprocess.Popen(
        _bridge_command(config),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        env=environment,
        text=True,
        bufsize=1,
    )


def _bridge_command(config: BackendConfig) -> list[str]:
    if config.runner_module == "team1_asr.avec_manifest_runner":
        command = [config.python_executable, str(_default_runner_path())]
    else:
        command = [config.python_executable, "-m", config.runner_module]
    command.extend(
        [
            "--repo-path",
            config.repo_path,
            "--config-file",
            config.config_file,
            "--checkpoint",
            config.checkpoint,
            "--modality",
            config.modality,
            "--model-name",
            config.model,
        ]
    )
    if config.device == "cpu":
        command.append("--cpu")
    if config.callback_path:
        command.extend(["--callback-path", config.callback_path])
    command.extend(config.extra_args)
    return command


def _default_runner_path() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "run_avec_manifest.py"


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metadata_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}
