from __future__ import annotations

from pathlib import Path

from team1_asr.backends.base import ASRBackend, TranscriptionRequest
from team1_asr.config import BackendConfig
from team1_asr.types import Segment, Transcript


class NoOpBackend(ASRBackend):
    """Useful for smoke tests before model dependencies are installed."""

    def __init__(self, config: BackendConfig) -> None:
        self._config = config

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        stem = Path(request.audio_path).stem.replace("_", " ")
        text = f"[noop transcript] {stem}".strip()
        return Transcript(
            text=text,
            language=request.language or self._config.language,
            duration=None,
            segments=[Segment(start=0.0, end=0.0, text=text)],
            metadata={"backend": "noop", "model": self._config.model},
        )
