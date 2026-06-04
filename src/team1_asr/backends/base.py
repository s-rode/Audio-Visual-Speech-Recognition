from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from team1_asr.types import Transcript


@dataclass
class TranscriptionRequest:
    audio_path: str
    language: str | None


class ASRBackend(Protocol):
    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        ...


class MissingDependencyError(RuntimeError):
    """Raised when an optional runtime dependency is unavailable."""


def require_dependency(package_name: str, exc: Exception) -> None:
    raise MissingDependencyError(
        f"Missing optional dependency '{package_name}'. Install project dependencies first."
    ) from exc
