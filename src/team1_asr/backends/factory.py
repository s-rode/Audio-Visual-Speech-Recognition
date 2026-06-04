from __future__ import annotations

from team1_asr.backends.base import ASRBackend
from team1_asr.backends.faster_whisper_backend import FasterWhisperBackend
from team1_asr.backends.granite_speech_backend import GraniteSpeechBackend
from team1_asr.backends.noop_backend import NoOpBackend
from team1_asr.backends.parakeet_coreml_backend import ParakeetCoreMLBackend
from team1_asr.config import BackendConfig


def create_backend(config: BackendConfig) -> ASRBackend:
    if config.kind == "faster-whisper":
        return FasterWhisperBackend(config)
    if config.kind == "parakeet-coreml":
        return ParakeetCoreMLBackend(config)
    if config.kind == "granite-speech":
        return GraniteSpeechBackend(config)
    if config.kind == "noop":
        return NoOpBackend(config)
    raise ValueError(f"Unsupported backend kind: {config.kind}")
