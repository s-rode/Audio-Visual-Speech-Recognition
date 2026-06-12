from __future__ import annotations

from team1_asr.backends.base import ASRBackend
from team1_asr.backends.faster_whisper_backend import FasterWhisperBackend
from team1_asr.backends.mms_llama_backend import MmsLlamaBackend
from team1_asr.backends.noop_backend import NoOpBackend
from team1_asr.config import BackendConfig


def create_backend(config: BackendConfig) -> ASRBackend:
    if config.kind == "faster-whisper":
        return FasterWhisperBackend(config)
    if config.kind == "noop":
        return NoOpBackend(config)
    if config.kind == "mms-llama":
        return MmsLlamaBackend(config)
    raise ValueError(f"Unsupported backend kind: {config.kind}")
