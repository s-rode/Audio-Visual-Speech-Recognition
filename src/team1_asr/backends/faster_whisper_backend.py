from __future__ import annotations

from team1_asr.backends.base import (
    ASRBackend,
    TranscriptionRequest,
    require_dependency,
)
from team1_asr.config import BackendConfig
from team1_asr.types import Segment, Transcript, WordSegment


class FasterWhisperBackend(ASRBackend):
    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            require_dependency("faster-whisper", exc)

        self._model = WhisperModel(
            model_size_or_path=config.model,
            device=config.device,
            compute_type=config.compute_type,
        )

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        segments, info = self._model.transcribe(
            request.audio_path,
            language=request.language or self._config.language,
            task=self._config.task,
            beam_size=self._config.beam_size,
            best_of=self._config.best_of,
            word_timestamps=self._config.word_timestamps,
            vad_filter=self._config.vad_filter,
            condition_on_previous_text=self._config.condition_on_previous_text,
        )

        transcript_segments: list[Segment] = []
        full_text: list[str] = []
        for item in segments:
            words = [
                WordSegment(
                    start=word.start,
                    end=word.end,
                    word=word.word,
                    probability=getattr(word, "probability", None),
                )
                for word in (item.words or [])
            ]
            transcript_segments.append(
                Segment(
                    start=item.start,
                    end=item.end,
                    text=item.text,
                    avg_logprob=getattr(item, "avg_logprob", None),
                    no_speech_prob=getattr(item, "no_speech_prob", None),
                    words=words,
                )
            )
            full_text.append(item.text.strip())

        return Transcript(
            text=" ".join(part for part in full_text if part).strip(),
            language=getattr(info, "language", None),
            duration=getattr(info, "duration", None),
            segments=transcript_segments,
            metadata={
                "backend": "faster-whisper",
                "model": self._config.model,
                "language_probability": getattr(info, "language_probability", None),
            },
        )
