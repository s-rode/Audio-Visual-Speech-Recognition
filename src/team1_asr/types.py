from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WordSegment:
    start: float | None
    end: float | None
    word: str
    probability: float | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: list[WordSegment] = field(default_factory=list)


@dataclass
class Transcript:
    text: str
    language: str | None
    duration: float | None
    segments: list[Segment]
    metadata: dict[str, Any] = field(default_factory=dict)
