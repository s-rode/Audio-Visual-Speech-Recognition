from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    utterances_scored: int
    wer: float | None
    cer: float | None


def compute_metrics(references: list[str], predictions: list[str]) -> Metrics:
    if not references:
        return Metrics(utterances_scored=0, wer=None, cer=None)

    try:
        from jiwer import cer, wer
    except ImportError as exc:
        from team1_asr.backends.base import require_dependency

        require_dependency("jiwer", exc)

    return Metrics(
        utterances_scored=len(references),
        wer=wer(references, predictions),
        cer=cer(references, predictions),
    )
