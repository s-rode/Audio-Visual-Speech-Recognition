from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    utterances_scored: int
    wer: float | None
    cer: float | None
    raw_wer: float | None
    raw_cer: float | None


def compute_metrics(references: list[str], predictions: list[str]) -> Metrics:
    if not references:
        return Metrics(
            utterances_scored=0,
            wer=None,
            cer=None,
            raw_wer=None,
            raw_cer=None,
        )

    try:
        from jiwer import (
            Compose,
            ReduceToListOfListOfChars,
            ReduceToListOfListOfWords,
            RemoveMultipleSpaces,
            RemovePunctuation,
            Strip,
            ToLowerCase,
            cer,
            wer,
        )
    except ImportError as exc:
        from team1_asr.backends.base import require_dependency

        require_dependency("jiwer", exc)

    word_transform = Compose(
        [
            ToLowerCase(),
            RemovePunctuation(),
            RemoveMultipleSpaces(),
            Strip(),
            ReduceToListOfListOfWords(),
        ]
    )
    char_transform = Compose(
        [
            ToLowerCase(),
            RemovePunctuation(),
            RemoveMultipleSpaces(),
            Strip(),
            ReduceToListOfListOfChars(),
        ]
    )
    return Metrics(
        utterances_scored=len(references),
        wer=wer(
            references,
            predictions,
            reference_transform=word_transform,
            hypothesis_transform=word_transform,
        ),
        cer=cer(
            references,
            predictions,
            reference_transform=char_transform,
            hypothesis_transform=char_transform,
        ),
        raw_wer=wer(references, predictions),
        raw_cer=cer(references, predictions),
    )
