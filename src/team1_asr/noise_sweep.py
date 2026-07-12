"""Metrics for controlled audio-noise comparisons between ASR and AVSR systems."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import unicodedata

from team1_asr.evaluation import compute_metrics


def read_manifest_references(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    references: dict[str, str] = {}
    for index, row in enumerate(rows):
        utterance_id = row.get("utterance_id") or f"utt-{index:05d}"
        if utterance_id in references:
            raise ValueError(f"Duplicate utterance_id in {path}: {utterance_id}")
        references[utterance_id] = row.get("reference", "")
    return references


def read_predictions(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            utterance_id = str(row.get("utterance_id") or "")
            if not utterance_id:
                raise ValueError(f"Transcript {path}:{line_number} has no utterance_id")
            if utterance_id in predictions:
                raise ValueError(f"Duplicate utterance_id in {path}: {utterance_id}")
            predictions[utterance_id] = str(row.get("prediction") or "")
    return predictions


def analyze_condition(
    *,
    name: str,
    snr_db: float | None,
    references: dict[str, str],
    whisper: dict[str, str],
    avec_ao: dict[str, str],
    avec_av: dict[str, str],
) -> dict[str, object]:
    missing_whisper = sorted(set(references) - set(whisper))
    if missing_whisper:
        preview = ", ".join(missing_whisper[:5])
        raise ValueError(f"Whisper is missing {len(missing_whisper)} transcript(s): {preview}")

    result = {
        "condition": name,
        "snr_db": snr_db,
        "whisper": _system_metrics(references, whisper),
        "avec_ao": _system_metrics(references, avec_ao),
        "avec_av": _system_metrics(references, avec_av),
    }
    result["av_vs_whisper"] = _pairwise_metrics(references, whisper, avec_av)
    result["av_vs_ao"] = _pairwise_metrics(references, avec_ao, avec_av)
    result["whisper_av_oracle"] = _oracle_metrics(references, whisper, avec_av)
    return result


def flatten_condition(result: dict[str, object]) -> dict[str, object]:
    whisper = _as_dict(result["whisper"])
    ao = _as_dict(result["avec_ao"])
    av = _as_dict(result["avec_av"])
    av_vs_whisper = _as_dict(result["av_vs_whisper"])
    av_vs_ao = _as_dict(result["av_vs_ao"])
    oracle = _as_dict(result["whisper_av_oracle"])
    return {
        "condition": result["condition"],
        "snr_db": result["snr_db"],
        "whisper_wer": whisper["wer"],
        "whisper_cer": whisper["cer"],
        "ao_wer": ao["wer"],
        "ao_cer": ao["cer"],
        "ao_coverage": ao["coverage"],
        "av_wer": av["wer"],
        "av_cer": av["cer"],
        "av_coverage": av["coverage"],
        "av_wins_over_whisper": av_vs_whisper["right_wins"],
        "av_wins_over_whisper_rate": av_vs_whisper["right_win_rate"],
        "av_wins_over_ao": av_vs_ao["right_wins"],
        "av_wins_over_ao_rate": av_vs_ao["right_win_rate"],
        "oracle_wer": oracle["wer"],
        "oracle_gap_from_whisper": oracle["gap_from_left_wer"],
    }


def _system_metrics(references: dict[str, str], predictions: dict[str, str]) -> dict[str, object]:
    ids = [utterance_id for utterance_id in references if utterance_id in predictions]
    metrics = compute_metrics(
        [references[utterance_id] for utterance_id in ids],
        [predictions[utterance_id] for utterance_id in ids],
    )
    return {
        "utterances_scored": metrics.utterances_scored,
        "coverage": len(ids) / len(references) if references else 0.0,
        "wer": metrics.wer,
        "cer": metrics.cer,
        "raw_wer": metrics.raw_wer,
        "raw_cer": metrics.raw_cer,
    }


def _pairwise_metrics(
    references: dict[str, str], left: dict[str, str], right: dict[str, str]
) -> dict[str, object]:
    ids = [
        utterance_id
        for utterance_id in references
        if utterance_id in left and utterance_id in right
    ]
    right_wins = 0
    left_wins = 0
    ties = 0
    for utterance_id in ids:
        left_errors, _ = word_error_stats(references[utterance_id], left[utterance_id])
        right_errors, _ = word_error_stats(
            references[utterance_id], right[utterance_id]
        )
        if right_errors < left_errors:
            right_wins += 1
        elif left_errors < right_errors:
            left_wins += 1
        else:
            ties += 1
    return {
        "utterances_compared": len(ids),
        "right_wins": right_wins,
        "left_wins": left_wins,
        "ties": ties,
        "right_win_rate": right_wins / len(ids) if ids else 0.0,
    }


def _oracle_metrics(
    references: dict[str, str], left: dict[str, str], right: dict[str, str]
) -> dict[str, object]:
    predictions = []
    for utterance_id, reference in references.items():
        left_prediction = left[utterance_id]
        right_prediction = right.get(utterance_id)
        if right_prediction is None:
            predictions.append(left_prediction)
            continue
        left_errors, _ = word_error_stats(reference, left_prediction)
        right_errors, _ = word_error_stats(reference, right_prediction)
        predictions.append(right_prediction if right_errors < left_errors else left_prediction)
    metrics = compute_metrics(list(references.values()), predictions)
    left_metrics = _system_metrics(references, left)
    return {
        "wer": metrics.wer,
        "cer": metrics.cer,
        "gap_from_left_wer": left_metrics["wer"] - metrics.wer,
    }


def word_error_stats(reference: str, prediction: str) -> tuple[int, int]:
    reference_words = _normalize_words(reference)
    prediction_words = _normalize_words(prediction)
    previous = list(range(len(prediction_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current = [reference_index]
        for prediction_index, prediction_word in enumerate(prediction_words, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[prediction_index] + 1,
                    previous[prediction_index - 1]
                    + int(reference_word != prediction_word),
                )
            )
        previous = current
    return previous[-1], len(reference_words)


def _normalize_words(text: str) -> list[str]:
    return "".join(
        character
        for character in text.lower()
        if not unicodedata.category(character).startswith("P")
    ).split()


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected a metrics object.")
    return value
