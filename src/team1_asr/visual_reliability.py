from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


RELIABILITY_FIELDNAMES = [
    "utterance_id",
    "total_frames",
    "face_detected_frames",
    "face_detection_rate",
    "retinaface_confidence_mean",
    "retinaface_confidence_p10",
    "landmark_detected_frames",
    "mouth_landmark_confidence_mean",
    "mouth_landmark_confidence_p10",
    "interpolated_frames",
    "interpolation_rate",
    "longest_missing_run",
    "multiple_face_frames",
    "multiple_face_rate",
    "roi_status",
    "fallback_used",
    "failure_reason",
    "detector_model",
    "detector_threshold",
]


@dataclass(frozen=True)
class FrameReliability:
    face_count: int
    face_confidence: float | None
    mouth_landmark_confidence: float | None


def summarize_visual_reliability(
    utterance_id: str,
    observations: Iterable[FrameReliability],
    *,
    roi_status: str,
    fallback_used: bool,
    failure_reason: str = "",
    detector_model: str = "resnet50",
    detector_threshold: float = 0.8,
) -> dict[str, object]:
    frames = list(observations)
    face_confidences = [
        item.face_confidence for item in frames if item.face_confidence is not None
    ]
    landmark_confidences = [
        item.mouth_landmark_confidence
        for item in frames
        if item.mouth_landmark_confidence is not None
    ]
    face_detected = len(face_confidences)
    landmark_detected = len(landmark_confidences)
    total = len(frames)
    interpolated = total - landmark_detected
    multiple_face_frames = sum(item.face_count > 1 for item in frames)

    return {
        "utterance_id": utterance_id,
        "total_frames": total,
        "face_detected_frames": face_detected,
        "face_detection_rate": _rate(face_detected, total),
        "retinaface_confidence_mean": _mean(face_confidences),
        "retinaface_confidence_p10": _percentile(face_confidences, 0.10),
        "landmark_detected_frames": landmark_detected,
        "mouth_landmark_confidence_mean": _mean(landmark_confidences),
        "mouth_landmark_confidence_p10": _percentile(landmark_confidences, 0.10),
        "interpolated_frames": interpolated,
        "interpolation_rate": _rate(interpolated, total),
        "longest_missing_run": _longest_missing_run(frames),
        "multiple_face_frames": multiple_face_frames,
        "multiple_face_rate": _rate(multiple_face_frames, total),
        "roi_status": roi_status,
        "fallback_used": fallback_used,
        "failure_reason": failure_reason,
        "detector_model": detector_model,
        "detector_threshold": detector_threshold,
    }


def is_real_mouth_roi(row: dict[str, object]) -> bool:
    return (
        str(row.get("roi_status", "")).strip().lower() == "success"
        and not parse_bool(row.get("fallback_used"))
    )


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(math.floor((len(ordered) - 1) * fraction))
    return ordered[index]


def _longest_missing_run(frames: list[FrameReliability]) -> int:
    longest = 0
    current = 0
    for item in frames:
        if item.mouth_landmark_confidence is None:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
