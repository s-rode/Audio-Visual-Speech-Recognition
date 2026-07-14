"""Build router metadata for a manifest containing supplied mouth ROI clips."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import av

from team1_asr.visual_reliability import RELIABILITY_FIELDNAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate pre-cropped mouth videos and emit the reliability schema "
            "required by team1-reliability."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--expected-width", type=int, default=88)
    parser.add_argument("--expected-height", type=int, default=88)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = build_rows(
        args.manifest,
        expected_width=args.expected_width,
        expected_height=args.expected_height,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RELIABILITY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    success_count = sum(row["roi_status"] == "success" for row in rows)
    print(
        f"Wrote {len(rows)} rows to {args.output_csv} "
        f"({success_count} valid mouth ROIs)"
    )


def build_rows(
    manifest_path: Path,
    *,
    expected_width: int,
    expected_height: int,
) -> list[dict[str, object]]:
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    return [
        _inspect_row(
            row,
            manifest_path=manifest_path,
            expected_width=expected_width,
            expected_height=expected_height,
        )
        for row in manifest_rows
    ]


def _inspect_row(
    manifest_row: dict[str, str],
    *,
    manifest_path: Path,
    expected_width: int,
    expected_height: int,
) -> dict[str, object]:
    utterance_id = str(manifest_row.get("utterance_id") or "").strip()
    video_path = _resolve_path(str(manifest_row.get("video_path") or ""), manifest_path)
    frame_count = 0
    failure_reason = ""
    try:
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                if frame.width != expected_width or frame.height != expected_height:
                    failure_reason = (
                        f"unexpected_roi_size:{frame.width}x{frame.height};"
                        f"expected:{expected_width}x{expected_height}"
                    )
                    break
                frame_count += 1
    except (av.error.FFmpegError, FileNotFoundError, IndexError, ValueError) as error:
        failure_reason = f"decode_failed:{type(error).__name__}"

    success = frame_count > 0 and not failure_reason
    # The archive contains finalized mouth crops, not full-frame detector traces.
    # A value of 1 marks the supplied crop as trusted; it is not a detector score.
    trusted = 1.0 if success else ""
    return {
        "utterance_id": utterance_id,
        "total_frames": frame_count,
        "face_detected_frames": frame_count if success else 0,
        "face_detection_rate": trusted,
        "retinaface_confidence_mean": "",
        "retinaface_confidence_p10": "",
        "landmark_detected_frames": frame_count if success else 0,
        "mouth_landmark_confidence_mean": trusted,
        "mouth_landmark_confidence_p10": trusted,
        "interpolated_frames": 0,
        "interpolation_rate": 0.0 if success else "",
        "longest_missing_run": 0,
        "multiple_face_frames": 0,
        "multiple_face_rate": 0.0 if success else "",
        "roi_status": "success" if success else "failed",
        "fallback_used": False,
        "failure_reason": failure_reason or "",
        "detector_model": "supplied_pre_cropped_roi",
        "detector_threshold": "",
    }


def _resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


if __name__ == "__main__":
    main()
