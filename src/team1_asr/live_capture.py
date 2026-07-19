from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Callable

import av
import numpy as np
import soundfile as sf


CaptureEventSink = Callable[[dict[str, object]], None]
_SUPPORTED_MEDIA_TYPES = {
    "video/webm": ".webm",
    "video/mp4": ".mp4",
    "video/x-matroska": ".mkv",
}


@dataclass(frozen=True)
class CaptureArtifacts:
    source_path: Path
    audio_path: Path
    video_path: Path
    reliability: dict[str, object]


class BrowserCapturePreparer:
    def __init__(
        self,
        *,
        real_preprocessing: bool,
        repo_path: str | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.real_preprocessing = real_preprocessing
        self.repo_path = Path(repo_path).resolve() if repo_path else None
        self.python_executable = (
            str(Path(python_executable).resolve())
            if python_executable and "/" in python_executable
            else python_executable
        )

    def prepare(
        self,
        payload: bytes,
        content_type: str,
        work_dir: Path,
        utterance_id: str,
        emit: CaptureEventSink,
    ) -> CaptureArtifacts:
        suffix = capture_suffix(content_type)
        recording_path = work_dir / f"{utterance_id}{suffix}"
        recording_path.write_bytes(payload)

        emit(
            {
                "stage": "capture",
                "status": "complete",
                "message": "Camera and microphone capture received.",
                "bytes": len(payload),
            }
        )
        emit(
            {
                "stage": "preprocess",
                "status": "running",
                "message": "Preparing synchronized audio and video.",
            }
        )
        audio_path = work_dir / f"{utterance_id}.wav"
        _extract_audio(recording_path, audio_path)
        _validate_video(recording_path)

        if self.real_preprocessing:
            artifacts = self._prepare_real(
                recording_path,
                audio_path,
                work_dir,
                utterance_id,
            )
            message = "Mouth ROI and visual reliability metadata are ready."
        else:
            artifacts = CaptureArtifacts(
                source_path=recording_path,
                audio_path=audio_path,
                video_path=recording_path,
                reliability={
                    "utterance_id": utterance_id,
                    "roi_status": "success",
                    "fallback_used": False,
                    "face_detection_rate": 1.0,
                    "mouth_landmark_confidence_mean": 0.9,
                    "mouth_landmark_confidence_p10": 0.85,
                },
            )
            message = "Live audio and camera motion features are ready."

        emit(
            {
                "stage": "preprocess",
                "status": "complete",
                "message": message,
                "roi_status": artifacts.reliability.get("roi_status"),
            }
        )
        return artifacts

    def _prepare_real(
        self,
        recording_path: Path,
        audio_path: Path,
        work_dir: Path,
        utterance_id: str,
    ) -> CaptureArtifacts:
        if self.repo_path is None or self.python_executable is None:
            raise RuntimeError("Real camera mode requires the AVEC repo and Python executable.")

        manifest_path = work_dir / "capture_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "utterance_id",
                    "audio_path",
                    "video_path",
                    "reference",
                    "language",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "utterance_id": utterance_id,
                    "audio_path": str(audio_path),
                    "video_path": str(recording_path),
                    "reference": "",
                    "language": "en",
                }
            )

        output_dir = work_dir / "avec_roi"
        runner = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "run_avec_preprocess_manifest.py"
        )
        command = [
            self.python_executable,
            str(runner),
            str(manifest_path),
            str(output_dir),
            "--repo-path",
            str(self.repo_path),
            "--force",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                "AVEC camera preprocessing failed"
                + (f": {detail[-1200:]}" if detail else ".")
            )

        manifest_rows = _read_csv(output_dir / "manifest.csv")
        reliability_rows = _read_csv(output_dir / "visual_reliability.csv")
        if not manifest_rows or not reliability_rows:
            raise RuntimeError("AVEC camera preprocessing produced no output row.")
        reliability = reliability_rows[0]
        if reliability.get("roi_status") != "success":
            reason = reliability.get("failure_reason") or reliability.get("roi_status")
            raise RuntimeError(f"No usable mouth ROI was detected: {reason}")
        roi_path = Path(manifest_rows[0]["video_path"])
        if not roi_path.is_absolute():
            roi_path = (output_dir / roi_path).resolve()
        return CaptureArtifacts(recording_path, audio_path, roi_path, reliability)


def capture_suffix(content_type: str) -> str:
    media_type = content_type.partition(";")[0].strip().lower()
    try:
        return _SUPPORTED_MEDIA_TYPES[media_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_SUPPORTED_MEDIA_TYPES))
        raise ValueError(
            f"Unsupported capture media type '{media_type}'. Expected one of: {supported}."
        ) from exc


def _extract_audio(recording_path: Path, audio_path: Path) -> None:
    chunks: list[np.ndarray] = []
    with av.open(str(recording_path)) as container:
        if not container.streams.audio:
            raise ValueError("Camera capture contains no microphone audio.")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray().reshape(-1).astype(np.float32))
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray().reshape(-1).astype(np.float32))
    if not chunks:
        raise ValueError("Camera capture contains no decodable microphone audio.")
    sf.write(audio_path, np.concatenate(chunks), 16000, subtype="PCM_16")


def _validate_video(recording_path: Path) -> None:
    with av.open(str(recording_path)) as container:
        if not container.streams.video:
            raise ValueError("Camera capture contains no video stream.")
        try:
            next(container.decode(video=0))
        except StopIteration as exc:
            raise ValueError("Camera capture contains no decodable video frames.") from exc


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
