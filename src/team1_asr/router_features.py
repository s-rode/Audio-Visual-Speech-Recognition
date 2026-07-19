"""Extract deployable audio, mouth-ROI, and synchronization routing features."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import unicodedata

import av
import numpy as np
import soundfile as sf

from team1_asr.types import Transcript
from team1_asr.visual_reliability import optional_float


def extract_live_features(
    audio_path: Path,
    video_path: Path | None,
    transcript: Transcript,
) -> dict[str, object]:
    """Extract one request's deployable features after Whisper completes."""
    samples, sample_rate = sf.read(audio_path, dtype="float64", always_2d=True)
    audio = samples.mean(axis=1)
    features: dict[str, object] = {
        **_transcript_features(transcript),
        **_audio_signal_features(audio),
    }
    audio_features, _ = _audio_features(audio, sample_rate)
    features.update(audio_features)

    if video_path is None:
        return features
    frames, timestamps = _read_gray_video(video_path)
    video_features, mouth_envelope = _video_features(frames, timestamps)
    features.update(video_features)
    features.update(
        _synchronization_features(audio, sample_rate, timestamps, mouth_envelope)
    )
    return features


def _transcript_features(transcript: Transcript) -> dict[str, object]:
    record = {"segments": [asdict(segment) for segment in transcript.segments]}
    logprobs = [
        value
        for segment in transcript.segments
        for value in [optional_float(segment.avg_logprob)]
        if value is not None
    ]
    no_speech = [
        value
        for segment in transcript.segments
        for value in [optional_float(segment.no_speech_prob)]
        if value is not None
    ]
    duration = optional_float(transcript.duration)
    normalized = "".join(
        character
        for character in transcript.text.lower()
        if not unicodedata.category(character).startswith("P")
    )
    word_count = len(normalized.split())
    return {
        **_whisper_features(record),
        "audio_avg_logprob_mean": _mean(logprobs),
        "audio_avg_logprob_min": min(logprobs) if logprobs else None,
        "audio_no_speech_prob_max": max(no_speech) if no_speech else None,
        "audio_prediction_word_count": word_count,
        "audio_duration": duration,
        "audio_prediction_words_per_second": (
            word_count / duration if duration is not None and duration > 0 else None
        ),
    }


def _audio_signal_features(samples: np.ndarray) -> dict[str, float | None]:
    if samples.size == 0:
        return {
            "audio_rms_dbfs": None,
            "audio_peak_dbfs": None,
            "audio_crest_factor_db": None,
            "audio_clipping_fraction": None,
        }
    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples)))
    epsilon = 1e-12
    return {
        "audio_rms_dbfs": _db(rms),
        "audio_peak_dbfs": _db(peak),
        "audio_crest_factor_db": 20.0
        * math.log10(max(peak, epsilon) / max(rms, epsilon)),
        "audio_clipping_fraction": float(np.mean(np.abs(samples) >= 0.999)),
    }


def extract_features(
    roi_manifest_path: Path,
    reliability_path: Path,
    conditions: list[tuple[str, Path]],
    output_dir: Path,
) -> dict[str, Path]:
    """Write condition-keyed, pre-AVEC routing features from completed Whisper runs."""
    manifest_rows = _read_csv_by_id(roi_manifest_path)
    reliability_rows = _read_csv_by_id(reliability_path)
    names = [name for name, _ in conditions]
    if len(set(names)) != len(names):
        raise ValueError("Feature-extraction condition names must be unique.")

    rows: list[dict[str, object]] = []
    for condition, transcript_path in conditions:
        transcripts = _read_transcripts(transcript_path)
        missing = sorted(set(manifest_rows) - set(transcripts))
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Whisper transcripts are missing {len(missing)} utterance(s) for "
                f"{condition}: {preview}"
            )
        for utterance_id, manifest_row in manifest_rows.items():
            if utterance_id not in reliability_rows:
                raise ValueError(f"Reliability metadata is missing {utterance_id}")
            transcript = transcripts[utterance_id]
            audio_path = Path(str(transcript.get("used_audio_path") or ""))
            if not audio_path.exists():
                raise ValueError(
                    f"Noisy audio for {condition}/{utterance_id} does not exist: {audio_path}"
                )
            video_path = _resolve_path(roi_manifest_path.parent, manifest_row.get("video_path", ""))
            if not video_path.exists():
                raise ValueError(
                    f"Mouth ROI for {condition}/{utterance_id} does not exist: {video_path}"
                )
            samples, sample_rate = sf.read(audio_path, dtype="float64", always_2d=True)
            audio = samples.mean(axis=1)
            frames, timestamps = _read_gray_video(video_path)
            whisper = _whisper_features(transcript)
            audio_features, _ = _audio_features(audio, sample_rate)
            video_features, mouth_envelope = _video_features(frames, timestamps)
            synchronization = _synchronization_features(
                audio,
                sample_rate,
                timestamps,
                mouth_envelope,
            )
            rows.append(
                {
                    "condition": condition,
                    "utterance_id": utterance_id,
                    **whisper,
                    **audio_features,
                    **video_features,
                    **synchronization,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "routing_features.csv"
    summary_path = output_dir / "routing_feature_summary.csv"
    summary_json_path = output_dir / "routing_feature_summary.json"
    _write_csv(features_path, rows)
    summary_rows = _feature_summary_rows(rows)
    _write_csv(summary_path, summary_rows)
    summary_json_path.write_text(
        json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "features": features_path,
        "summary": summary_path,
        "summary_json": summary_json_path,
    }


def _whisper_features(record: dict[str, object]) -> dict[str, float | None]:
    segments = record.get("segments")
    if not isinstance(segments, list):
        segments = []
    compression = [
        value
        for item in segments
        if isinstance(item, dict)
        for value in [optional_float(item.get("compression_ratio"))]
        if value is not None
    ]
    logprobs = [
        value
        for item in segments
        if isinstance(item, dict)
        for value in [optional_float(item.get("avg_logprob"))]
        if value is not None
    ]
    temperatures = [
        value
        for item in segments
        if isinstance(item, dict)
        for value in [optional_float(item.get("temperature"))]
        if value is not None
    ]
    return {
        "audio_compression_ratio_mean": _mean(compression),
        "audio_compression_ratio_max": max(compression) if compression else None,
        "audio_segment_count": float(len(segments)),
        "audio_avg_logprob_std": _std(logprobs),
        "audio_temperature_mean": _mean(temperatures),
    }


def _audio_features(
    samples: np.ndarray, sample_rate: int
) -> tuple[dict[str, float | None], np.ndarray]:
    if samples.size == 0 or sample_rate <= 0:
        return ({name: None for name in _AUDIO_FEATURES}, np.array([], dtype=float))
    frame_size = max(1, round(0.025 * sample_rate))
    hop_size = max(1, round(0.010 * sample_rate))
    frames = _frame_signal(samples, frame_size, hop_size)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    frame_mean = np.mean(frames, axis=1, keepdims=True)
    zero_crossing = np.mean(
        np.diff(np.signbit(frames - frame_mean), axis=1) != 0,
        axis=1,
    )
    windowed = frames * np.hanning(frame_size)
    power = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    mean_power = power.mean(axis=0) + 1e-12
    total_power = float(mean_power.sum())
    centroid = float((frequencies * mean_power).sum() / total_power)
    bandwidth = float(
        np.sqrt((((frequencies - centroid) ** 2) * mean_power).sum() / total_power)
    )
    cumulative = np.cumsum(mean_power) / total_power
    rolloff_index = int(np.searchsorted(cumulative, 0.85, side="left"))
    high_frequency = float(mean_power[frequencies >= 4000].sum() / total_power)
    spectral_flatness = float(
        np.exp(np.mean(np.log(mean_power))) / np.mean(mean_power)
    )
    peak_rms = float(np.max(rms))
    silence_threshold = max(peak_rms * 0.1, 1e-6)
    return (
        {
            "audio_rms_p10_dbfs": _db(np.percentile(rms, 10)),
            "audio_rms_p50_dbfs": _db(np.percentile(rms, 50)),
            "audio_rms_p90_dbfs": _db(np.percentile(rms, 90)),
            "audio_rms_dynamic_range_db": _db(np.percentile(rms, 90))
            - _db(np.percentile(rms, 10)),
            "audio_silence_fraction": float(np.mean(rms <= silence_threshold)),
            "audio_zero_crossing_rate": float(np.mean(zero_crossing)),
            "audio_spectral_centroid_hz": centroid,
            "audio_spectral_bandwidth_hz": bandwidth,
            "audio_spectral_rolloff_hz": float(frequencies[min(rolloff_index, len(frequencies) - 1)]),
            "audio_spectral_flatness": spectral_flatness,
            "audio_high_frequency_energy_ratio": high_frequency,
        },
        rms,
    )


def _video_features(
    frames: np.ndarray, timestamps: np.ndarray
) -> tuple[dict[str, float | None], np.ndarray]:
    if frames.size == 0:
        return ({name: None for name in _VIDEO_FEATURES}, np.array([], dtype=float))
    brightness = frames.mean(axis=(1, 2))
    contrast = frames.std(axis=(1, 2))
    horizontal = np.diff(frames, axis=2)
    vertical = np.diff(frames, axis=1)
    sharpness = np.mean(horizontal**2, axis=(1, 2)) + np.mean(vertical**2, axis=(1, 2))
    motion = (
        np.mean(np.abs(np.diff(frames, axis=0)), axis=(1, 2))
        if len(frames) > 1
        else np.array([], dtype=float)
    )
    motion_with_first = np.concatenate(([0.0], motion))
    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    return (
        {
            "video_mouth_motion_mean": _mean(motion),
            "video_mouth_motion_p10": _percentile(motion, 10),
            "video_mouth_motion_p90": _percentile(motion, 90),
            "video_still_frame_rate": float(np.mean(motion_with_first < 0.01)),
            "video_luminance_mean": _mean(brightness),
            "video_luminance_std": _std(brightness),
            "video_contrast_mean": _mean(contrast),
            "video_sharpness_mean": _mean(sharpness),
            "video_roi_frame_rate": float((len(frames) - 1) / duration) if duration > 0 else None,
        },
        motion_with_first,
    )


def _synchronization_features(
    audio: np.ndarray,
    sample_rate: int,
    timestamps: np.ndarray,
    mouth_envelope: np.ndarray,
) -> dict[str, float | None]:
    if len(timestamps) < 3 or len(mouth_envelope) != len(timestamps):
        return {name: None for name in _SYNC_FEATURES}
    audio_envelope = np.array(
        [_local_rms(audio, sample_rate, float(timestamp)) for timestamp in timestamps],
        dtype=float,
    )
    if np.std(audio_envelope) < 1e-10 or np.std(mouth_envelope) < 1e-10:
        return {name: None for name in _SYNC_FEATURES}
    frame_interval = float(np.median(np.diff(timestamps)))
    max_shift = max(1, round(0.4 / max(frame_interval, 1e-3)))
    best_correlation = -1.0
    best_shift = 0
    for shift in range(-max_shift, max_shift + 1):
        left, right = _shifted_pair(audio_envelope, mouth_envelope, shift)
        if len(left) < 3 or np.std(left) < 1e-10 or np.std(right) < 1e-10:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        if correlation > best_correlation:
            best_correlation = correlation
            best_shift = shift
    return {
        "audio_video_sync_correlation": best_correlation if best_correlation >= -1.0 else None,
        "audio_video_sync_lag_ms": best_shift * frame_interval * 1000.0,
    }


def _read_gray_video(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frames = []
    timestamps = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            frames.append(frame.to_ndarray(format="gray").astype(np.float64) / 255.0)
            timestamps.append(float(frame.time) if frame.time is not None else float(index) / float(stream.average_rate))
    if not frames:
        raise ValueError(f"Mouth ROI video has no frames: {path}")
    return np.stack(frames), np.array(timestamps, dtype=float)


def _frame_signal(samples: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if len(samples) < frame_size:
        return np.pad(samples, (0, frame_size - len(samples))).reshape(1, frame_size)
    starts = np.arange(0, len(samples) - frame_size + 1, hop_size)
    return np.stack([samples[start : start + frame_size] for start in starts])


def _local_rms(audio: np.ndarray, sample_rate: int, timestamp: float) -> float:
    half_window = max(1, round(0.020 * sample_rate))
    center = round(timestamp * sample_rate)
    start = max(0, center - half_window)
    end = min(len(audio), center + half_window)
    return float(np.sqrt(np.mean(audio[start:end] ** 2))) if end > start else 0.0


def _shifted_pair(left: np.ndarray, right: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    if shift > 0:
        return left[shift:], right[:-shift]
    if shift < 0:
        return left[:shift], right[-shift:]
    return left, right


def _read_transcripts(path: Path) -> dict[str, dict[str, object]]:
    transcripts = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            utterance_id = str(record.get("utterance_id") or "")
            if not utterance_id:
                raise ValueError(f"Transcript {path}:{line_number} has no utterance_id")
            if utterance_id in transcripts:
                raise ValueError(f"Duplicate utterance_id in {path}: {utterance_id}")
            transcripts[utterance_id] = record
    return transcripts


def _read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            utterance_id = str(row.get("utterance_id") or "")
            if not utterance_id:
                raise ValueError(f"CSV row in {path} has no utterance_id")
            if utterance_id in rows:
                raise ValueError(f"Duplicate utterance_id in {path}: {utterance_id}")
            rows[utterance_id] = row
    return rows


def _resolve_path(base: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _feature_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows = []
    feature_names = [name for name in rows[0] if name not in {"condition", "utterance_id"}] if rows else []
    for condition in sorted({str(row["condition"]) for row in rows}):
        condition_rows = [row for row in rows if row["condition"] == condition]
        for feature in feature_names:
            values = [optional_float(row.get(feature)) for row in condition_rows]
            present = [value for value in values if value is not None]
            summary_rows.append(
                {
                    "condition": condition,
                    "feature": feature,
                    "count": len(condition_rows),
                    "missing_count": len(condition_rows) - len(present),
                    "p10": _percentile(np.array(present), 10),
                    "median": _percentile(np.array(present), 50),
                    "p90": _percentile(np.array(present), 90),
                }
            )
    return summary_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = _union_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _union_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    names = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    return names


def _mean(values: np.ndarray | list[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _std(values: np.ndarray | list[float]) -> float | None:
    return float(np.std(values)) if len(values) else None


def _percentile(values: np.ndarray | list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if len(values) else None


def _db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


_AUDIO_FEATURES = (
    "audio_rms_p10_dbfs",
    "audio_rms_p50_dbfs",
    "audio_rms_p90_dbfs",
    "audio_rms_dynamic_range_db",
    "audio_silence_fraction",
    "audio_zero_crossing_rate",
    "audio_spectral_centroid_hz",
    "audio_spectral_bandwidth_hz",
    "audio_spectral_rolloff_hz",
    "audio_spectral_flatness",
    "audio_high_frequency_energy_ratio",
)
_VIDEO_FEATURES = (
    "video_mouth_motion_mean",
    "video_mouth_motion_p10",
    "video_mouth_motion_p90",
    "video_still_frame_rate",
    "video_luminance_mean",
    "video_luminance_std",
    "video_contrast_mean",
    "video_sharpness_mean",
    "video_roi_frame_rate",
)
_SYNC_FEATURES = ("audio_video_sync_correlation", "audio_video_sync_lag_ms")
