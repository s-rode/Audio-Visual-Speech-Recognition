#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from itertools import islice
from pathlib import Path
import shutil
import subprocess

import numpy as np
import soundfile as sf


DATASET_NAME = "mattymchen/lrs3-test"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export the experimental Hugging Face mattymchen/lrs3-test mirror "
            "into WAV/MP4 files plus a team1 manifest.csv."
        )
    )
    parser.add_argument("output_dir", help="Directory to write audio, video, and manifest.csv into.")
    parser.add_argument("--split", default="train", help="Dataset split to load. The mirror exposes train.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to export.")
    parser.add_argument("--offset", type=int, default=0, help="Rows to skip before exporting.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face datasets cache directory.")
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        export_dataset(
            output_dir=Path(args.output_dir),
            split=args.split,
            limit=args.limit,
            offset=args.offset,
            cache_dir=args.cache_dir,
            audio_sample_rate=args.audio_sample_rate,
            video_fps=args.video_fps,
            ffmpeg=args.ffmpeg,
            force=args.force,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None


def export_dataset(
    output_dir: Path,
    split: str,
    limit: int | None,
    offset: int,
    cache_dir: str | None,
    audio_sample_rate: int,
    video_fps: int,
    ffmpeg: str,
    force: bool,
) -> None:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1.")
    if offset < 0:
        raise ValueError("--offset must be at least 0.")
    if audio_sample_rate < 1:
        raise ValueError("--audio-sample-rate must be positive.")
    if video_fps < 1:
        raise ValueError("--video-fps must be positive.")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --force to overwrite.")
        shutil.rmtree(output_dir)

    audio_dir = output_dir / "audio"
    video_dir = output_dir / "video"
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_NAME} split={split}...")
    dataset = _load_dataset(split=split, cache_dir=cache_dir)
    rows = []
    selected = islice(dataset, offset, None)
    for sample in selected:
        if limit is not None and len(rows) >= limit:
            break
        rows.append(
            _export_sample(
                sample=sample,
                output_dir=output_dir,
                audio_dir=audio_dir,
                video_dir=video_dir,
                audio_sample_rate=audio_sample_rate,
                video_fps=video_fps,
                ffmpeg=ffmpeg,
            )
        )
        if len(rows) == 1:
            print(f"Exporting first sample: {rows[-1]['utterance_id']}")
        if len(rows) % 25 == 0:
            print(f"Exported {len(rows)} samples...")

    if not rows:
        raise RuntimeError("No rows were exported.")

    manifest_path = output_dir / "manifest.csv"
    _write_manifest(manifest_path, rows)
    print(f"Exported {len(rows)} utterances to {output_dir}")
    print(f"Manifest: {manifest_path}")


def _load_dataset(split: str, cache_dir: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'datasets'. Run 'uv sync' first.") from exc
    return load_dataset(DATASET_NAME, split=split, cache_dir=cache_dir)


def _export_sample(
    sample: dict[str, object],
    output_dir: Path,
    audio_dir: Path,
    video_dir: Path,
    audio_sample_rate: int,
    video_fps: int,
    ffmpeg: str,
) -> dict[str, object]:
    idx = int(sample["idx"])
    utterance_id = f"mattymchen_lrs3_test_{idx:05d}"
    audio_path = audio_dir / f"{utterance_id}.wav"
    video_path = video_dir / f"{utterance_id}.mp4"

    audio = np.asarray(sample["audio"], dtype=np.int16)
    video = _normalize_video_array(np.asarray(sample["video"]))
    reference = " ".join(str(sample["label"]).split())
    if not reference:
        raise ValueError(f"Sample {idx} has an empty label.")

    sf.write(audio_path, audio, audio_sample_rate, subtype="PCM_16")
    _write_video(video=video, output_path=video_path, fps=video_fps, ffmpeg=ffmpeg)

    return {
        "utterance_id": utterance_id,
        "audio_path": str(Path("audio") / audio_path.name),
        "video_path": str(Path("video") / video_path.name),
        "reference": reference,
        "language": "en",
        "source_index": idx,
        "dataset": DATASET_NAME,
        "audio_sample_rate": audio_sample_rate,
        "video_fps": video_fps,
    }


def _normalize_video_array(video: np.ndarray) -> np.ndarray:
    if video.ndim == 3:
        video = video[:, :, :, None]
    if video.ndim != 4:
        raise ValueError(f"Unsupported video array shape: {video.shape}")
    if video.shape[-1] not in {1, 3}:
        raise ValueError(f"Unsupported video channel count: {video.shape[-1]}")
    if video.dtype != np.uint8:
        video = np.clip(video, 0, 255).astype(np.uint8)
    return video


def _write_video(video: np.ndarray, output_path: Path, fps: int, ffmpeg: str) -> None:
    frames, height, width, channels = video.shape
    if frames < 1:
        raise ValueError("Video array contains no frames.")

    pix_fmt = "gray" if channels == 1 else "rgb24"
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    completed = subprocess.run(command, input=video.tobytes(), capture_output=True)
    if completed.returncode != 0:
        details = completed.stderr.decode("utf-8", "replace").strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg failed for {output_path.name}: {details}")


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "utterance_id",
        "audio_path",
        "video_path",
        "reference",
        "language",
        "source_index",
        "dataset",
        "audio_sample_rate",
        "video_fps",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
