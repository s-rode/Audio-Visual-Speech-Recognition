from __future__ import annotations

import argparse
import csv
from itertools import islice
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

_DATASET_NAME = "Ainncy/LRS3"
_DEFAULT_SHUFFLE_BUFFER = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream Ainncy/LRS3 from Hugging Face, extract audio from MP4 clips, "
            "and write a WAV+manifest subset for ASR experiments."
        )
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write extracted WAV files and manifest.csv into.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Hugging Face split to stream. Ainncy/LRS3 currently exposes train.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of utterances to export.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of streamed samples to skip before exporting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional seed for streaming shuffle before offset/limit. "
            f"Uses a buffer of {_DEFAULT_SHUFFLE_BUFFER} samples."
        ),
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language tag to write into the manifest.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip samples with missing text/video or failed ffmpeg extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifest/audio files in the output directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        export_lrs3_hf_subset(
            output_dir=Path(args.output_dir),
            split=args.split,
            limit=args.limit,
            offset=args.offset,
            seed=args.seed,
            language=args.language,
            cache_dir=args.cache_dir,
            skip_invalid=args.skip_invalid,
            force=args.force,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

    # HF streaming can leave shutdown hooks waiting after a limited export.
    # The CLI has already written all files, so flush and exit immediately.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def export_lrs3_hf_subset(
    output_dir: Path,
    split: str,
    limit: int | None,
    offset: int,
    seed: int | None,
    language: str,
    cache_dir: str | None,
    skip_invalid: bool,
    force: bool,
) -> None:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1.")
    if offset < 0:
        raise ValueError("--offset must be at least 0.")

    manifest_path = output_dir / "manifest.csv"
    audio_dir = output_dir / "audio"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --force to overwrite."
            )
        shutil.rmtree(output_dir)

    print(f"Streaming {_DATASET_NAME} split={split}...")
    dataset = _load_dataset(split=split, cache_dir=cache_dir)
    if seed is not None:
        print(
            f"Applying streaming shuffle with seed={seed} "
            f"and buffer={_DEFAULT_SHUFFLE_BUFFER}..."
        )
        dataset = dataset.shuffle(seed=seed, buffer_size=_DEFAULT_SHUFFLE_BUFFER)

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    skipped = 0
    sample_iterable = islice(dataset, offset, None)
    for sample in sample_iterable:
        if limit is not None and len(rows) >= limit:
            break

        try:
            row = _export_sample(
                sample=sample,
                audio_dir=audio_dir,
                output_dir=output_dir,
                language=language,
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            if not skip_invalid:
                raise
            skipped += 1
            print(f"Skipping invalid sample: {exc}", file=sys.stderr)
            continue

        rows.append(row)
        if len(rows) == 1:
            print(f"Exporting first sample: {row['utterance_id']}")
        if len(rows) % 25 == 0:
            print(f"Exported {len(rows)} samples...")

    _write_manifest(manifest_path, rows)
    print(f"Exported {len(rows)} utterances from {_DATASET_NAME} to {output_dir}.")
    if skipped:
        print(f"Skipped {skipped} invalid samples.")
    print(f"Manifest: {manifest_path}")


def _load_dataset(split: str, cache_dir: str | None):
    try:
        from datasets import Video, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Run 'uv sync' to install it."
        ) from exc

    dataset = load_dataset(
        _DATASET_NAME,
        split=split,
        streaming=True,
        cache_dir=cache_dir,
    )
    return dataset.cast_column("mp4", Video(decode=False))


def _export_sample(
    sample: dict[str, object],
    audio_dir: Path,
    output_dir: Path,
    language: str,
) -> dict[str, object]:
    source_key = str(sample["__key__"])
    utterance_id = _sanitize_key(source_key)
    audio_path = audio_dir / f"{utterance_id}.wav"
    reference = _parse_transcript(str(sample["txt"]))
    video = sample["mp4"]
    if not isinstance(video, dict):
        raise ValueError(f"Sample {source_key} has unsupported mp4 payload.")

    _extract_audio_to_wav(video, audio_path, temp_dir=output_dir)
    return {
        "utterance_id": utterance_id,
        "audio_path": str(Path("audio") / audio_path.name),
        "reference": reference,
        "language": language,
        "source_key": source_key,
        "source_url": sample.get("__url__", ""),
        "dataset": _DATASET_NAME,
    }


def _parse_transcript(raw_text: str) -> str:
    match = re.search(r"Text:\s*(.*?)\s*Conf:", raw_text, flags=re.DOTALL)
    if not match:
        match = re.search(r"Text:\s*(.*)", raw_text, flags=re.DOTALL)
    if not match:
        raise ValueError("LRS3 transcript does not contain a Text: field.")

    text = " ".join(match.group(1).split())
    if not text:
        raise ValueError("LRS3 transcript Text: field is empty.")
    return text


def _extract_audio_to_wav(video: dict[str, object], output_path: Path, temp_dir: Path) -> None:
    video_path, should_delete = _video_path(video, temp_dir)
    try:
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip() or "unknown ffmpeg error"
            raise RuntimeError(f"ffmpeg failed for {video_path.name}: {details}")
    finally:
        if should_delete and video_path.exists():
            video_path.unlink()


def _video_path(video: dict[str, object], temp_dir: Path) -> tuple[Path, bool]:
    path = video.get("path")
    if path:
        return Path(str(path)), False

    video_bytes = video.get("bytes")
    if not isinstance(video_bytes, bytes) or not video_bytes:
        raise ValueError("LRS3 video sample does not contain mp4 bytes or a path.")

    temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".mp4",
        prefix="lrs3_",
        dir=temp_dir,
        delete=False,
    ) as handle:
        handle.write(video_bytes)
        return Path(handle.name), True


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "utterance_id",
                "audio_path",
                "reference",
                "language",
                "source_key",
                "source_url",
                "dataset",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _sanitize_key(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


if __name__ == "__main__":
    main()
