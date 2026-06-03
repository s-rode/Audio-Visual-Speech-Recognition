"""Prepare the LRS3 test set from mattymchen/lrs3-test on Hugging Face.

This dataset contains the official LRS3 test split (1,321 utterances) with
audio stored as raw int16 PCM bytes at 16 kHz mono and transcripts in a
'label' field — no ffmpeg required.

Usage:
    uv run python prepare_lrs3_test_hf.py data/lrs3/hf_test_full
    uv run python prepare_lrs3_test_hf.py data/lrs3/hf_test_full --limit 300
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import struct
import sys
from itertools import islice
from pathlib import Path

_DATASET_NAME = "mattymchen/lrs3-test"
_SAMPLE_RATE = 16_000
_NUM_CHANNELS = 1
_SAMPLE_WIDTH = 2  # int16


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Stream {_DATASET_NAME} from Hugging Face and write a "
            "WAV + manifest.csv subset for ASR experiments."
        )
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write extracted WAV files and manifest.csv into.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of utterances to export (default: all 1321).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of samples to skip before exporting (default: 0).",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language tag to write into the manifest (default: en).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip samples with missing audio/label instead of aborting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory if it already exists.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        export_lrs3_test_subset(
            output_dir=Path(args.output_dir),
            limit=args.limit,
            offset=args.offset,
            language=args.language,
            cache_dir=args.cache_dir,
            skip_invalid=args.skip_invalid,
            force=args.force,
        )
    except (RuntimeError, ValueError, FileExistsError) as exc:
        raise SystemExit(f"error: {exc}") from None

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def export_lrs3_test_subset(
    output_dir: Path,
    limit: int | None,
    offset: int,
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

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {_DATASET_NAME} (LRS3 test set)...")
    dataset = _load_dataset(cache_dir=cache_dir)

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
                language=language,
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            if not skip_invalid:
                raise
            skipped += 1
            print(f"Skipping invalid sample: {exc}", file=sys.stderr)
            continue

        rows.append(row)
        if len(rows) == 1:
            print(f"Exporting first sample: {row['utterance_id']}")
        if len(rows) % 100 == 0:
            print(f"Exported {len(rows)} samples...")

    _write_manifest(manifest_path, rows)
    print(f"\nExported {len(rows)} utterances to {output_dir}.")
    if skipped:
        print(f"Skipped {skipped} invalid samples.")
    print(f"Manifest: {manifest_path}")


def _load_dataset(cache_dir: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Run 'uv sync' to install it."
        ) from exc

    return load_dataset(
        _DATASET_NAME,
        split="train",  # this dataset exposes the test split as 'train'
        streaming=True,
        cache_dir=cache_dir,
    )


def _export_sample(
    sample: dict[str, object],
    audio_dir: Path,
    language: str,
) -> dict[str, object]:
    idx = sample.get("idx")
    if idx is None:
        raise ValueError("Sample missing 'idx' field.")

    label = sample.get("label")
    if not label:
        raise ValueError(f"Sample {idx} has empty label.")

    raw_audio = sample.get("audio")
    if not raw_audio:
        raise ValueError(f"Sample {idx} has missing audio.")

    utterance_id = f"lrs3_test_{idx:05d}"
    audio_path = audio_dir / f"{utterance_id}.wav"

    pcm_bytes = _sequence_to_pcm(raw_audio)
    _write_wav(audio_path, pcm_bytes)

    return {
        "utterance_id": utterance_id,
        "audio_path": str(Path("audio") / audio_path.name),
        "reference": str(label).strip(),
        "language": language,
        "source_key": str(idx),
        "source_url": "",
        "dataset": _DATASET_NAME,
    }


def _sequence_to_pcm(audio_sequence) -> bytes:
    """Convert a sequence of int16 samples to raw PCM bytes."""
    samples = list(audio_sequence)
    return struct.pack(f"<{len(samples)}h", *samples)


def _write_wav(path: Path, pcm_bytes: bytes) -> None:
    """Write a minimal 16 kHz mono int16 WAV file."""
    num_samples = len(pcm_bytes) // _SAMPLE_WIDTH
    byte_rate = _SAMPLE_RATE * _NUM_CHANNELS * _SAMPLE_WIDTH
    block_align = _NUM_CHANNELS * _SAMPLE_WIDTH
    data_size = len(pcm_bytes)
    riff_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16,
        1,                  # PCM
        _NUM_CHANNELS,
        _SAMPLE_RATE,
        byte_rate,
        block_align,
        _SAMPLE_WIDTH * 8,  # bits per sample
        b"data", data_size,
    )
    path.write_bytes(header + pcm_bytes)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "utterance_id",
        "audio_path",
        "reference",
        "language",
        "source_key",
        "source_url",
        "dataset",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
