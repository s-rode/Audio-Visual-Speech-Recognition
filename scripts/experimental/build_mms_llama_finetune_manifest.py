#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import subprocess
import wave
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a team1 preprocessed AVSR manifest into MMS-LLaMA "
            "train/test TSV+WRD files for experimental fine-tuning."
        )
    )
    parser.add_argument("manifest", help="Input manifest.csv with audio_path, video_path, reference.")
    parser.add_argument("output_dir", help="Directory to write train/test.tsv and train/test.wrd.")
    parser.add_argument("--train-count", type=int, default=None, help="Number of rows to put in train split.")
    parser.add_argument("--valid-count", type=int, default=200, help="Number of rows to put in test/valid split.")
    parser.add_argument("--seed", type=int, default=1, help="Shuffle seed before splitting. Use --no-shuffle to preserve order.")
    parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle before splitting.")
    parser.add_argument("--id-column", default="utterance_id")
    parser.add_argument("--audio-column", default="audio_path")
    parser.add_argument("--video-column", default="video_path")
    parser.add_argument("--reference-column", default="reference")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        build_manifest(
            manifest_path=Path(args.manifest),
            output_dir=Path(args.output_dir),
            train_count=args.train_count,
            valid_count=args.valid_count,
            seed=args.seed,
            shuffle=not args.no_shuffle,
            id_column=args.id_column,
            audio_column=args.audio_column,
            video_column=args.video_column,
            reference_column=args.reference_column,
            ffprobe=args.ffprobe,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None


def build_manifest(
    manifest_path: Path,
    output_dir: Path,
    train_count: int | None,
    valid_count: int,
    seed: int,
    shuffle: bool,
    id_column: str,
    audio_column: str,
    video_column: str,
    reference_column: str,
    ffprobe: str,
) -> None:
    if valid_count < 1:
        raise ValueError("--valid-count must be at least 1.")
    if train_count is not None and train_count < 1:
        raise ValueError("--train-count must be at least 1.")

    manifest_path = manifest_path.resolve()
    manifest_dir = manifest_path.parent
    rows = _read_rows(manifest_path)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    if train_count is None:
        train_count = len(rows) - valid_count
    if train_count < 1 or train_count + valid_count > len(rows):
        raise ValueError(
            f"Cannot split {len(rows)} row(s) into train_count={train_count}, valid_count={valid_count}."
        )

    train_rows = rows[:train_count]
    valid_rows = rows[train_count:train_count + valid_count]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_split(
        output_dir=output_dir,
        split="train",
        rows=train_rows,
        manifest_dir=manifest_dir,
        id_column=id_column,
        audio_column=audio_column,
        video_column=video_column,
        reference_column=reference_column,
        ffprobe=ffprobe,
    )
    _write_split(
        output_dir=output_dir,
        split="test",
        rows=valid_rows,
        manifest_dir=manifest_dir,
        id_column=id_column,
        audio_column=audio_column,
        video_column=video_column,
        reference_column=reference_column,
        ffprobe=ffprobe,
    )
    print(f"Wrote MMS-LLaMA fine-tune manifests to {output_dir}")
    print(f"train: {len(train_rows)} utterance(s)")
    print(f"test: {len(valid_rows)} utterance(s)")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest contains no rows: {path}")
    return rows


def _write_split(
    output_dir: Path,
    split: str,
    rows: list[dict[str, str]],
    manifest_dir: Path,
    id_column: str,
    audio_column: str,
    video_column: str,
    reference_column: str,
    ffprobe: str,
) -> None:
    tsv_path = output_dir / f"{split}.tsv"
    wrd_path = output_dir / f"{split}.wrd"
    with tsv_path.open("w", encoding="utf-8") as tsv, wrd_path.open("w", encoding="utf-8") as wrd:
        tsv.write("/\n")
        for index, row in enumerate(rows):
            utterance_id = _safe_id(row.get(id_column) or f"{split}_{index:06d}")
            audio_path = _resolve(row.get(audio_column, ""), manifest_dir)
            video_path = _resolve(row.get(video_column, ""), manifest_dir)
            reference = " ".join(row.get(reference_column, "").split())
            if not reference:
                raise ValueError(f"Row {utterance_id} has empty reference text.")
            sample_count = _audio_sample_count(audio_path)
            frame_count = _video_frame_count(video_path, ffprobe=ffprobe)
            speech_rate = _speech_rate(reference, sample_count)
            tsv.write(
                "\t".join(
                    [
                        utterance_id,
                        str(video_path),
                        str(audio_path),
                        str(frame_count),
                        str(sample_count),
                        f"{speech_rate:.2f}",
                    ]
                )
                + "\n"
            )
            wrd.write(reference + "\n")


def _resolve(value: str, manifest_dir: Path) -> Path:
    if not value:
        raise ValueError("Manifest row is missing a required path value.")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _audio_sample_count(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != 16000:
            raise ValueError(f"Expected 16 kHz audio, got {handle.getframerate()} Hz: {path}")
        if handle.getnchannels() != 1:
            raise ValueError(f"Expected mono audio, got {handle.getnchannels()} channel(s): {path}")
        return handle.getnframes()


def _video_frame_count(path: Path, ffprobe: str) -> int:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or "unknown ffprobe error"
        raise RuntimeError(f"ffprobe failed for {path}: {details}")
    value = completed.stdout.strip().splitlines()[0]
    if value == "N/A":
        raise RuntimeError(f"ffprobe could not count frames for {path}")
    return int(value)


def _speech_rate(reference: str, sample_count: int) -> float:
    duration = sample_count / 16000
    if duration <= 0:
        return 0.0
    return len(reference.split()) / duration


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


if __name__ == "__main__":
    main()
