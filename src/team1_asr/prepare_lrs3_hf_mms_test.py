from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import sys

from team1_asr.prepare_lrs3_hf import (
    _DATASET_NAME,
    _extract_audio_to_wav_from_path,
    _load_dataset,
    _materialize_video,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream Ainncy/LRS3 and export only the utterances listed in an "
            "MMS-LLaMA TSV/WRD manifest."
        )
    )
    parser.add_argument("tsv", help="MMS-LLaMA TSV manifest, e.g. external/MMS-LLaMA/manifest/433h/test.tsv")
    parser.add_argument("wrd", help="MMS-LLaMA WRD references, e.g. external/MMS-LLaMA/manifest/433h/test.wrd")
    parser.add_argument("output_dir", help="Directory to write audio/video files and manifest.csv into.")
    parser.add_argument("--split", default="train", help="Hugging Face split to stream. Ainncy/LRS3 exposes train.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of target rows to export.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face datasets cache directory.")
    parser.add_argument("--language", default="en", help="Language tag to write into the manifest.")
    parser.add_argument("--force", action="store_true", help="Overwrite output_dir if it already exists.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        export_mms_manifest_from_hf(
            tsv_path=Path(args.tsv),
            wrd_path=Path(args.wrd),
            output_dir=Path(args.output_dir),
            split=args.split,
            limit=args.limit,
            cache_dir=args.cache_dir,
            language=args.language,
            force=args.force,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def export_mms_manifest_from_hf(
    tsv_path: Path,
    wrd_path: Path,
    output_dir: Path,
    split: str,
    limit: int | None,
    cache_dir: str | None,
    language: str,
    force: bool,
) -> None:
    targets = _read_targets(tsv_path, wrd_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1.")
        targets = dict(list(targets.items())[:limit])

    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --force to overwrite.")
        shutil.rmtree(output_dir)

    audio_dir = output_dir / "audio"
    video_dir = output_dir / "video"
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {_DATASET_NAME} split={split} for {len(targets)} target utterance(s)...")
    dataset = _load_dataset(split=split, cache_dir=cache_dir)
    rows: list[dict[str, object]] = []
    remaining = set(targets)
    scanned = 0

    for sample in dataset:
        scanned += 1
        source_key = str(sample.get("__key__", ""))
        if source_key not in remaining:
            if scanned % 10000 == 0:
                print(f"Scanned {scanned} sample(s); found {len(rows)}/{len(targets)}", flush=True)
            continue

        target = targets[source_key]
        utterance_id = _safe_id(source_key)
        audio_path = audio_dir / f"{utterance_id}.wav"
        video = sample.get("mp4")
        if not isinstance(video, dict):
            raise ValueError(f"Sample {source_key} has unsupported mp4 payload.")

        video_path, cleanup_video = _materialize_video(video, utterance_id, output_dir, video_dir)
        try:
            _extract_audio_to_wav_from_path(video_path, audio_path)
        finally:
            if cleanup_video and video_path.exists():
                video_path.unlink()

        rows.append(
            {
                "utterance_id": utterance_id,
                "audio_path": str(Path("audio") / audio_path.name),
                "video_path": str(Path("video") / video_path.name),
                "reference": target["reference"],
                "language": language,
                "source_key": source_key,
                "source_url": sample.get("__url__", ""),
                "dataset": _DATASET_NAME,
                "num_video_frames": target["num_video_frames"],
                "num_audio_frames": target["num_audio_frames"],
                "speech_rate": target["speech_rate"],
            }
        )
        remaining.remove(source_key)
        print(f"Exported {len(rows)}/{len(targets)}: {source_key}", flush=True)
        if not remaining:
            break

    if remaining:
        examples = ", ".join(sorted(list(remaining))[:5])
        raise RuntimeError(f"Could not find {len(remaining)} target key(s) in HF stream. Examples: {examples}")

    _write_manifest(output_dir / "manifest.csv", rows)
    print(f"Wrote {len(rows)} utterance(s) after scanning {scanned} sample(s).")
    print(f"Manifest: {output_dir / 'manifest.csv'}")


def _read_targets(tsv_path: Path, wrd_path: Path) -> dict[str, dict[str, str]]:
    tsv_lines = tsv_path.read_text(encoding="utf-8").splitlines()[1:]
    refs = wrd_path.read_text(encoding="utf-8").splitlines()
    if len(tsv_lines) != len(refs):
        raise ValueError(f"Mismatched TSV/WRD rows: {len(tsv_lines)} vs {len(refs)}")

    targets: dict[str, dict[str, str]] = {}
    for line, reference in zip(tsv_lines, refs):
        parts = line.split("\t")
        if len(parts) != 6:
            raise ValueError(f"Invalid MMS-LLaMA TSV row: {line}")
        _, video_path, _, video_frames, audio_frames, speech_rate = parts
        key = _key_from_mms_video_path(video_path)
        targets[key] = {
            "reference": reference.strip(),
            "num_video_frames": video_frames,
            "num_audio_frames": audio_frames,
            "speech_rate": speech_rate,
        }
    return targets


def _key_from_mms_video_path(value: str) -> str:
    marker = "lrs3_video_seg24s/"
    if marker not in value:
        raise ValueError(f"Cannot derive LRS3 key from path: {value}")
    return value.split(marker, 1)[1].removesuffix(".mp4")


def _safe_id(value: str) -> str:
    return value.replace("/", "_").replace(".", "_")


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "utterance_id",
                "audio_path",
                "video_path",
                "reference",
                "language",
                "source_key",
                "source_url",
                "dataset",
                "num_video_frames",
                "num_audio_frames",
                "speech_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
