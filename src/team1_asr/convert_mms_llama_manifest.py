from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MMS-LLaMA TSV/WRD manifests into the team1 ASR CSV manifest format."
    )
    parser.add_argument("tsv", help="MMS-LLaMA split TSV, e.g. external/MMS-LLaMA/manifest/433h/test.tsv")
    parser.add_argument("wrd", help="MMS-LLaMA split WRD, e.g. external/MMS-LLaMA/manifest/433h/test.wrd")
    parser.add_argument("output", help="Output CSV manifest path.")
    parser.add_argument(
        "--lrs3-root",
        required=True,
        help="Local root replacing {LRS3_ROOT}; this should contain lrs3_video_seg24s/.",
    )
    parser.add_argument("--language", default="en", help="Language tag to write into the CSV manifest.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        convert_manifest(
            tsv_path=Path(args.tsv),
            wrd_path=Path(args.wrd),
            output_path=Path(args.output),
            lrs3_root=Path(args.lrs3_root),
            language=args.language,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None


def convert_manifest(
    tsv_path: Path,
    wrd_path: Path,
    output_path: Path,
    lrs3_root: Path,
    language: str,
) -> None:
    rows = _read_tsv(tsv_path, lrs3_root)
    references = wrd_path.read_text(encoding="utf-8").splitlines()
    if len(rows) != len(references):
        raise ValueError(
            f"Mismatched manifest/reference lengths: {len(rows)} TSV rows vs {len(references)} WRD rows"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "utterance_id",
                "audio_path",
                "video_path",
                "reference",
                "language",
                "dataset",
                "source_root",
                "num_video_frames",
                "num_audio_frames",
                "speech_rate",
            ],
        )
        writer.writeheader()
        for row, reference in zip(rows, references):
            writer.writerow(
                {
                    "utterance_id": row["utterance_id"],
                    "audio_path": row["audio_path"],
                    "video_path": row["video_path"],
                    "reference": reference.strip(),
                    "language": language,
                    "dataset": row["dataset"],
                    "source_root": row["source_root"],
                    "num_video_frames": row["num_video_frames"],
                    "num_audio_frames": row["num_audio_frames"],
                    "speech_rate": row["speech_rate"],
                }
            )
    print(f"Wrote {len(rows)} rows to {output_path}")


def _read_tsv(path: Path, lrs3_root: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty TSV manifest: {path}")

    rows = []
    for index, line in enumerate(lines[1:], start=1):
        parts = line.split("\t")
        if len(parts) != 6:
            raise ValueError(f"Invalid TSV row {index}: expected 6 tab-separated fields, got {len(parts)}")
        dataset, video_path, audio_path, video_frames, audio_frames, speech_rate = parts
        video_path = _replace_lrs3_root(video_path, lrs3_root)
        audio_path = _replace_lrs3_root(audio_path, lrs3_root)
        rows.append(
            {
                "utterance_id": _utterance_id(video_path),
                "audio_path": audio_path,
                "video_path": video_path,
                "dataset": dataset,
                "source_root": lines[0],
                "num_video_frames": video_frames,
                "num_audio_frames": audio_frames,
                "speech_rate": speech_rate,
            }
        )
    return rows


def _replace_lrs3_root(value: str, lrs3_root: Path) -> Path:
    replaced = value.replace("{LRS3_ROOT}", str(lrs3_root.resolve()))
    return Path(replaced)


def _utterance_id(video_path: Path) -> str:
    parts = video_path.parts
    if len(parts) >= 3:
        return "_".join(parts[-3:]).replace(".mp4", "")
    return video_path.stem


if __name__ == "__main__":
    main()
