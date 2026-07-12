"""Build an AVEC-style CSV manifest (utterance_id, audio_path, video_path, reference, language)
for the LRS-VoxMM dev split, matching the format of manifest_test.csv.
"""
from __future__ import annotations

import argparse
import csv
import random
import wave
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dev_dir", help="Path to lrs-voxmm/original/dev")
    parser.add_argument("output_csv", help="Output manifest CSV path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample this many utterances after filtering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used with --sample.",
    )
    parser.add_argument(
        "--balance-by-video",
        action="store_true",
        help="With --sample, spread selected utterances across source video folders.",
    )
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=0.0,
        help="Skip utterances shorter than this many seconds, based on the wav file.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.limit is not None and args.sample is not None:
        raise ValueError("Use either --limit or --sample, not both.")
    if args.balance_by_video and args.sample is None:
        raise ValueError("--balance-by-video requires --sample.")

    dev_dir = Path(args.dev_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows(dev_dir, min_duration_sec=args.min_duration_sec)
    rows = select_rows(
        rows,
        limit=args.limit,
        sample=args.sample,
        seed=args.seed,
        balance_by_video=args.balance_by_video,
    )

    output_rows = [_output_row(row) for row in rows]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utterance_id", "audio_path", "video_path", "reference", "language"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(rows)} utterances to {output_csv}")


def build_rows(dev_dir: Path, *, min_duration_sec: float = 0.0) -> list[dict[str, str]]:
    rows = []
    video_dirs = sorted(d for d in dev_dir.iterdir() if d.is_dir())
    for video_dir in video_dirs:
        youtube_id = video_dir.name
        for txt_path in sorted(video_dir.glob("*.txt")):
            utt_id_suffix = txt_path.stem
            wav_path = video_dir / f"{utt_id_suffix}.wav"
            mp4_path = video_dir / f"{utt_id_suffix}.mp4"
            if not (wav_path.exists() and mp4_path.exists()):
                continue
            duration_sec = _wav_duration_sec(wav_path)
            if duration_sec < min_duration_sec:
                continue
            lines = txt_path.read_text().splitlines()
            text = lines[0].strip() if lines else ""
            if text.startswith("Text:"):
                text = text[len("Text:"):].strip()
            if not text:
                continue
            rows.append({
                "_video_id": youtube_id,
                "utterance_id": f"{youtube_id}_{utt_id_suffix}",
                "audio_path": str(wav_path.resolve()),
                "video_path": str(mp4_path.resolve()),
                "reference": text.upper(),
                "language": "en",
            })
    return rows


def select_rows(
    rows: list[dict[str, str]],
    *,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 0,
    balance_by_video: bool = False,
) -> list[dict[str, str]]:
    if limit is not None:
        return rows[:limit]
    if sample is None:
        return rows
    sample = min(sample, len(rows))
    rng = random.Random(seed)
    if not balance_by_video:
        return sorted(rng.sample(rows, sample), key=lambda row: row["utterance_id"])
    return _balanced_sample(rows, sample=sample, rng=rng)


def _balanced_sample(
    rows: list[dict[str, str]],
    *,
    sample: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["_video_id"], []).append(row)
    video_ids = sorted(grouped)
    rng.shuffle(video_ids)
    for group in grouped.values():
        rng.shuffle(group)

    selected = []
    while len(selected) < sample and video_ids:
        next_video_ids = []
        for video_id in video_ids:
            group = grouped[video_id]
            if group and len(selected) < sample:
                selected.append(group.pop())
            if group:
                next_video_ids.append(video_id)
        video_ids = next_video_ids
    return sorted(selected, key=lambda row: row["utterance_id"])


def _output_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "utterance_id": row["utterance_id"],
        "audio_path": row["audio_path"],
        "video_path": row["video_path"],
        "reference": row["reference"],
        "language": row["language"],
    }


def _wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        if sample_rate <= 0:
            return 0.0
        return wav.getnframes() / sample_rate


if __name__ == "__main__":
    main()
