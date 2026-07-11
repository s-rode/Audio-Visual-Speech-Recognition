"""Build an AVEC-style CSV manifest (utterance_id, audio_path, video_path, reference, language)
for the LRS-VoxMM dev split, matching the format of manifest_test.csv.
"""
from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dev_dir", help="Path to lrs-voxmm/original/dev")
    parser.add_argument("output_csv", help="Output manifest CSV path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=0.0,
        help="Skip utterances shorter than this many seconds, based on the wav file.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    dev_dir = Path(args.dev_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

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
            if duration_sec < args.min_duration_sec:
                continue
            lines = txt_path.read_text().splitlines()
            text = lines[0].strip() if lines else ""
            if text.startswith("Text:"):
                text = text[len("Text:"):].strip()
            if not text:
                continue
            rows.append({
                "utterance_id": f"{youtube_id}_{utt_id_suffix}",
                "audio_path": str(wav_path.resolve()),
                "video_path": str(mp4_path.resolve()),
                "reference": text.upper(),
                "language": "en",
            })

    if args.limit:
        rows = rows[:args.limit]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["utterance_id", "audio_path", "video_path", "reference", "language"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} utterances to {output_csv}")


def _wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        if sample_rate <= 0:
            return 0.0
        return wav.getnframes() / sample_rate


if __name__ == "__main__":
    main()
