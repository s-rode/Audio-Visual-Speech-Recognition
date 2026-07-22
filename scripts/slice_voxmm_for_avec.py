"""
Slice VoxMM raw YouTube videos into per-utterance clips ready for avec_preprocess_runner.py.

Unlike preprocess_voxmm.py, this script:
  - Includes ALL segments (on-screen True and False) for router training
  - Only filters out segments with invalid/zero-duration timestamps
  - Does NOT run any face detection — that is left to avec_preprocess_runner.py
  - Outputs full-frame color MP4 clips (not mouth ROIs) + WAV slices + manifest CSV

Output structure:
  <output_dir>/
    video/   <utterance_id>.mp4    (full-frame color clip, e.g. 360p)
    wav/     <utterance_id>.wav    (16kHz mono WAV slice)
    manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import wave
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Slice VoxMM YouTube videos into per-utterance clips for AVEC preprocessing."
    )
    parser.add_argument("--video-dir", required=True, help="Directory of downloaded YouTube MP4s")
    parser.add_argument("--wav-dir", required=True, help="Directory of full-length VoxMM WAVs")
    parser.add_argument("--metadata-dir", required=True, help="Directory of VoxMM metadata JSONs")
    parser.add_argument("--split-file", required=True, help="train.txt or test.txt with video IDs")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos (for smoke tests)")
    parser.add_argument("--min-duration", type=float, default=0.1, help="Minimum segment duration in seconds")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    video_dir = Path(args.video_dir)
    wav_dir = Path(args.wav_dir)
    meta_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)
    video_out = output_dir / "video"
    wav_out = output_dir / "wav"

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.force:
            import shutil
            shutil.rmtree(output_dir)
        elif not args.skip_existing:
            raise FileExistsError(
                f"Output dir not empty: {output_dir}. Use --force or --skip-existing."
            )

    video_out.mkdir(parents=True, exist_ok=True)
    wav_out.mkdir(parents=True, exist_ok=True)

    split_ids = [
        l.strip()
        for l in Path(args.split_file).read_text().splitlines()
        if l.strip()
    ]
    if args.limit:
        split_ids = split_ids[: args.limit]

    manifest_rows: list[dict] = []
    total_written = 0
    total_failed = 0
    total_skipped_no_video = 0
    total_skipped_invalid = 0

    for vid_idx, vid_id in enumerate(split_ids):
        meta_path = meta_dir / f"{vid_id}.json"
        video_path = video_dir / f"{vid_id}.mp4"
        wav_path = wav_dir / f"{vid_id}.wav"

        if not meta_path.exists():
            print(f"[{vid_idx+1}/{len(split_ids)}] SKIP {vid_id}: no metadata", flush=True)
            continue

        if not video_path.exists():
            print(f"[{vid_idx+1}/{len(split_ids)}] SKIP {vid_id}: no video", flush=True)
            total_skipped_no_video += 1
            continue

        if not wav_path.exists():
            print(f"[{vid_idx+1}/{len(split_ids)}] SKIP {vid_id}: no wav", flush=True)
            continue

        meta = json.loads(meta_path.read_text())
        segments = meta.get("segments", [])

        # Include ALL segments — on-screen True and False
        # Only filter out segments with invalid timestamps
        valid_segments = []
        for seg in segments:
            start = seg.get("start")
            end = seg.get("end")
            if start is None or end is None:
                continue
            if end - start < args.min_duration:
                total_skipped_invalid += 1
                continue
            valid_segments.append(seg)

        print(
            f"[{vid_idx+1}/{len(split_ids)}] {vid_id}: {len(valid_segments)} segments "
            f"({len(segments) - len(valid_segments)} skipped invalid)",
            flush=True,
        )

        # Read WAV metadata for slicing
        try:
            with wave.open(str(wav_path), "rb") as wf:
                sr = wf.getframerate()
                total_frames = wf.getnframes()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                wav_data = wf.readframes(total_frames)
        except Exception as e:
            print(f"  ERROR reading WAV {wav_path}: {e}", file=sys.stderr)
            continue

        for seg_idx, seg in enumerate(valid_segments):
            start: float = seg["start"]
            end: float = seg["end"]
            on_screen: bool = bool(seg.get("on-screen", False))
            text: str = seg.get("text", "").strip()
            utt_id = f"{vid_id}_{seg_idx:05d}"

            video_out_path = video_out / f"{utt_id}.mp4"
            wav_out_path = wav_out / f"{utt_id}.wav"

            if args.skip_existing and video_out_path.exists() and wav_out_path.exists():
                manifest_rows.append({
                    "utterance_id": utt_id,
                    "audio_path": str(wav_out_path),
                    "video_path": str(video_out_path),
                    "reference": text,
                    "language": "en",
                    "on_screen": str(on_screen).lower(),
                    "source_video": vid_id,
                    "segment_start": start,
                    "segment_end": end,
                })
                total_written += 1
                continue

            # Slice WAV
            try:
                _slice_wav(
                    wav_data=wav_data,
                    sr=sr,
                    n_channels=n_channels,
                    sampwidth=sampwidth,
                    start=start,
                    end=end,
                    dst=wav_out_path,
                )
            except Exception as e:
                print(f"  ERROR slicing WAV for {utt_id}: {e}", file=sys.stderr)
                total_failed += 1
                continue

            # Slice video using FFmpeg (fast, no re-encode)
            try:
                _slice_video_ffmpeg(
                    src=video_path,
                    dst=video_out_path,
                    start=start,
                    end=end,
                )
            except Exception as e:
                print(f"  ERROR slicing video for {utt_id}: {e}", file=sys.stderr)
                # Remove partial WAV if video failed
                wav_out_path.unlink(missing_ok=True)
                total_failed += 1
                continue

            manifest_rows.append({
                "utterance_id": utt_id,
                "audio_path": str(wav_out_path),
                "video_path": str(video_out_path),
                "reference": text,
                "language": "en",
                "on_screen": str(on_screen).lower(),
                "source_video": vid_id,
                "segment_start": start,
                "segment_end": end,
            })
            total_written += 1

        print(
            f"  Done {vid_id}: total={total_written} elapsed",
            flush=True,
        )

    # Write manifest
    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "utterance_id", "audio_path", "video_path", "reference", "language",
        "on_screen", "source_video", "segment_start", "segment_end",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nFinished.")
    print(f"  Written:              {total_written}")
    print(f"  Failed:               {total_failed}")
    print(f"  Skipped (no video):   {total_skipped_no_video}")
    print(f"  Skipped (invalid ts): {total_skipped_invalid}")
    print(f"  Manifest:             {manifest_path}")


def _slice_wav(
    wav_data: bytes,
    sr: int,
    n_channels: int,
    sampwidth: int,
    start: float,
    end: float,
    dst: Path,
) -> None:
    """Slice a WAV from pre-read raw bytes."""
    frame_start = min(int(start * sr), len(wav_data) // (sampwidth * n_channels))
    frame_end = min(int(end * sr), len(wav_data) // (sampwidth * n_channels))
    byte_start = frame_start * sampwidth * n_channels
    byte_end = frame_end * sampwidth * n_channels
    sliced = wav_data[byte_start:byte_end]
    with wave.open(str(dst), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(sliced)


def _slice_video_ffmpeg(src: Path, dst: Path, start: float, end: float) -> None:
    """Slice a video segment using FFmpeg stream copy (fast, no re-encode)."""
    duration = end - start
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", str(src),
        "-t", str(duration),
        "-c:v", "libx264",     # re-encode for exact timestamp cuts
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "aac",
        "-ar", "44100",
        "-movflags", "+faststart",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed for {src} [{start:.2f}-{end:.2f}]: "
            f"{result.stderr.decode(errors='replace')[-500:]}"
        )


if __name__ == "__main__":
    main()
