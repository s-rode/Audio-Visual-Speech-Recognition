#!/usr/bin/env python3

import csv
import json
import os
import argparse
from pathlib import Path


def extract_transcript(content):
    """Pull just the transcription line out of an LRS-style .txt file, e.g.

        Text:  SO MANY OF US LOVE THEM
        Conf:  6.837

        WORD START END FACONF
        SO 0.08 0.26 0.98
        ...

    Returns only the text after 'Text:' on the first line (case-insensitive),
    never the confidence score or the word-alignment table below it.
    """
    first_line = content.splitlines()[0] if content.strip() else ""
    stripped = first_line.strip()
    if stripped.lower().startswith("text:"):
        return stripped.split(":", 1)[1].strip()
    return stripped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate manifest.csv for AVEC preprocessing from LRS-VoxMM structure, "
                     "using lip-ROI video crops in place of the original face-crop videos."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing video folders (e.g., 'original/test/'). "
             "Determines which audio/text noise variant is used."
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        help="Path to folder containing the metadata JSON files (defaults to --data-dir if omitted)."
    )
    parser.add_argument(
        "--roi-dir",
        type=str,
        required=True,
        help="Path to the directory containing lip-ROI crops, e.g. "
             "'data/voxmm_and_lrs-voxmm/lrs-voxmm_avec_roi'. Expects ROI clips under "
             "'<roi-dir>/video/{video_id}_{stem}_mouth.mp4'."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="manifest.csv",
        help="Output path for the generated manifest file."
    )
    parser.add_argument(
        "--default-lang",
        type=str,
        default="en",
        help="Language flag value for the manifest."
    )
    parser.add_argument(
        "--video-preprocessing",
        type=str,
        default="avec_mouth_roi",
        help="Value written to the video_preprocessing column, identifying which "
             "video preprocessing pipeline produced video_path."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_dir).resolve()
    json_path = Path(args.json_dir).resolve() if args.json_dir else data_path
    roi_video_dir = (Path(args.roi_dir) / "video").resolve()

    if not data_path.exists():
        raise SystemExit(f"Error: Data directory '{data_path}' does not exist.")

    if not roi_video_dir.exists():
        raise SystemExit(f"Error: ROI video directory '{roi_video_dir}' does not exist.")

    output_csv_path = Path(args.output_csv).resolve()
    output_csv_dir = output_csv_path.parent
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    headers = ["video_path", "audio_path", "utterance_id", "reference", "language", "video_preprocessing"]
    manifest_rows = []

    # Identify video folders (each directory represents a YouTube Video ID)
    video_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(f"Found {len(video_dirs)} video folders in '{data_path}'")

    skipped_missing_roi = 0
    skipped_missing_audio = 0
    skipped_missing_text = 0

    for video_dir in video_dirs:
        video_id = video_dir.name

        # Look up the matching JSON file to parse pristine text transcripts
        segments_text_map = {}
        target_json = json_path / f"{video_id}.json"
        if not target_json.exists():
            # Fallback: check inside the video folder itself
            target_json = video_dir / f"{video_id}.json"

        if target_json.exists():
            try:
                with open(target_json, "r", encoding="utf-8") as jf:
                    meta_data = json.load(jf)
                    for seg in meta_data.get("segments", []):
                        idx = seg.get("segment_index")
                        if idx is not None:
                            segments_text_map[int(idx)] = seg.get("text", "").strip()
            except Exception as e:
                print(f"Warning: Could not parse JSON file {target_json}: {e}")

        # Use the original .mp4 files purely to enumerate which utterances (stems) exist.
        # The actual video_path written to the manifest comes from the ROI crop, not this file.
        mp4_files = sorted(video_dir.glob("*.mp4"))
        for mp4_file in mp4_files:
            stem = mp4_file.stem  # e.g., '00019'

            # Locate sibling audio and text tracks (still driven by --data-dir, so switching
            # the noise variant is just a matter of pointing --data-dir/--json-dir elsewhere)
            wav_file = video_dir / f"{stem}.wav"
            txt_file = video_dir / f"{stem}.txt"

            # Locate the corresponding lip-ROI crop
            roi_file = roi_video_dir / f"{video_id}_{stem}_mouth.mp4"

            if not roi_file.exists():
                print(f"Warning: ROI video missing for {video_id}/{stem}. Skipping.")
                skipped_missing_roi += 1
                continue

            if not wav_file.exists():
                print(f"Warning: Audio file missing for {mp4_file}. Skipping.")
                skipped_missing_audio += 1
                continue

            # Extract transcript text
            reference_text = ""
            try:
                segment_idx = int(stem)
                if segment_idx in segments_text_map:
                    reference_text = segments_text_map[segment_idx]
                elif txt_file.exists():
                    # Fallback to direct file read if JSON lacks it — only the
                    # transcription line, never the Conf/word-alignment table below it.
                    with open(txt_file, "r", encoding="utf-8") as tf:
                        content = tf.read()
                        reference_text = extract_transcript(content)
                else:
                    print(f"Warning: No text reference found for {mp4_file}. Skipping.")
                    skipped_missing_text += 1
                    continue
            except ValueError:
                print(f"Warning: Filename stem '{stem}' is not a valid integer index. Skipping.")
                continue

            utterance_id = f"{video_id}_{stem}"

            # Paths are written relative to the output CSV's location, so the
            # manifest stays portable as long as that relative layout is preserved.
            manifest_rows.append({
                "video_path": os.path.relpath(roi_file, output_csv_dir),
                "audio_path": os.path.relpath(wav_file, output_csv_dir),
                "utterance_id": utterance_id,
                "reference": reference_text,
                "language": args.default_lang,
                "video_preprocessing": args.video_preprocessing
            })

    # Write everything out to a CSV file
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Success! Generated '{output_csv_path}' with {len(manifest_rows)} rows.")
    if skipped_missing_roi or skipped_missing_audio or skipped_missing_text:
        print(
            f"Skipped: {skipped_missing_roi} missing ROI video, "
            f"{skipped_missing_audio} missing audio, "
            f"{skipped_missing_text} missing text."
        )


if __name__ == "__main__":
    main()