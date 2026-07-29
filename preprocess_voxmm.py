import os
import json
import csv
import subprocess
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
METADATA_DIR = Path("./data/voxmm_and_lrs-voxmm/metadata")
VIDEO_INPUT_DIR = Path("./data/voxmm_raw/video")
AUDIO_INPUT_DIR = Path("./data/voxmm_and_lrs-voxmm/wav")
OUTPUT_DIR = Path("./data/voxmm_training")

# Manifest settings
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

# Adjust these prefixes to match what your training code expects in the CSV
# (Currently configured to write relative paths starting from 'voxmm_training/')
MANIFEST_VIDEO_PREFIX = "voxmm_training"
MANIFEST_AUDIO_PREFIX = "voxmm_training"

# ==========================================
# PROCEDURAL ENGINE
# ==========================================

def cut_segment(input_path, output_path, start, end, is_video=True):
    """
    Slices a precise segment from a media file using ffmpeg.
    Placing -ss AFTER the input -i ensures frame-accurate extraction,
    which is critical for visual lip alignment in AVSR.
    """
    if is_video:
        # Re-encode video slightly to guarantee sync; fast preset keeps it snappy
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-c:v", "libx264",
            "-preset", "superfast",
            "-crf", "18",
            "-c:a", "aac",
            str(output_path)
        ]
    else:
        # Export uncompressed WAV audio
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-c:a", "pcm_s16le",
            str(output_path)
        ]

    # Run silently, only capture errors
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # We open/create the manifest and write to it line-by-line.
    # This allows you to safely stop and resume the script without losing progress.
    file_exists = MANIFEST_PATH.exists()
    
    with open(MANIFEST_PATH, "a" if file_exists else "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        
        if not file_exists:
            writer.writerow(["video_path", "audio_path", "utterance_id", "reference"])

        json_files = list(METADATA_DIR.glob("*.json"))
        if not json_files:
            print(f"[-] No JSON files found in {METADATA_DIR}. Please check your paths.")
            return

        print(f"[*] Found {len(json_files)} metadata files. Starting processing...")

        for idx, json_file in enumerate(json_files, 1):
            url = json_file.stem  # Get the <url> name
            
            # 1. Match with raw source video and audio
            raw_video_path = VIDEO_INPUT_DIR / f"{url}.mp4"
            raw_audio_path = AUDIO_INPUT_DIR / f"{url}.wav"

            if not raw_video_path.exists() or not raw_audio_path.exists():
                print(f"[!] Missing raw video or audio for {url}. Skipping.")
                continue

            # Load metadata
            with open(json_file, "r", encoding="utf-8") as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    print(f"[-] Corrupt JSON: {json_file.name}. Skipping.")
                    continue

            segments = metadata.get("segments", [])
            url_output_dir = OUTPUT_DIR / url
            url_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{idx}/{len(json_files)}] Processing {url} ({len(segments)} segments total)...")

            for segment in segments:
                # Filter only on-screen segments
                if not segment.get("on-screen", False):
                    continue

                segment_idx = segment.get("segment_index")
                start = segment.get("start")
                end = segment.get("end")
                text = str(segment.get("text", "")).upper().strip()  # Standardize to uppercase

                # Format segment index (e.g., 00023)
                padded_idx = f"{segment_idx:05d}"
                utterance_id = f"{url}_{padded_idx}"

                # Output file naming
                out_video_name = f"{padded_idx}.mp4"
                out_audio_name = f"{padded_idx}.wav"
                
                out_video_path = url_output_dir / out_video_name
                out_audio_path = url_output_dir / out_audio_name

                # Resuming capability: skip rendering if files already exist
                video_sliced = out_video_path.exists()
                audio_sliced = out_audio_path.exists()

                if not video_sliced:
                    video_sliced = cut_segment(raw_video_path, out_video_path, start, end, is_video=True)
                if not audio_sliced:
                    audio_sliced = cut_segment(raw_audio_path, out_audio_path, start, end, is_video=False)

                if video_sliced and audio_sliced:
                    # Write to Manifest with custom path formatting
                    manifest_video_val = f"{MANIFEST_VIDEO_PREFIX}/{url}/{out_video_name}"
                    manifest_audio_val = f"{MANIFEST_AUDIO_PREFIX}/{url}/{out_audio_name}"
                    
                    writer.writerow([manifest_video_val, manifest_audio_val, utterance_id, text])
                    # Force write to disk immediately so progress is saved
                    csvfile.flush()
                else:
                    print(f"  [-] Failed to process segment {padded_idx} due to an ffmpeg error.")

    print(f"\n[+] Processing complete! Manifest saved to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()