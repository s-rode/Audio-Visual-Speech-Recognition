import os
import json
import csv
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# CONFIGURATION
# ==========================================
METADATA_DIR = Path("./data/metadata")
VIDEO_INPUT_DIR = Path("./data/voxmm_raw/video")
AUDIO_INPUT_DIR = Path("./data/wav")
OUTPUT_DIR = Path("./data/voxmm_training")

# Manifest settings
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

# Adjust these prefixes to match your model training expectations
MANIFEST_VIDEO_PREFIX = "voxmm_training"
MANIFEST_AUDIO_PREFIX = "voxmm_training"


# ==========================================
# WORKER FUNCTION (Runs on a single core)
# ==========================================
def cut_segment(input_path, output_path, start, end, is_video=True):
    """
    Slices a precise segment from a media file using ffmpeg.
    -ss placed after -i guarantees frame-accurate visual sync.
    """
    if is_video:
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
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-c:a", "pcm_s16le",
            str(output_path)
        ]

    # Run quietly to prevent terminal output flooding from parallel jobs
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0


def process_single_url(json_file, existing_ids):
    """
    Processes all on-screen segments for a single JSON metadata file.
    Returns a list of successful manifest rows and a list of errors.
    """
    url = json_file.stem
    raw_video_path = VIDEO_INPUT_DIR / f"{url}.mp4"
    raw_audio_path = AUDIO_INPUT_DIR / f"{url}.wav"

    if not raw_video_path.exists() or not raw_audio_path.exists():
        return url, [], [f"Missing raw files for {url}"]

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        return url, [], [f"Corrupt JSON {json_file.name}: {e}"]

    segments = metadata.get("segments", [])
    url_output_dir = OUTPUT_DIR / url
    url_output_dir.mkdir(parents=True, exist_ok=True)

    rows_to_write = []
    errors = []

    for segment in segments:
        if not segment.get("on-screen", False):
            continue

        segment_idx = segment.get("segment_index")
        start = segment.get("start")
        end = segment.get("end")
        text = str(segment.get("text", "")).upper().strip()

        padded_idx = f"{segment_idx:05d}"
        utterance_id = f"{url}_{padded_idx}"

        # 1. Skip if this utterance was already successfully added to the CSV in a past run
        if utterance_id in existing_ids:
            continue

        out_video_name = f"{padded_idx}.mp4"
        out_audio_name = f"{padded_idx}.wav"
        
        out_video_path = url_output_dir / out_video_name
        out_audio_path = url_output_dir / out_audio_name

        # 2. Check if output files already exist on disk
        video_sliced = out_video_path.exists()
        audio_sliced = out_audio_path.exists()

        if not video_sliced:
            video_sliced = cut_segment(raw_video_path, out_video_path, start, end, is_video=True)
        if not audio_sliced:
            audio_sliced = cut_segment(raw_audio_path, out_audio_path, start, end, is_video=False)

        if video_sliced and audio_sliced:
            manifest_video_val = f"{MANIFEST_VIDEO_PREFIX}/{url}/{out_video_name}"
            manifest_audio_val = f"{MANIFEST_AUDIO_PREFIX}/{url}/{out_audio_name}"
            rows_to_write.append([manifest_video_val, manifest_audio_val, utterance_id, text])
        else:
            errors.append(f"Failed ffmpeg slicing for {utterance_id}")

    return url, rows_to_write, errors


# ==========================================
# MAIN EXECUTION ENGINE
# ==========================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Read existing manifest to enable safe resuming
    existing_ids = set()
    manifest_exists = MANIFEST_PATH.exists()
    
    if manifest_exists:
        print("[*] Found existing manifest.csv. Indexing processed segments to resume safely...")
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    existing_ids.add(row[2])  # Add the utterance_id
        print(f"[+] Loaded {len(existing_ids)} already processed segments. They will be skipped.")

    # 2. Find all target metadata files
    json_files = list(METADATA_DIR.glob("*.json"))
    if not json_files:
        print(f"[-] No JSON files found in {METADATA_DIR}.")
        return

    # 3. Detect available CPU cores (Slurm-aware)
    num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 2))
    print(f"[*] Initializing parallel process pool with {num_cores} workers.")

    # 4. Open the CSV file (Append mode if resuming, Write mode if new)
    open_mode = "a" if manifest_exists else "w"
    with open(MANIFEST_PATH, open_mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        
        if not manifest_exists:
            writer.writerow(["video_path", "audio_path", "utterance_id", "reference"])

        # 5. Distribute jobs to the ProcessPool
        total_files = len(json_files)
        completed_count = 0

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Map futures to track progress
            futures = {
                executor.submit(process_single_url, j_file, existing_ids): j_file 
                for j_file in json_files
            }

            # As each worker finishes processing a JSON, harvest its results
            for future in as_completed(futures):
                completed_count += 1
                url, rows, errors = future.result()

                # Write results to the single CSV file in the main thread (No thread locks required!)
                if rows:
                    writer.writerows(rows)
                    csvfile.flush()  # Push to disk immediately

                # Print progress logs
                for error in errors:
                    print(f"  [Error] {error}")
                
                print(f"[{completed_count}/{total_files}] Completed {url} | Added {len(rows)} segments.")

    print(f"\n[+] Processing complete! Manifest saved to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()