import os
import json
import csv
import subprocess
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# CONFIGURATION
# ==========================================
METADATA_DIR = Path("./data/metadata")
FACE_TRACK_DIR = Path("./data/face_track")
VIDEO_INPUT_DIR = Path("./data/voxmm_raw/video")
AUDIO_INPUT_DIR = Path("./data/wav")
OUTPUT_DIR = Path("./data/voxmm_training")

MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
MANIFEST_VIDEO_PREFIX = "voxmm_training"
MANIFEST_AUDIO_PREFIX = "voxmm_training"

# Set to True if you want to strictly crop the bottom half of the provided 
# face bounding box to isolate the mouth. If False, it resizes the entire face box to 96x96.
CROP_LOWER_HALF = False 

# ==========================================
# WORKER FUNCTIONS
# ==========================================
def extract_audio(input_path, output_path, start, end):
    """Extracts the audio segment using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(input_path),
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:a", "pcm_s16le",
        str(output_path)
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def extract_and_crop_video(input_path, output_path, start_time, end_time, fps, track_bboxes):
    """
    Reads the video frame-by-frame, dynamically crops the bounding box, 
    resizes to 96x96, and pipes to ffmpeg for encoding.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return False
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # 1. FRAME-ACCURATE SEEKING
    # OpenCV can jump to the wrong frame due to H264 I-frames. 
    # We seek to the closest keyframe, then manually grab frames until we hit our target.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    while current_frame < start_frame:
        ret = cap.grab()
        if not ret: break
        current_frame += 1

    # 2. FFMPEG PIPE SETUP
    # Piping raw frames to ffmpeg avoids OpenCV codec issues on Linux clusters
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', '96x96', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-c:v', 'libx264', '-preset', 'superfast', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-v', 'error', str(output_path)
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    
    last_bbox = None

    # 3. DYNAMIC CROP LOOP
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get the bounding box for this specific frame, fallback to last known if missing
        bbox = track_bboxes.get(frame_idx, last_bbox)
        
        if bbox is not None:
            # bbox is [x_min, y_min, x_max, y_max] in normalized coordinates (0 to 1)
            x1 = max(0, int(bbox[0] * width))
            y1 = max(0, int(bbox[1] * height))
            x2 = min(width, int(bbox[2] * width))
            y2 = min(height, int(bbox[3] * height))
            
            # Isolate mouth region by taking lower half of the face box (if toggled)
            if CROP_LOWER_HALF:
                y1 = y1 + ((y2 - y1) // 2)
                
            # Prevent zero-size crops from crashing the script
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                try:
                    crop_resized = cv2.resize(crop, (96, 96))
                    process.stdin.write(crop_resized.tobytes())
                    last_bbox = bbox
                except Exception:
                    pass
            
    # Close pipe to finalize the mp4
    process.stdin.close()
    process.wait()
    cap.release()
    
    return process.returncode == 0


def process_single_url(json_file, existing_ids):
    url = json_file.stem
    raw_video_path = VIDEO_INPUT_DIR / f"{url}.mp4"
    raw_audio_path = AUDIO_INPUT_DIR / f"{url}.wav"
    face_track_path = FACE_TRACK_DIR / f"{url}.json"

    if not all([raw_video_path.exists(), raw_audio_path.exists(), face_track_path.exists()]):
        return url, [], [f"Missing raw/face_track files for {url}"]

    # Load Metadata
    with open(json_file, "r") as f:
        metadata = json.load(f)
        
    # Load Face Tracks
    with open(face_track_path, "r") as f:
        face_data = json.load(f)

    fps = face_data.get("FPS", 25)
    
    # Pre-map all face tracks for O(1) lookup: dict[track_index] -> dict[frame_number -> bbox]
    track_lookup = {}
    for track in face_data.get("face_tracks", []):
        t_id = track.get("track_index")
        track_lookup[t_id] = {f: b for f, b in zip(track["frame"], track["bbox"])}

    segments = metadata.get("segments", [])
    url_output_dir = OUTPUT_DIR / url
    url_output_dir.mkdir(parents=True, exist_ok=True)

    rows_to_write = []
    errors = []

    for segment in segments:
        if not segment.get("on-screen", False):
            continue

        # Safely extract track ID (handles both dict and list structures)
        ft_info = segment.get("face_track")
        if isinstance(ft_info, list) and len(ft_info) > 0:
            track_id = ft_info[0].get("index")
        elif isinstance(ft_info, dict):
            track_id = ft_info.get("index")
        else:
            errors.append(f"Invalid face_track format in segment {segment.get('segment_index')}")
            continue

        if track_id not in track_lookup:
            errors.append(f"Track ID {track_id} not found in face_track JSON.")
            continue

        segment_idx = segment.get("segment_index")
        start = segment.get("start")
        end = segment.get("end")
        text = str(segment.get("text", "")).upper().strip()

        padded_idx = f"{segment_idx:05d}"
        utterance_id = f"{url}_{padded_idx}"

        if utterance_id in existing_ids:
            continue

        out_video_name = f"{padded_idx}_mouth.mp4"
        out_audio_name = f"{padded_idx}.wav"
        
        out_video_path = url_output_dir / out_video_name
        out_audio_path = url_output_dir / out_audio_name

        video_sliced = out_video_path.exists()
        audio_sliced = out_audio_path.exists()

        if not video_sliced:
            track_bboxes = track_lookup[track_id]
            video_sliced = extract_and_crop_video(raw_video_path, out_video_path, start, end, fps, track_bboxes)
            
        if not audio_sliced:
            audio_sliced = extract_audio(raw_audio_path, out_audio_path, start, end)

        if video_sliced and audio_sliced:
            manifest_video_val = f"{MANIFEST_VIDEO_PREFIX}/{url}/{out_video_name}"
            manifest_audio_val = f"{MANIFEST_AUDIO_PREFIX}/{url}/{out_audio_name}"
            rows_to_write.append([manifest_video_val, manifest_audio_val, utterance_id, text])
        else:
            errors.append(f"Failed slicing for {utterance_id}")

    return url, rows_to_write, errors


# ==========================================
# MAIN EXECUTION ENGINE
# ==========================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    manifest_exists = MANIFEST_PATH.exists()
    
    if manifest_exists:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3: existing_ids.add(row[2])

    json_files = list(METADATA_DIR.glob("*.json"))
    num_cores = 2 #int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 2))
    
    print(f"[*] Initializing parallel face-crop pool with {num_cores} workers.")

    with open(MANIFEST_PATH, "a" if manifest_exists else "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        if not manifest_exists:
            writer.writerow(["video_path", "audio_path", "utterance_id", "reference"])

        completed_count = 0
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = {executor.submit(process_single_url, j, existing_ids): j for j in json_files}

            for future in as_completed(futures):
                completed_count += 1
                url, rows, errors = future.result()

                if rows:
                    writer.writerows(rows)
                    csvfile.flush()

                for error in errors:
                    print(f"  [Error] {error}")
                print(f"[{completed_count}/{len(json_files)}] Completed {url} | Added {len(rows)} segments.")

if __name__ == "__main__":
    main()