import os
import json
from datasets import load_dataset

# 1. Create directory to hold files
output_dir = "lrs3_test_samples"
os.makedirs(output_dir, exist_ok=True)

# 2. Load the dataset stream
dataset = load_dataset("Ainncy/LRS3", split="train", streaming=True)

# 3. CRITICAL: Add .decode(False) to stop torchcodec from parsing it into a decoder object
dataset = dataset.decode(False).take(10)

metadata = []

print("Extracting raw video bytes and transcripts from stream...")
for idx, example in enumerate(dataset):
    file_key = example.get('__key__', f"sample_{idx + 1}")
    video_data = example.get('mp4')
    txt_data = example.get('txt')  # Extract the transcript data
    print(file_key)
    
    # Format safe filenames
    safe_base = file_key.replace("/", "_")
    safe_video_filename = safe_base + ".mp4"
    safe_txt_filename = safe_base + ".txt"
    
    local_video_path = os.path.join(output_dir, safe_video_filename)
    local_txt_path = os.path.join(output_dir, safe_txt_filename)
    
    # --- Step A: Save the Video ---
    if video_data:
        if isinstance(video_data, dict) and "bytes" in video_data:
            with open(local_video_path, "wb") as f:
                f.write(video_data["bytes"])
            print(f" -> Saved Video: {safe_video_filename}")
        elif isinstance(video_data, bytes):
            with open(local_video_path, "wb") as f:
                f.write(video_data)
            print(f" -> Saved Video: {safe_video_filename}")
        else:
            print(f" ! Unexpected video format for item {idx+1}: {type(video_data)}")
    else:
        print(f" ! Missing video data for item {idx+1}. Available keys: {list(example.keys())}")

    # --- Step B: Parse and Save the Transcript ---
    transcript_str = ""
    if txt_data is not None:
        # Decode the raw transcript data depending on its streaming format
        if isinstance(txt_data, dict) and "bytes" in txt_data:
            transcript_str = txt_data["bytes"].decode("utf-8").strip()
        elif isinstance(txt_data, bytes):
            transcript_str = txt_data.decode("utf-8").strip()
        elif isinstance(txt_data, str):
            transcript_str = txt_data.strip()
        else:
            print(f" ! Unexpected text format for item {idx+1}: {type(txt_data)}")

        # Save as a standalone text file companion
        with open(local_txt_path, "w", encoding="utf-8") as f:
            f.write(transcript_str)
        print(f" -> Saved Transcript: {safe_txt_filename} (\"{transcript_str[:30]}...\")")
    else:
        print(f" ! Missing transcript data for item {idx+1}")

    # --- Step C: Append to Metadata list ---
    metadata.append({
        "id": idx + 1,
        "video_file": safe_video_filename,
        "transcript_file": safe_txt_filename if txt_data is not None else None,
        "transcript": transcript_str,
        "original_key": file_key
    })

# 4. Save the metadata file (now including the transcript string!)
with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print(f"\nFinished! Your '{output_dir}/' folder now contains both .mp4 and companion .txt files, alongside a detailed metadata.json.")