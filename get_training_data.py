import os
import json
from pathlib import Path

def calculate_onscreen_duration(folder_path):
    total_onscreen_duration = 0.0
    speaker_durations = {}  # Track individual speaker durations globally
    processed_files = 0

    # Convert to Path object for easier handling
    dir_path = Path(folder_path)

    if not dir_path.exists():
        print(f"Error: The directory '{folder_path}' does not exist.")
        return

    # Loop through all JSON files in the folder
    for json_file in dir_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if speaker_infos exists in this JSON
            speaker_infos = data.get("speaker_infos", {})
            
            for speaker_id, info in speaker_infos.items():
                # Extract the on-screen duration (default to 0 if missing)
                duration = info.get("utterance_duration", 0.0)
                
                # Update global sum
                total_onscreen_duration += duration
                
                # Track per-speaker sum (optional, but helpful to see balance)
                speaker_durations[speaker_id] = speaker_durations.get(speaker_id, 0.0) + duration
            
            processed_files += 1

        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read/parse file {json_file.name}. Error: {e}")

    # Output results
    print("=" * 50)
    print(f"Data Scan Complete!")
    print(f"Processed files: {processed_files}")
    print(f"Total unique speaker IDs found: {len(speaker_durations)}")
    print("=" * 50)
    
    # Format the final duration beautifully
    hours = int(total_onscreen_duration // 3600)
    minutes = int((total_onscreen_duration % 3600) // 60)
    seconds = total_onscreen_duration % 60
    
    print(f"Total Useable On-Screen Training Data:")
    print(f"  - {total_onscreen_duration:.3f} seconds")
    print(f"  - (~{hours} hours, {minutes} minutes, and {seconds:.2f} seconds)")
    print("=" * 50)

# Replace this with the actual path to your folder
folder_path = "./data/voxmm_and_lrs-voxmm/metadata" 
calculate_onscreen_duration(folder_path)