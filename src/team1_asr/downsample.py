import os
import wave
import numpy as np

# Set this to the path where your dataset folders are stored
target_dir = "/Users/alexiswisniewski/Desktop/AVSR/data/voxmm_and_lrs-voxmm/lrs-voxmm/original/test/"

def convert_24bit_to_16bit_numpy(file_path):
    with wave.open(file_path, 'rb') as wav_in:
        params = wav_in.getparams()
        
        # If it's already 16-bit (2 bytes), skip it
        if params.sampwidth == 2:
            return False
            
        frames = wav_in.readframes(params.nframes)
        
        # If it's 24-bit (3 bytes)
        if params.sampwidth == 3:
            # Convert raw 3-byte data into a numpy array of 24-bit integers
            raw_bytes = np.frombuffer(frames, dtype=np.uint8)
            
            # Reshape so each row is 3 bytes (1 sample)
            samples_3bytes = raw_bytes.reshape(-1, 3)
            
            # Pad a 4th zero-byte to make it a standard 32-bit integer (4 bytes)
            padding = np.zeros((samples_3bytes.shape[0], 1), dtype=np.uint8)
            samples_4bytes = np.hstack((padding, samples_3bytes))
            
            # View as 32-bit signed integers and shift right to restore scale
            ints_32 = samples_4bytes.view(dtype=np.int32).squeeze()
            ints_32 >>= 8 
            
            # Normalize and downscale to 16-bit signed integers (-32768 to 32767)
            ints_16 = (ints_32 / (2**23 - 1) * 32767).astype(np.int16)
            converted_frames = ints_16.tobytes()
        else:
            return False

    # Overwrite the original file safely
    with wave.open(file_path, 'wb') as wav_out:
        wav_out.setparams((params.nchannels, 2, params.framerate, len(converted_frames), params.comptype, params.compname))
        wav_out.writeframes(converted_frames)
    return True

# Walk through directories
converted_count = 0
for root, _, files in os.walk(target_dir):
    for file in files:
        if file.endswith('.wav'):
            full_path = os.path.join(root, file)
            try:
                if convert_24bit_to_16bit_numpy(full_path):
                    print(f"Successfully downsampled: {file}")
                    converted_count += 1
            except Exception as e:
                print(f"Error processing {file}: {e}")

print(f"\nFinished! Downsampled {converted_count} files to 16-bit PCM.")