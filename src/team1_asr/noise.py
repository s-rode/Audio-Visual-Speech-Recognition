import glob
import os
from pathlib import Path

def apply_demand_noise(audio_path: str, config, workspace: Path) -> str:
    """
    Loads a clean audio file and mixes it with a random segment from a random
    noise track inside a randomly selected environment folder within the DEMAND 16k root directory.
    """
    # 1. Early exit if noise is not enabled
    if not getattr(config, "enabled", False):
        return audio_path

    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        from team1_asr.backends.base import require_dependency
        require_dependency("numpy and soundfile", exc)

    # 2. Setup seeded random number generator for reproducibility
    seed = getattr(config, "seed", None)
    rng = np.random.default_rng(seed)

    # 3. Read the clean speech file
    audio, sample_rate = sf.read(audio_path, always_2d=False)
    audio = audio.astype("float32", copy=False)
    
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  # Ensure mono

    # 4. Navigate DEMAND 16k Directory Structure
    demand_root = config.demand_dir
    if not demand_root or not os.path.isdir(demand_root):
        raise ValueError(f"noise_path must point to a valid DEMAND 16k root directory. Got: {demand_root}")

    # Get all environment folders (e.g., DKITCHEN, TMETRO, OOFFICE)
    env_folders = [
        os.path.join(demand_root, d) 
        for d in os.listdir(demand_root) 
        if os.path.isdir(os.path.join(demand_root, d))
    ]
    if not env_folders:
        raise FileNotFoundError(f"No environment subfolders found inside: {demand_root}")

    # Step A: Pick a random environment folder
    chosen_env_dir = rng.choice(env_folders)

    # Step B: Pick a random .wav track inside that folder (e.g., ch01.wav)
    noise_tracks = glob.glob(os.path.join(chosen_env_dir, "*.wav"))
    if not noise_tracks:
        raise FileNotFoundError(f"No .wav tracks found inside selected environment folder: {chosen_env_dir}")
    
    chosen_noise_path = rng.choice(noise_tracks)

    # 5. Extract a random snippet from the selected noise file
    with sf.SoundFile(chosen_noise_path) as f:
        noise_rate = f.samplerate
        noise_total_frames = f.frames
        
        if noise_rate != sample_rate:
            raise ValueError(
                f"DEMAND track sample rate ({noise_rate}Hz) does not match clean audio rate ({sample_rate}Hz). "
                "Double check that your root path targets the 16k variant."
            )
            
        target_frames = audio.shape[0]
        if noise_total_frames > target_frames:
            # Pick a random starting point within bounds and slice it out
            max_start = noise_total_frames - target_frames
            start_frame = rng.integers(0, max_start)
            f.seek(start_frame)
            noise_audio = f.read(target_frames, dtype="float32")
        else:
            # If the noise file is unexpectedly shorter, load it fully and loop
            f.seek(0)
            noise_audio = f.read(dtype="float32")
            repeats = int(np.ceil(target_frames / noise_total_frames))
            noise_audio = np.tile(noise_audio, repeats)[:target_frames]

    if len(noise_audio.shape) > 1:
        noise_audio = np.mean(noise_audio, axis=1)  # Ensure mono

    # 6. Mix at target SNR
    speech_power = np.mean(np.square(audio))
    noise_power = np.mean(np.square(noise_audio))
    
    if speech_power > 0 and noise_power > 0:
        target_noise_power = speech_power / (10 ** (config.snr_db / 10.0))
        scale = np.sqrt(target_noise_power / noise_power)
        mixed = audio + scale * noise_audio
        mixed = np.clip(mixed, -1.0, 1.0).astype("float32")
    else:
        mixed = audio

    # 7. Write to workspace and return path
    noisy_dir = workspace / "noisy_audio"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    
    # Capture environment name for clean logging filenames (e.g., TMETRO)
    env_name = Path(chosen_env_dir).name
    noisy_path = noisy_dir / f"{Path(audio_path).stem}_{env_name}_{int(config.snr_db)}dB.wav"
    
    sf.write(noisy_path, mixed, sample_rate)
    return str(noisy_path)