from __future__ import annotations

import glob
import os
from pathlib import Path
from team1_asr.config import NoiseConfig, ReverbNoiseConfig, DemandNoiseConfig, BandwidthNoiseConfig


def apply_noise(audio_path: str, config: NoiseConfig, workspace: Path) -> str:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        from team1_asr.backends.base import require_dependency
        require_dependency("numpy and soundfile", exc)

    # 1. Read the clean audio file into memory exactly once
    audio, sample_rate = sf.read(audio_path, always_2d=False)
    audio = audio.astype("float32", copy=False)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    rng = config._rng
    filename_tags = []

    # 2. Sequential in-memory processing chain
    if config.reverb.enabled:
        audio, tag = _apply_reverb_noise(audio, sample_rate, config.reverb, rng)
        filename_tags.append(tag)

    if config.demand.enabled:
        audio, tag = _apply_demand_noise(audio, sample_rate, config.demand, rng)
        filename_tags.append(tag)

    if config.bandwidth.enabled:
        audio, tag = _apply_bandwidth_noise(audio, sample_rate, config.bandwidth, rng)
        filename_tags.append(tag)

    # If no noise types were enabled, just return the original file path
    if not filename_tags:
        return audio_path

    # 3. Save the single, final aggregated audio array to disk
    noisy_dir = workspace / "noisy_audio"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine tags to track all applied modifications (e.g., "speech_reverb_rt60_0.45s_TMETRO_-5dB_butter_2000Hz.wav")
    tags_suffix = "_".join(filename_tags)
    noisy_path = noisy_dir / f"{Path(audio_path).stem}_{tags_suffix}.wav"
    
    sf.write(noisy_path, audio, sample_rate)
    return str(noisy_path)


def _apply_reverb_noise(audio, sample_rate: int, config: ReverbNoiseConfig, rng) -> tuple:
    import numpy as np
    import pyroomacoustics as pra

    room_length = rng.uniform(config.length_range[0], config.length_range[1])
    room_width = rng.uniform(config.width_range[0], config.width_range[1])
    room_height = rng.uniform(config.height_range[0], config.height_range[1])
    room_dim = [room_length, room_width, room_height]

    rt60_target = rng.uniform(config.rt60_range[0], config.rt60_range[1])

    try:
        e_absorption, max_order = pra.inverse_sabine(rt60_target, room_dim)
    except ValueError:
        e_absorption, max_order = 0.2, 10

    room = pra.ShoeBox(room_dim, fs=sample_rate, materials=pra.Material(e_absorption), max_order=max_order)
    source_pos = [room_length * 0.4, room_width * 0.5, room_height * 0.5]
    mic_pos = [room_length * 0.6, room_width * 0.5, room_height * 0.5]

    room.add_source(source_pos, signal=audio)
    room.add_microphone(np.array(mic_pos).reshape(3, 1))
    room.compute_rir()
    room.simulate()

    reverb_audio = room.mic_array.signals[0].astype("float32")[:audio.shape[0]]
    if np.max(np.abs(reverb_audio)) > 0:
        reverb_audio = reverb_audio / np.max(np.abs(reverb_audio))

    return reverb_audio, f"reverb_rt60_{rt60_target:.2f}s"


def _apply_demand_noise(audio, sample_rate: int, config: DemandNoiseConfig, rng) -> tuple:
    import numpy as np
    import soundfile as sf

    demand_root = config.noise_path
    if not demand_root or not os.path.isdir(demand_root):
        raise ValueError(f"noise_path must point to a valid DEMAND directory. Got: {demand_root}")

    env_folders = [
        os.path.join(demand_root, d) 
        for d in os.listdir(demand_root) 
        if os.path.isdir(os.path.join(demand_root, d))
    ]
    
    chosen_env_dir = rng.choice(env_folders)
    noise_tracks = glob.glob(os.path.join(chosen_env_dir, "*.wav"))
    chosen_noise_path = rng.choice(noise_tracks)

    with sf.SoundFile(chosen_noise_path) as f:
        noise_rate = f.samplerate
        noise_total_frames = f.frames
        if noise_rate != sample_rate:
            raise ValueError(f"DEMAND track sample rate ({noise_rate}Hz) does not match clean audio rate ({sample_rate}Hz).")
            
        target_frames = audio.shape[0]
        if noise_total_frames > target_frames:
            max_start = noise_total_frames - target_frames
            start_frame = rng.integers(0, max_start)
            f.seek(start_frame)
            noise_audio = f.read(target_frames, dtype="float32")
        else:
            f.seek(0)
            noise_audio = f.read(dtype="float32")
            repeats = int(np.ceil(target_frames / noise_total_frames))
            noise_audio = np.tile(noise_audio, repeats)[:target_frames]

    if len(noise_audio.shape) > 1:
        noise_audio = np.mean(noise_audio, axis=1)

    speech_power = np.mean(np.square(audio))
    noise_power = np.mean(np.square(noise_audio))
    
    if speech_power > 0 and noise_power > 0:
        target_noise_power = speech_power / (10 ** (config.snr_db / 10.0))
        scale = np.sqrt(target_noise_power / noise_power)
        mixed = audio + scale * noise_audio
        mixed = np.clip(mixed, -1.0, 1.0).astype("float32")
    else:
        mixed = audio

    env_name = Path(chosen_env_dir).name
    return mixed, f"{env_name}_{int(config.snr_db)}dB"


def _apply_bandwidth_noise(audio, sample_rate: int, config: BandwidthNoiseConfig, rng) -> tuple:
    import numpy as np
    import scipy.signal as signal

    chosen_filter = rng.choice(config.available_filters)
    chosen_cutoff = rng.choice(config.available_cutoffs_hz)

    nyquist = 0.5 * sample_rate
    if chosen_cutoff >= nyquist:
        chosen_cutoff = nyquist - 100
    Wn = chosen_cutoff / nyquist

    if chosen_filter == "butter":
        sos = signal.butter(config.filter_order, Wn, btype="low", output="sos")
    elif chosen_filter == "bessel":
        sos = signal.bessel(config.filter_order, Wn, btype="low", norm="mag", output="sos")
    elif chosen_filter == "cheby1":
        sos = signal.cheby1(config.filter_order, rp=1.0, Wn=Wn, btype="low", output="sos")
    else:
        raise ValueError(f"Unsupported filter type: {chosen_filter}")

    filtered_audio = signal.sosfilt(sos, audio).astype("float32")

    if np.max(np.abs(filtered_audio)) > 0:
        filtered_audio = filtered_audio / np.max(np.abs(filtered_audio))

    return filtered_audio, f"{chosen_filter}_{chosen_cutoff}Hz"