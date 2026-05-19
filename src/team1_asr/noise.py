from __future__ import annotations

from pathlib import Path

from team1_asr.config import NoiseConfig


def maybe_apply_noise(audio_path: str, config: NoiseConfig, workspace: Path) -> str:
    if not config.enabled:
        return audio_path

    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        from team1_asr.backends.base import require_dependency

        require_dependency("numpy and soundfile", exc)

    audio, sample_rate = sf.read(audio_path, always_2d=False)
    audio = audio.astype("float32", copy=False)
    noise = _build_noise_track(audio, sample_rate, config, np)
    mixed = _mix_at_snr(audio, noise, config.snr_db, np)

    noisy_dir = workspace / "noisy_audio"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    noisy_path = noisy_dir / f"{Path(audio_path).stem}_{config.kind}_{int(config.snr_db)}dB.wav"
    sf.write(noisy_path, mixed, sample_rate)
    return str(noisy_path)


def _build_noise_track(audio, sample_rate: int, config: NoiseConfig, np):
    rng = np.random.default_rng(config.seed)
    if config.kind == "white":
        return rng.normal(0.0, 1.0, size=audio.shape).astype("float32")

    if config.kind == "external":
        if not config.noise_path:
            raise ValueError("noise.noise_path must be set when noise.kind='external'.")
        try:
            import soundfile as sf
        except ImportError as exc:
            from team1_asr.backends.base import require_dependency

            require_dependency("soundfile", exc)
        noise_audio, noise_rate = sf.read(config.noise_path, always_2d=False)
        if noise_rate != sample_rate:
            raise ValueError(
                "External noise sample rate does not match speech sample rate. "
                "Resample it beforehand to keep the scaffold simple."
            )
        noise_audio = noise_audio.astype("float32", copy=False)
        if noise_audio.shape[0] < audio.shape[0]:
            repeats = int(np.ceil(audio.shape[0] / noise_audio.shape[0]))
            noise_audio = np.tile(noise_audio, repeats)
        return noise_audio[: audio.shape[0]]

    raise ValueError(f"Unsupported noise kind: {config.kind}")


def _mix_at_snr(audio, noise, snr_db: float, np):
    speech_power = np.mean(np.square(audio))
    noise_power = np.mean(np.square(noise))
    if speech_power <= 0 or noise_power <= 0:
        return audio

    target_noise_power = speech_power / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    mixed = audio + scale * noise
    return np.clip(mixed, -1.0, 1.0).astype("float32")
