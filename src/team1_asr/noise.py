from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from team1_asr.config import (
    BandwidthNoiseConfig,
    DemandNoiseConfig,
    NoiseConfig,
    ReverbNoiseConfig,
)


def maybe_apply_noise(
    audio_path: str,
    config: NoiseConfig,
    workspace: Path,
    utterance_id: str | None = None,
) -> str:
    if not config.enabled and not config.advanced_enabled:
        return audio_path

    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        from team1_asr.backends.base import require_dependency

        require_dependency("numpy and soundfile", exc)

    audio, sample_rate = sf.read(audio_path, always_2d=False)
    audio = audio.astype("float32", copy=False)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1).astype("float32")

    rng = np.random.default_rng(_seed_for_utterance(config.seed, utterance_id or audio_path))
    tags: list[str] = []

    if config.advanced_enabled:
        audio, tags = _apply_advanced_noise(audio, sample_rate, config, rng, np)
    elif config.enabled:
        noise = _build_legacy_noise_track(audio, sample_rate, config, rng, np)
        audio = _mix_at_snr(audio, noise, config.snr_db, np)
        tags.append(f"{config.kind}_{_format_snr(config.snr_db)}")

    if not tags:
        return audio_path

    noisy_dir = workspace / "noisy_audio"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    noisy_path = noisy_dir / f"{Path(audio_path).stem}_{'_'.join(tags)}.wav"
    sf.write(noisy_path, audio, sample_rate)
    return str(noisy_path)


def validate_demand_noise_path(noise_path: str) -> list[Path]:
    root = Path(noise_path)
    if not root.is_dir():
        raise ValueError(f"DEMAND noise path does not exist or is not a directory: {root}")

    tracks = sorted(root.glob("*/*.wav"))
    if not tracks:
        raise ValueError(
            f"DEMAND noise path contains no WAV files under environment folders: {root}"
        )
    return tracks


def _apply_advanced_noise(audio, sample_rate: int, config: NoiseConfig, rng, np):
    tags: list[str] = []

    if config.reverb.enabled:
        audio, tag = _apply_reverb_noise(audio, sample_rate, config.reverb, rng)
        tags.append(tag)

    if config.demand.enabled:
        audio, tag = _apply_demand_noise(audio, sample_rate, config.demand, rng, np)
        tags.append(tag)

    if config.bandwidth.enabled:
        audio, tag = _apply_bandwidth_noise(audio, sample_rate, config.bandwidth, rng, np)
        tags.append(tag)

    return audio, tags


def _build_legacy_noise_track(audio, sample_rate: int, config: NoiseConfig, rng, np):
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
        if len(noise_audio.shape) > 1:
            noise_audio = np.mean(noise_audio, axis=1).astype("float32")
        return _fit_audio_length(noise_audio, audio.shape[0], rng, np)

    raise ValueError(f"Unsupported noise kind: {config.kind}")


def _apply_reverb_noise(
    audio,
    sample_rate: int,
    config: ReverbNoiseConfig,
    rng,
) -> tuple[Any, str]:
    try:
        import numpy as np
        import pyroomacoustics as pra
    except ImportError as exc:
        from team1_asr.backends.base import require_dependency

        require_dependency("pyroomacoustics", exc)

    room_length = _uniform_range(rng, config.length_range, "noise.reverb.length_range")
    room_width = _uniform_range(rng, config.width_range, "noise.reverb.width_range")
    room_height = _uniform_range(rng, config.height_range, "noise.reverb.height_range")
    rt60_target = _uniform_range(rng, config.rt60_range, "noise.reverb.rt60_range")
    room_dim = [room_length, room_width, room_height]

    try:
        e_absorption, max_order = pra.inverse_sabine(rt60_target, room_dim)
    except ValueError:
        e_absorption, max_order = 0.2, 10

    room = pra.ShoeBox(
        room_dim,
        fs=sample_rate,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )
    source_pos = [room_length * 0.4, room_width * 0.5, room_height * 0.5]
    mic_pos = [room_length * 0.6, room_width * 0.5, room_height * 0.5]

    room.add_source(source_pos, signal=audio)
    room.add_microphone(np.array(mic_pos).reshape(3, 1))
    room.compute_rir()
    room.simulate()

    reverb_audio = room.mic_array.signals[0].astype("float32")
    reverb_audio = _align_processed_speech(reverb_audio, audio.shape[0], np)
    reverb_audio = _match_rms(reverb_audio, audio, np)

    return reverb_audio, f"reverb_rt60_{rt60_target:.2f}s"


def _apply_demand_noise(
    audio,
    sample_rate: int,
    config: DemandNoiseConfig,
    rng,
    np,
) -> tuple[Any, str]:
    try:
        import soundfile as sf
    except ImportError as exc:
        from team1_asr.backends.base import require_dependency

        require_dependency("soundfile", exc)

    tracks = validate_demand_noise_path(config.noise_path)
    noise_path = tracks[int(rng.integers(0, len(tracks)))]

    noise_audio, noise_rate = sf.read(noise_path, always_2d=False)
    if noise_rate != sample_rate:
        raise ValueError(
            f"DEMAND track sample rate ({noise_rate} Hz) does not match "
            f"speech sample rate ({sample_rate} Hz): {noise_path}"
        )

    noise_audio = noise_audio.astype("float32", copy=False)
    if len(noise_audio.shape) > 1:
        noise_audio = np.mean(noise_audio, axis=1).astype("float32")

    noise_audio = _fit_audio_length(noise_audio, audio.shape[0], rng, np)
    mixed = _mix_at_snr(audio, noise_audio, config.snr_db, np)
    env_name = noise_path.parent.name
    return mixed, f"{env_name}_{noise_path.stem}_{_format_snr(config.snr_db)}"


def _apply_bandwidth_noise(
    audio,
    sample_rate: int,
    config: BandwidthNoiseConfig,
    rng,
    np,
) -> tuple[Any, str]:
    try:
        import scipy.signal as signal
    except ImportError as exc:
        from team1_asr.backends.base import require_dependency

        require_dependency("scipy", exc)

    if not config.available_filters:
        raise ValueError("noise.bandwidth.available_filters must not be empty.")
    if not config.available_cutoffs_hz:
        raise ValueError("noise.bandwidth.available_cutoffs_hz must not be empty.")

    chosen_filter = str(config.available_filters[int(rng.integers(0, len(config.available_filters)))])
    chosen_cutoff = int(
        config.available_cutoffs_hz[int(rng.integers(0, len(config.available_cutoffs_hz)))]
    )

    nyquist = 0.5 * sample_rate
    if chosen_cutoff >= nyquist:
        chosen_cutoff = int(nyquist - 100)
    if chosen_cutoff <= 0:
        raise ValueError(
            "noise.bandwidth.available_cutoffs_hz must contain positive values below Nyquist."
        )

    normalized_cutoff = chosen_cutoff / nyquist
    if chosen_filter == "butter":
        sos = signal.butter(config.filter_order, normalized_cutoff, btype="low", output="sos")
    elif chosen_filter == "bessel":
        sos = signal.bessel(
            config.filter_order,
            normalized_cutoff,
            btype="low",
            norm="mag",
            output="sos",
        )
    elif chosen_filter == "cheby1":
        sos = signal.cheby1(
            config.filter_order,
            rp=1.0,
            Wn=normalized_cutoff,
            btype="low",
            output="sos",
        )
    else:
        raise ValueError(f"Unsupported bandwidth filter type: {chosen_filter}")

    filtered_audio = signal.sosfilt(sos, audio).astype("float32")
    filtered_audio = _match_rms(filtered_audio, audio, np)

    return filtered_audio, f"{chosen_filter}_{chosen_cutoff}Hz"


def _fit_audio_length(audio, target_length: int, rng, np):
    if audio.shape[0] <= 0:
        raise ValueError("Noise audio must contain at least one sample.")
    if audio.shape[0] == target_length:
        return audio
    if audio.shape[0] > target_length:
        max_start = audio.shape[0] - target_length
        start = int(rng.integers(0, max_start + 1))
        return audio[start : start + target_length]

    repeats = int(np.ceil(target_length / audio.shape[0]))
    return np.tile(audio, repeats)[:target_length]


def _align_processed_speech(audio, target_length: int, np):
    if audio.shape[0] <= 0:
        raise ValueError("Processed speech audio must contain at least one sample.")
    if audio.shape[0] >= target_length:
        return audio[:target_length]
    return np.pad(audio, (0, target_length - audio.shape[0]))


def _match_rms(audio, reference, np):
    audio_power = np.mean(np.square(audio))
    reference_power = np.mean(np.square(reference))
    if audio_power <= 0 or reference_power <= 0:
        return audio.astype("float32")
    return (audio * np.sqrt(reference_power / audio_power)).astype("float32")


def _mix_at_snr(audio, noise, snr_db: float, np):
    speech_power = np.mean(np.square(audio))
    noise_power = np.mean(np.square(noise))
    if speech_power <= 0 or noise_power <= 0:
        return audio

    target_noise_power = speech_power / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    mixed = audio + scale * noise
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak * 0.99
    return mixed.astype("float32")


def _seed_for_utterance(seed: int, utterance_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{utterance_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _uniform_range(rng, values: list[float], name: str) -> float:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    low, high = values
    if high < low:
        raise ValueError(f"{name} upper bound must be >= lower bound.")
    return float(rng.uniform(low, high))


def _format_snr(snr_db: float) -> str:
    value = int(snr_db) if float(snr_db).is_integer() else snr_db
    return f"{value}dB"
