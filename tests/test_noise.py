from __future__ import annotations

from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import soundfile as sf

from team1_asr.config import NoiseConfig
from team1_asr.noise import (
    _align_processed_speech,
    maybe_apply_noise,
    validate_demand_noise_path,
)


class NoiseTest(unittest.TestCase):
    def test_missing_demand_path_fails_helpfully(self) -> None:
        with self.assertRaisesRegex(ValueError, "DEMAND noise path"):
            validate_demand_noise_path("missing/DEMAND/16k")

    def test_legacy_white_noise_still_writes_noisy_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            speech_path = _write_wav(root / "speech.wav")
            config = NoiseConfig(enabled=True, kind="white", snr_db=10.0, seed=1)

            noisy_path = maybe_apply_noise(
                str(speech_path),
                config,
                root / "outputs",
                utterance_id="utt-1",
            )

            self.assertTrue(Path(noisy_path).exists())
            self.assertIn("white_10dB", Path(noisy_path).name)

    def test_same_seed_and_utterance_select_same_demand_track_across_snr(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            speech_path = _write_wav(root / "speech.wav")
            demand_root = _write_demand(root / "DEMAND" / "16k")

            config_10 = NoiseConfig(seed=17)
            config_10.demand.enabled = True
            config_10.demand.snr_db = 10.0
            config_10.demand.noise_path = str(demand_root)

            config_0 = NoiseConfig(seed=17)
            config_0.demand.enabled = True
            config_0.demand.snr_db = 0.0
            config_0.demand.noise_path = str(demand_root)

            path_10 = maybe_apply_noise(
                str(speech_path),
                config_10,
                root / "out10",
                utterance_id="same-utt",
            )
            path_0 = maybe_apply_noise(
                str(speech_path),
                config_0,
                root / "out0",
                utterance_id="same-utt",
            )

            self.assertEqual(_without_snr(Path(path_10).stem), _without_snr(Path(path_0).stem))

    def test_processed_speech_alignment_keeps_start_of_signal(self) -> None:
        audio = np.arange(10, dtype=np.float32)

        aligned = _align_processed_speech(audio, 4, np)

        np.testing.assert_array_equal(aligned, np.array([0, 1, 2, 3], dtype=np.float32))


def _write_wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.linspace(-0.25, 0.25, 1600, dtype=np.float32)
    sf.write(path, audio, 16_000)
    return path


def _write_demand(root: Path) -> Path:
    for env_name, offset in [("NRIVER", 0.1), ("TMETRO", 0.2)]:
        env_dir = root / env_name
        env_dir.mkdir(parents=True, exist_ok=True)
        audio = np.full(3200, offset, dtype=np.float32)
        sf.write(env_dir / "ch01.wav", audio, 16_000)
    return root


def _without_snr(stem: str) -> str:
    return re.sub(r"_[-0-9.]+dB$", "", stem)


if __name__ == "__main__":
    unittest.main()
