from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from team1_asr.router_features import (
    _audio_features,
    _synchronization_features,
    _video_features,
    extract_features,
)


class RouterFeaturesTest(unittest.TestCase):
    def test_audio_features_identify_silence_and_spectral_content(self) -> None:
        sample_rate = 16000
        samples = np.concatenate(
            [np.zeros(sample_rate // 2), 0.4 * np.sin(2 * np.pi * 1000 * np.arange(sample_rate // 2) / sample_rate)]
        )
        features, envelope = _audio_features(samples, sample_rate)

        self.assertGreater(float(features["audio_silence_fraction"]), 0.35)
        self.assertGreater(float(features["audio_spectral_centroid_hz"]), 500.0)
        self.assertGreater(len(envelope), 1)

    def test_video_features_identify_motionless_roi(self) -> None:
        frames = np.full((5, 8, 8), 0.4, dtype=float)
        features, envelope = _video_features(frames, np.arange(5) / 25.0)

        self.assertEqual(float(features["video_mouth_motion_mean"]), 0.0)
        self.assertEqual(float(features["video_still_frame_rate"]), 1.0)
        self.assertEqual(len(envelope), 5)

    def test_synchronization_features_recover_known_lag(self) -> None:
        sample_rate = 1000
        timestamps = np.arange(10) * 0.1 + 0.05
        mouth = np.array([0.0, 0.1, 0.8, 0.2, 0.9, 0.1, 0.7, 0.2, 0.8, 0.1])
        audio_envelope = np.concatenate([np.full(100, value) for value in mouth])
        features = _synchronization_features(audio_envelope, sample_rate, timestamps, mouth)

        self.assertGreater(float(features["audio_video_sync_correlation"]), 0.9)
        self.assertAlmostEqual(float(features["audio_video_sync_lag_ms"]), 0.0, delta=100.0)

    def test_extract_features_uses_whisper_and_roi_inputs_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "audio.wav"
            sf.write(audio_path, np.linspace(-0.1, 0.1, 1600), 16000)
            video_path = root / "roi.mp4"
            video_path.touch()
            roi_manifest = root / "manifest.csv"
            roi_manifest.write_text(
                "utterance_id,audio_path,video_path,reference,language\n"
                "u1,audio.wav,roi.mp4,ignored reference,en\n"
            )
            reliability = root / "reliability.csv"
            reliability.write_text("utterance_id,roi_status\nu1,success\n")
            transcripts = root / "whisper.jsonl"
            transcripts.write_text(
                json.dumps(
                    {
                        "utterance_id": "u1",
                        "used_audio_path": str(audio_path),
                        "segments": [
                            {
                                "avg_logprob": -0.4,
                                "compression_ratio": 1.2,
                                "no_speech_prob": 0.01,
                            }
                        ],
                    }
                )
                + "\n"
            )
            frames = np.stack(
                [np.full((8, 8), value, dtype=float) for value in (0.2, 0.4, 0.6)]
            )
            with patch(
                "team1_asr.router_features._read_gray_video",
                return_value=(frames, np.arange(3) / 25.0),
            ):
                paths = extract_features(
                    roi_manifest,
                    reliability,
                    [("clean", transcripts)],
                    root / "features",
                )

            with paths["features"].open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["condition"], "clean")
            self.assertEqual(rows[0]["utterance_id"], "u1")
            self.assertEqual(rows[0]["audio_compression_ratio_mean"], "1.2")
            self.assertTrue(paths["summary"].exists())
            self.assertTrue(paths["summary_json"].exists())


if __name__ == "__main__":
    unittest.main()
