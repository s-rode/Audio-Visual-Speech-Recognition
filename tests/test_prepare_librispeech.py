from __future__ import annotations

import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import soundfile as sf

from team1_asr import prepare_librispeech


class LibriSpeechSplitTest(unittest.TestCase):
    def test_normalizes_dotted_clean_alias(self) -> None:
        config_name, split, warning = prepare_librispeech._normalize_config_and_split(
            "clean",
            "validation.clean",
        )

        self.assertEqual(config_name, "clean")
        self.assertEqual(split, "validation")
        self.assertIsNotNone(warning)
        self.assertIn("validation.clean", warning)
        self.assertIn("validation", warning)

    def test_rejects_invalid_split_with_hint(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported LibriSpeech split 'validation.clean.other'",
        ) as context:
            prepare_librispeech._normalize_config_and_split(
                "clean",
                "validation.clean.other",
            )

        self.assertIn("Valid splits:", str(context.exception))
        self.assertIn("--config all", str(context.exception))


class LibriSpeechExportTest(unittest.TestCase):
    def test_exports_raw_audio_bytes_and_relative_manifest_path(self) -> None:
        sample = {
            "id": "speaker/chapter/utt-1",
            "audio": {
                "bytes": _audio_bytes(),
                "path": "speaker-chapter-utt-1.flac",
            },
            "text": "HELLO WORLD",
            "speaker_id": 1,
            "chapter_id": 2,
            "file": "speaker-chapter-utt-1.flac",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "librispeech"
            with patch.object(
                prepare_librispeech,
                "_load_dataset",
                return_value=iter([sample]),
            ) as load_dataset:
                prepare_librispeech.export_librispeech_split(
                    output_dir=output_dir,
                    config_name="clean",
                    split="validation.clean",
                    limit=1,
                    seed=0,
                    language="en",
                    cache_dir=None,
                    save_dataset=False,
                    force=False,
                )

            load_dataset.assert_called_once_with(
                config_name="clean",
                split="validation",
                cache_dir=None,
                streaming=True,
            )

            manifest_path = output_dir / "manifest.csv"
            self.assertTrue(manifest_path.exists())
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["audio_path"], "audio/speaker_chapter_utt-1.wav")
            exported_audio = output_dir / rows[0]["audio_path"]
            self.assertTrue(exported_audio.exists())

            audio, sample_rate = sf.read(exported_audio)
            self.assertEqual(sample_rate, 16000)
            self.assertGreater(len(audio), 0)


def _audio_bytes() -> bytes:
    handle = io.BytesIO()
    sf.write(handle, [0.0, 0.1, -0.1, 0.0], 16000, format="WAV")
    return handle.getvalue()


if __name__ == "__main__":
    unittest.main()
