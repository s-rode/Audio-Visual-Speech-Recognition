from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from team1_asr.config import InputConfig
from team1_asr.manifest import load_manifest


class ManifestTest(unittest.TestCase):
    def test_loads_optional_video_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "utterance_id,audio_path,video_path,reference,language\n"
                "utt-1,audio/utt.wav,video/utt.mp4,hello,en\n",
                encoding="utf-8",
            )

            rows = load_manifest(InputConfig(manifest_path=str(manifest)))

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].audio_path.endswith("audio/utt.wav"))
        self.assertTrue(rows[0].video_path.endswith("video/utt.mp4"))

    def test_missing_video_column_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "manifest.csv"
            manifest.write_text(
                "utterance_id,audio_path,reference,language\n"
                "utt-1,audio/utt.wav,hello,en\n",
                encoding="utf-8",
            )

            rows = load_manifest(InputConfig(manifest_path=str(manifest)))

        self.assertIsNone(rows[0].video_path)


if __name__ == "__main__":
    unittest.main()
