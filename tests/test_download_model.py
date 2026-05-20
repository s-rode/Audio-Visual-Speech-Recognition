from __future__ import annotations

import unittest
from unittest.mock import patch

from team1_asr.download_model import download_model


class DownloadModelTest(unittest.TestCase):
    def test_resolves_model_alias_and_downloads_with_progress_enabled(self) -> None:
        with (
            patch("faster_whisper.utils._MODELS", {"tiny.en": "Systran/faster-whisper-tiny.en"}),
            patch("huggingface_hub.snapshot_download", return_value="/tmp/model") as snapshot_download,
        ):
            model_path = download_model("tiny.en")

        self.assertEqual(model_path, "/tmp/model")
        snapshot_download.assert_called_once()
        args, kwargs = snapshot_download.call_args
        self.assertEqual(args[0], "Systran/faster-whisper-tiny.en")
        self.assertEqual(kwargs["local_files_only"], False)
        self.assertIn("model.bin", kwargs["allow_patterns"])

    def test_rejects_unknown_model_alias(self) -> None:
        with patch("faster_whisper.utils._MODELS", {"tiny.en": "Systran/faster-whisper-tiny.en"}):
            with self.assertRaisesRegex(ValueError, "Unknown faster-whisper model"):
                download_model("not-a-model")


if __name__ == "__main__":
    unittest.main()
