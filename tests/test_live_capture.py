from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from team1_asr.live_capture import BrowserCapturePreparer, capture_suffix


class LiveCaptureTest(unittest.TestCase):
    def test_capture_suffix_accepts_browser_codec_parameter(self) -> None:
        self.assertEqual(capture_suffix("video/webm;codecs=vp8,opus"), ".webm")
        self.assertEqual(capture_suffix("video/mp4"), ".mp4")

    def test_capture_suffix_rejects_audio_only_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported capture media type"):
            capture_suffix("audio/webm")

    def test_real_preparer_preserves_virtualenv_symlink(self) -> None:
        with TemporaryDirectory() as temp_dir:
            executable = Path(temp_dir) / ".venv" / "bin" / "python"
            executable.parent.mkdir(parents=True)
            executable.symlink_to(Path("/usr/bin/python3"))

            preparer = BrowserCapturePreparer(
                real_preprocessing=True,
                repo_path=".",
                python_executable=str(executable),
            )

            self.assertEqual(preparer.python_executable, str(executable.absolute()))

    def test_mock_preparer_emits_capture_and_preprocess_events(self) -> None:
        events: list[dict[str, object]] = []
        with TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            preparer = BrowserCapturePreparer(real_preprocessing=False)
            with (
                patch("team1_asr.live_capture._extract_audio") as extract_audio,
                patch("team1_asr.live_capture._validate_video") as validate_video,
            ):
                artifacts = preparer.prepare(
                    b"browser recording",
                    "video/webm;codecs=vp8,opus",
                    work_dir,
                    "live-test",
                    events.append,
                )

            recording_path = work_dir / "live-test.webm"
            self.assertEqual(recording_path.read_bytes(), b"browser recording")
            extract_audio.assert_called_once_with(
                recording_path,
                work_dir / "live-test.wav",
            )
            validate_video.assert_called_once_with(recording_path)
            self.assertEqual(artifacts.source_path, recording_path)
            self.assertEqual(artifacts.video_path, recording_path)
            self.assertEqual(artifacts.reliability["roi_status"], "success")

        self.assertEqual(
            [(event["stage"], event["status"]) for event in events],
            [
                ("capture", "complete"),
                ("preprocess", "running"),
                ("preprocess", "complete"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
