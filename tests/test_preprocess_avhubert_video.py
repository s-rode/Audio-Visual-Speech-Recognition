from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from team1_asr.preprocess_avhubert_video import preprocess_manifest


class PreprocessAvhubertVideoTests(unittest.TestCase):
    def test_preprocess_manifest_stages_files_and_rewrites_video_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            data_dir.mkdir()
            video = data_dir / "clip.mp4"
            audio = data_dir / "clip.wav"
            video.write_bytes(b"video")
            audio.write_bytes(b"audio")
            manifest = data_dir / "manifest.csv"
            manifest.write_text(
                "utterance_id,audio_path,video_path,reference,language\n"
                "utt1,clip.wav,clip.mp4,hello,en\n",
                encoding="utf-8",
            )

            mms_repo = root / "MMS-LLaMA"
            prep_dir = mms_repo / "avhubert" / "preparation"
            prep_dir.mkdir(parents=True)
            (prep_dir / "detect_landmark.py").write_text("", encoding="utf-8")
            (prep_dir / "align_mouth.py").write_text("", encoding="utf-8")
            venv_python = mms_repo / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            venv_python.write_text("", encoding="utf-8")
            cnn = root / "mmod_human_face_detector.dat"
            predictor = root / "shape_predictor_68_face_landmarks.dat"
            mean_face = root / "20words_mean_face.npy"
            for path in [cnn, predictor, mean_face]:
                path.write_bytes(b"model")

            output_dir = root / "prepared"

            def fake_run(command, cwd):
                roi = output_dir / "video" / "utt1.mp4"
                roi.parent.mkdir(parents=True, exist_ok=True)
                roi.write_bytes(b"roi")

            with patch("team1_asr.preprocess_avhubert_video._run", side_effect=fake_run):
                preprocess_manifest(
                    manifest_path=manifest,
                    output_dir=output_dir,
                    mms_repo=mms_repo,
                    ffmpeg="ffmpeg",
                    python_executable=None,
                    cnn_detector=cnn,
                    face_predictor=predictor,
                    mean_face=mean_face,
                    video_column="video_path",
                    audio_column="audio_path",
                    id_column="utterance_id",
                    reference_column="reference",
                    language_column="language",
                    copy=True,
                    force=False,
                    skip_existing=False,
                )

            with (output_dir / "manifest.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["utterance_id"], "utt1")
            self.assertEqual(rows[0]["video_path"], "video/utt1.mp4")
            self.assertTrue((output_dir / rows[0]["video_path"]).exists())
            self.assertEqual((output_dir / "raw_video" / "utt1.mp4").read_bytes(), b"video")


if __name__ == "__main__":
    unittest.main()
