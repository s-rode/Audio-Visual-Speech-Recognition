from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from team1_asr.reliability import calibrate, evaluate, split_manifests


class ReliabilityTest(unittest.TestCase):
    def test_calibrates_detection_threshold_and_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _write_fixture(Path(temp_dir))
            output_dir = Path(temp_dir) / "calibration"

            policy = calibrate(
                paths["manifest"],
                paths["reliability"],
                paths["audio"],
                paths["av"],
                output_dir,
                audio_system_name="whisper_base_en",
                av_system_name="avec_voxmm_finetuned_av",
            )

            self.assertEqual(policy["policy_type"], "threshold")
            self.assertEqual(policy["feature"], "face_detection_rate")
            self.assertEqual(policy["threshold"], 0.9)
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertEqual(metrics["routed"]["wer"], 0.0)
            self.assertEqual(policy["audio_system"], "whisper_base_en")
            self.assertEqual(policy["av_system"], "avec_voxmm_finetuned_av")
            self.assertEqual(metrics["audio"]["system_name"], "whisper_base_en")
            self.assertEqual(
                metrics["audio_visual"]["system_name"],
                "avec_voxmm_finetuned_av",
            )
            self.assertEqual(metrics["routed_to_audio_visual_count"], 1)
            self.assertEqual(metrics["preprocessing_failure_count"], 1)
            self.assertTrue((output_dir / "threshold_sweep.csv").exists())
            self.assertTrue((output_dir / "routing_decisions.csv").exists())

    def test_split_and_evaluate_use_frozen_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_fixture(root)
            calibration_dir = root / "calibration"
            calibrate(
                paths["manifest"],
                paths["reliability"],
                paths["audio"],
                paths["av"],
                calibration_dir,
            )

            split_dir = root / "split"
            audio_manifest, fallback_manifest, av_manifest = split_manifests(
                paths["manifest"],
                paths["roi_manifest"],
                paths["reliability"],
                calibration_dir / "policy.json",
                split_dir,
            )

            self.assertEqual(_manifest_ids(audio_manifest), ["u1", "u2", "u3"])
            self.assertEqual(_manifest_ids(fallback_manifest), ["u2", "u3"])
            self.assertEqual(_manifest_ids(av_manifest), ["u1"])
            self.assertEqual(audio_manifest.name, "audio_manifest.csv")
            self.assertEqual(fallback_manifest.name, "audio_fallback_manifest.csv")
            self.assertEqual(av_manifest.name, "audio_visual_manifest.csv")
            metrics = evaluate(
                paths["manifest"],
                paths["reliability"],
                paths["audio"],
                paths["av"],
                calibration_dir / "policy.json",
                root / "evaluation",
            )
            self.assertEqual(metrics["routed"]["wer"], 0.0)

    def test_requires_complete_audio_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_fixture(root)
            paths["audio"].write_text(
                json.dumps({"utterance_id": "u1", "prediction": "hello world"}) + "\n"
            )

            with self.assertRaisesRegex(ValueError, "audio transcripts are missing"):
                calibrate(
                    paths["manifest"],
                    paths["reliability"],
                    paths["audio"],
                    paths["av"],
                    root / "calibration",
                )

    def test_missing_av_prediction_falls_back_to_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_fixture(root)
            _write_jsonl(
                paths["av"],
                [
                    {"utterance_id": "u2", "prediction": "bad bad"},
                ],
            )

            calibrate(
                paths["manifest"],
                paths["reliability"],
                paths["audio"],
                paths["av"],
                root / "calibration",
            )
            metrics = json.loads((root / "calibration" / "metrics.json").read_text())

            self.assertEqual(metrics["audio_visual"]["coverage"], 1 / 3)
            self.assertEqual(metrics["routed_to_audio_visual_count"], 0)

    def test_tie_prefers_roi_success_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_fixture(root)
            _write_jsonl(
                paths["audio"],
                [
                    {"utterance_id": "u1", "prediction": "wrong wrong"},
                    {"utterance_id": "u2", "prediction": "bad bad"},
                    {"utterance_id": "u3", "prediction": "red blue"},
                ],
            )
            _write_jsonl(
                paths["av"],
                [
                    {"utterance_id": "u1", "prediction": "hello world"},
                    {"utterance_id": "u2", "prediction": "good day"},
                ],
            )

            policy = calibrate(
                paths["manifest"],
                paths["reliability"],
                paths["audio"],
                paths["av"],
                root / "calibration",
            )

            self.assertEqual(policy["policy_type"], "roi_success")
            self.assertIsNone(policy["feature"])

    def test_uses_explicit_audio_only_policy_when_audio_is_better(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_fixture(root)
            _write_jsonl(
                paths["audio"],
                [
                    {"utterance_id": "u1", "prediction": "hello world"},
                    {"utterance_id": "u2", "prediction": "good day"},
                    {"utterance_id": "u3", "prediction": "red blue"},
                ],
            )
            _write_jsonl(
                paths["av"],
                [
                    {"utterance_id": "u1", "prediction": "wrong wrong"},
                    {"utterance_id": "u2", "prediction": "bad bad"},
                ],
            )

            policy = calibrate(
                paths["manifest"],
                paths["reliability"],
                paths["audio"],
                paths["av"],
                root / "calibration",
            )

            self.assertEqual(policy["policy_type"], "audio_only")
            self.assertIsNone(policy["threshold"])

    def test_rejects_duplicate_transcript_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_fixture(root)
            duplicate = json.dumps({"utterance_id": "u1", "prediction": "hello"})
            paths["audio"].write_text(duplicate + "\n" + duplicate + "\n")

            with self.assertRaisesRegex(ValueError, "Duplicate utterance_id"):
                calibrate(
                    paths["manifest"],
                    paths["reliability"],
                    paths["audio"],
                    paths["av"],
                    root / "calibration",
                )

    def test_evaluates_older_policy_system_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_fixture(root)
            policy_path = root / "old_policy.json"
            policy_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "policy_type": "roi_success",
                        "feature": None,
                        "threshold": None,
                        "audio_system": "parakeet",
                        "av_system": "avec_av",
                    }
                )
            )

            metrics = evaluate(
                paths["manifest"],
                paths["reliability"],
                paths["audio"],
                paths["av"],
                policy_path,
                root / "evaluation",
            )
            with (root / "evaluation" / "routing_decisions.csv").open() as handle:
                decisions = list(csv.DictReader(handle))

            self.assertEqual(metrics["audio"]["system_name"], "parakeet")
            self.assertEqual(metrics["audio_visual"]["system_name"], "avec_av")
            self.assertEqual(decisions[0]["selected_system"], "avec_av")
            self.assertIn("audio_prediction", decisions[0])
            self.assertIn("audio_visual_prediction", decisions[0])


def _write_fixture(root: Path) -> dict[str, Path]:
    manifest = root / "manifest.csv"
    manifest.write_text(
        "utterance_id,audio_path,video_path,reference,language\n"
        "u1,audio/u1.wav,raw/u1.mp4,hello world,en\n"
        "u2,audio/u2.wav,raw/u2.mp4,good day,en\n"
        "u3,audio/u3.wav,raw/u3.mp4,red blue,en\n"
    )
    roi_manifest = root / "roi_manifest.csv"
    roi_manifest.write_text(
        "utterance_id,audio_path,video_path,reference,language,video_preprocessing\n"
        "u1,audio/u1.wav,roi/u1.mp4,hello world,en,avec_mouth_roi\n"
        "u2,audio/u2.wav,roi/u2.mp4,good day,en,avec_mouth_roi\n"
        "u3,audio/u3.wav,roi/u3.mp4,red blue,en,avec_mouth_roi\n"
    )
    reliability = root / "visual_reliability.csv"
    reliability.write_text(
        "utterance_id,roi_status,fallback_used,face_detection_rate,"
        "retinaface_confidence_mean,mouth_landmark_confidence_mean\n"
        "u1,success,false,0.9,0.9,0.8\n"
        "u2,success,false,0.2,0.8,0.7\n"
        "u3,fallback_full_frame,true,0.0,,\n"
    )
    audio = root / "audio.jsonl"
    _write_jsonl(
        audio,
        [
            {"utterance_id": "u1", "prediction": "wrong wrong"},
            {"utterance_id": "u2", "prediction": "good day"},
            {"utterance_id": "u3", "prediction": "red blue"},
        ],
    )
    av = root / "av.jsonl"
    _write_jsonl(
        av,
        [
            {"utterance_id": "u1", "prediction": "hello world"},
            {"utterance_id": "u2", "prediction": "bad bad"},
            {"utterance_id": "u3", "prediction": "wrong again"},
        ],
    )
    return {
        "manifest": manifest,
        "roi_manifest": roi_manifest,
        "reliability": reliability,
        "audio": audio,
        "av": av,
    }


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _manifest_ids(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return [row["utterance_id"] for row in csv.DictReader(handle)]


if __name__ == "__main__":
    unittest.main()
