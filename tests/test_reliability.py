from __future__ import annotations

import csv
import json
import math
import pickle
from pathlib import Path
import struct
import tempfile
import unittest
import wave
from unittest.mock import patch

from team1_asr.reliability import (
    _benefit_prediction,
    _cumulative_routing_thresholds,
    _fit_benefit_policy,
    _load_evaluation_inputs,
    _score_policy,
    _score_threshold_candidates,
    _select_routing_threshold,
    _benefit_target,
    calibrate,
    diagnose_failures,
    evaluate,
    finalize_train_dev_study,
    learn,
    noise_study,
    select_train_dev_candidate,
    split_manifests,
    study,
    train_dev_candidate,
    train_dev_oof_fold,
)


def _routing_decision(
    score: float | None,
    audio_errors: int,
    av_errors: int | None,
    *,
    roi_status: str = "success",
) -> dict[str, object]:
    return {
        "reliability_score": "" if score is None else score,
        "audio_word_errors": audio_errors,
        "audio_visual_word_errors": "" if av_errors is None else av_errors,
        "audio_visual_prediction": "avec" if av_errors is not None else "",
        "roi_status": roi_status,
        "fallback_used": "false",
    }


def _bruteforce_routing_threshold(decisions: list[dict[str, object]]) -> float:
    scores = sorted({
        float(row["reliability_score"])
        for row in decisions
        if row["reliability_score"] != ""
        and row["audio_visual_word_errors"] != ""
        and row["audio_visual_prediction"]
        and row["roi_status"] == "success"
    })
    if not scores:
        return float("inf")
    candidates = [max(scores) + 1.0, *scores]
    return min(
        candidates,
        key=lambda threshold: (
            sum(
                int(row["audio_visual_word_errors"])
                if row["reliability_score"] != ""
                and row["audio_visual_word_errors"] != ""
                and row["audio_visual_prediction"]
                and row["roi_status"] == "success"
                and float(row["reliability_score"]) >= threshold
                else int(row["audio_word_errors"])
                for row in decisions
            ),
            sum(
                row["reliability_score"] != ""
                and row["audio_visual_word_errors"] != ""
                and row["audio_visual_prediction"]
                and row["roi_status"] == "success"
                and float(row["reliability_score"]) >= threshold
                for row in decisions
            ),
            -threshold,
        ),
    )


def _precomputed_threshold_row(
    score: float | None,
    audio_errors: int,
    av_errors: int | None,
    *,
    roi_status: str = "success",
) -> dict[str, object]:
    return {
        "feature_row": {
            "score": "" if score is None else score,
            "roi_status": roi_status,
            "fallback_used": "false",
        },
        "audio_errors": audio_errors,
        "av_errors": av_errors,
        "av_prediction": "avec" if av_errors is not None else None,
        "reference_words": 1,
    }


class ReliabilityTest(unittest.TestCase):
    def test_cumulative_routing_threshold_matches_bruteforce(self) -> None:
        decisions = [
            _routing_decision(0.9, 4, 1),
            _routing_decision(0.9, 3, 3),
            _routing_decision(0.4, 2, 0),
            _routing_decision(0.2, 1, 0, roi_status="fallback"),
            _routing_decision(None, 1, 0),
            _routing_decision(0.1, 1, None),
        ]
        expected = _bruteforce_routing_threshold(decisions)
        self.assertEqual(_select_routing_threshold(decisions), expected)
        self.assertEqual(
            _cumulative_routing_thresholds(decisions),
            [
                (12, 0, 1.9),
                (9, 2, 0.9),
                (7, 3, 0.4),
            ],
        )

    def test_cumulative_routing_threshold_uses_existing_tiebreaks(self) -> None:
        decisions = [_routing_decision(0.7, 2, 2), _routing_decision(0.3, 1, 1)]
        self.assertEqual(_select_routing_threshold(decisions), 1.7)
        self.assertEqual(_select_routing_threshold([_routing_decision(None, 1, 0)]), float("inf"))

    def test_cumulative_generic_threshold_scores_match_bruteforce(self) -> None:
        precomputed = [
            _precomputed_threshold_row(0.8, 4, 1),
            _precomputed_threshold_row(0.8, 2, 2),
            _precomputed_threshold_row(0.2, 1, 0),
            _precomputed_threshold_row(0.1, 1, 0, roi_status="fallback"),
            _precomputed_threshold_row(None, 1, 0),
            _precomputed_threshold_row(0.4, 1, None),
        ]
        policies = [
            {"policy_type": "threshold", "feature": "score", "threshold": threshold,
             "direction": direction, "priority": 1, "feature_family": "visual"}
            for direction in ("higher", "lower")
            for threshold in (0.2, 0.8)
        ]
        optimized = _score_threshold_candidates(policies, precomputed)
        for policy in policies:
            self.assertEqual(
                optimized[id(policy)],
                _score_policy({}, policy, precomputed=precomputed),
            )

    def test_benefit_target_represents_expected_av_error_reduction(self) -> None:
        self.assertEqual(_benefit_target(5, 2), 3.0)
        self.assertEqual(_benefit_target(3, 3), 0.0)
        self.assertEqual(_benefit_target(1, 4), -3.0)

    def test_neural_benefit_router_is_grouped_deterministic_and_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _write_learned_fixture(Path(temp_dir))
            inputs = _load_evaluation_inputs(
                paths["manifest"],
                paths["reliability"],
                paths["audio"],
                paths["av"],
                "audio",
            )
            features = ["audio_avg_logprob_mean", "face_detection_rate"]
            first = _fit_benefit_policy(
                inputs, feature_candidates=features, model="mlp", seed=7,
                compute_feature_importances=True,
            )
            second = _fit_benefit_policy(
                inputs, feature_candidates=features, model="mlp", seed=7,
                compute_feature_importances=False,
            )

            self.assertEqual(first["hidden_layer_sizes"], [64, 32])
            self.assertEqual(first["selected_epochs"], second["selected_epochs"])
            self.assertTrue(first["feature_importances"])
            fit_groups = set(first["early_stopping_fit_source_groups"])
            validation_groups = set(first["early_stopping_validation_source_groups"])
            self.assertFalse(fit_groups & validation_groups)
            self.assertEqual(fit_groups | validation_groups, {f"video{i}" for i in range(5)})

            missing_prediction = _benefit_prediction(first, {})
            repeated_prediction = _benefit_prediction(second, {})
            self.assertIsNotNone(missing_prediction)
            self.assertIsNotNone(repeated_prediction)
            self.assertTrue(math.isfinite(float(missing_prediction)))
            self.assertAlmostEqual(float(missing_prediction), float(repeated_prediction), places=12)

            normalized = [
                (first["feature_medians"][feature] - first["feature_means"][feature])
                / first["feature_scales"][feature]
                for feature in first["feature_names"]
            ]
            expected = float(first["estimator"].predict([normalized])[0])
            expected = expected * first["target_scale"] + first["target_mean"]
            self.assertAlmostEqual(float(missing_prediction), expected, places=12)

            restored = pickle.loads(pickle.dumps(first))
            self.assertAlmostEqual(
                float(_benefit_prediction(restored, {}) or 0.0),
                float(missing_prediction),
                places=12,
            )
            corrupt = dict(first)
            corrupt.pop("target_scale")
            self.assertIsNone(_benefit_prediction(corrupt, {}))

    def test_neural_candidate_writes_complete_dev_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_learned_fixture(root)
            candidate_root = root / "candidates"
            config = {
                "name": "benefit_mlp_enriched",
                "model": "mlp",
                "features": ["audio_avg_logprob_mean", "face_detection_rate"],
            }
            with patch("team1_asr.reliability.CLEAN_ROUTER_CONFIGS", (config,)):
                result = train_dev_candidate(
                    train_manifest_path=paths["manifest"],
                    train_reliability_path=paths["reliability"],
                    train_audio_transcripts_path=paths["audio"],
                    train_av_transcripts_path=paths["av"],
                    dev_manifest_path=paths["manifest"],
                    dev_reliability_path=paths["reliability"],
                    dev_audio_transcripts_path=paths["audio"],
                    dev_av_transcripts_path=paths["av"],
                    output_dir=candidate_root,
                    configuration="benefit_mlp_enriched",
                    folds=5,
                    inner_folds=2,
                    seed=11,
                )
            output = candidate_root / "benefit_mlp_enriched"
            self.assertEqual(result["dev"]["utterances_scored"], 10)
            self.assertEqual(result["policy"]["benefit_model"], "mlp")
            self.assertTrue((output / "policy.pkl").exists())
            self.assertTrue((output / "candidate.json").exists())
            with (output / "dev_decisions.csv").open() as handle:
                decisions = list(csv.DictReader(handle))
            self.assertEqual(len(decisions), 10)
            self.assertNotIn("audio_visual_prediction", result["features"])

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
            self.assertIn("missed_audio_visual_wins", metrics)
            self.assertIn("false_audio_visual_routes", metrics)
            self.assertIn("oracle_gap_wer", metrics)
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

    def test_diagnoses_missing_audio_visual_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_fixture(root)
            _write_jsonl(
                paths["av"],
                [
                    {"utterance_id": "u1", "prediction": "hello world"},
                ],
            )

            summary = diagnose_failures(
                paths["manifest"],
                paths["reliability"],
                paths["av"],
                root / "diagnostics",
            )
            with (root / "diagnostics" / "avec_failure_diagnostics.csv").open() as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(summary["audio_visual_success_count"], 1)
            self.assertEqual(summary["audio_visual_failure_count"], 2)
            self.assertEqual(rows[0]["status"], "success")
            self.assertEqual(rows[1]["status"], "failed")
            self.assertTrue(
                (root / "diagnostics" / "avec_failure_summary.json").exists()
            )

    def test_calibrates_lower_is_better_multiple_face_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "utterance_id,audio_path,video_path,reference,language\n"
                "u1,audio/u1.wav,raw/u1.mp4,hello world,en\n"
                "u2,audio/u2.wav,raw/u2.mp4,good day,en\n"
            )
            reliability = root / "visual_reliability.csv"
            reliability.write_text(
                "utterance_id,roi_status,fallback_used,face_detection_rate,"
                "retinaface_confidence_mean,retinaface_confidence_p10,"
                "mouth_landmark_confidence_mean,mouth_landmark_confidence_p10,"
                "multiple_face_rate\n"
                "u1,success,false,1.0,0.9,0.9,0.8,0.8,0.0\n"
                "u2,success,false,1.0,0.9,0.9,0.8,0.8,0.5\n"
            )
            audio = root / "audio.jsonl"
            _write_jsonl(
                audio,
                [
                    {"utterance_id": "u1", "prediction": "wrong wrong"},
                    {"utterance_id": "u2", "prediction": "good day"},
                ],
            )
            av = root / "av.jsonl"
            _write_jsonl(
                av,
                [
                    {"utterance_id": "u1", "prediction": "hello world"},
                    {"utterance_id": "u2", "prediction": "bad bad"},
                ],
            )

            policy = calibrate(
                manifest,
                reliability,
                audio,
                av,
                root / "calibration",
            )

            self.assertEqual(policy["policy_type"], "threshold")
            self.assertEqual(policy["feature"], "multiple_face_rate")
            self.assertEqual(policy["direction"], "lower")
            self.assertEqual(policy["threshold"], 0.0)
            metrics = json.loads((root / "calibration" / "metrics.json").read_text())
            self.assertEqual(metrics["routed"]["wer"], 0.0)

    def test_calibrates_audio_uncertainty_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "utterance_id,audio_path,video_path,reference,language\n"
                "u1,audio/u1.wav,raw/u1.mp4,hello world,en\n"
                "u2,audio/u2.wav,raw/u2.mp4,good day,en\n"
            )
            reliability = root / "visual_reliability.csv"
            reliability.write_text(
                "utterance_id,roi_status,fallback_used,face_detection_rate,"
                "retinaface_confidence_mean,retinaface_confidence_p10,"
                "mouth_landmark_confidence_mean,mouth_landmark_confidence_p10,"
                "multiple_face_rate\n"
                "u1,success,false,1.0,0.9,0.9,0.8,0.8,0.0\n"
                "u2,success,false,1.0,0.9,0.9,0.8,0.8,0.0\n"
            )
            audio = root / "audio.jsonl"
            _write_jsonl(
                audio,
                [
                    {
                        "utterance_id": "u1",
                        "prediction": "wrong wrong",
                        "duration": 2.0,
                        "segments": [{"avg_logprob": -1.2, "no_speech_prob": 0.1}],
                    },
                    {
                        "utterance_id": "u2",
                        "prediction": "good day",
                        "duration": 2.0,
                        "segments": [{"avg_logprob": -0.1, "no_speech_prob": 0.0}],
                    },
                ],
            )
            av = root / "av.jsonl"
            _write_jsonl(
                av,
                [
                    {"utterance_id": "u1", "prediction": "hello world"},
                    {"utterance_id": "u2", "prediction": "bad bad"},
                ],
            )

            policy = calibrate(
                manifest,
                reliability,
                audio,
                av,
                root / "calibration",
            )
            metrics = json.loads((root / "calibration" / "metrics.json").read_text())
            with (root / "calibration" / "routing_decisions.csv").open() as handle:
                decisions = list(csv.DictReader(handle))

            self.assertEqual(policy["policy_type"], "threshold")
            self.assertEqual(policy["feature_family"], "audio_uncertainty")
            self.assertEqual(policy["feature"], "audio_avg_logprob_mean")
            self.assertEqual(policy["direction"], "lower")
            self.assertEqual(policy["threshold"], -1.2)
            self.assertEqual(metrics["routed"]["wer"], 0.0)
            self.assertEqual(decisions[0]["feature_family"], "audio_uncertainty")
            self.assertIn("audio_avg_logprob_mean", decisions[0])

    def test_writes_missed_audio_visual_wins_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_fixture(root)
            _write_jsonl(
                paths["audio"],
                [
                    {
                        "utterance_id": "u1",
                        "prediction": "wrong wrong",
                        "duration": 2.0,
                        "segments": [{"avg_logprob": -1.0, "no_speech_prob": 0.2}],
                    },
                    {"utterance_id": "u2", "prediction": "good day"},
                    {"utterance_id": "u3", "prediction": "red blue"},
                ],
            )
            _write_jsonl(
                paths["av"],
                [
                    {"utterance_id": "u1", "prediction": "hello world"},
                    {"utterance_id": "u2", "prediction": "bad bad"},
                ],
            )

            evaluate(
                paths["manifest"],
                paths["reliability"],
                paths["audio"],
                paths["av"],
                _write_policy(root, "audio_only"),
                root / "evaluation",
            )
            with (root / "evaluation" / "missed_av_wins.csv").open() as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual([row["utterance_id"] for row in rows], ["u1"])
            self.assertEqual(rows[0]["audio_prediction"], "wrong wrong")
            self.assertEqual(rows[0]["audio_visual_prediction"], "hello world")
            self.assertIn("audio_avg_logprob_mean", rows[0])

    def test_learns_grouped_router_and_splits_from_audio_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_learned_fixture(root)
            output_dir = root / "learned"

            policy = learn(
                paths["manifest"],
                paths["reliability"],
                paths["audio"],
                paths["av"],
                output_dir,
                folds=5,
                seed=4,
            )

            self.assertEqual(policy["policy_type"], "learned_logistic")
            self.assertEqual(policy["feature_family"], "combined")
            self.assertEqual(policy["cross_validation"]["folds"], 5)
            self.assertTrue((output_dir / "cross_validation_decisions.csv").exists())
            self.assertTrue((output_dir / "cross_validation_metrics.json").exists())
            self.assertTrue((output_dir / "learned_router_coefficients.csv").exists())
            self.assertIn("recommended_for_test", policy["cross_validation"])
            with (output_dir / "cross_validation_decisions.csv").open() as handle:
                decisions = list(csv.DictReader(handle))
            self.assertEqual(len(decisions), 10)
            self.assertEqual(
                {row["fold"] for row in decisions if row["utterance_id"].startswith("video0_")},
                {next(row["fold"] for row in decisions if row["utterance_id"] == "video0_00000")},
            )

            with self.assertRaisesRegex(ValueError, "require --audio-transcripts"):
                split_manifests(
                    paths["manifest"],
                    paths["roi_manifest"],
                    paths["reliability"],
                    output_dir / "policy.json",
                    root / "missing-audio",
                )
            _, _, av_manifest = split_manifests(
                paths["manifest"],
                paths["roi_manifest"],
                paths["reliability"],
                output_dir / "policy.json",
                root / "split",
                audio_transcripts_path=paths["audio"],
            )
            self.assertEqual(
                _manifest_ids(av_manifest),
                [f"video{index}_00000" for index in range(5)],
            )

    def test_study_compares_compact_feature_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_learned_fixture(root)
            output_dir = root / "study"

            result = study(
                paths["manifest"],
                paths["reliability"],
                paths["audio"],
                paths["av"],
                output_dir,
                folds=5,
                inner_folds=2,
                seed=2,
            )

            self.assertEqual(result["baseline"]["utterances_scored"], 10)
            self.assertTrue((output_dir / "router_study.csv").exists())
            self.assertTrue((output_dir / "router_study_decisions.csv").exists())
            self.assertTrue((output_dir / "router_study.json").exists())
            self.assertTrue((output_dir / "best_policy.json").exists())
            with (output_dir / "router_study.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            self.assertEqual(
                {row["feature_set"] for row in rows},
                {"audio_compact", "visual_compact", "combined_compact"},
            )

    def test_checkpointed_train_dev_workflow_merges_grouped_folds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = _write_learned_fixture(root)
            candidate_root = root / "candidates"
            oof_root = root / "oof"
            final_root = root / "final"
            kwargs = {
                "train_manifest_path": paths["manifest"],
                "train_reliability_path": paths["reliability"],
                "train_audio_transcripts_path": paths["audio"],
                "train_av_transcripts_path": paths["av"],
                "dev_manifest_path": paths["manifest"],
                "dev_reliability_path": paths["reliability"],
                "dev_audio_transcripts_path": paths["audio"],
                "dev_av_transcripts_path": paths["av"],
                "folds": 5,
                "inner_folds": 2,
                "seed": 3,
                "configuration": "logistic_core",
            }
            with patch(
                "team1_asr.reliability.CLEAN_ROUTER_CONFIGS",
                (
                    {
                        "name": "logistic_core",
                        "model": "logistic",
                        "features": ["face_detection_rate"],
                    },
                ),
            ):
                train_dev_candidate(output_dir=candidate_root, **kwargs)
                selection = select_train_dev_candidate(
                    candidate_dir=candidate_root, output_dir=root / "selection"
                )
                self.assertEqual(selection["selected_configuration"], "logistic_core")
                for fold_index in range(5):
                    train_dev_oof_fold(
                        output_dir=oof_root, fold_index=fold_index, **kwargs
                    )
                result = finalize_train_dev_study(
                    candidate_dir=candidate_root,
                    oof_dir=oof_root,
                    output_dir=final_root,
                    folds=5,
                    inner_folds=2,
                    seed=3,
                )
            self.assertEqual(result["selected_configuration"], "logistic_core")
            self.assertTrue((final_root / "selected_policy.pkl").exists())
            self.assertTrue((final_root / "router_train_dev_decisions.csv").exists())

    def test_noise_study_groups_conditions_by_source_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, reliability, conditions = _write_noise_study_fixture(root)
            output_dir = root / "noise-study"

            result = noise_study(
                manifest,
                reliability,
                conditions,
                output_dir,
                folds=5,
                inner_folds=2,
                seed=1,
            )

            self.assertIn(result["best_configuration"], {
                "whisper_confidence",
                "audio_quality",
                "combined_noise",
                "benefit_ridge",
                "benefit_hist_gradient_boosting",
            })
            with (output_dir / "noise_router_decisions.csv").open() as handle:
                decisions = list(csv.DictReader(handle))
            self.assertEqual(len(decisions), 50)
            for configuration in {row["configuration"] for row in decisions}:
                rows = [row for row in decisions if row["configuration"] == configuration]
                for video_index in range(5):
                    folds = {
                        row["fold"]
                        for row in rows
                        if row["utterance_id"].endswith(f"video{video_index}_00000")
                    }
                    self.assertEqual(len(folds), 1)
            with (output_dir / "noise_router_metrics.csv").open() as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual(len(metrics), 10)
            self.assertEqual({row["condition"] for row in metrics}, {"clean", "demand_-5db"})
            self.assertTrue(
                all("oracle_wer" in row and "router_to_oracle_gap_wer" in row for row in metrics)
            )
            study_json = json.loads((output_dir / "noise_router_study.json").read_text())
            self.assertEqual(study_json["bootstrap_samples"], 1000)
            self.assertIn(
                "comparison_vs_combined_noise",
                study_json["configurations"]["benefit_ridge"],
            )

    def test_noise_study_boosted_tree_is_deterministic_and_imputes_missing_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, reliability, conditions = _write_noise_study_fixture(root)
            with reliability.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["mouth_landmark_confidence_p10"] = ""
            with reliability.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            first = root / "first"
            second = root / "second"
            noise_study(manifest, reliability, conditions, first, folds=5, inner_folds=2, seed=3)
            noise_study(manifest, reliability, conditions, second, folds=5, inner_folds=2, seed=3)

            def boosted_decisions(path: Path) -> list[tuple[str, str, str]]:
                with (path / "noise_router_decisions.csv").open() as handle:
                    return [
                        (row["utterance_id"], row["selected_system"], row["nested_threshold"])
                        for row in csv.DictReader(handle)
                        if row["configuration"] == "benefit_hist_gradient_boosting"
                    ]

            self.assertEqual(boosted_decisions(first), boosted_decisions(second))

    def test_noise_study_accepts_extracted_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, reliability, conditions = _write_noise_study_fixture(root)
            features = root / "routing_features.csv"
            with features.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "condition",
                        "utterance_id",
                        "audio_compression_ratio_mean",
                        "video_mouth_motion_mean",
                    ],
                )
                writer.writeheader()
                for condition, _, _ in conditions:
                    for index in range(5):
                        writer.writerow(
                            {
                                "condition": condition,
                                "utterance_id": f"video{index}_00000",
                                "audio_compression_ratio_mean": 1.0 + index / 10,
                                "video_mouth_motion_mean": index / 10,
                            }
                        )

            result = noise_study(
                manifest,
                reliability,
                conditions,
                root / "noise-study",
                features_path=features,
                folds=5,
                inner_folds=2,
                seed=1,
            )

            self.assertIn(result["best_configuration"], {
                "whisper_confidence",
                "audio_quality",
                "combined_noise",
                "whisper_enriched",
                "multimodal_enriched",
                "benefit_ridge",
                "benefit_hist_gradient_boosting",
                "multimodal_enriched_hist_gradient_boosting",
            })
            study = json.loads(
                (root / "noise-study" / "noise_router_study.json").read_text()
            )
            self.assertIn(
                "feature_metadata",
                study["configurations"]["multimodal_enriched"],
            )


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
        "retinaface_confidence_mean,retinaface_confidence_p10,"
        "mouth_landmark_confidence_mean,mouth_landmark_confidence_p10,"
        "multiple_face_rate\n"
        "u1,success,false,0.9,0.9,0.8,0.8,0.7,0.0\n"
        "u2,success,false,0.2,0.8,0.7,0.7,0.6,0.2\n"
        "u3,fallback_full_frame,true,0.0,,,,,\n"
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


def _write_learned_fixture(root: Path) -> dict[str, Path]:
    manifest_rows = ["utterance_id,audio_path,video_path,reference,language"]
    reliability_rows = [
        "utterance_id,roi_status,fallback_used,face_detection_rate,"
        "retinaface_confidence_mean,retinaface_confidence_p10,"
        "mouth_landmark_confidence_mean,mouth_landmark_confidence_p10,"
        "multiple_face_rate"
    ]
    audio_rows = []
    av_rows = []
    for index in range(5):
        for utterance_index, is_av_win in ((0, True), (1, False)):
            utterance_id = f"video{index}_{utterance_index:05d}"
            manifest_rows.append(
                f"{utterance_id},audio/{utterance_id}.wav,raw/{utterance_id}.mp4,hello world,en"
            )
            reliability_rows.append(
                f"{utterance_id},success,false,1.0,0.9,0.9,0.8,0.8,0.0"
            )
            audio_rows.append(
                {
                    "utterance_id": utterance_id,
                    "prediction": "wrong words" if is_av_win else "hello world",
                    "duration": 2.0,
                    "segments": [
                        {
                            "avg_logprob": -1.0 if is_av_win else -0.1,
                            "no_speech_prob": 0.1,
                        }
                    ],
                }
            )
            av_rows.append(
                {
                    "utterance_id": utterance_id,
                    "prediction": "hello world" if is_av_win else "wrong words",
                }
            )
    manifest = root / "manifest.csv"
    manifest.write_text("\n".join(manifest_rows) + "\n")
    roi_manifest = root / "roi_manifest.csv"
    roi_manifest.write_text("\n".join(manifest_rows) + "\n")
    reliability = root / "visual_reliability.csv"
    reliability.write_text("\n".join(reliability_rows) + "\n")
    audio = root / "audio.jsonl"
    av = root / "av.jsonl"
    _write_jsonl(audio, audio_rows)
    _write_jsonl(av, av_rows)
    return {
        "manifest": manifest,
        "roi_manifest": roi_manifest,
        "reliability": reliability,
        "audio": audio,
        "av": av,
    }


def _write_noise_study_fixture(
    root: Path,
) -> tuple[Path, Path, list[tuple[str, Path, Path]]]:
    manifest_rows = ["utterance_id,audio_path,video_path,reference,language"]
    reliability_rows = [
        "utterance_id,roi_status,fallback_used,mouth_landmark_confidence_mean,mouth_landmark_confidence_p10"
    ]
    for index in range(5):
        utterance_id = f"video{index}_00000"
        manifest_rows.append(f"{utterance_id},audio.wav,video.mp4,hello world,en")
        reliability_rows.append(f"{utterance_id},success,false,0.8,0.7")
    manifest = root / "manifest.csv"
    reliability = root / "visual_reliability.csv"
    manifest.write_text("\n".join(manifest_rows) + "\n")
    reliability.write_text("\n".join(reliability_rows) + "\n")

    conditions = []
    for condition, amplitude in (("clean", 0.1), ("demand_-5db", 0.7)):
        audio_path = root / f"{condition}_audio.jsonl"
        av_path = root / f"{condition}_av.jsonl"
        waveform_path = root / f"{condition}.wav"
        _write_test_wav(waveform_path, amplitude)
        audio_rows = []
        av_rows = []
        for index in range(5):
            utterance_id = f"video{index}_00000"
            av_win = (index + (condition != "clean")) % 2 == 0
            audio_rows.append(
                {
                    "utterance_id": utterance_id,
                    "prediction": "wrong words" if av_win else "hello world",
                    "used_audio_path": str(waveform_path),
                    "duration": 1.0,
                    "segments": [{"avg_logprob": -1.0 if av_win else -0.1, "no_speech_prob": 0.1}],
                }
            )
            av_rows.append(
                {"utterance_id": utterance_id, "prediction": "hello world" if av_win else "wrong words"}
            )
        _write_jsonl(audio_path, audio_rows)
        _write_jsonl(av_path, av_rows)
        conditions.append((condition, audio_path, av_path))
    return manifest, reliability, conditions


def _write_test_wav(path: Path, amplitude: float) -> None:
    samples = [int(32767 * amplitude), -int(32767 * amplitude)] * 80
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(struct.pack("<" + "h" * len(samples), *samples))


def _write_policy(root: Path, policy_type: str) -> Path:
    policy = root / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "version": 1,
                "policy_type": policy_type,
                "feature": None,
                "threshold": None,
                "audio_system": "audio",
                "av_system": "audio_visual",
            }
        )
    )
    return policy


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _manifest_ids(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return [row["utterance_id"] for row in csv.DictReader(handle)]


if __name__ == "__main__":
    unittest.main()
