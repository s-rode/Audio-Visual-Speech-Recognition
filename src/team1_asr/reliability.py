from __future__ import annotations

import argparse
import csv
import copy
import json
import math
from pathlib import Path
import shutil
import random
import re
import statistics
import unicodedata
import wave

import numpy as np
import soundfile as sf
import pickle
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from team1_asr.evaluation import compute_metrics
from team1_asr.router_features import extract_features as extract_router_features
from team1_asr.visual_reliability import (
    is_real_mouth_roi,
    optional_float,
)


POLICY_FEATURES = [
    ("face_detection_rate", 1, "higher", "visual"),
    ("retinaface_confidence_mean", 2, "higher", "visual"),
    ("retinaface_confidence_p10", 3, "higher", "visual"),
    ("mouth_landmark_confidence_mean", 4, "higher", "visual"),
    ("mouth_landmark_confidence_p10", 5, "higher", "visual"),
    ("multiple_face_rate", 6, "lower", "visual"),
]
AUDIO_UNCERTAINTY_FEATURES = [
    ("audio_avg_logprob_mean", 10, "lower"),
    ("audio_avg_logprob_min", 11, "lower"),
    ("audio_no_speech_prob_max", 12, "higher"),
    ("audio_prediction_words_per_second", 13, "lower"),
    ("audio_prediction_words_per_second", 14, "higher"),
    ("audio_prediction_word_count", 15, "lower"),
    ("audio_prediction_word_count", 16, "higher"),
    ("audio_duration", 17, "lower"),
    ("audio_duration", 18, "higher"),
]
LEARNED_FEATURES = [
    *(item[0] for item in POLICY_FEATURES),
    "audio_avg_logprob_mean",
    "audio_avg_logprob_min",
    "audio_no_speech_prob_max",
    "audio_prediction_word_count",
    "audio_duration",
    "audio_prediction_words_per_second",
]
COMPACT_LEARNED_FEATURE_SETS = {
    "audio_compact": [
        "audio_avg_logprob_mean",
        "audio_no_speech_prob_max",
        "audio_prediction_word_count",
        "audio_prediction_words_per_second",
    ],
    "visual_compact": [
        "mouth_landmark_confidence_mean",
        "mouth_landmark_confidence_p10",
    ],
    "combined_compact": [
        "mouth_landmark_confidence_mean",
        "mouth_landmark_confidence_p10",
        "audio_avg_logprob_mean",
        "audio_no_speech_prob_max",
        "audio_prediction_word_count",
        "audio_prediction_words_per_second",
    ],
}
NOISE_ROUTER_FEATURE_SETS = {
    "whisper_confidence": COMPACT_LEARNED_FEATURE_SETS["audio_compact"],
    "audio_quality": [
        "audio_rms_dbfs",
        "audio_peak_dbfs",
        "audio_crest_factor_db",
        "audio_clipping_fraction",
        "audio_spectral_flatness",
    ],
    "combined_noise": [
        *COMPACT_LEARNED_FEATURE_SETS["audio_compact"],
        "audio_rms_dbfs",
        "audio_peak_dbfs",
        "audio_crest_factor_db",
        "audio_clipping_fraction",
        "audio_spectral_flatness",
        "mouth_landmark_confidence_mean",
        "mouth_landmark_confidence_p10",
    ],
}
NOISE_BENEFIT_ROUTER_CONFIGS = (
    {
        "name": "benefit_ridge",
        "model": "ridge",
        "features": NOISE_ROUTER_FEATURE_SETS["combined_noise"],
    },
    {
        "name": "benefit_hist_gradient_boosting",
        "model": "hist_gradient_boosting",
        "features": NOISE_ROUTER_FEATURE_SETS["combined_noise"],
    },
)
ENRICHED_WHISPER_FEATURES = [
    *COMPACT_LEARNED_FEATURE_SETS["audio_compact"],
    "audio_compression_ratio_mean",
    "audio_compression_ratio_max",
    "audio_segment_count",
    "audio_avg_logprob_std",
    "audio_temperature_mean",
]
ENRICHED_MULTIMODAL_FEATURES = [
    *ENRICHED_WHISPER_FEATURES,
    "audio_rms_p10_dbfs",
    "audio_rms_p50_dbfs",
    "audio_rms_p90_dbfs",
    "audio_rms_dynamic_range_db",
    "audio_silence_fraction",
    "audio_zero_crossing_rate",
    "audio_spectral_centroid_hz",
    "audio_spectral_bandwidth_hz",
    "audio_spectral_rolloff_hz",
    "audio_spectral_flatness",
    "audio_high_frequency_energy_ratio",
    "video_mouth_motion_mean",
    "video_mouth_motion_p10",
    "video_mouth_motion_p90",
    "video_still_frame_rate",
    "video_luminance_mean",
    "video_luminance_std",
    "video_contrast_mean",
    "video_sharpness_mean",
    "video_roi_frame_rate",
    "audio_video_sync_correlation",
    "audio_video_sync_lag_ms",
    "mouth_landmark_confidence_mean",
    "mouth_landmark_confidence_p10",
]
CLEAN_ROUTER_CONFIGS = (
    {
        "name": "logistic_core",
        "model": "logistic",
        "features": NOISE_ROUTER_FEATURE_SETS["combined_noise"],
    },
    {
        "name": "logistic_enriched",
        "model": "logistic",
        "features": ENRICHED_MULTIMODAL_FEATURES,
    },
    {
        "name": "benefit_ridge_core",
        "model": "ridge",
        "features": NOISE_ROUTER_FEATURE_SETS["combined_noise"],
    },
    {
        "name": "benefit_ridge_enriched",
        "model": "ridge",
        "features": ENRICHED_MULTIMODAL_FEATURES,
    },
    {
        "name": "benefit_hist_gradient_boosting_enriched",
        "model": "hist_gradient_boosting",
        "features": ENRICHED_MULTIMODAL_FEATURES,
    },
    {
        "name": "benefit_mlp_enriched",
        "model": "mlp",
        "features": ENRICHED_MULTIMODAL_FEATURES,
    },
)
MLP_HIDDEN_LAYER_SIZES = (64, 32)
MLP_ALPHA = 1e-3
MLP_BATCH_SIZE = 256
MLP_LEARNING_RATE = 1e-3
MLP_MAX_EPOCHS = 150
MLP_PATIENCE = 12
MLP_IMPROVEMENT_TOLERANCE = 1e-4
NOISE_STUDY_BOOTSTRAP_SAMPLES = 1000
STUDY_L2_STRENGTHS = (0.3, 1.0, 3.0)
STUDY_CLASS_WEIGHTS = ("unweighted",)
LEARNED_POLICY_TYPE = "learned_logistic"
LEARNED_POLICY_VERSION = 1
BENEFIT_POLICY_TYPE = "benefit_regression"
BENEFIT_POLICY_VERSION = 1
DEFAULT_AUDIO_SYSTEM = "audio"
DEFAULT_AV_SYSTEM = "audio_visual"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate and evaluate visual-reliability ASR routing."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate")
    _add_evaluation_inputs(calibrate, require_policy=False)
    calibrate.add_argument("--audio-system-name", default=DEFAULT_AUDIO_SYSTEM)
    calibrate.add_argument("--av-system-name", default=DEFAULT_AV_SYSTEM)

    learn = subparsers.add_parser("learn")
    _add_evaluation_inputs(learn, require_policy=False)
    learn.add_argument("--audio-system-name", default=DEFAULT_AUDIO_SYSTEM)
    learn.add_argument("--av-system-name", default=DEFAULT_AV_SYSTEM)
    learn.add_argument("--folds", type=int, default=5)
    learn.add_argument("--seed", type=int, default=0)

    study = subparsers.add_parser("study")
    _add_evaluation_inputs(study, require_policy=False)
    study.add_argument("--audio-system-name", default=DEFAULT_AUDIO_SYSTEM)
    study.add_argument("--av-system-name", default=DEFAULT_AV_SYSTEM)
    study.add_argument("--folds", type=int, default=5)
    study.add_argument("--inner-folds", type=int, default=4)
    study.add_argument("--seed", type=int, default=0)

    train_dev = subparsers.add_parser("train-dev-study")
    train_dev.add_argument("--train-manifest", required=True)
    train_dev.add_argument("--train-reliability", required=True)
    train_dev.add_argument("--train-audio-transcripts", required=True)
    train_dev.add_argument("--train-av-transcripts", required=True)
    train_dev.add_argument("--dev-manifest", required=True)
    train_dev.add_argument("--dev-reliability", required=True)
    train_dev.add_argument("--dev-audio-transcripts", required=True)
    train_dev.add_argument("--dev-av-transcripts", required=True)
    train_dev.add_argument("--train-features")
    train_dev.add_argument("--dev-features")
    train_dev.add_argument("--output-dir", required=True)
    train_dev.add_argument("--audio-system-name", default=DEFAULT_AUDIO_SYSTEM)
    train_dev.add_argument("--av-system-name", default=DEFAULT_AV_SYSTEM)
    train_dev.add_argument("--folds", type=int, default=5)
    train_dev.add_argument("--inner-folds", type=int, default=4)
    train_dev.add_argument("--seed", type=int, default=0)

    train_dev_candidate = subparsers.add_parser("train-dev-candidate")
    _add_train_dev_study_inputs(train_dev_candidate)
    train_dev_candidate.add_argument("--configuration", required=True)

    train_dev_oof = subparsers.add_parser("train-dev-oof-fold")
    _add_train_dev_study_inputs(train_dev_oof)
    train_dev_oof.add_argument("--configuration", required=True)
    train_dev_oof.add_argument("--fold-index", type=int, required=True)

    finalize_train_dev = subparsers.add_parser("finalize-train-dev-study")
    finalize_train_dev.add_argument("--candidate-dir", required=True)
    finalize_train_dev.add_argument("--oof-dir", required=True)
    finalize_train_dev.add_argument("--output-dir", required=True)
    finalize_train_dev.add_argument("--folds", type=int, default=5)
    finalize_train_dev.add_argument("--inner-folds", type=int, default=4)
    finalize_train_dev.add_argument("--seed", type=int, default=0)

    select_train_dev = subparsers.add_parser("select-train-dev-candidate")
    select_train_dev.add_argument("--candidate-dir", required=True)
    select_train_dev.add_argument("--output-dir", required=True)

    noise_study = subparsers.add_parser("noise-study")
    noise_study.add_argument("--manifest", required=True)
    noise_study.add_argument("--reliability", required=True)
    noise_study.add_argument(
        "--condition",
        action="append",
        nargs=3,
        metavar=("NAME", "AUDIO_TRANSCRIPTS", "AV_TRANSCRIPTS"),
        required=True,
        help="Repeat for clean, demand_10db, demand_0db, and demand_-5db.",
    )
    noise_study.add_argument("--output-dir", required=True)
    noise_study.add_argument(
        "--features",
        help="Optional routing_features.csv written by extract-features.",
    )
    noise_study.add_argument("--audio-system-name", default=DEFAULT_AUDIO_SYSTEM)
    noise_study.add_argument("--av-system-name", default=DEFAULT_AV_SYSTEM)
    noise_study.add_argument("--folds", type=int, default=5)
    noise_study.add_argument("--inner-folds", type=int, default=4)
    noise_study.add_argument("--seed", type=int, default=0)

    extract_features = subparsers.add_parser("extract-features")
    extract_features.add_argument("--roi-manifest", required=True)
    extract_features.add_argument("--reliability", required=True)
    extract_features.add_argument(
        "--condition",
        action="append",
        nargs=2,
        metavar=("NAME", "WHISPER_TRANSCRIPTS"),
        required=True,
    )
    extract_features.add_argument("--output-dir", required=True)

    split = subparsers.add_parser("split")
    split.add_argument("--source-manifest", required=True)
    split.add_argument("--roi-manifest", required=True)
    split.add_argument("--reliability", required=True)
    split.add_argument("--policy", required=True)
    split.add_argument(
        "--audio-transcripts",
        help="Complete audio-system transcripts; required for learned policies.",
    )
    split.add_argument("--output-dir", required=True)

    evaluate = subparsers.add_parser("evaluate")
    _add_evaluation_inputs(evaluate, require_policy=True)

    diagnose = subparsers.add_parser("diagnose-failures")
    diagnose.add_argument("--manifest", required=True)
    diagnose.add_argument("--reliability", required=True)
    diagnose.add_argument("--audio-visual-transcripts", "--av-transcripts", dest="av_transcripts", required=True)
    diagnose.add_argument("--output-dir", required=True)
    return parser


def _add_train_dev_study_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--train-reliability", required=True)
    parser.add_argument("--train-audio-transcripts", required=True)
    parser.add_argument("--train-av-transcripts", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--dev-reliability", required=True)
    parser.add_argument("--dev-audio-transcripts", required=True)
    parser.add_argument("--dev-av-transcripts", required=True)
    parser.add_argument("--train-features")
    parser.add_argument("--dev-features")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio-system-name", default=DEFAULT_AUDIO_SYSTEM)
    parser.add_argument("--av-system-name", default=DEFAULT_AV_SYSTEM)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)


def _add_evaluation_inputs(
    parser: argparse.ArgumentParser,
    *,
    require_policy: bool,
) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reliability", required=True)
    parser.add_argument("--audio-transcripts", required=True)
    parser.add_argument(
        "--audio-visual-transcripts",
        "--av-transcripts",
        dest="av_transcripts",
        required=True,
    )
    if require_policy:
        parser.add_argument("--policy", required=True)
    parser.add_argument("--features")
    parser.add_argument("--output-dir", required=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "calibrate":
        calibrate(
            manifest_path=Path(args.manifest),
            reliability_path=Path(args.reliability),
            audio_transcripts_path=Path(args.audio_transcripts),
            av_transcripts_path=Path(args.av_transcripts),
            output_dir=Path(args.output_dir),
            audio_system_name=args.audio_system_name,
            av_system_name=args.av_system_name,
        )
    elif args.command == "learn":
        learn(
            manifest_path=Path(args.manifest),
            reliability_path=Path(args.reliability),
            audio_transcripts_path=Path(args.audio_transcripts),
            av_transcripts_path=Path(args.av_transcripts),
            output_dir=Path(args.output_dir),
            audio_system_name=args.audio_system_name,
            av_system_name=args.av_system_name,
            folds=args.folds,
            seed=args.seed,
        )
    elif args.command == "study":
        study(
            manifest_path=Path(args.manifest),
            reliability_path=Path(args.reliability),
            audio_transcripts_path=Path(args.audio_transcripts),
            av_transcripts_path=Path(args.av_transcripts),
            output_dir=Path(args.output_dir),
            audio_system_name=args.audio_system_name,
            av_system_name=args.av_system_name,
            folds=args.folds,
            inner_folds=args.inner_folds,
            seed=args.seed,
        )
    elif args.command == "noise-study":
        noise_study(
            manifest_path=Path(args.manifest),
            reliability_path=Path(args.reliability),
            conditions=[(name, Path(audio), Path(av)) for name, audio, av in args.condition],
            output_dir=Path(args.output_dir),
            features_path=Path(args.features) if args.features else None,
            audio_system_name=args.audio_system_name,
            av_system_name=args.av_system_name,
            folds=args.folds,
            inner_folds=args.inner_folds,
            seed=args.seed,
        )
    elif args.command == "train-dev-study":
        train_dev_study(
            train_manifest_path=Path(args.train_manifest),
            train_reliability_path=Path(args.train_reliability),
            train_audio_transcripts_path=Path(args.train_audio_transcripts),
            train_av_transcripts_path=Path(args.train_av_transcripts),
            dev_manifest_path=Path(args.dev_manifest),
            dev_reliability_path=Path(args.dev_reliability),
            dev_audio_transcripts_path=Path(args.dev_audio_transcripts),
            dev_av_transcripts_path=Path(args.dev_av_transcripts),
            output_dir=Path(args.output_dir),
            train_features_path=Path(args.train_features) if args.train_features else None,
            dev_features_path=Path(args.dev_features) if args.dev_features else None,
            audio_system_name=args.audio_system_name,
            av_system_name=args.av_system_name,
            folds=args.folds,
            inner_folds=args.inner_folds,
            seed=args.seed,
        )
    elif args.command == "train-dev-candidate":
        train_dev_candidate(
            **_train_dev_study_paths_from_args(args),
            configuration=args.configuration,
        )
    elif args.command == "train-dev-oof-fold":
        train_dev_oof_fold(
            **_train_dev_study_paths_from_args(args),
            configuration=args.configuration,
            fold_index=args.fold_index,
        )
    elif args.command == "finalize-train-dev-study":
        finalize_train_dev_study(
            candidate_dir=Path(args.candidate_dir),
            oof_dir=Path(args.oof_dir),
            output_dir=Path(args.output_dir),
            folds=args.folds,
            inner_folds=args.inner_folds,
            seed=args.seed,
        )
    elif args.command == "select-train-dev-candidate":
        select_train_dev_candidate(
            candidate_dir=Path(args.candidate_dir),
            output_dir=Path(args.output_dir),
        )
    elif args.command == "extract-features":
        paths = extract_router_features(
            roi_manifest_path=Path(args.roi_manifest),
            reliability_path=Path(args.reliability),
            conditions=[(name, Path(transcripts)) for name, transcripts in args.condition],
            output_dir=Path(args.output_dir),
        )
        print(f"Wrote routing features: {paths['features']}")
        print(f"Wrote feature summary: {paths['summary']}")
        print(f"Wrote feature summary JSON: {paths['summary_json']}")
    elif args.command == "split":
        split_manifests(
            source_manifest_path=Path(args.source_manifest),
            roi_manifest_path=Path(args.roi_manifest),
            reliability_path=Path(args.reliability),
            policy_path=Path(args.policy),
            output_dir=Path(args.output_dir),
            audio_transcripts_path=(
                Path(args.audio_transcripts) if args.audio_transcripts else None
            ),
        )
    elif args.command == "evaluate":
        evaluate(
            manifest_path=Path(args.manifest),
            reliability_path=Path(args.reliability),
            audio_transcripts_path=Path(args.audio_transcripts),
            av_transcripts_path=Path(args.av_transcripts),
            policy_path=Path(args.policy),
            output_dir=Path(args.output_dir),
            features_path=Path(args.features) if args.features else None,
        )
    else:
        diagnose_failures(
            manifest_path=Path(args.manifest),
            reliability_path=Path(args.reliability),
            av_transcripts_path=Path(args.av_transcripts),
            output_dir=Path(args.output_dir),
        )


def _train_dev_study_paths_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "train_manifest_path": Path(args.train_manifest),
        "train_reliability_path": Path(args.train_reliability),
        "train_audio_transcripts_path": Path(args.train_audio_transcripts),
        "train_av_transcripts_path": Path(args.train_av_transcripts),
        "dev_manifest_path": Path(args.dev_manifest),
        "dev_reliability_path": Path(args.dev_reliability),
        "dev_audio_transcripts_path": Path(args.dev_audio_transcripts),
        "dev_av_transcripts_path": Path(args.dev_av_transcripts),
        "output_dir": Path(args.output_dir),
        "train_features_path": Path(args.train_features) if args.train_features else None,
        "dev_features_path": Path(args.dev_features) if args.dev_features else None,
        "audio_system_name": args.audio_system_name,
        "av_system_name": args.av_system_name,
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
    }


def calibrate(
    manifest_path: Path,
    reliability_path: Path,
    audio_transcripts_path: Path,
    av_transcripts_path: Path,
    output_dir: Path,
    audio_system_name: str = DEFAULT_AUDIO_SYSTEM,
    av_system_name: str = DEFAULT_AV_SYSTEM,
) -> dict[str, object]:
    audio_system_name = _require_system_name(audio_system_name, "audio")
    av_system_name = _require_system_name(av_system_name, "audio-visual")
    if audio_system_name == av_system_name:
        raise ValueError("Audio and audio-visual system names must be different.")
    inputs = _load_evaluation_inputs(
        manifest_path,
        reliability_path,
        audio_transcripts_path,
        av_transcripts_path,
        audio_system_name,
    )
    sweep: list[dict[str, object]] = []
    candidates = _threshold_candidates(inputs)
    selected_policy, scored = _select_threshold_policy(inputs, candidates)
    for _, policy, result in scored:
        threshold = policy.get("threshold")
        sweep.append(
            {
                "policy_type": policy["policy_type"],
                "feature_family": policy.get("feature_family") or "",
                "feature": policy.get("feature") or "",
                "direction": policy.get("direction") or "",
                "threshold": "" if threshold is None else threshold,
                "word_errors": result["total_word_errors"],
                "reference_words": result["total_reference_words"],
                "wer": result["word_error_rate"],
                "audio_visual_selected": result["audio_visual_selected"],
            }
        )
    selected_policy.update(
        {
            "version": 1,
            "audio_system": audio_system_name,
            "av_system": av_system_name,
            "calibrated_on": str(manifest_path),
            "calibrated_utterances": len(inputs["rows"]),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "policy.json", selected_policy)
    _write_csv(output_dir / "threshold_sweep.csv", sweep)
    _write_reports(inputs, selected_policy, output_dir)
    return selected_policy


def _threshold_candidates(inputs: dict[str, object]) -> list[dict[str, object]]:
    candidates = [_policy("roi_success", None, None, priority=0, family="visual")]
    for feature, priority, direction, family in POLICY_FEATURES:
        values = sorted(
            {
                value
                for row in inputs["reliability"].values()
                if is_real_mouth_roi(row)
                for value in [optional_float(row.get(feature))]
                if value is not None
            }
        )
        for threshold in values:
            candidates.append(
                _policy(
                    "threshold",
                    feature,
                    threshold,
                    priority=priority,
                    direction=direction,
                    family=family,
                )
            )
    for feature, priority, direction in AUDIO_UNCERTAINTY_FEATURES:
        values = sorted(
            {
                value
                for utterance_id, row in inputs["reliability"].items()
                if is_real_mouth_roi(row)
                for value in [
                    optional_float(
                        inputs["audio"].get(utterance_id, {}).get(feature)
                    )
                ]
                if value is not None
            }
        )
        for threshold in values:
            candidates.append(
                _policy(
                    "threshold",
                    feature,
                    threshold,
                    priority=priority,
                    direction=direction,
                    family="audio_uncertainty",
                )
            )
    candidates.append(_policy("audio_only", None, None, priority=99, family="audio"))
    return candidates


def _select_threshold_policy(
    inputs: dict[str, object],
    candidates: list[dict[str, object]] | None = None,
) -> tuple[
    dict[str, object],
    list[tuple[tuple[float, int, float], dict[str, object], dict[str, object]]],
]:
    candidates = candidates or _threshold_candidates(inputs)
    precomputed = _precompute_policy_rows(inputs)

    scored: list[
        tuple[tuple[float, int, float], dict[str, object], dict[str, object]]
    ] = []
    threshold_results = _score_threshold_candidates(candidates, precomputed)
    for policy in candidates:
        result = threshold_results.get(id(policy))
        if result is None:
            result = _score_policy(inputs, policy, precomputed=precomputed)
        threshold = policy.get("threshold")
        scored.append(
            (
                (
                    float(result["total_word_errors"]),
                    int(policy["priority"]),
                    float(threshold or 0.0),
                ),
                policy,
                result,
            )
        )
    selected_policy = dict(min(scored, key=lambda item: item[0])[1])
    selected_policy = {
        key: value for key, value in selected_policy.items() if key != "priority"
    }
    return selected_policy, scored


def learn(
    manifest_path: Path,
    reliability_path: Path,
    audio_transcripts_path: Path,
    av_transcripts_path: Path,
    output_dir: Path,
    audio_system_name: str = DEFAULT_AUDIO_SYSTEM,
    av_system_name: str = DEFAULT_AV_SYSTEM,
    folds: int = 5,
    seed: int = 0,
) -> dict[str, object]:
    """Fit a small router and report grouped out-of-fold development results."""
    audio_system_name = _require_system_name(audio_system_name, "audio")
    av_system_name = _require_system_name(av_system_name, "audio-visual")
    if audio_system_name == av_system_name:
        raise ValueError("Audio and audio-visual system names must be different.")
    if folds < 2:
        raise ValueError("--folds must be at least 2.")
    inputs = _load_evaluation_inputs(
        manifest_path,
        reliability_path,
        audio_transcripts_path,
        av_transcripts_path,
        audio_system_name,
    )
    fold_ids = _grouped_folds(list(inputs["rows"]), folds=folds, seed=seed)
    if len(fold_ids) < 2:
        raise ValueError("Need at least two source-video groups to learn a router.")

    cross_validation_rows: list[dict[str, object]] = []
    fold_reports: list[dict[str, object]] = []
    all_ids = set(inputs["rows"])
    for fold_index, validation_ids in enumerate(fold_ids):
        training_ids = all_ids - set(validation_ids)
        training_inputs = _subset_inputs(inputs, training_ids)
        validation_inputs = _subset_inputs(inputs, set(validation_ids))
        learned_policy = _fit_learned_policy(training_inputs)
        learned_policy.update(
            {
                "audio_system": audio_system_name,
                "av_system": av_system_name,
            }
        )
        threshold_policy, _ = _select_threshold_policy(training_inputs)
        threshold_policy.update(
            {
                "audio_system": audio_system_name,
                "av_system": av_system_name,
            }
        )
        learned_result = _route(validation_inputs, learned_policy)
        threshold_result = _route(validation_inputs, threshold_policy)
        threshold_decisions = {
            str(row["utterance_id"]): row for row in threshold_result["decisions"]
        }
        for row in learned_result["decisions"]:
            threshold_row = threshold_decisions[str(row["utterance_id"])]
            cross_validation_rows.append(
                {
                    "fold": fold_index,
                    "audio_system": audio_system_name,
                    "audio_visual_system": av_system_name,
                    "learned_probability": row["reliability_score"],
                    "learned_selected_system": row["selected_system"],
                    "learned_selected_prediction": row["selected_prediction"],
                    "learned_selected_word_errors": row["selected_word_errors"],
                    "threshold_policy_type": threshold_policy["policy_type"],
                    "threshold_feature": threshold_policy.get("feature") or "",
                    "threshold_value": threshold_policy.get("threshold", ""),
                    "threshold_selected_system": threshold_row["selected_system"],
                    "threshold_selected_prediction": threshold_row["selected_prediction"],
                    "threshold_selected_word_errors": threshold_row[
                        "selected_word_errors"
                    ],
                    **row,
                }
            )
        fold_reports.append(
            {
                "fold": fold_index,
                "validation_utterances": len(validation_ids),
                "source_videos": len({_source_video_id(item) for item in validation_ids}),
                "learned": _report_metrics_without_policy(learned_result["metrics"]),
                "single_threshold": _report_metrics_without_policy(
                    threshold_result["metrics"]
                ),
            }
        )

    full_policy = _fit_learned_policy(inputs)
    cross_validation_metrics = _cross_validation_metrics(cross_validation_rows)
    recommendation = _learned_router_recommendation(cross_validation_metrics)
    full_policy.update(
        {
            "version": LEARNED_POLICY_VERSION,
            "audio_system": audio_system_name,
            "av_system": av_system_name,
            "calibrated_on": str(manifest_path),
            "calibrated_utterances": len(inputs["rows"]),
            "cross_validation": {
                "folds": len(fold_ids),
                "seed": seed,
                **recommendation,
            },
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "policy.json", full_policy)
    _write_csv(output_dir / "cross_validation_decisions.csv", cross_validation_rows)
    _write_json(
        output_dir / "cross_validation_metrics.json",
        {
            "folds": fold_reports,
            "aggregate": cross_validation_metrics,
            "recommendation": recommendation,
        },
    )
    _write_learned_coefficients(output_dir / "learned_router_coefficients.csv", full_policy)
    _write_reports(inputs, full_policy, output_dir)
    return full_policy


def study(
    manifest_path: Path,
    reliability_path: Path,
    audio_transcripts_path: Path,
    av_transcripts_path: Path,
    output_dir: Path,
    audio_system_name: str = DEFAULT_AUDIO_SYSTEM,
    av_system_name: str = DEFAULT_AV_SYSTEM,
    folds: int = 5,
    inner_folds: int = 4,
    seed: int = 0,
) -> dict[str, object]:
    """Compare compact learned-router variants with nested grouped validation."""
    audio_system_name = _require_system_name(audio_system_name, "audio")
    av_system_name = _require_system_name(av_system_name, "audio-visual")
    if audio_system_name == av_system_name:
        raise ValueError("Audio and audio-visual system names must be different.")
    if folds < 2 or inner_folds < 2:
        raise ValueError("--folds and --inner-folds must both be at least 2.")
    inputs = _load_evaluation_inputs(
        manifest_path,
        reliability_path,
        audio_transcripts_path,
        av_transcripts_path,
        audio_system_name,
    )
    outer_folds = _grouped_folds(list(inputs["rows"]), folds=folds, seed=seed)
    all_ids = set(inputs["rows"])
    baseline_decisions = _outer_threshold_decisions(
        inputs,
        outer_folds,
        all_ids,
        audio_system_name,
        av_system_name,
    )
    baseline_summary = _decision_summary(baseline_decisions, av_system_name)

    configurations = [
        {
            "name": f"{feature_set_name}_l2_{str(l2_strength).replace('.', '_')}_{class_weight}",
            "feature_set": feature_set_name,
            "features": features,
            "l2_strength": l2_strength,
            "class_weight": class_weight,
        }
        for feature_set_name, features in COMPACT_LEARNED_FEATURE_SETS.items()
        for l2_strength in STUDY_L2_STRENGTHS
        for class_weight in STUDY_CLASS_WEIGHTS
    ]
    summary_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    policies: dict[str, dict[str, object]] = {}
    for config_index, config in enumerate(configurations):
        config_decisions: list[dict[str, object]] = []
        outer_thresholds: list[float] = []
        for fold_index, validation_ids in enumerate(outer_folds):
            training_inputs = _subset_inputs(inputs, all_ids - set(validation_ids))
            validation_inputs = _subset_inputs(inputs, set(validation_ids))
            threshold = _nested_probability_threshold(
                training_inputs,
                features=list(config["features"]),
                l2_strength=float(config["l2_strength"]),
                class_weight=str(config["class_weight"]),
                folds=inner_folds,
                seed=seed + config_index * 100 + fold_index,
                audio_system_name=audio_system_name,
                av_system_name=av_system_name,
            )
            policy = _fit_learned_policy(
                training_inputs,
                feature_candidates=list(config["features"]),
                l2_strength=float(config["l2_strength"]),
                class_weight=str(config["class_weight"]),
            )
            policy.update(
                {
                    "threshold": threshold,
                    "audio_system": audio_system_name,
                    "av_system": av_system_name,
                }
            )
            result = _route(validation_inputs, policy)
            outer_thresholds.append(threshold)
            for row in result["decisions"]:
                config_decisions.append(row)
                decision_rows.append(
                    {
                        "configuration": config["name"],
                        "feature_set": config["feature_set"],
                        "l2_strength": config["l2_strength"],
                        "class_weight": config["class_weight"],
                        "fold": fold_index,
                        "nested_threshold": threshold,
                        **row,
                    }
                )
        summary = _decision_summary(config_decisions, av_system_name)
        summary_row = {
            "configuration": config["name"],
            "feature_set": config["feature_set"],
            "l2_strength": config["l2_strength"],
            "class_weight": config["class_weight"],
            "nested_threshold_mean": sum(outer_thresholds) / len(outer_thresholds),
            **summary,
            "audio_wer": baseline_summary["audio_wer"],
            "single_threshold_wer": baseline_summary["routed_wer"],
            "recommended": (
                summary["routed_wer"] < baseline_summary["audio_wer"]
                and summary["routed_wer"] < baseline_summary["routed_wer"]
            ),
        }
        summary_rows.append(summary_row)
        policies[str(config["name"])] = _fit_learned_policy(
            inputs,
            feature_candidates=list(config["features"]),
            l2_strength=float(config["l2_strength"]),
            class_weight=str(config["class_weight"]),
        )

    best = min(summary_rows, key=lambda row: float(row["routed_wer"]))
    best_policy = policies[str(best["configuration"])]
    full_threshold = _nested_probability_threshold(
        inputs,
        features=list(COMPACT_LEARNED_FEATURE_SETS[str(best["feature_set"])]),
        l2_strength=float(best["l2_strength"]),
        class_weight=str(best["class_weight"]),
        folds=inner_folds,
        seed=seed + 10_000,
        audio_system_name=audio_system_name,
        av_system_name=av_system_name,
    )
    best_policy.update(
        {
            "threshold": full_threshold,
            "version": LEARNED_POLICY_VERSION,
            "audio_system": audio_system_name,
            "av_system": av_system_name,
            "calibrated_on": str(manifest_path),
            "calibrated_utterances": len(inputs["rows"]),
            "study": {
                "configuration": best["configuration"],
                "outer_folds": len(outer_folds),
                "inner_folds": inner_folds,
                "seed": seed,
                "recommended_for_test": bool(best["recommended"]),
            },
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "router_study.csv", summary_rows)
    _write_csv(output_dir / "router_study_decisions.csv", decision_rows)
    _write_json(
        output_dir / "router_study.json",
        {
            "baseline": baseline_summary,
            "configurations": summary_rows,
            "best_configuration": best,
            "recommended_for_test": bool(best["recommended"]),
        },
    )
    _write_json(output_dir / "best_policy.json", best_policy)
    return {
        "baseline": baseline_summary,
        "best_configuration": best,
        "recommended_for_test": bool(best["recommended"]),
    }


def train_dev_study(
    *,
    train_manifest_path: Path,
    train_reliability_path: Path,
    train_audio_transcripts_path: Path,
    train_av_transcripts_path: Path,
    dev_manifest_path: Path,
    dev_reliability_path: Path,
    dev_audio_transcripts_path: Path,
    dev_av_transcripts_path: Path,
    output_dir: Path,
    train_features_path: Path | None = None,
    dev_features_path: Path | None = None,
    audio_system_name: str = DEFAULT_AUDIO_SYSTEM,
    av_system_name: str = DEFAULT_AV_SYSTEM,
    folds: int = 5,
    inner_folds: int = 4,
    seed: int = 0,
) -> dict[str, object]:
    """Select a deployable router on dev after grouped train-only fitting."""
    if folds < 2 or inner_folds < 2:
        raise ValueError("--folds and --inner-folds must both be at least 2.")
    train_inputs = _load_evaluation_inputs(
        train_manifest_path, train_reliability_path, train_audio_transcripts_path,
        train_av_transcripts_path, audio_system_name, features_path=train_features_path,
    )
    dev_inputs = _load_evaluation_inputs(
        dev_manifest_path, dev_reliability_path, dev_audio_transcripts_path,
        dev_av_transcripts_path, audio_system_name, features_path=dev_features_path,
    )
    train_ids = set(train_inputs["rows"])
    outer_folds = _grouped_folds(list(train_ids), folds=folds, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_policy = _audio_only_policy(audio_system_name, av_system_name)
    dev_audio = _route(dev_inputs, audio_policy)["decisions"]
    rows: list[dict[str, object]] = []
    all_decisions: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for index, config in enumerate(CLEAN_ROUTER_CONFIGS):
        name = str(config["name"])
        model = str(config["model"])
        features = list(config["features"])
        oof = _clean_oof_decisions(
            train_inputs, outer_folds, train_ids, model=model, features=features,
            inner_folds=inner_folds, seed=seed + index * 100,
            audio_system_name=audio_system_name, av_system_name=av_system_name,
        )
        policy = _fit_clean_policy(
            train_inputs, model=model, features=features, inner_folds=inner_folds,
            seed=seed + index * 1000, audio_system_name=audio_system_name,
            av_system_name=av_system_name,
        )
        dev_result = _route(dev_inputs, policy)
        dev_summary = _decision_summary(dev_result["decisions"], av_system_name)
        comparison = _bootstrap_router_comparison(
            dev_audio, dev_result["decisions"], seed=seed + index,
            conditions=["clean"],
        )["overall"]
        supported = float(comparison["ci_95_low_pp"]) > 0.0
        row = {
            "configuration": name,
            "model": model,
            "features": features,
            "train_oof": _decision_summary(oof, av_system_name),
            "dev": dev_summary,
            "dev_vs_whisper": comparison,
            "supported_over_whisper": supported,
            "policy": _policy_metadata(policy),
        }
        rows.append(row)
        candidates.append({**row, "policy_object": policy, "dev_decisions": dev_result["decisions"]})
        all_decisions.extend({"split": "train_oof", "configuration": name, **item} for item in oof)
        all_decisions.extend({"split": "dev", "configuration": name, **item} for item in dev_result["decisions"])

    supported = [item for item in candidates if item["supported_over_whisper"]]
    selected_pool = supported or candidates
    best_wer = min(float(item["dev"]["routed_wer"]) for item in selected_pool)
    tied = [
        item for item in selected_pool
        if float(item["dev"]["routed_wer"]) <= best_wer + 0.001
    ]
    selected = min(tied, key=lambda item: float(item["dev"]["routed_to_audio_visual_rate"]))
    policy = selected["policy_object"]
    if selected["model"] == "hist_gradient_boosting":
        policy = _fit_clean_policy(
            train_inputs,
            model="hist_gradient_boosting",
            features=list(selected["features"]),
            inner_folds=inner_folds,
            seed=seed + CLEAN_ROUTER_CONFIGS.index(
                next(config for config in CLEAN_ROUTER_CONFIGS if config["name"] == selected["configuration"])
            ) * 1000,
            audio_system_name=audio_system_name,
            av_system_name=av_system_name,
            compute_feature_importances=True,
        )
    with (output_dir / "selected_policy.pkl").open("wb") as handle:
        pickle.dump(policy, handle)
    _write_json(output_dir / "selected_policy.json", _policy_metadata(policy))
    _write_csv(output_dir / "router_train_dev_decisions.csv", all_decisions)
    _write_json(
        output_dir / "router_train_dev_study.json",
        {
            "train_utterances": len(train_inputs["rows"]),
            "dev_utterances": len(dev_inputs["rows"]),
            "outer_folds": len(outer_folds),
            "inner_folds": inner_folds,
            "seed": seed,
            "features": {"train": str(train_features_path or ""), "dev": str(dev_features_path or "")},
            "whisper_dev": _decision_summary(dev_audio, av_system_name),
            "configurations": rows,
            "selected_configuration": selected["configuration"],
            "selected_supported_over_whisper": selected["supported_over_whisper"],
            "selection_tie_band_wer": 0.001,
        },
    )
    return {"selected_configuration": selected["configuration"], "supported": selected["supported_over_whisper"]}


def _clean_router_config(name: str) -> dict[str, object]:
    for config in CLEAN_ROUTER_CONFIGS:
        if config["name"] == name:
            return config
    choices = ", ".join(str(config["name"]) for config in CLEAN_ROUTER_CONFIGS)
    raise ValueError(f"Unknown clean router configuration {name!r}; choose one of: {choices}.")


def _load_train_dev_study_inputs(
    *,
    train_manifest_path: Path,
    train_reliability_path: Path,
    train_audio_transcripts_path: Path,
    train_av_transcripts_path: Path,
    dev_manifest_path: Path,
    dev_reliability_path: Path,
    dev_audio_transcripts_path: Path,
    dev_av_transcripts_path: Path,
    train_features_path: Path | None,
    dev_features_path: Path | None,
    audio_system_name: str,
    av_system_name: str,
    folds: int,
    inner_folds: int,
) -> tuple[dict[str, object], dict[str, object]]:
    if folds < 2 or inner_folds < 2:
        raise ValueError("--folds and --inner-folds must both be at least 2.")
    audio_system_name = _require_system_name(audio_system_name, "audio")
    av_system_name = _require_system_name(av_system_name, "audio-visual")
    if audio_system_name == av_system_name:
        raise ValueError("Audio and audio-visual system names must be different.")
    print("[router] loading train inputs", flush=True)
    train_inputs = _load_evaluation_inputs(
        train_manifest_path, train_reliability_path, train_audio_transcripts_path,
        train_av_transcripts_path, audio_system_name, features_path=train_features_path,
    )
    print("[router] loading dev inputs", flush=True)
    dev_inputs = _load_evaluation_inputs(
        dev_manifest_path, dev_reliability_path, dev_audio_transcripts_path,
        dev_av_transcripts_path, audio_system_name, features_path=dev_features_path,
    )
    return train_inputs, dev_inputs


def train_dev_candidate(
    *,
    train_manifest_path: Path,
    train_reliability_path: Path,
    train_audio_transcripts_path: Path,
    train_av_transcripts_path: Path,
    dev_manifest_path: Path,
    dev_reliability_path: Path,
    dev_audio_transcripts_path: Path,
    dev_av_transcripts_path: Path,
    output_dir: Path,
    configuration: str,
    train_features_path: Path | None = None,
    dev_features_path: Path | None = None,
    audio_system_name: str = DEFAULT_AUDIO_SYSTEM,
    av_system_name: str = DEFAULT_AV_SYSTEM,
    folds: int = 5,
    inner_folds: int = 4,
    seed: int = 0,
) -> dict[str, object]:
    """Fit one candidate on train, evaluate it once on dev, and checkpoint it."""
    config = _clean_router_config(configuration)
    train_inputs, dev_inputs = _load_train_dev_study_inputs(
        train_manifest_path=train_manifest_path,
        train_reliability_path=train_reliability_path,
        train_audio_transcripts_path=train_audio_transcripts_path,
        train_av_transcripts_path=train_av_transcripts_path,
        dev_manifest_path=dev_manifest_path,
        dev_reliability_path=dev_reliability_path,
        dev_audio_transcripts_path=dev_audio_transcripts_path,
        dev_av_transcripts_path=dev_av_transcripts_path,
        train_features_path=train_features_path,
        dev_features_path=dev_features_path,
        audio_system_name=audio_system_name,
        av_system_name=av_system_name,
        folds=folds,
        inner_folds=inner_folds,
    )
    config_index = list(CLEAN_ROUTER_CONFIGS).index(config)
    candidate_dir = output_dir / configuration
    candidate_dir.mkdir(parents=True, exist_ok=True)
    print(f"[router] candidate {configuration}: fitting train policy", flush=True)
    policy = _fit_clean_policy(
        train_inputs,
        model=str(config["model"]),
        features=list(config["features"]),
        inner_folds=inner_folds,
        seed=seed + config_index * 1000,
        audio_system_name=audio_system_name,
        av_system_name=av_system_name,
        compute_feature_importances=str(config["model"]) in {
            "hist_gradient_boosting", "mlp"
        },
    )
    print(f"[router] candidate {configuration}: evaluating dev", flush=True)
    dev_result = _route(dev_inputs, policy)
    dev_audio = _route(
        dev_inputs, _audio_only_policy(audio_system_name, av_system_name)
    )["decisions"]
    comparison = _bootstrap_router_comparison(
        dev_audio, dev_result["decisions"], seed=seed + config_index,
        conditions=["clean"],
    )["overall"]
    result = {
        "configuration": configuration,
        "model": config["model"],
        "features": list(config["features"]),
        "train_utterances": len(train_inputs["rows"]),
        "dev_utterances": len(dev_inputs["rows"]),
        "outer_folds": folds,
        "inner_folds": inner_folds,
        "seed": seed,
        "whisper_dev": _decision_summary(dev_audio, av_system_name),
        "dev": _decision_summary(dev_result["decisions"], av_system_name),
        "dev_vs_whisper": comparison,
        "supported_over_whisper": float(comparison["ci_95_low_pp"]) > 0.0,
        "policy": _policy_metadata(policy),
    }
    with (candidate_dir / "policy.pkl").open("wb") as handle:
        pickle.dump(policy, handle)
    _write_csv(candidate_dir / "dev_decisions.csv", dev_result["decisions"])
    _write_json(candidate_dir / "candidate.json", result)
    print(f"[router] candidate {configuration}: checkpoint written", flush=True)
    return result


def train_dev_oof_fold(
    *,
    train_manifest_path: Path,
    train_reliability_path: Path,
    train_audio_transcripts_path: Path,
    train_av_transcripts_path: Path,
    dev_manifest_path: Path,
    dev_reliability_path: Path,
    dev_audio_transcripts_path: Path,
    dev_av_transcripts_path: Path,
    output_dir: Path,
    configuration: str,
    fold_index: int,
    train_features_path: Path | None = None,
    dev_features_path: Path | None = None,
    audio_system_name: str = DEFAULT_AUDIO_SYSTEM,
    av_system_name: str = DEFAULT_AV_SYSTEM,
    folds: int = 5,
    inner_folds: int = 4,
    seed: int = 0,
) -> dict[str, object]:
    """Write one source-video grouped OOF fold for the selected candidate."""
    config = _clean_router_config(configuration)
    train_inputs, _ = _load_train_dev_study_inputs(
        train_manifest_path=train_manifest_path,
        train_reliability_path=train_reliability_path,
        train_audio_transcripts_path=train_audio_transcripts_path,
        train_av_transcripts_path=train_av_transcripts_path,
        dev_manifest_path=dev_manifest_path,
        dev_reliability_path=dev_reliability_path,
        dev_audio_transcripts_path=dev_audio_transcripts_path,
        dev_av_transcripts_path=dev_av_transcripts_path,
        train_features_path=train_features_path,
        dev_features_path=dev_features_path,
        audio_system_name=audio_system_name,
        av_system_name=av_system_name,
        folds=folds,
        inner_folds=inner_folds,
    )
    outer_folds = _grouped_folds(list(train_inputs["rows"]), folds=folds, seed=seed)
    if fold_index < 0 or fold_index >= len(outer_folds):
        raise ValueError(f"--fold-index must be in [0, {len(outer_folds) - 1}].")
    config_index = list(CLEAN_ROUTER_CONFIGS).index(config)
    validation_ids = set(outer_folds[fold_index])
    print(f"[router] {configuration} OOF fold {fold_index}: fitting", flush=True)
    policy = _fit_clean_policy(
        _subset_inputs(train_inputs, set(train_inputs["rows"]) - validation_ids),
        model=str(config["model"]),
        features=list(config["features"]),
        inner_folds=inner_folds,
        seed=seed + config_index * 100 + fold_index,
        audio_system_name=audio_system_name,
        av_system_name=av_system_name,
    )
    decisions = [
        {"split": "train_oof", "configuration": configuration, "fold": fold_index, **row}
        for row in _route(_subset_inputs(train_inputs, validation_ids), policy)["decisions"]
    ]
    fold_dir = output_dir / configuration
    fold_dir.mkdir(parents=True, exist_ok=True)
    path = fold_dir / f"fold-{fold_index:02d}.csv"
    _write_csv(path, decisions)
    print(f"[router] {configuration} OOF fold {fold_index}: wrote {len(decisions)} decisions", flush=True)
    return {"configuration": configuration, "fold": fold_index, "utterances": len(decisions), "path": str(path)}


def finalize_train_dev_study(
    *,
    candidate_dir: Path,
    oof_dir: Path,
    output_dir: Path,
    folds: int,
    inner_folds: int,
    seed: int,
) -> dict[str, object]:
    """Select from checkpointed dev results and merge selected OOF fold outputs."""
    candidates, selected = _select_checkpointed_train_dev_candidate(candidate_dir)
    selected_name = str(selected["configuration"])
    oof_decisions: list[dict[str, object]] = []
    for fold_index in range(folds):
        path = oof_dir / selected_name / f"fold-{fold_index:02d}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing selected OOF fold: {path}")
        oof_decisions.extend(_read_csv(path))
    oof_ids = [str(row["utterance_id"]) for row in oof_decisions]
    if len(oof_ids) != len(set(oof_ids)):
        raise ValueError("Selected OOF folds contain duplicate utterance IDs.")
    dev_decisions = _read_csv(candidate_dir / selected_name / "dev_decisions.csv")
    all_decisions = [*oof_decisions, *[
        {"split": "dev", "configuration": selected_name, **row}
        for row in dev_decisions
    ]]
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_dir / selected_name / "policy.pkl", output_dir / "selected_policy.pkl")
    _write_json(output_dir / "selected_policy.json", selected["policy"])
    _write_csv(output_dir / "router_train_dev_decisions.csv", all_decisions)
    _write_json(
        output_dir / "router_train_dev_study.json",
        {
            "train_utterances": len(oof_decisions),
            "dev_utterances": len(dev_decisions),
            "outer_folds": folds,
            "inner_folds": inner_folds,
            "seed": seed,
            "whisper_dev": candidates[0]["whisper_dev"],
            "configurations": candidates,
            "selected_configuration": selected_name,
            "selected_supported_over_whisper": selected["supported_over_whisper"],
            "selection_tie_band_wer": 0.001,
            "selected_train_oof": _decision_summary(oof_decisions, DEFAULT_AV_SYSTEM),
        },
    )
    return {"selected_configuration": selected_name, "supported": selected["supported_over_whisper"]}


def _select_checkpointed_train_dev_candidate(
    candidate_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for config in CLEAN_ROUTER_CONFIGS:
        name = str(config["name"])
        path = candidate_dir / name / "candidate.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate checkpoint: {path}")
        candidate = _read_json(path)
        if candidate.get("configuration") != name:
            raise ValueError(f"Candidate checkpoint has wrong configuration: {path}")
        candidates.append(candidate)
    supported = [item for item in candidates if item["supported_over_whisper"]]
    selected_pool = supported or candidates
    best_wer = min(float(item["dev"]["routed_wer"]) for item in selected_pool)
    tied = [
        item for item in selected_pool
        if float(item["dev"]["routed_wer"]) <= best_wer + 0.001
    ]
    return candidates, min(
        tied, key=lambda item: float(item["dev"]["routed_to_audio_visual_rate"])
    )


def select_train_dev_candidate(
    *, candidate_dir: Path, output_dir: Path
) -> dict[str, object]:
    """Persist the dev-selected candidate for a dependent grouped OOF array."""
    candidates, selected = _select_checkpointed_train_dev_candidate(candidate_dir)
    selected_name = str(selected["configuration"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "selection.json",
        {
            "selected_configuration": selected_name,
            "selected_supported_over_whisper": selected["supported_over_whisper"],
            "selection_tie_band_wer": 0.001,
            "configurations": candidates,
        },
    )
    (output_dir / "selected_configuration.txt").write_text(f"{selected_name}\n")
    print(f"[router] selected dev candidate: {selected_name}", flush=True)
    return {"selected_configuration": selected_name, "supported": selected["supported_over_whisper"]}


def _audio_only_policy(audio_system_name: str, av_system_name: str) -> dict[str, object]:
    return {
        "policy_type": "audio_only", "feature_family": "audio", "feature": None,
        "direction": "higher", "threshold": None,
        "audio_system": audio_system_name, "av_system": av_system_name,
    }


def _fit_clean_policy(
    inputs: dict[str, object], *, model: str, features: list[str], inner_folds: int,
    seed: int, audio_system_name: str, av_system_name: str,
    compute_feature_importances: bool = False,
) -> dict[str, object]:
    if model == "threshold":
        policy, _ = _select_threshold_policy(inputs)
    elif model == "logistic":
        policy = _fit_learned_policy(inputs, feature_candidates=features, l2_strength=1.0, class_weight="unweighted")
        policy["threshold"] = _nested_probability_threshold(
            inputs, features=features, l2_strength=1.0, class_weight="unweighted",
            folds=inner_folds, seed=seed, audio_system_name=audio_system_name,
            av_system_name=av_system_name,
        )
    else:
        policy = _fit_benefit_policy(
            inputs,
            feature_candidates=features,
            model=model,
            seed=seed,
            compute_feature_importances=compute_feature_importances,
        )
        policy["threshold"] = _nested_benefit_threshold(
            inputs, features=features, model=model, folds=inner_folds, seed=seed,
            audio_system_name=audio_system_name, av_system_name=av_system_name,
        )
    policy.update({"audio_system": audio_system_name, "av_system": av_system_name})
    return policy


def _clean_oof_decisions(
    inputs: dict[str, object], folds: list[list[str]], all_ids: set[str], *, model: str,
    features: list[str], inner_folds: int, seed: int, audio_system_name: str,
    av_system_name: str,
) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []
    for index, validation_ids in enumerate(folds):
        training = _subset_inputs(inputs, all_ids - set(validation_ids))
        validation = _subset_inputs(inputs, set(validation_ids))
        policy = _fit_clean_policy(
            training, model=model, features=features, inner_folds=inner_folds,
            seed=seed + index, audio_system_name=audio_system_name, av_system_name=av_system_name,
        )
        decisions.extend({"fold": index, **row} for row in _route(validation, policy)["decisions"])
    return decisions


def _policy_metadata(policy: dict[str, object]) -> dict[str, object]:
    metadata = copy.deepcopy(policy)
    metadata.pop("estimator", None)
    return metadata


def noise_study(
    manifest_path: Path,
    reliability_path: Path,
    conditions: list[tuple[str, Path, Path]],
    output_dir: Path,
    features_path: Path | None = None,
    audio_system_name: str = DEFAULT_AUDIO_SYSTEM,
    av_system_name: str = DEFAULT_AV_SYSTEM,
    folds: int = 5,
    inner_folds: int = 4,
    seed: int = 0,
) -> dict[str, object]:
    """Evaluate noise-aware router features with source-video grouped CV."""
    audio_system_name = _require_system_name(audio_system_name, "audio")
    av_system_name = _require_system_name(av_system_name, "audio-visual")
    if audio_system_name == av_system_name:
        raise ValueError("Audio and audio-visual system names must be different.")
    if len(conditions) < 2:
        raise ValueError("Noise study needs at least two conditions.")
    if folds < 2 or inner_folds < 2:
        raise ValueError("--folds and --inner-folds must both be at least 2.")

    inputs = _load_noise_study_inputs(
        manifest_path,
        reliability_path,
        conditions,
        audio_system_name,
        features_path=features_path,
    )
    outer_folds = _grouped_folds(list(inputs["rows"]), folds=folds, seed=seed)
    all_ids = set(inputs["rows"])
    summary_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    config_results: dict[str, dict[str, object]] = {}
    decisions_by_configuration: dict[str, list[dict[str, object]]] = {}

    feature_sets = dict(NOISE_ROUTER_FEATURE_SETS)
    benefit_configs = list(NOISE_BENEFIT_ROUTER_CONFIGS)
    if features_path is not None:
        feature_sets["whisper_enriched"] = ENRICHED_WHISPER_FEATURES
        feature_sets["multimodal_enriched"] = ENRICHED_MULTIMODAL_FEATURES
        benefit_configs.append(
            {
                "name": "multimodal_enriched_hist_gradient_boosting",
                "model": "hist_gradient_boosting",
                "features": ENRICHED_MULTIMODAL_FEATURES,
            }
        )

    for config_index, (feature_set, features) in enumerate(feature_sets.items()):
        summary, decisions = _run_noise_router_configuration(
            inputs=inputs,
            outer_folds=outer_folds,
            all_ids=all_ids,
            name=feature_set,
            model="logistic",
            features=features,
            conditions=conditions,
            inner_folds=inner_folds,
            seed=seed + config_index * 100,
            audio_system_name=audio_system_name,
            av_system_name=av_system_name,
        )
        config_results[feature_set] = summary
        decisions_by_configuration[feature_set] = decisions

    offset = len(feature_sets)
    for config_index, config in enumerate(benefit_configs, start=offset):
        name = str(config["name"])
        summary, decisions = _run_noise_router_configuration(
            inputs=inputs,
            outer_folds=outer_folds,
            all_ids=all_ids,
            name=name,
            model=str(config["model"]),
            features=list(config["features"]),
            conditions=conditions,
            inner_folds=inner_folds,
            seed=seed + config_index * 100,
            audio_system_name=audio_system_name,
            av_system_name=av_system_name,
        )
        config_results[name] = summary
        decisions_by_configuration[name] = decisions

    baseline_name = "combined_noise"
    baseline_decisions = decisions_by_configuration[baseline_name]
    for name, summary in config_results.items():
        comparison = _bootstrap_router_comparison(
            baseline_decisions,
            decisions_by_configuration[name],
            seed=seed,
            conditions=[condition for condition, _, _ in conditions],
        )
        summary["comparison_vs_combined_noise"] = comparison
        summary["meaningful_improvement"] = _is_meaningful_noise_improvement(
            summary,
            config_results[baseline_name],
        )
        for condition, condition_summary in summary["by_condition"].items():
            condition_comparison = comparison["by_condition"][condition]
            summary_rows.append(
                {
                    "configuration": name,
                    "model": summary["model"],
                    "condition": condition,
                    "snr_db": _condition_snr_db(condition),
                    "nested_threshold_mean": summary["nested_threshold_mean"],
                    "meaningful_improvement": summary["meaningful_improvement"],
                    "wer_reduction_vs_combined_noise_pp": condition_comparison[
                        "wer_reduction_pp"
                    ],
                    "wer_reduction_ci_95_low_pp": condition_comparison[
                        "ci_95_low_pp"
                    ],
                    "wer_reduction_ci_95_high_pp": condition_comparison[
                        "ci_95_high_pp"
                    ],
                    **condition_summary,
                }
            )
        for row in decisions_by_configuration[name]:
            decision_rows.append({"configuration": name, **row})

    best_name = min(
        config_results,
        key=lambda name: float(config_results[name]["overall"]["routed_wer"]),
    )
    best = config_results[best_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "noise_router_metrics.csv", summary_rows)
    _write_csv(output_dir / "noise_router_decisions.csv", decision_rows)
    _write_json(
        output_dir / "noise_router_study.json",
        {
            "conditions": [
                {"name": name, "snr_db": _condition_snr_db(name)}
                for name, _, _ in conditions
            ],
            "feature_sets": feature_sets,
            "benefit_router_configurations": benefit_configs,
            "features_path": str(features_path) if features_path else None,
            "outer_folds": len(outer_folds),
            "inner_folds": inner_folds,
            "seed": seed,
            "bootstrap_samples": NOISE_STUDY_BOOTSTRAP_SAMPLES,
            "baseline_configuration": baseline_name,
            "best_configuration": best_name,
            "recommended_configuration": (
                best_name if best["meaningful_improvement"] else None
            ),
            "configurations": config_results,
        },
    )
    return {"best_configuration": best_name, "overall": best["overall"]}


def _run_noise_router_configuration(
    *,
    inputs: dict[str, object],
    outer_folds: list[list[str]],
    all_ids: set[str],
    name: str,
    model: str,
    features: list[str],
    conditions: list[tuple[str, Path, Path]],
    inner_folds: int,
    seed: int,
    audio_system_name: str,
    av_system_name: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    decisions: list[dict[str, object]] = []
    thresholds: list[float] = []
    for fold_index, validation_ids in enumerate(outer_folds):
        training_inputs = _subset_inputs(inputs, all_ids - set(validation_ids))
        validation_inputs = _subset_inputs(inputs, set(validation_ids))
        fold_seed = seed + fold_index
        if model == "logistic":
            threshold = _nested_probability_threshold(
                training_inputs,
                features=features,
                l2_strength=1.0,
                class_weight="unweighted",
                folds=inner_folds,
                seed=fold_seed,
                audio_system_name=audio_system_name,
                av_system_name=av_system_name,
            )
            policy = _fit_learned_policy(
                training_inputs,
                feature_candidates=features,
                l2_strength=1.0,
                class_weight="unweighted",
            )
        else:
            threshold = _nested_benefit_threshold(
                training_inputs,
                features=features,
                model=model,
                folds=inner_folds,
                seed=fold_seed,
                audio_system_name=audio_system_name,
                av_system_name=av_system_name,
            )
            policy = _fit_benefit_policy(
                training_inputs,
                feature_candidates=features,
                model=model,
                seed=fold_seed,
            )
        policy.update(
            {
                "threshold": threshold,
                "audio_system": audio_system_name,
                "av_system": av_system_name,
            }
        )
        result = _route(validation_inputs, policy)
        thresholds.append(threshold)
        decisions.extend(
            {
                "fold": fold_index,
                "nested_threshold": threshold,
                **row,
            }
            for row in result["decisions"]
        )

    overall = _decision_summary(decisions, av_system_name)
    by_condition = {
        condition: _decision_summary(
            [row for row in decisions if row.get("condition") == condition],
            av_system_name,
        )
        for condition, _, _ in conditions
    }
    if model == "logistic":
        diagnostic_policy = _fit_learned_policy(
            inputs,
            feature_candidates=features,
            l2_strength=1.0,
            class_weight="unweighted",
        )
    else:
        diagnostic_policy = _fit_benefit_policy(
            inputs,
            feature_candidates=features,
            model=model,
            seed=seed,
        )
    return (
        {
            "configuration": name,
            "model": model,
            "features": features,
            "nested_threshold_mean": sum(thresholds) / len(thresholds),
            "feature_metadata": _study_feature_metadata(diagnostic_policy),
            "overall": overall,
            "by_condition": by_condition,
        },
        decisions,
    )


def _outer_threshold_decisions(
    inputs: dict[str, object],
    folds: list[list[str]],
    all_ids: set[str],
    audio_system_name: str,
    av_system_name: str,
) -> list[dict[str, object]]:
    decisions = []
    for validation_ids in folds:
        training_inputs = _subset_inputs(inputs, all_ids - set(validation_ids))
        validation_inputs = _subset_inputs(inputs, set(validation_ids))
        policy, _ = _select_threshold_policy(training_inputs)
        policy.update(
            {"audio_system": audio_system_name, "av_system": av_system_name}
        )
        decisions.extend(_route(validation_inputs, policy)["decisions"])
    return decisions


def _nested_probability_threshold(
    inputs: dict[str, object],
    *,
    features: list[str],
    l2_strength: float,
    class_weight: str,
    folds: int,
    seed: int,
    audio_system_name: str,
    av_system_name: str,
) -> float:
    inner_folds = _grouped_folds(list(inputs["rows"]), folds=folds, seed=seed)
    all_ids = set(inputs["rows"])
    decisions = []
    for validation_ids in inner_folds:
        print("[router] nested probability calibration fold", flush=True)
        training_inputs = _subset_inputs(inputs, all_ids - set(validation_ids))
        validation_inputs = _subset_inputs(inputs, set(validation_ids))
        policy = _fit_learned_policy(
            training_inputs,
            feature_candidates=features,
            l2_strength=l2_strength,
            class_weight=class_weight,
        )
        policy.update(
            {
                "audio_system": audio_system_name,
                "av_system": av_system_name,
            }
        )
        decisions.extend(_route(validation_inputs, policy)["decisions"])
    return _select_routing_threshold(decisions)


def _nested_benefit_threshold(
    inputs: dict[str, object],
    *,
    features: list[str],
    model: str,
    folds: int,
    seed: int,
    audio_system_name: str,
    av_system_name: str,
) -> float:
    inner_folds = _grouped_folds(list(inputs["rows"]), folds=folds, seed=seed)
    all_ids = set(inputs["rows"])
    decisions = []
    for fold_index, validation_ids in enumerate(inner_folds):
        print(f"[router] nested benefit calibration fold {fold_index}", flush=True)
        training_inputs = _subset_inputs(inputs, all_ids - set(validation_ids))
        validation_inputs = _subset_inputs(inputs, set(validation_ids))
        policy = _fit_benefit_policy(
            training_inputs,
            feature_candidates=features,
            model=model,
            seed=seed + fold_index,
            compute_feature_importances=False,
        )
        policy.update(
            {
                "audio_system": audio_system_name,
                "av_system": av_system_name,
            }
        )
        decisions.extend(_route(validation_inputs, policy)["decisions"])
    return _select_routing_threshold(decisions)


def _select_routing_threshold(decisions: list[dict[str, object]]) -> float:
    candidates = _cumulative_routing_thresholds(decisions)
    if not candidates:
        return float("inf")
    _, _, threshold = min(
        candidates, key=lambda item: (item[0], item[1], -item[2])
    )
    return threshold


def _cumulative_routing_thresholds(
    decisions: list[dict[str, object]],
) -> list[tuple[int, int, float]]:
    """Score deployable ``score >= threshold`` routes in one sorted sweep."""
    total_errors = sum(int(row["audio_word_errors"]) for row in decisions)
    switchable: list[tuple[float, int]] = []
    for row in decisions:
        score = optional_float(row.get("reliability_score"))
        av_errors = optional_float(row.get("audio_visual_word_errors"))
        if (
            score is None
            or av_errors is None
            or not row.get("audio_visual_prediction")
            or not is_real_mouth_roi(row)
        ):
            continue
        switchable.append((score, int(av_errors) - int(row["audio_word_errors"])))
    if not switchable:
        return []

    switchable.sort(key=lambda item: item[0], reverse=True)
    fallback_threshold = switchable[0][0] + 1.0
    candidates = [(total_errors, 0, fallback_threshold)]
    av_selected = 0
    index = 0
    while index < len(switchable):
        threshold = switchable[index][0]
        while index < len(switchable) and switchable[index][0] == threshold:
            total_errors += switchable[index][1]
            av_selected += 1
            index += 1
        candidates.append((total_errors, av_selected, threshold))
    return candidates


def _score_threshold_candidates(
    candidates: list[dict[str, object]],
    precomputed: list[dict[str, object]],
) -> dict[int, dict[str, object]]:
    """Cumulatively score generic visual/audio threshold policies.

    Threshold policies are only allowed to switch rows that deployment can route to
    AVEC: a real ROI, an AVEC transcript, a numeric score, and known AVEC errors.
    """
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for policy in candidates:
        if policy.get("policy_type") != "threshold":
            continue
        feature = str(policy.get("feature") or "")
        grouped.setdefault((feature, _policy_direction(policy)), []).append(policy)

    results: dict[int, dict[str, object]] = {}
    baseline_errors = sum(int(row["audio_errors"]) for row in precomputed)
    reference_words = sum(int(row["reference_words"]) for row in precomputed)
    for (feature, direction), policies in grouped.items():
        groups: dict[float, list[int]] = {}
        for row in precomputed:
            score = optional_float(row["feature_row"].get(feature))
            if (
                score is None
                or row["av_prediction"] is None
                or row["av_errors"] is None
                or not is_real_mouth_roi(row["feature_row"])
            ):
                continue
            groups.setdefault(score, []).append(
                int(row["av_errors"]) - int(row["audio_errors"])
            )
        totals: dict[float, tuple[int, int]] = {}
        errors = baseline_errors
        selected = 0
        for threshold in sorted(groups, reverse=direction == "higher"):
            for delta in groups[threshold]:
                errors += delta
                selected += 1
            totals[threshold] = (errors, selected)
        for policy in policies:
            threshold = optional_float(policy.get("threshold"))
            errors, selected = totals.get(
                threshold, (baseline_errors, 0)
            )
            results[id(policy)] = {
                "total_word_errors": errors,
                "total_reference_words": reference_words,
                "word_error_rate": errors / reference_words if reference_words else None,
                "audio_visual_selected": selected,
            }
    return results


def _decision_summary(
    decisions: list[dict[str, object]], av_system_name: str
) -> dict[str, object]:
    references = [str(row["reference"]) for row in decisions]
    audio_predictions = [str(row["audio_prediction"]) for row in decisions]
    routed_predictions = [str(row["selected_prediction"]) for row in decisions]
    oracle_predictions = [
        str(row["audio_visual_prediction"])
        if row["oracle_system"] == av_system_name
        else str(row["audio_prediction"])
        for row in decisions
    ]
    routed_metrics = _metrics_dict(references, routed_predictions)
    oracle_metrics = _metrics_dict(references, oracle_predictions)
    routed_to_av = [row["selected_system"] == av_system_name for row in decisions]
    av_wins = [
        bool(row["audio_visual_prediction"])
        and int(row["audio_visual_word_errors"]) < int(row["audio_word_errors"])
        for row in decisions
    ]
    return {
        "utterances_scored": len(decisions),
        "audio_wer": _metrics_dict(references, audio_predictions)["wer"],
        "routed_wer": routed_metrics["wer"],
        "routed_cer": routed_metrics["cer"],
        "oracle_wer": oracle_metrics["wer"],
        "oracle_cer": oracle_metrics["cer"],
        "router_to_oracle_gap_wer": _metric_gap(
            routed_metrics, oracle_metrics, "wer"
        ),
        "routed_to_audio_visual_count": sum(routed_to_av),
        "routed_to_audio_visual_rate": (
            sum(routed_to_av) / len(decisions) if decisions else 0.0
        ),
        "false_audio_visual_routes": sum(
            routed and not av_win for routed, av_win in zip(routed_to_av, av_wins)
        ),
        "missed_audio_visual_wins": sum(
            not routed and av_win for routed, av_win in zip(routed_to_av, av_wins)
        ),
    }


def _bootstrap_router_comparison(
    baseline: list[dict[str, object]],
    candidate: list[dict[str, object]],
    *,
    seed: int,
    conditions: list[str],
) -> dict[str, object]:
    baseline_by_id = {str(row["utterance_id"]): row for row in baseline}
    candidate_by_id = {str(row["utterance_id"]): row for row in candidate}
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("Router configurations do not cover the same utterances.")
    return {
        "overall": _bootstrap_wer_reduction(
            baseline_by_id, candidate_by_id, seed=seed
        ),
        "by_condition": {
            condition: _bootstrap_wer_reduction(
                baseline_by_id,
                candidate_by_id,
                seed=seed + index + 1,
                condition=condition,
            )
            for index, condition in enumerate(conditions)
        },
    }


def _bootstrap_wer_reduction(
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
    *,
    seed: int,
    condition: str | None = None,
) -> dict[str, object]:
    by_video: dict[str, list[float]] = {}
    for utterance_id in sorted(baseline):
        baseline_row = baseline[utterance_id]
        candidate_row = candidate[utterance_id]
        if condition is not None and baseline_row.get("condition") != condition:
            continue
        reference_words = _word_error_stats(
            str(baseline_row["reference"]), str(baseline_row["audio_prediction"])
        )[1]
        if reference_words == 0:
            continue
        video_id = _source_video_id(utterance_id)
        totals = by_video.setdefault(video_id, [0.0, 0.0, 0.0])
        totals[0] += float(baseline_row["selected_word_errors"])
        totals[1] += float(candidate_row["selected_word_errors"])
        totals[2] += reference_words
    if not by_video:
        return {
            "wer_reduction_pp": 0.0,
            "ci_95_low_pp": 0.0,
            "ci_95_high_pp": 0.0,
            "source_videos": 0,
        }
    values = np.array(list(by_video.values()), dtype=float)
    baseline_errors, candidate_errors, words = values.sum(axis=0)
    point_estimate = 100.0 * (baseline_errors - candidate_errors) / words
    random_state = np.random.default_rng(seed)
    samples = random_state.integers(
        0,
        len(values),
        size=(NOISE_STUDY_BOOTSTRAP_SAMPLES, len(values)),
    )
    sampled = values[samples].sum(axis=1)
    reductions = 100.0 * (sampled[:, 0] - sampled[:, 1]) / sampled[:, 2]
    low, high = np.percentile(reductions, [2.5, 97.5])
    return {
        "wer_reduction_pp": float(point_estimate),
        "ci_95_low_pp": float(low),
        "ci_95_high_pp": float(high),
        "source_videos": len(values),
    }


def _is_meaningful_noise_improvement(
    candidate: dict[str, object], baseline: dict[str, object]
) -> bool:
    if candidate["configuration"] == baseline["configuration"]:
        return False
    comparison = candidate["comparison_vs_combined_noise"]["overall"]
    if comparison["ci_95_low_pp"] <= 0.0:
        return False
    candidate_conditions = candidate["by_condition"]
    baseline_conditions = baseline["by_condition"]
    return all(
        float(candidate_conditions[condition]["routed_wer"])
        <= float(baseline_conditions[condition]["routed_wer"])
        for condition in ("demand_0db", "demand_-5db")
        if condition in candidate_conditions and condition in baseline_conditions
    )


def _fit_learned_policy(
    inputs: dict[str, object],
    *,
    feature_candidates: list[str] | None = None,
    l2_strength: float = 0.1,
    class_weight: str = "balanced",
) -> dict[str, object]:
    feature_candidates = feature_candidates or LEARNED_FEATURES
    if l2_strength < 0:
        raise ValueError("L2 strength cannot be negative.")
    if class_weight not in {"balanced", "unweighted"}:
        raise ValueError(f"Unsupported class weighting: {class_weight}")
    eligible = []
    for utterance_id, manifest_row in inputs["rows"].items():
        if not manifest_row.get("reference", ""):
            continue
        reliability_row = inputs["reliability"].get(utterance_id, {})
        av_prediction = inputs["av"].get(utterance_id, {}).get("prediction")
        if not is_real_mouth_roi(reliability_row) or av_prediction is None:
            continue
        audio_prediction = inputs["audio"][utterance_id]["prediction"]
        audio_errors, _ = _word_error_stats(
            manifest_row.get("reference", ""), audio_prediction
        )
        av_errors, _ = _word_error_stats(
            manifest_row.get("reference", ""), str(av_prediction)
        )
        eligible.append(
            (
                utterance_id,
                _feature_row(reliability_row, inputs["audio"][utterance_id]),
                int(av_errors < audio_errors),
            )
        )
    labels = [item[2] for item in eligible]
    if len(eligible) < 4 or len(set(labels)) < 2:
        raise ValueError(
            "Need at least four AVEC-ready utterances with both routing outcomes."
        )

    feature_names = [
        feature
        for feature in feature_candidates
        if any(optional_float(item[1].get(feature)) is not None for item in eligible)
    ]
    if not feature_names:
        raise ValueError("No numeric routing features are available.")
    medians = {
        feature: float(
            statistics.median(
                optional_float(item[1].get(feature))
                for item in eligible
                if optional_float(item[1].get(feature)) is not None
            )
        )
        for feature in feature_names
    }
    raw = np.array(
        [
            [
                optional_float(feature_row.get(feature))
                if optional_float(feature_row.get(feature)) is not None
                else medians[feature]
                for feature in feature_names
            ]
            for _, feature_row, _ in eligible
        ],
        dtype=float,
    )
    means = raw.mean(axis=0)
    scales = raw.std(axis=0)
    scales[scales < 1e-12] = 1.0
    features = (raw - means) / scales
    labels_array = np.array(labels, dtype=float)
    positives = float(labels_array.sum())
    negatives = float(len(labels_array) - positives)
    if class_weight == "balanced":
        weights = np.where(
            labels_array == 1.0,
            len(labels_array) / (2.0 * positives),
            len(labels_array) / (2.0 * negatives),
        )
    else:
        weights = np.ones(len(labels_array), dtype=float)
    coefficients, intercept = _fit_logistic_regression(
        features,
        labels_array,
        weights,
        l2_strength=l2_strength,
    )
    return {
        "policy_type": LEARNED_POLICY_TYPE,
        "feature_family": "combined",
        "feature": None,
        "direction": "higher",
        "threshold": 0.5,
        "model_version": LEARNED_POLICY_VERSION,
        "feature_names": feature_names,
        "feature_medians": medians,
        "feature_means": {
            feature: float(value) for feature, value in zip(feature_names, means)
        },
        "feature_scales": {
            feature: float(value) for feature, value in zip(feature_names, scales)
        },
        "coefficients": {
            feature: float(value)
            for feature, value in zip(feature_names, coefficients)
        },
        "intercept": float(intercept),
        "l2_strength": l2_strength,
        "class_weight": class_weight,
        "training_utterances": len(eligible),
        "training_audio_visual_wins": int(positives),
    }


def _fit_benefit_policy(
    inputs: dict[str, object],
    *,
    feature_candidates: list[str],
    model: str,
    seed: int,
    compute_feature_importances: bool = True,
) -> dict[str, object]:
    if model not in {"ridge", "hist_gradient_boosting", "mlp"}:
        raise ValueError(f"Unsupported benefit router model: {model}")
    eligible = []
    for utterance_id, manifest_row in inputs["rows"].items():
        reference = str(manifest_row.get("reference", ""))
        av_prediction = inputs["av"].get(utterance_id, {}).get("prediction")
        reliability_row = inputs["reliability"].get(utterance_id, {})
        if not reference or not is_real_mouth_roi(reliability_row) or av_prediction is None:
            continue
        audio_prediction = str(inputs["audio"][utterance_id]["prediction"])
        audio_errors, _ = _word_error_stats(reference, audio_prediction)
        av_errors, _ = _word_error_stats(reference, str(av_prediction))
        eligible.append(
            (
                utterance_id,
                _feature_row(reliability_row, inputs["audio"][utterance_id]),
                _benefit_target(audio_errors, av_errors),
            )
        )
    if len(eligible) < 4:
        raise ValueError("Need at least four AVEC-ready utterances for benefit routing.")

    feature_names = [
        feature
        for feature in feature_candidates
        if any(
            optional_float(feature_row.get(feature)) is not None
            for _, feature_row, _ in eligible
        )
    ]
    if not feature_names:
        raise ValueError("No numeric benefit-routing features are available.")
    medians = {
        feature: float(
            statistics.median(
                optional_float(feature_row.get(feature))
                for _, feature_row, _ in eligible
                if optional_float(feature_row.get(feature)) is not None
            )
        )
        for feature in feature_names
    }
    raw = np.array(
        [
            [
                optional_float(feature_row.get(feature))
                if optional_float(feature_row.get(feature)) is not None
                else medians[feature]
                for feature in feature_names
            ]
            for _, feature_row, _ in eligible
        ],
        dtype=float,
    )
    targets = np.array([target for _, _, target in eligible], dtype=float)
    policy: dict[str, object] = {
        "policy_type": BENEFIT_POLICY_TYPE,
        "feature_family": "combined",
        "feature": None,
        "direction": "higher",
        "threshold": 0.0,
        "model_version": BENEFIT_POLICY_VERSION,
        "benefit_model": model,
        "feature_names": feature_names,
        "feature_medians": medians,
        "training_utterances": len(eligible),
    }
    if model in {"ridge", "mlp"}:
        means = raw.mean(axis=0)
        scales = raw.std(axis=0)
        scales[scales < 1e-12] = 1.0
        normalized = (raw - means) / scales
        if model == "ridge":
            estimator = Ridge(alpha=1.0)
            estimator.fit(normalized, targets)
        else:
            target_mean = float(targets.mean())
            target_scale = float(targets.std())
            if target_scale < 1e-12:
                target_scale = 1.0
            normalized_targets = (targets - target_mean) / target_scale
            estimator, mlp_metadata = _fit_grouped_mlp(
                normalized,
                normalized_targets,
                [utterance_id for utterance_id, _, _ in eligible],
                seed=seed,
            )
        policy.update(
            {
                "estimator": estimator,
                "feature_means": {
                    feature: float(value) for feature, value in zip(feature_names, means)
                },
                "feature_scales": {
                    feature: float(value) for feature, value in zip(feature_names, scales)
                },
            }
        )
        if model == "ridge":
            policy["coefficients"] = {
                feature: float(value)
                for feature, value in zip(feature_names, estimator.coef_)
            }
        else:
            policy.update(
                {
                    "target_mean": target_mean,
                    "target_scale": target_scale,
                    **mlp_metadata,
                }
            )
            if compute_feature_importances:
                policy["feature_importances"] = _permutation_importances(
                    estimator, normalized, normalized_targets, feature_names, seed
                )
    else:
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=50,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        )
        estimator.fit(raw, targets)
        policy["estimator"] = estimator
        if compute_feature_importances:
            policy["feature_importances"] = _permutation_importances(
                estimator,
                raw,
                targets,
                feature_names,
                seed,
            )
    return policy


def _new_mlp_regressor(seed: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
        activation="relu",
        solver="adam",
        alpha=MLP_ALPHA,
        batch_size=MLP_BATCH_SIZE,
        learning_rate_init=MLP_LEARNING_RATE,
        max_iter=1,
        shuffle=True,
        random_state=seed,
    )


def _fit_grouped_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    utterance_ids: list[str],
    *,
    seed: int,
) -> tuple[MLPRegressor, dict[str, object]]:
    source_groups = sorted({_source_video_id(item) for item in utterance_ids})
    if len(source_groups) < 2:
        selected_epochs = MLP_MAX_EPOCHS
        fit_groups = source_groups
        validation_groups: list[str] = []
    else:
        fold_count = min(5, len(source_groups))
        validation_ids = set(
            _grouped_folds(utterance_ids, folds=fold_count, seed=seed)[0]
        )
        validation_mask = np.array(
            [item in validation_ids for item in utterance_ids], dtype=bool
        )
        fit_mask = ~validation_mask
        fit_groups = sorted(
            {_source_video_id(item) for item, keep in zip(utterance_ids, fit_mask) if keep}
        )
        validation_groups = sorted(
            {
                _source_video_id(item)
                for item, keep in zip(utterance_ids, validation_mask)
                if keep
            }
        )
        selector = _new_mlp_regressor(seed)
        best_error = float("inf")
        selected_epochs = 1
        epochs_without_improvement = 0
        for epoch in range(1, MLP_MAX_EPOCHS + 1):
            selector.partial_fit(features[fit_mask], targets[fit_mask])
            predictions = selector.predict(features[validation_mask])
            error = float(np.mean((predictions - targets[validation_mask]) ** 2))
            if error < best_error - MLP_IMPROVEMENT_TOLERANCE:
                best_error = error
                selected_epochs = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= MLP_PATIENCE:
                break

    estimator = _new_mlp_regressor(seed)
    for _ in range(selected_epochs):
        estimator.partial_fit(features, targets)
    print(
        f"[router] neural benefit model selected {selected_epochs} epoch(s)",
        flush=True,
    )
    return estimator, {
        "hidden_layer_sizes": list(MLP_HIDDEN_LAYER_SIZES),
        "activation": "relu",
        "solver": "adam",
        "alpha": MLP_ALPHA,
        "batch_size": MLP_BATCH_SIZE,
        "learning_rate_init": MLP_LEARNING_RATE,
        "max_epochs": MLP_MAX_EPOCHS,
        "patience": MLP_PATIENCE,
        "selected_epochs": selected_epochs,
        "random_seed": seed,
        "training_source_groups": len(source_groups),
        "early_stopping_fit_source_groups": fit_groups,
        "early_stopping_validation_source_groups": validation_groups,
    }


def _permutation_importances(
    estimator: object,
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
    seed: int,
) -> dict[str, float]:
    baseline_error = float(np.mean((estimator.predict(features) - targets) ** 2))
    randomizer = np.random.default_rng(seed)
    importances: dict[str, float] = {}
    for index, feature in enumerate(feature_names):
        shuffled = features.copy()
        shuffled[:, index] = randomizer.permutation(shuffled[:, index])
        shuffled_error = float(
            np.mean((estimator.predict(shuffled) - targets) ** 2)
        )
        importances[feature] = shuffled_error - baseline_error
    return importances


def _study_feature_metadata(policy: dict[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {
        "feature_names": list(policy.get("feature_names", [])),
    }
    coefficients = policy.get("coefficients")
    if isinstance(coefficients, dict):
        metadata["coefficients"] = coefficients
    importances = policy.get("feature_importances")
    if isinstance(importances, dict):
        metadata["importances"] = importances
    return metadata


def _benefit_target(audio_errors: int, av_errors: int) -> float:
    """Positive values mean AVEC reduced the utterance's word errors."""
    return float(audio_errors - av_errors)


def _fit_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    l2_strength: float,
) -> tuple[np.ndarray, float]:
    coefficients = np.zeros(features.shape[1], dtype=float)
    intercept = 0.0
    learning_rate = 0.1
    total_weight = float(weights.sum())
    for _ in range(800):
        logits = np.clip(features @ coefficients + intercept, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        residual = weights * (probabilities - labels)
        coefficient_gradient = (features.T @ residual) / total_weight
        coefficient_gradient += l2_strength * coefficients
        intercept_gradient = float(residual.sum() / total_weight)
        update = learning_rate * coefficient_gradient
        coefficients -= update
        intercept -= learning_rate * intercept_gradient
        if max(float(np.max(np.abs(update))), abs(learning_rate * intercept_gradient)) < 1e-6:
            break
    return coefficients, intercept


def _grouped_folds(
    utterance_ids: list[str], *, folds: int, seed: int
) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for utterance_id in utterance_ids:
        groups.setdefault(_source_video_id(utterance_id), []).append(utterance_id)
    if len(groups) < folds:
        raise ValueError(
            f"Cannot make {folds} folds from only {len(groups)} source videos."
        )
    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)
    result: list[list[str]] = [[] for _ in range(folds)]
    for _, group_ids in group_items:
        target = min(range(folds), key=lambda index: (len(result[index]), index))
        result[target].extend(sorted(group_ids))
    return [fold for fold in result if fold]


def _source_video_id(utterance_id: str) -> str:
    if "::" in utterance_id:
        utterance_id = utterance_id.split("::", 1)[1]
    return utterance_id.rsplit("_", 1)[0] if "_" in utterance_id else utterance_id


def _subset_inputs(inputs: dict[str, object], utterance_ids: set[str]) -> dict[str, object]:
    return {
        key: {
            utterance_id: row
            for utterance_id, row in value.items()
            if utterance_id in utterance_ids
        }
        for key, value in inputs.items()
    }


def _cross_validation_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    references = [str(row["reference"]) for row in rows]
    audio_predictions = [str(row["audio_prediction"]) for row in rows]
    learned_predictions = [str(row["learned_selected_prediction"]) for row in rows]
    threshold_predictions = [str(row["threshold_selected_prediction"]) for row in rows]
    return {
        "utterances_scored": len(rows),
        "audio": _metrics_dict(references, audio_predictions),
        "learned": _metrics_dict(references, learned_predictions),
        "single_threshold": _metrics_dict(references, threshold_predictions),
        "learned_audio_visual_rate": (
            sum(
                row["learned_selected_system"] == row["audio_visual_system"]
                for row in rows
            )
            / len(rows)
            if rows
            else 0.0
        ),
    }


def _report_metrics_without_policy(metrics: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in metrics.items() if key != "policy"}


def _learned_router_recommendation(
    metrics: dict[str, object],
) -> dict[str, object]:
    audio = metrics.get("audio", {})
    learned = metrics.get("learned", {})
    threshold = metrics.get("single_threshold", {})
    audio_wer = optional_float(audio.get("wer")) if isinstance(audio, dict) else None
    learned_wer = optional_float(learned.get("wer")) if isinstance(learned, dict) else None
    threshold_wer = (
        optional_float(threshold.get("wer")) if isinstance(threshold, dict) else None
    )
    recommended = (
        audio_wer is not None
        and learned_wer is not None
        and threshold_wer is not None
        and learned_wer < audio_wer
        and learned_wer < threshold_wer
    )
    return {
        "recommended_for_test": recommended,
        "reason": (
            "Learned out-of-fold WER beats both audio-only and the single-threshold baseline."
            if recommended
            else "Keep the simpler baseline: learned out-of-fold WER did not beat both comparators."
        ),
    }


def _write_learned_coefficients(path: Path, policy: dict[str, object]) -> None:
    coefficients = policy.get("coefficients", {})
    if not isinstance(coefficients, dict):
        return
    _write_csv(
        path,
        [
            {"feature": feature, "standardized_coefficient": value}
            for feature, value in coefficients.items()
        ],
    )


def evaluate(
    manifest_path: Path,
    reliability_path: Path,
    audio_transcripts_path: Path,
    av_transcripts_path: Path,
    policy_path: Path,
    output_dir: Path,
    features_path: Path | None = None,
) -> dict[str, object]:
    policy = _read_policy(policy_path)
    _validate_policy(policy)
    inputs = _load_evaluation_inputs(
        manifest_path,
        reliability_path,
        audio_transcripts_path,
        av_transcripts_path,
        _policy_audio_system(policy),
        features_path=features_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return _write_reports(inputs, policy, output_dir)


def diagnose_failures(
    manifest_path: Path,
    reliability_path: Path,
    av_transcripts_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    rows = _read_csv_by_id(manifest_path)
    reliability = _read_csv_by_id(reliability_path)
    av = _read_transcripts(av_transcripts_path)
    diagnostics = [
        _diagnostic_row(
            utterance_id=utterance_id,
            manifest_row=row,
            reliability_row=reliability.get(utterance_id, {}),
            manifest_path=manifest_path,
            status="success" if utterance_id in av else "failed",
        )
        for utterance_id, row in rows.items()
    ]
    summary = {
        "num_utterances": len(diagnostics),
        "audio_visual_success_count": sum(
            row["status"] == "success" for row in diagnostics
        ),
        "audio_visual_failure_count": sum(
            row["status"] == "failed" for row in diagnostics
        ),
        "groups": {
            status: _diagnostic_summary(
                [row for row in diagnostics if row["status"] == status]
            )
            for status in ("success", "failed")
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "avec_failure_diagnostics.csv", diagnostics)
    _write_json(output_dir / "avec_failure_summary.json", summary)
    return summary


def split_manifests(
    source_manifest_path: Path,
    roi_manifest_path: Path,
    reliability_path: Path,
    policy_path: Path,
    output_dir: Path,
    audio_transcripts_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    source_rows = _read_csv_by_id(source_manifest_path)
    roi_rows = _read_csv_by_id(roi_manifest_path)
    reliability = _read_csv_by_id(reliability_path)
    policy = _read_policy(policy_path)
    _validate_policy(policy)
    audio = (
        _read_transcripts(audio_transcripts_path)
        if audio_transcripts_path is not None
        else {}
    )
    if policy.get("policy_type") == LEARNED_POLICY_TYPE:
        if audio_transcripts_path is None:
            raise ValueError("Learned policies require --audio-transcripts for split.")
        missing_audio = sorted(set(source_rows) - set(audio))
        if missing_audio:
            preview = ", ".join(missing_audio[:5])
            raise ValueError(
                "Audio transcripts are missing "
                f"{len(missing_audio)} utterance(s): {preview}"
            )

    audio_rows = [
        _absolutize_manifest_row(row, source_manifest_path)
        for row in source_rows.values()
    ]
    fallback_rows: list[dict[str, str]] = []
    av_rows: list[dict[str, str]] = []
    for utterance_id, source_row in source_rows.items():
        reliability_row = reliability.get(utterance_id, {})
        feature_row = _feature_row(reliability_row, audio.get(utterance_id, {}))
        if (
            _policy_selects_av(policy, feature_row)
            and utterance_id in roi_rows
        ):
            av_rows.append(_absolutize_manifest_row(roi_rows[utterance_id], roi_manifest_path))
        else:
            fallback_rows.append(
                _absolutize_manifest_row(source_row, source_manifest_path)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "audio_manifest.csv"
    fallback_path = output_dir / "audio_fallback_manifest.csv"
    av_path = output_dir / "audio_visual_manifest.csv"
    _write_manifest(audio_path, audio_rows, source_rows)
    _write_manifest(fallback_path, fallback_rows, source_rows)
    _write_manifest(av_path, av_rows, roi_rows)
    return audio_path, fallback_path, av_path


def _load_evaluation_inputs(
    manifest_path: Path,
    reliability_path: Path,
    audio_transcripts_path: Path,
    av_transcripts_path: Path,
    audio_system_name: str,
    *,
    features_path: Path | None = None,
) -> dict[str, object]:
    rows = _read_csv_by_id(manifest_path)
    reliability = _read_csv_by_id(reliability_path)
    audio = _read_transcripts(audio_transcripts_path)
    av = _read_transcripts(av_transcripts_path)
    missing_audio = sorted(set(rows) - set(audio))
    if missing_audio:
        preview = ", ".join(missing_audio[:5])
        raise ValueError(
            f"{audio_system_name} transcripts are missing "
            f"{len(missing_audio)} utterance(s): {preview}"
        )
    if features_path is not None:
        features = _read_noise_features(features_path)
        for utterance_id, manifest_row in rows.items():
            condition = str(manifest_row.get("condition") or "clean")
            feature_row = features.get(f"{condition}::{utterance_id}")
            if feature_row is None:
                raise ValueError(
                    f"Extracted routing features are missing {condition}/{utterance_id}"
                )
            audio[utterance_id].update(feature_row)
    return {
        "rows": rows,
        "reliability": reliability,
        "audio": audio,
        "av": av,
    }


def _write_reports(
    inputs: dict[str, object],
    policy: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    result = _route(inputs, policy)
    _write_csv(output_dir / "routing_decisions.csv", result["decisions"])
    _write_csv(output_dir / "missed_av_wins.csv", result["missed_av_wins"])
    metrics = _report_metrics_without_policy(result["metrics"])
    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def _route(
    inputs: dict[str, object],
    policy: dict[str, object],
) -> dict[str, object]:
    rows = inputs["rows"]
    reliability = inputs["reliability"]
    audio = inputs["audio"]
    av = inputs["av"]
    audio_system = _policy_audio_system(policy)
    av_system = _policy_av_system(policy)

    decisions = []
    routed_predictions: dict[str, str] = {}
    oracle_predictions: dict[str, str] = {}
    total_word_errors = 0
    total_reference_words = 0
    audio_visual_selected = 0
    routing_correct = 0
    scored_decisions = 0
    missed_audio_visual_wins = 0
    false_audio_visual_routes = 0
    missed_av_wins = []

    for utterance_id, row in rows.items():
        reference = row.get("reference", "")
        audio_prediction = audio[utterance_id]["prediction"]
        av_prediction = av.get(utterance_id, {}).get("prediction")
        reliability_row = reliability.get(utterance_id, {})
        feature_row = _feature_row(reliability_row, audio[utterance_id])
        selects_av = (
            av_prediction is not None
            and _policy_selects_av(policy, feature_row)
        )
        selected_system = av_system if selects_av else audio_system
        selected_prediction = av_prediction if selects_av else audio_prediction
        if selected_prediction is None:
            selected_prediction = audio_prediction
            selected_system = audio_system
        if selected_system == av_system:
            audio_visual_selected += 1

        audio_errors, reference_words = _word_error_stats(
            reference,
            audio_prediction,
        )
        av_errors = (
            _word_error_stats(reference, av_prediction)[0]
            if av_prediction is not None
            else None
        )
        selected_errors = av_errors if selected_system == av_system else audio_errors
        oracle_system = (
            av_system
            if av_errors is not None and av_errors < audio_errors
            else audio_system
        )
        if reference and oracle_system == av_system and selected_system != av_system:
            missed_audio_visual_wins += 1
            missed_av_wins.append(
                {
                    "utterance_id": utterance_id,
                    "reference": reference,
                    "audio_prediction": audio_prediction,
                    "audio_visual_prediction": av_prediction or "",
                    "audio_word_errors": audio_errors,
                    "audio_visual_word_errors": av_errors,
                    "selected_system": selected_system,
                    **_decision_feature_values(feature_row),
                }
            )
        if reference and selected_system == av_system and oracle_system != av_system:
            false_audio_visual_routes += 1
        oracle_prediction = (
            av_prediction if oracle_system == av_system else audio_prediction
        )
        routed_predictions[utterance_id] = selected_prediction
        oracle_predictions[utterance_id] = oracle_prediction

        if reference:
            total_word_errors += selected_errors
            total_reference_words += reference_words
            routing_correct += int(
                selected_errors == min(
                    audio_errors,
                    av_errors if av_errors is not None else audio_errors,
                )
            )
            scored_decisions += 1

        feature = policy.get("feature")
        reliability_score = _policy_score(policy, feature_row)
        decisions.append(
            {
                "utterance_id": utterance_id,
                "condition": row.get("condition", ""),
                "policy_type": policy["policy_type"],
                "feature_family": _policy_feature_family(policy),
                "feature": feature or "",
                "direction": _policy_direction(policy),
                "reliability_score": (
                    reliability_score
                ),
                "threshold": policy.get("threshold", ""),
                "selected_system": selected_system,
                "reference": reference,
                "audio_prediction": audio_prediction,
                "audio_visual_prediction": av_prediction or "",
                "audio_word_errors": audio_errors,
                "audio_visual_word_errors": "" if av_errors is None else av_errors,
                "selected_prediction": selected_prediction,
                "selected_word_errors": selected_errors,
                "oracle_system": oracle_system,
                "roi_status": feature_row.get("roi_status", ""),
                "fallback_used": feature_row.get("fallback_used", ""),
                **_decision_feature_values(feature_row),
            }
        )

    ordered_ids = list(rows)
    references = [rows[item].get("reference", "") for item in ordered_ids]
    audio_predictions = [audio[item]["prediction"] for item in ordered_ids]
    routed = [routed_predictions[item] for item in ordered_ids]
    oracle = [oracle_predictions[item] for item in ordered_ids]
    av_ids = [item for item in ordered_ids if item in av]
    av_references = [rows[item].get("reference", "") for item in av_ids]
    av_predictions = [av[item]["prediction"] for item in av_ids]
    real_roi_count = sum(
        is_real_mouth_roi(reliability.get(item, {})) for item in ordered_ids
    )

    routed_metrics = _metrics_dict(references, routed)
    oracle_metrics = _metrics_dict(references, oracle)
    metrics = {
        "num_utterances": len(ordered_ids),
        "policy": policy,
        "audio": {
            "system_name": audio_system,
            **_metrics_dict(references, audio_predictions),
            "coverage": 1.0 if ordered_ids else 0.0,
        },
        "audio_visual": {
            "system_name": av_system,
            **_metrics_dict(av_references, av_predictions),
            "coverage": len(av_ids) / len(ordered_ids) if ordered_ids else 0.0,
        },
        "routed": routed_metrics,
        "oracle": oracle_metrics,
        "oracle_gap_wer": _metric_gap(routed_metrics, oracle_metrics, "wer"),
        "oracle_gap_cer": _metric_gap(routed_metrics, oracle_metrics, "cer"),
        "routed_to_audio_visual_count": audio_visual_selected,
        "routed_to_audio_visual_rate": (
            audio_visual_selected / len(ordered_ids) if ordered_ids else 0.0
        ),
        "missed_audio_visual_wins": missed_audio_visual_wins,
        "false_audio_visual_routes": false_audio_visual_routes,
        "routing_accuracy": (
            routing_correct / scored_decisions if scored_decisions else None
        ),
        "preprocessing_failure_count": len(ordered_ids) - real_roi_count,
    }
    return {
        "decisions": decisions,
        "missed_av_wins": missed_av_wins,
        "metrics": metrics,
        "total_word_errors": total_word_errors,
        "total_reference_words": total_reference_words,
        "word_error_rate": (
            total_word_errors / total_reference_words
            if total_reference_words
            else None
        ),
        "audio_visual_selected": audio_visual_selected,
    }


def _score_policy(
    inputs: dict[str, object],
    policy: dict[str, object],
    *,
    precomputed: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    precomputed = precomputed or _precompute_policy_rows(inputs)
    total_word_errors = 0
    total_reference_words = 0
    audio_visual_selected = 0

    for row in precomputed:
        av_prediction = row["av_prediction"]
        selects_av = (
            av_prediction is not None
            and _policy_selects_av(policy, row["feature_row"])
        )
        if selects_av:
            selected_errors = int(row["av_errors"])
            audio_visual_selected += 1
        else:
            selected_errors = int(row["audio_errors"])
        total_word_errors += selected_errors
        total_reference_words += int(row["reference_words"])

    return {
        "total_word_errors": total_word_errors,
        "total_reference_words": total_reference_words,
        "word_error_rate": (
            total_word_errors / total_reference_words
            if total_reference_words
            else None
        ),
        "audio_visual_selected": audio_visual_selected,
    }


def _precompute_policy_rows(inputs: dict[str, object]) -> list[dict[str, object]]:
    result = []
    for utterance_id, manifest_row in inputs["rows"].items():
        reference = manifest_row.get("reference", "")
        if not reference:
            continue
        audio_row = inputs["audio"][utterance_id]
        audio_prediction = str(audio_row["prediction"])
        av_prediction = inputs["av"].get(utterance_id, {}).get("prediction")
        audio_errors, reference_words = _word_error_stats(reference, audio_prediction)
        av_errors = (
            _word_error_stats(reference, str(av_prediction))[0]
            if av_prediction is not None
            else None
        )
        result.append(
            {
                "feature_row": _feature_row(
                    inputs["reliability"].get(utterance_id, {}), audio_row
                ),
                "audio_errors": audio_errors,
                "av_errors": av_errors,
                "av_prediction": av_prediction,
                "reference_words": reference_words,
            }
        )
    return result


def _policy(
    policy_type: str,
    feature: str | None,
    threshold: float | None,
    *,
    priority: int,
    direction: str = "higher",
    family: str = "visual",
) -> dict[str, object]:
    return {
        "policy_type": policy_type,
        "feature_family": family,
        "feature": feature,
        "threshold": threshold,
        "direction": direction,
        "priority": priority,
    }


def _policy_selects_av(
    policy: dict[str, object],
    reliability_row: dict[str, object],
) -> bool:
    if not is_real_mouth_roi(reliability_row):
        return False
    if policy.get("policy_type") == "audio_only":
        return False
    if policy.get("policy_type") == "roi_success":
        return True
    if policy.get("policy_type") == LEARNED_POLICY_TYPE:
        probability = _learned_probability(policy, reliability_row)
        threshold = optional_float(policy.get("threshold"))
        return probability is not None and threshold is not None and probability >= threshold
    if policy.get("policy_type") == BENEFIT_POLICY_TYPE:
        benefit = _benefit_prediction(policy, reliability_row)
        threshold = optional_float(policy.get("threshold"))
        return benefit is not None and threshold is not None and benefit >= threshold
    feature = str(policy.get("feature") or "")
    threshold = optional_float(policy.get("threshold"))
    value = optional_float(reliability_row.get(feature))
    if value is None or threshold is None:
        return False
    if _policy_direction(policy) == "lower":
        return value <= threshold
    return value >= threshold


def _policy_score(policy: dict[str, object], feature_row: dict[str, object]) -> object:
    if policy.get("policy_type") == LEARNED_POLICY_TYPE:
        probability = _learned_probability(policy, feature_row)
        return "" if probability is None else probability
    if policy.get("policy_type") == BENEFIT_POLICY_TYPE:
        benefit = _benefit_prediction(policy, feature_row)
        return "" if benefit is None else benefit
    feature = policy.get("feature")
    return feature_row.get(str(feature), "") if feature else ""


def _learned_probability(
    policy: dict[str, object], feature_row: dict[str, object]
) -> float | None:
    feature_names = policy.get("feature_names")
    medians = policy.get("feature_medians")
    means = policy.get("feature_means")
    scales = policy.get("feature_scales")
    coefficients = policy.get("coefficients")
    intercept = optional_float(policy.get("intercept"))
    if (
        not isinstance(feature_names, list)
        or not isinstance(medians, dict)
        or not isinstance(means, dict)
        or not isinstance(scales, dict)
        or not isinstance(coefficients, dict)
        or intercept is None
    ):
        return None
    logit = intercept
    for feature in feature_names:
        if not isinstance(feature, str):
            return None
        median = optional_float(medians.get(feature))
        mean = optional_float(means.get(feature))
        scale = optional_float(scales.get(feature))
        coefficient = optional_float(coefficients.get(feature))
        if (
            median is None
            or mean is None
            or scale is None
            or scale <= 0
            or coefficient is None
        ):
            return None
        value = optional_float(feature_row.get(feature))
        logit += coefficient * ((median if value is None else value) - mean) / scale
    logit = max(-30.0, min(30.0, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def _benefit_prediction(
    policy: dict[str, object], feature_row: dict[str, object]
) -> float | None:
    feature_names = policy.get("feature_names")
    medians = policy.get("feature_medians")
    estimator = policy.get("estimator")
    model = policy.get("benefit_model")
    if (
        not isinstance(feature_names, list)
        or not isinstance(medians, dict)
        or estimator is None
        or model not in {"ridge", "hist_gradient_boosting", "mlp"}
    ):
        return None
    values = []
    for feature in feature_names:
        if not isinstance(feature, str):
            return None
        median = optional_float(medians.get(feature))
        if median is None:
            return None
        value = optional_float(feature_row.get(feature))
        values.append(median if value is None else value)
    features = np.array([values], dtype=float)
    if model in {"ridge", "mlp"}:
        means = policy.get("feature_means")
        scales = policy.get("feature_scales")
        if not isinstance(means, dict) or not isinstance(scales, dict):
            return None
        normalized = []
        for index, feature in enumerate(feature_names):
            mean = optional_float(means.get(feature))
            scale = optional_float(scales.get(feature))
            if mean is None or scale is None or scale <= 0:
                return None
            normalized.append((features[0, index] - mean) / scale)
        features = np.array([normalized], dtype=float)
    try:
        prediction = float(estimator.predict(features)[0])
        if model == "mlp":
            target_mean = optional_float(policy.get("target_mean"))
            target_scale = optional_float(policy.get("target_scale"))
            if target_mean is None or target_scale is None or target_scale <= 0:
                return None
            prediction = prediction * target_scale + target_mean
        return prediction
    except (AttributeError, TypeError, ValueError):
        return None


def _validate_policy(policy: dict[str, object]) -> None:
    if _policy_audio_system(policy) == _policy_av_system(policy):
        raise ValueError("Audio and audio-visual system names must be different.")
    policy_type = policy.get("policy_type")
    if policy_type in {"roi_success", "audio_only"}:
        return
    if policy_type == LEARNED_POLICY_TYPE:
        if _policy_feature_family(policy) != "combined":
            raise ValueError("Learned reliability policies must use combined features.")
        if optional_float(policy.get("threshold")) is None:
            raise ValueError("Learned reliability policy has no probability threshold.")
        if _learned_probability(policy, {}) is None:
            raise ValueError("Learned reliability policy has an invalid model.")
        return
    if policy_type == BENEFIT_POLICY_TYPE:
        if _policy_feature_family(policy) != "combined":
            raise ValueError("Benefit routers must use combined features.")
        if optional_float(policy.get("threshold")) is None:
            raise ValueError("Benefit router has no numeric threshold.")
        if _benefit_prediction(policy, {}) is None:
            raise ValueError("Benefit router has an invalid model.")
        return
    if policy_type != "threshold":
        raise ValueError(f"Unsupported reliability policy: {policy_type}")
    supported_features = {item[0] for item in POLICY_FEATURES} | {
        item[0] for item in AUDIO_UNCERTAINTY_FEATURES
    }
    if policy.get("feature") not in supported_features:
        raise ValueError(f"Unsupported reliability feature: {policy.get('feature')}")
    if optional_float(policy.get("threshold")) is None:
        raise ValueError("Threshold reliability policy has no numeric threshold.")
    _policy_direction(policy)
    _policy_feature_family(policy)


def _policy_feature_family(policy: dict[str, object]) -> str:
    family = str(policy.get("feature_family") or "visual")
    if family not in {"visual", "audio_uncertainty", "combined", "audio"}:
        raise ValueError(f"Unsupported reliability feature family: {family}")
    return family


def _policy_direction(policy: dict[str, object]) -> str:
    direction = str(policy.get("direction") or "higher")
    if direction not in {"higher", "lower"}:
        raise ValueError(f"Unsupported reliability threshold direction: {direction}")
    return direction


def _policy_audio_system(policy: dict[str, object]) -> str:
    return _require_system_name(
        str(policy.get("audio_system") or DEFAULT_AUDIO_SYSTEM),
        "audio",
    )


def _policy_av_system(policy: dict[str, object]) -> str:
    return _require_system_name(
        str(policy.get("av_system") or DEFAULT_AV_SYSTEM),
        "audio-visual",
    )


def _require_system_name(value: str, kind: str) -> str:
    name = value.strip()
    if not name:
        raise ValueError(f"The {kind} system name cannot be empty.")
    return name


def _metrics_dict(references: list[str], predictions: list[str]) -> dict[str, object]:
    pairs = [
        (reference, prediction)
        for reference, prediction in zip(references, predictions)
        if reference
    ]
    metrics = compute_metrics(
        [item[0] for item in pairs],
        [item[1] for item in pairs],
    )
    return {
        "utterances_scored": metrics.utterances_scored,
        "wer": metrics.wer,
        "cer": metrics.cer,
        "raw_wer": metrics.raw_wer,
        "raw_cer": metrics.raw_cer,
    }


def _metric_gap(
    baseline: dict[str, object],
    upper_bound: dict[str, object],
    name: str,
) -> float | None:
    baseline_value = baseline.get(name)
    upper_bound_value = upper_bound.get(name)
    if baseline_value is None or upper_bound_value is None:
        return None
    return float(baseline_value) - float(upper_bound_value)


def _word_error_stats(reference: str, prediction: str) -> tuple[int, int]:
    reference_words = _normalize_words(reference)
    prediction_words = _normalize_words(prediction)
    previous = list(range(len(prediction_words) + 1))
    for ref_index, ref_word in enumerate(reference_words, start=1):
        current = [ref_index]
        for hyp_index, hyp_word in enumerate(prediction_words, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + int(ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1], len(reference_words)


def _normalize_words(text: str) -> list[str]:
    without_punctuation = "".join(
        character
        for character in text.lower()
        if not unicodedata.category(character).startswith("P")
    )
    return without_punctuation.split()


def _read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    indexed: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows):
        utterance_id = row.get("utterance_id") or f"utt-{index:05d}"
        if utterance_id in indexed:
            raise ValueError(f"Duplicate utterance_id in {path}: {utterance_id}")
        indexed[utterance_id] = row
    return indexed


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_transcripts(path: Path) -> dict[str, dict[str, str]]:
    transcripts: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            utterance_id = str(record.get("utterance_id") or "")
            if not utterance_id:
                raise ValueError(f"Transcript {path}:{line_number} has no utterance_id")
            if utterance_id in transcripts:
                raise ValueError(f"Duplicate utterance_id in {path}: {utterance_id}")
            transcripts[utterance_id] = _transcript_features(record)
    return transcripts


def _load_noise_study_inputs(
    manifest_path: Path,
    reliability_path: Path,
    conditions: list[tuple[str, Path, Path]],
    audio_system_name: str,
    *,
    features_path: Path | None = None,
) -> dict[str, object]:
    manifest_rows = _read_csv_by_id(manifest_path)
    reliability_rows = _read_csv_by_id(reliability_path)
    external_features = _read_noise_features(features_path) if features_path else {}
    names = [name for name, _, _ in conditions]
    if len(set(names)) != len(names):
        raise ValueError("Noise-study condition names must be unique.")

    rows: dict[str, dict[str, str]] = {}
    reliability: dict[str, dict[str, str]] = {}
    audio: dict[str, dict[str, object]] = {}
    av: dict[str, dict[str, object]] = {}
    waveform_cache: dict[Path, dict[str, float | None]] = {}
    for condition, audio_path, av_path in conditions:
        audio_transcripts = _read_transcripts(audio_path)
        av_transcripts = _read_transcripts(av_path)
        missing_audio = sorted(set(manifest_rows) - set(audio_transcripts))
        if missing_audio:
            preview = ", ".join(missing_audio[:5])
            raise ValueError(
                f"{audio_system_name} transcripts are missing {len(missing_audio)} "
                f"utterance(s) for {condition}: {preview}"
            )
        for utterance_id, manifest_row in manifest_rows.items():
            key = f"{condition}::{utterance_id}"
            rows[key] = {**manifest_row, "condition": condition}
            reliability[key] = dict(reliability_rows.get(utterance_id, {}))
            audio_row = dict(audio_transcripts[utterance_id])
            used_audio = str(audio_row.get("used_audio_path") or "")
            if used_audio:
                path = Path(used_audio)
                if path not in waveform_cache:
                    waveform_cache[path] = _waveform_features(path)
                audio_row.update(waveform_cache[path])
            if features_path is not None:
                feature_row = external_features.get(key)
                if feature_row is None:
                    raise ValueError(
                        f"Extracted routing features are missing {condition}/{utterance_id}"
                    )
                audio_row.update(feature_row)
            audio[key] = audio_row
            if utterance_id in av_transcripts:
                av[key] = dict(av_transcripts[utterance_id])
    return {"rows": rows, "reliability": reliability, "audio": audio, "av": av}


def _read_noise_features(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            condition = str(row.get("condition") or "")
            utterance_id = str(row.get("utterance_id") or "")
            if not condition or not utterance_id:
                raise ValueError(f"Feature row in {path} needs condition and utterance_id")
            key = f"{condition}::{utterance_id}"
            if key in rows:
                raise ValueError(f"Duplicate feature row in {path}: {key}")
            rows[key] = {
                feature: value
                for feature, value in row.items()
                if feature not in {"condition", "utterance_id"}
            }
    return rows


def _waveform_features(path: Path) -> dict[str, float | None]:
    if not path.exists():
        raise ValueError(f"Noisy audio file does not exist: {path}")
    samples, _ = sf.read(path, dtype="float64", always_2d=True)
    samples = samples.mean(axis=1)
    if samples.size == 0:
        return {field: None for field in _NOISE_WAVEFORM_FEATURES}
    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples)))
    epsilon = 1e-12
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size))) ** 2
    spectrum = spectrum[1:] if spectrum.size > 1 else spectrum
    flatness = float(np.exp(np.mean(np.log(spectrum + epsilon))) / (np.mean(spectrum + epsilon)))
    return {
        "audio_rms_dbfs": 20.0 * math.log10(max(rms, epsilon)),
        "audio_peak_dbfs": 20.0 * math.log10(max(peak, epsilon)),
        "audio_crest_factor_db": 20.0 * math.log10(max(peak, epsilon) / max(rms, epsilon)),
        "audio_clipping_fraction": float(np.mean(np.abs(samples) >= 0.999)),
        "audio_spectral_flatness": flatness,
    }


_NOISE_WAVEFORM_FEATURES = (
    "audio_rms_dbfs",
    "audio_peak_dbfs",
    "audio_crest_factor_db",
    "audio_clipping_fraction",
    "audio_spectral_flatness",
)


def _condition_snr_db(condition: str) -> float | None:
    if condition == "clean":
        return None
    match = re.fullmatch(r"demand_(-?\d+)db", condition)
    return float(match.group(1)) if match else None


def _transcript_features(record: dict[str, object]) -> dict[str, object]:
    prediction = str(record.get("prediction") or "")
    duration = optional_float(record.get("duration"))
    prediction_word_count = len(_normalize_words(prediction))
    avg_logprobs = []
    no_speech_probs = []
    segments = record.get("segments")
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            avg_logprob = optional_float(segment.get("avg_logprob"))
            if avg_logprob is not None:
                avg_logprobs.append(avg_logprob)
            no_speech_prob = optional_float(segment.get("no_speech_prob"))
            if no_speech_prob is not None:
                no_speech_probs.append(no_speech_prob)
    return {
        "prediction": prediction,
        "used_audio_path": str(record.get("used_audio_path") or ""),
        "audio_avg_logprob_mean": (
            sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else None
        ),
        "audio_avg_logprob_min": min(avg_logprobs) if avg_logprobs else None,
        "audio_no_speech_prob_max": max(no_speech_probs) if no_speech_probs else None,
        "audio_prediction_word_count": prediction_word_count,
        "audio_duration": duration,
        "audio_prediction_words_per_second": (
            prediction_word_count / duration
            if duration is not None and duration > 0
            else None
        ),
    }


def _feature_row(
    reliability_row: dict[str, object],
    audio_row: dict[str, object],
) -> dict[str, object]:
    result = dict(reliability_row)
    for key, value in audio_row.items():
        if key.startswith(("audio_", "video_")):
            result[key] = value
    return result


def _decision_feature_values(feature_row: dict[str, object]) -> dict[str, object]:
    fields = [
        "face_detection_rate",
        "retinaface_confidence_mean",
        "retinaface_confidence_p10",
        "mouth_landmark_confidence_mean",
        "mouth_landmark_confidence_p10",
        "multiple_face_rate",
        "audio_avg_logprob_mean",
        "audio_avg_logprob_min",
        "audio_no_speech_prob_max",
        "audio_prediction_word_count",
        "audio_duration",
        "audio_prediction_words_per_second",
        *_NOISE_WAVEFORM_FEATURES,
    ]
    return {field: feature_row.get(field, "") for field in fields}


def _diagnostic_row(
    utterance_id: str,
    manifest_row: dict[str, str],
    reliability_row: dict[str, str],
    manifest_path: Path,
    status: str,
) -> dict[str, object]:
    reference = manifest_row.get("reference", "")
    word_count = len(_normalize_words(reference))
    audio_path = _resolve_manifest_path(manifest_row.get("audio_path", ""), manifest_path)
    duration = _wav_duration_sec(audio_path)
    words_per_second = (
        word_count / duration if isinstance(duration, float) and duration > 0 else ""
    )
    return {
        "utterance_id": utterance_id,
        "status": status,
        "audio_path": audio_path,
        "video_path": _resolve_manifest_path(manifest_row.get("video_path", ""), manifest_path),
        "reference": reference,
        "word_count": word_count,
        "audio_duration": duration,
        "words_per_second": words_per_second,
        "roi_status": reliability_row.get("roi_status", ""),
        "face_detection_rate": reliability_row.get("face_detection_rate", ""),
        "retinaface_confidence_mean": reliability_row.get("retinaface_confidence_mean", ""),
        "retinaface_confidence_p10": reliability_row.get("retinaface_confidence_p10", ""),
        "mouth_landmark_confidence_mean": reliability_row.get("mouth_landmark_confidence_mean", ""),
        "mouth_landmark_confidence_p10": reliability_row.get("mouth_landmark_confidence_p10", ""),
        "multiple_face_rate": reliability_row.get("multiple_face_rate", ""),
    }


def _diagnostic_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "count": len(rows),
        "word_count": _numeric_summary(rows, "word_count"),
        "audio_duration": _numeric_summary(rows, "audio_duration"),
        "words_per_second": _numeric_summary(rows, "words_per_second"),
        "face_detection_rate": _numeric_summary(rows, "face_detection_rate"),
        "retinaface_confidence_mean": _numeric_summary(
            rows,
            "retinaface_confidence_mean",
        ),
        "retinaface_confidence_p10": _numeric_summary(
            rows,
            "retinaface_confidence_p10",
        ),
        "mouth_landmark_confidence_mean": _numeric_summary(
            rows,
            "mouth_landmark_confidence_mean",
        ),
        "mouth_landmark_confidence_p10": _numeric_summary(
            rows,
            "mouth_landmark_confidence_p10",
        ),
        "multiple_face_rate": _numeric_summary(rows, "multiple_face_rate"),
    }


def _numeric_summary(
    rows: list[dict[str, object]],
    field: str,
) -> dict[str, float | int | None]:
    values = [
        value
        for row in rows
        for value in [optional_float(row.get(field))]
        if value is not None
    ]
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _resolve_manifest_path(value: str, manifest_path: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((manifest_path.parent / path).resolve())


def _wav_duration_sec(path: str) -> float | str:
    if not path:
        return ""
    try:
        with wave.open(path, "rb") as handle:
            sample_rate = handle.getframerate()
            if sample_rate <= 0:
                return ""
            return handle.getnframes() / sample_rate
    except (OSError, EOFError, wave.Error):
        return ""


def _absolutize_manifest_row(
    row: dict[str, str],
    source_manifest: Path,
) -> dict[str, str]:
    result = dict(row)
    for column in ("audio_path", "video_path"):
        value = result.get(column)
        if value and not Path(value).is_absolute():
            result[column] = str((source_manifest.parent / value).resolve())
    return result


def _write_manifest(
    path: Path,
    rows: list[dict[str, str]],
    fallback_rows: dict[str, dict[str, str]],
) -> None:
    names = _union_fieldnames(rows or list(fallback_rows.values()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    names = _union_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _union_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    return names


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value




def _read_policy(path: Path) -> dict[str, object]:
    """Load a portable JSON policy summary or its fitted pickle artifact."""
    if path.suffix != ".pkl":
        return _read_json(path)
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a policy dictionary in {path}")
    return value

def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
