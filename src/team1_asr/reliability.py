from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics
import unicodedata
import wave

import numpy as np

from team1_asr.evaluation import compute_metrics
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
STUDY_L2_STRENGTHS = (0.3, 1.0, 3.0)
STUDY_CLASS_WEIGHTS = ("unweighted",)
LEARNED_POLICY_TYPE = "learned_logistic"
LEARNED_POLICY_VERSION = 1
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
        )
    else:
        diagnose_failures(
            manifest_path=Path(args.manifest),
            reliability_path=Path(args.reliability),
            av_transcripts_path=Path(args.av_transcripts),
            output_dir=Path(args.output_dir),
        )


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
    for policy in candidates:
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
    return _select_probability_threshold(decisions)


def _select_probability_threshold(decisions: list[dict[str, object]]) -> float:
    scores = sorted(
        {
            score
            for row in decisions
            for score in [optional_float(row.get("reliability_score"))]
            if score is not None and row.get("audio_visual_prediction")
        }
    )
    if not scores:
        return 1.1
    best: tuple[int, int, float] | None = None
    for threshold in [1.1, *scores]:
        total_errors = 0
        av_selected = 0
        for row in decisions:
            score = optional_float(row.get("reliability_score"))
            av_errors = optional_float(row.get("audio_visual_word_errors"))
            use_av = (
                score is not None
                and av_errors is not None
                and score >= threshold
            )
            total_errors += int(av_errors) if use_av else int(row["audio_word_errors"])
            av_selected += int(use_av)
        candidate = (total_errors, av_selected, -threshold)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return -best[2]


def _decision_summary(
    decisions: list[dict[str, object]], av_system_name: str
) -> dict[str, object]:
    references = [str(row["reference"]) for row in decisions]
    audio_predictions = [str(row["audio_prediction"]) for row in decisions]
    routed_predictions = [str(row["selected_prediction"]) for row in decisions]
    routed_metrics = _metrics_dict(references, routed_predictions)
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
) -> dict[str, object]:
    policy = _read_json(policy_path)
    _validate_policy(policy)
    inputs = _load_evaluation_inputs(
        manifest_path,
        reliability_path,
        audio_transcripts_path,
        av_transcripts_path,
        _policy_audio_system(policy),
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
    policy = _read_json(policy_path)
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
    _write_json(output_dir / "metrics.json", result["metrics"])
    return result["metrics"]


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
        if key.startswith("audio_"):
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


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
