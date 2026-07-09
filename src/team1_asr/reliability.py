from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import unicodedata

from team1_asr.evaluation import compute_metrics
from team1_asr.visual_reliability import (
    is_real_mouth_roi,
    optional_float,
)


POLICY_FEATURES = [
    ("face_detection_rate", 1),
    ("retinaface_confidence_mean", 2),
    ("mouth_landmark_confidence_mean", 3),
]
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

    split = subparsers.add_parser("split")
    split.add_argument("--source-manifest", required=True)
    split.add_argument("--roi-manifest", required=True)
    split.add_argument("--reliability", required=True)
    split.add_argument("--policy", required=True)
    split.add_argument("--output-dir", required=True)

    evaluate = subparsers.add_parser("evaluate")
    _add_evaluation_inputs(evaluate, require_policy=True)
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
    elif args.command == "split":
        split_manifests(
            source_manifest_path=Path(args.source_manifest),
            roi_manifest_path=Path(args.roi_manifest),
            reliability_path=Path(args.reliability),
            policy_path=Path(args.policy),
            output_dir=Path(args.output_dir),
        )
    else:
        evaluate(
            manifest_path=Path(args.manifest),
            reliability_path=Path(args.reliability),
            audio_transcripts_path=Path(args.audio_transcripts),
            av_transcripts_path=Path(args.av_transcripts),
            policy_path=Path(args.policy),
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
    candidates = [_policy("roi_success", None, None, priority=0)]
    for feature, priority in POLICY_FEATURES:
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
                _policy("threshold", feature, threshold, priority=priority)
            )
    candidates.append(_policy("audio_only", None, None, priority=4))

    scored: list[tuple[tuple[float, int, float], dict[str, object]]] = []
    for policy in candidates:
        result = _route(inputs, policy)
        threshold = policy.get("threshold")
        sweep_row = {
            "policy_type": policy["policy_type"],
            "feature": policy.get("feature") or "",
            "threshold": "" if threshold is None else threshold,
            "word_errors": result["total_word_errors"],
            "reference_words": result["total_reference_words"],
            "wer": result["word_error_rate"],
            "audio_visual_selected": result["audio_visual_selected"],
        }
        sweep.append(sweep_row)
        scored.append(
            (
                (
                    float(result["total_word_errors"]),
                    int(policy["priority"]),
                    float(threshold or 0.0),
                ),
                policy,
            )
        )

    selected_policy = min(scored, key=lambda item: item[0])[1]
    selected_policy = {
        key: value for key, value in selected_policy.items() if key != "priority"
    }
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


def split_manifests(
    source_manifest_path: Path,
    roi_manifest_path: Path,
    reliability_path: Path,
    policy_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    source_rows = _read_csv_by_id(source_manifest_path)
    roi_rows = _read_csv_by_id(roi_manifest_path)
    reliability = _read_csv_by_id(reliability_path)
    policy = _read_json(policy_path)
    _validate_policy(policy)

    audio_rows = [
        _absolutize_manifest_row(row, source_manifest_path)
        for row in source_rows.values()
    ]
    fallback_rows: list[dict[str, str]] = []
    av_rows: list[dict[str, str]] = []
    for utterance_id, source_row in source_rows.items():
        reliability_row = reliability.get(utterance_id, {})
        if _policy_selects_av(policy, reliability_row) and utterance_id in roi_rows:
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

    for utterance_id, row in rows.items():
        reference = row.get("reference", "")
        audio_prediction = audio[utterance_id]["prediction"]
        av_prediction = av.get(utterance_id, {}).get("prediction")
        reliability_row = reliability.get(utterance_id, {})
        selects_av = (
            av_prediction is not None
            and _policy_selects_av(policy, reliability_row)
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
        decisions.append(
            {
                "utterance_id": utterance_id,
                "policy_type": policy["policy_type"],
                "feature": feature or "",
                "reliability_score": (
                    reliability_row.get(str(feature), "") if feature else ""
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
        "routed": _metrics_dict(references, routed),
        "oracle": _metrics_dict(references, oracle),
        "routed_to_audio_visual_count": audio_visual_selected,
        "routed_to_audio_visual_rate": (
            audio_visual_selected / len(ordered_ids) if ordered_ids else 0.0
        ),
        "routing_accuracy": (
            routing_correct / scored_decisions if scored_decisions else None
        ),
        "preprocessing_failure_count": len(ordered_ids) - real_roi_count,
    }
    return {
        "decisions": decisions,
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


def _policy(
    policy_type: str,
    feature: str | None,
    threshold: float | None,
    *,
    priority: int,
) -> dict[str, object]:
    return {
        "policy_type": policy_type,
        "feature": feature,
        "threshold": threshold,
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
    feature = str(policy.get("feature") or "")
    threshold = optional_float(policy.get("threshold"))
    value = optional_float(reliability_row.get(feature))
    return value is not None and threshold is not None and value >= threshold


def _validate_policy(policy: dict[str, object]) -> None:
    if _policy_audio_system(policy) == _policy_av_system(policy):
        raise ValueError("Audio and audio-visual system names must be different.")
    policy_type = policy.get("policy_type")
    if policy_type in {"roi_success", "audio_only"}:
        return
    if policy_type != "threshold":
        raise ValueError(f"Unsupported reliability policy: {policy_type}")
    if policy.get("feature") not in {item[0] for item in POLICY_FEATURES}:
        raise ValueError(f"Unsupported reliability feature: {policy.get('feature')}")
    if optional_float(policy.get("threshold")) is None:
        raise ValueError("Threshold reliability policy has no numeric threshold.")


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
    transcripts: dict[str, dict[str, str]] = {}
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
            transcripts[utterance_id] = {
                "prediction": str(record.get("prediction") or ""),
            }
    return transcripts


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
