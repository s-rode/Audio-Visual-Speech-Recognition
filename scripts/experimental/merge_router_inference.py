#!/usr/bin/env python3
"""Merge sharded inference outputs into strict router transcript files."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from team1_asr.evaluation import compute_metrics


def _read_jsonl(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        utterance_id = str(record.get("utterance_id", ""))
        if not utterance_id or utterance_id in records:
            raise ValueError(f"{path}:{line_number}: missing or duplicate utterance_id")
        records[utterance_id] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--system", choices=("whisper", "avec"), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-relative-path", default="roberta/fixed/best.jsonl")
    parser.add_argument("--metadata-json", default="{}")
    args = parser.parse_args()
    metadata = json.loads(args.metadata_json)
    if not isinstance(metadata, dict):
        raise ValueError("--metadata-json must encode an object")

    spec = json.loads(args.tasks.read_text(encoding="utf-8"))
    tasks = [task for task in spec["tasks"] if task["split"] == args.split]
    expected: dict[str, dict] = {}
    source_task: dict[str, dict] = {}
    for task in tasks:
        with Path(task["manifest_path"]).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                utterance_id = row["utterance_id"]
                if utterance_id in expected:
                    raise ValueError(f"duplicate task-manifest ID: {utterance_id}")
                expected[utterance_id] = row
                source_task[utterance_id] = task

    predictions: dict[str, dict] = {}
    for task in tasks:
        relative = "transcripts.jsonl" if args.system == "whisper" else args.prediction_relative_path
        path = args.input_root / task["name"] / relative
        if not path.exists():
            raise FileNotFoundError(path)
        for utterance_id, record in _read_jsonl(path).items():
            if utterance_id in predictions:
                raise ValueError(f"duplicate output ID: {utterance_id}")
            predictions[utterance_id] = record

    missing = sorted(set(expected) - set(predictions))
    extra = sorted(set(predictions) - set(expected))
    if missing or extra:
        raise ValueError({"missing_count": len(missing), "extra_count": len(extra)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(expected)
    merged = []
    for utterance_id in ordered:
        row = expected[utterance_id]
        record = predictions[utterance_id]
        if args.system == "whisper":
            merged.append(record)
            continue
        merged.append(
            {
                "utterance_id": utterance_id,
                "clean_audio_path": row["audio_path"],
                "used_audio_path": row["audio_path"],
                "video_path": row["video_path"],
                "reference": row["reference"],
                "prediction": str(record.get("prediction", "")),
                "language": row.get("language", "en"),
                "duration": None,
                "segments": [],
                "metadata": {
                    "backend": "avec",
                    "source_task": source_task[utterance_id]["name"],
                    "lm_path": source_task[utterance_id].get("lm_path"),
                    **metadata,
                },
            }
        )
    transcript_path = args.output_dir / "transcripts.jsonl"
    transcript_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in merged),
        encoding="utf-8",
    )
    metrics = compute_metrics(
        [expected[utterance_id]["reference"] for utterance_id in ordered],
        [str(predictions[utterance_id].get("prediction", "")) for utterance_id in ordered],
    )
    report = {
        "split": args.split,
        "system": args.system,
        "utterances": len(ordered),
        "coverage": 1.0,
        "wer": metrics.wer,
        "cer": metrics.cer,
        "raw_wer": metrics.raw_wer,
        "raw_cer": metrics.raw_cer,
        "transcript_path": str(transcript_path),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
