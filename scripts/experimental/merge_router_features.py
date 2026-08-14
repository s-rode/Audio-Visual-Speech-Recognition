#!/usr/bin/env python3
"""Strictly merge per-task pre-AVEC router features."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BLOCKED = ("avec", "audio_visual", "disagreement", "reference", "prediction")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()
    tasks = [task for task in json.loads(args.tasks.read_text(encoding="utf-8"))["tasks"] if task["split"] == args.split]
    expected: dict[str, dict[str, str]] = {}
    records: dict[str, dict[str, str]] = {}
    for task in tasks:
        with Path(task["manifest_path"]).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                utterance_id = row["utterance_id"]
                if utterance_id in expected:
                    raise ValueError(f"duplicate task-manifest ID {utterance_id}")
                expected[utterance_id] = row
        path = args.input_root / str(task["name"]) / "routing_features.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                utterance_id = row.pop("utterance_id", "")
                condition = row.pop("condition", "")
                if condition != args.condition or not utterance_id or utterance_id in records:
                    raise ValueError(f"invalid feature row {path}: {utterance_id}")
                blocked = [name for name in row if any(token in name.lower() for token in BLOCKED)]
                if blocked:
                    raise ValueError(f"non-deployable feature(s): {blocked}")
                records[utterance_id] = row
    if set(expected) != set(records):
        raise ValueError({"missing": len(set(expected) - set(records)), "extra": len(set(records) - set(expected))})
    rows = [{"condition": args.condition, "utterance_id": utterance_id, **records[utterance_id]} for utterance_id in sorted(expected)]
    fields = list(rows[0]) if rows else ["condition", "utterance_id"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "routing_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "coverage.json").write_text(json.dumps({"split": args.split, "condition": args.condition, "utterances": len(rows), "feature_columns": fields[2:]}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split": args.split, "condition": args.condition, "utterances": len(rows), "columns": len(fields) - 2}), flush=True)


if __name__ == "__main__":
    main()
