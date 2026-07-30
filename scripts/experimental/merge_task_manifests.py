#!/usr/bin/env python3
"""Merge a task specification's disjoint manifests with strict ID coverage."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = [task for task in json.loads(args.tasks.read_text(encoding="utf-8"))["tasks"] if task["split"] == args.split]
    fields: list[str] | None = None
    rows: list[dict[str, str]] = []
    utterance_ids: set[str] = set()
    for task in tasks:
        with Path(task["manifest_path"]).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fields is None:
                fields = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fields:
                raise ValueError(f"incompatible header in {task['manifest_path']}")
            for row in reader:
                utterance_id = row.get("utterance_id", "")
                if not utterance_id or utterance_id in utterance_ids:
                    raise ValueError(f"missing or duplicate utterance ID: {utterance_id}")
                utterance_ids.add(utterance_id)
                rows.append(row)
    if not fields:
        raise ValueError(f"no {args.split} tasks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"split": args.split, "utterances": len(rows), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
