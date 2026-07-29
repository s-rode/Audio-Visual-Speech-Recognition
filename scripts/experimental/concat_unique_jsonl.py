#!/usr/bin/env python3
"""Concatenate JSONL shards while enforcing unique utterance IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    seen: set[str] = set()
    records: list[dict] = []
    for path in args.inputs:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            utterance_id = str(record.get("utterance_id", ""))
            if not utterance_id or utterance_id in seen:
                raise ValueError(f"{path}:{line_number}: missing or duplicate utterance_id")
            seen.add(utterance_id)
            records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "utterances": len(records)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
