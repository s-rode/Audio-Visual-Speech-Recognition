from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, TextIO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print reference and prediction pairs from a transcripts.jsonl file."
    )
    parser.add_argument(
        "transcripts",
        nargs="?",
        default="outputs/librispeech_demo_tiny/transcripts.jsonl",
        help="Path to a transcripts.jsonl file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of transcript records to print.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print_transcripts(Path(args.transcripts), limit=args.limit)


def print_transcripts(path: Path, limit: int, output: TextIO | None = None) -> None:
    if limit < 1:
        raise ValueError("--limit must be at least 1.")
    stream = output or sys.stdout
    for index, row in enumerate(_read_jsonl(path)):
        if index >= limit:
            break
        utterance_id = row.get("utterance_id", f"utterance-{index}")
        reference = row.get("reference") or ""
        prediction = row.get("prediction") or ""
        print(f"[{utterance_id}]", file=stream)
        print(f"Reference : {reference}", file=stream)
        print(f"Prediction: {prediction}", file=stream)
        if index + 1 < limit:
            print(file=stream)


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
