"""Analyze the standard VoxMM-200 DEMAND noise sweep."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from team1_asr.noise_sweep import (
    analyze_condition,
    flatten_condition,
    read_manifest_references,
    read_predictions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clean-whisper", type=Path, required=True)
    parser.add_argument("--clean-ao", type=Path, required=True)
    parser.add_argument("--clean-av", type=Path, required=True)
    parser.add_argument(
        "--noise-root",
        type=Path,
        required=True,
        help="Contains whisper_demand_*, avec_ao_demand_*, and avec_av_demand_* outputs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    references = read_manifest_references(args.manifest)
    conditions = [
        _condition(
            "clean",
            None,
            args.clean_whisper,
            args.clean_ao,
            args.clean_av,
        ),
        *[
            _condition_from_root(args.noise_root, snr_db)
            for snr_db in (10, 0, -5)
        ],
    ]
    results = [
        analyze_condition(
            name=name,
            snr_db=snr_db,
            references=references,
            whisper=read_predictions(whisper_path),
            avec_ao=read_predictions(ao_path),
            avec_av=read_predictions(av_path),
        )
        for name, snr_db, whisper_path, ao_path, av_path in conditions
    ]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "noise_sweep_metrics.json").write_text(
        json.dumps({"conditions": results}, indent=2), encoding="utf-8"
    )
    rows = [flatten_condition(result) for result in results]
    with (output_dir / "noise_sweep_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_dir / 'noise_sweep_metrics.json'}")
    print(f"Wrote {output_dir / 'noise_sweep_metrics.csv'}")


def _condition_from_root(root: Path, snr_db: int) -> tuple[str, float, Path, Path, Path]:
    label = f"demand_{snr_db}db"
    return _condition(
        label,
        float(snr_db),
        root / f"whisper_{label}" / "transcripts.jsonl",
        root / f"avec_ao_{label}" / "transcripts.jsonl",
        root / f"avec_av_{label}" / "transcripts.jsonl",
    )


def _condition(
    name: str,
    snr_db: float | None,
    whisper_path: Path,
    ao_path: Path,
    av_path: Path,
) -> tuple[str, float | None, Path, Path, Path]:
    return name, snr_db, whisper_path, ao_path, av_path


if __name__ == "__main__":
    main()
