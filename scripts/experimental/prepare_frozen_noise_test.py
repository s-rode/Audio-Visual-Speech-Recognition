#!/usr/bin/env python3
"""Create deterministic DEMAND-noise task manifests for a frozen test run."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from team1_asr.config import DemandNoiseConfig, NoiseConfig
from team1_asr.noise import maybe_apply_noise


def _tasks(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [task for task in data["tasks"] if task["split"] == "test"]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_spec(args: argparse.Namespace) -> None:
    tasks = _tasks(args.tasks)
    output = args.output_root / "conditions" / args.condition / "noise_tasks.json"
    rewritten = []
    for task in tasks:
        task = dict(task)
        task["manifest_path"] = str(
            args.output_root / "conditions" / args.condition / "manifests" / str(task["name"]) / "manifest.csv"
        )
        rewritten.append(task)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"version": "frozen_demand_test_v1", "tasks": rewritten}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "tasks": len(rewritten)}, indent=2), flush=True)


def materialize(args: argparse.Namespace) -> None:
    tasks = {int(task["task_id"]): task for task in _tasks(args.tasks)}
    task = tasks[args.task_id]
    source = Path(str(task["manifest_path"]))
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"utterance_id", "audio_path", "reference", "video_path"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"{source}: missing fields {sorted(missing)}")
    if len({row["utterance_id"] for row in rows}) != len(rows):
        raise ValueError(f"{source}: duplicate utterance IDs")
    if "clean_audio_path" not in fields:
        fields.append("clean_audio_path")
    if "condition" not in fields:
        fields.append("condition")

    noise = NoiseConfig(
        seed=args.seed,
        demand=DemandNoiseConfig(enabled=True, snr_db=args.snr_db, noise_path=str(args.demand_root)),
    )
    task_root = args.output_root / "conditions" / args.condition / "tasks" / str(task["name"])
    prepared = []
    for index, row in enumerate(rows, 1):
        original = row["audio_path"]
        noisy = maybe_apply_noise(original, noise, task_root, utterance_id=row["utterance_id"])
        row = dict(row)
        row["clean_audio_path"] = original
        row["audio_path"] = noisy
        row["condition"] = args.condition
        prepared.append(row)
        if index % 100 == 0 or index == len(rows):
            print(f"{args.condition} {task['name']}: materialized {index}/{len(rows)}", flush=True)
    output = args.output_root / "conditions" / args.condition / "manifests" / str(task["name"]) / "manifest.csv"
    _write_csv(output, fields, prepared)
    print(json.dumps({"task": task["name"], "condition": args.condition, "utterances": len(prepared), "manifest": str(output)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-spec")
    build.add_argument("--tasks", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--condition", required=True)
    build.set_defaults(func=build_spec)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--tasks", type=Path, required=True)
    materialize_parser.add_argument("--output-root", type=Path, required=True)
    materialize_parser.add_argument("--task-id", type=int, required=True)
    materialize_parser.add_argument("--condition", required=True)
    materialize_parser.add_argument("--snr-db", type=float, required=True)
    materialize_parser.add_argument("--seed", type=int, default=0)
    materialize_parser.add_argument("--demand-root", type=Path, required=True)
    materialize_parser.set_defaults(func=materialize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
