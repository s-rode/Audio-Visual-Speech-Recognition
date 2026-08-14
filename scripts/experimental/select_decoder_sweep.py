#!/usr/bin/env python3
"""Select a decoder setting from sharded sweep summaries by corpus WER."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--task-name", action="append", required=True)
    parser.add_argument("--part", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    totals: dict[str, dict] = defaultdict(lambda: {"hits": 0, "substitutions": 0, "deletions": 0, "insertions": 0})
    expected_paths = [
        args.result_root / task / part / "results.json"
        for task in args.task_name
        for part in args.part
    ]
    missing = [str(path) for path in expected_paths if not path.exists()]
    if missing:
        raise FileNotFoundError({"missing_result_count": len(missing), "first_missing": missing[:5]})
    for path in expected_paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        for result in summary["results"]:
            name = result["name"]
            total = totals[name]
            for field in ("hits", "substitutions", "deletions", "insertions"):
                total[field] += int(result[field])
            total["lm_weight"] = float(result["lm_weight"])
            total["word_bonus"] = float(result["word_bonus"])
            total["beam_size"] = int(result["beam_size"])
            total["token_cutoff"] = int(result["token_cutoff"])
            total["beam_threshold"] = float(result["beam_threshold"])
            total["log_add"] = bool(result["log_add"])
            total["temperature"] = float(result["temperature"])

    candidates = []
    for name, total in totals.items():
        errors = total["substitutions"] + total["deletions"] + total["insertions"]
        words = total["hits"] + total["substitutions"] + total["deletions"]
        if words <= 0:
            raise ValueError(f"no reference words for {name}")
        candidate = {"name": name, **total, "errors": errors, "reference_words": words, "wer": 100.0 * errors / words}
        candidates.append(candidate)
    if not candidates:
        raise ValueError("no sweep results found")
    candidates.sort(key=lambda item: (item["errors"], item["word_bonus"], item["lm_weight"], item["name"]))
    best = candidates[0]
    setting = ":".join(
        [
            "selected",
            str(best["lm_weight"]),
            str(best["word_bonus"]),
            str(best["beam_size"]),
            str(best["token_cutoff"]),
            str(best["beam_threshold"]),
            str(best["log_add"]).lower(),
            str(best["temperature"]),
        ]
    )
    output = {
        "selection_metric": "corpus_word_error",
        "tie_break": ["lower_word_bonus", "lower_lm_weight", "setting_name"],
        "task_names": args.task_name,
        "parts": args.part,
        "selected": best,
        "setting": setting,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"setting": setting, "wer": best["wer"], "errors": best["errors"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
