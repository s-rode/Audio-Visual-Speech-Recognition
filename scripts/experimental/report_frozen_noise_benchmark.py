#!/usr/bin/env python3
"""Summarize a frozen router policy across clean and DEMAND-noise test audio."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from team1_asr.reliability import _bootstrap_router_comparison


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_decisions(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bootstrap(decisions: list[dict[str, object]], seed: int) -> dict[str, object]:
    baseline = []
    for row in decisions:
        copy = dict(row)
        copy["selected_word_errors"] = row["audio_word_errors"]
        baseline.append(copy)
    return _bootstrap_router_comparison(baseline, decisions, seed=seed, conditions=[])["overall"]


def _percent(value: object) -> float:
    return 100.0 * float(value)


def _summary(name: str, snr: float | None, final_dir: Path, seed: int) -> dict[str, object]:
    metrics = _read_json(final_dir / "metrics.json")
    decisions = _read_decisions(final_dir / "routing_decisions.csv")
    if int(metrics["num_utterances"]) != len(decisions):
        raise ValueError(f"{final_dir}: metrics/decision coverage mismatch")
    comparison = _bootstrap(decisions, seed)
    return {
        "condition": name,
        "snr_db": snr,
        "utterances": int(metrics["num_utterances"]),
        "whisper_wer": _percent(metrics["audio"]["wer"]),
        "whisper_cer": _percent(metrics["audio"]["cer"]),
        "avec_wer": _percent(metrics["audio_visual"]["wer"]),
        "avec_cer": _percent(metrics["audio_visual"]["cer"]),
        "routed_wer": _percent(metrics["routed"]["wer"]),
        "routed_cer": _percent(metrics["routed"]["cer"]),
        "oracle_wer": _percent(metrics["oracle"]["wer"]),
        "oracle_cer": _percent(metrics["oracle"]["cer"]),
        "oracle_gap_wer_pp": _percent(metrics["oracle_gap_wer"]),
        "avec_route_rate": _percent(metrics["routed_to_audio_visual_rate"]),
        "avec_route_count": int(metrics["routed_to_audio_visual_count"]),
        "false_avec_routes": int(metrics["false_audio_visual_routes"]),
        "missed_avec_wins": int(metrics["missed_audio_visual_wins"]),
        "routed_wer_reduction_vs_whisper_pp": comparison["wer_reduction_pp"],
        "routed_wer_reduction_ci_95_low_pp": comparison["ci_95_low_pp"],
        "routed_wer_reduction_ci_95_high_pp": comparison["ci_95_high_pp"],
        "source_videos": comparison["source_videos"],
        "final_dir": str(final_dir),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, object]], output_dir: Path) -> None:
    cache = Path(tempfile.gettempdir()) / "team1_asr_plot_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    import matplotlib.pyplot as plt

    labels = ["clean" if row["snr_db"] is None else f"{int(float(row['snr_db']))} dB" for row in rows]
    figure, (wer_axis, route_axis) = plt.subplots(1, 2, figsize=(12, 4.6))
    for key, label, color, style in (
        ("whisper_wer", "Whisper", "#1f77b4", "o-"),
        ("avec_wer", "AVEC", "#e76f51", "o-"),
        ("routed_wer", "Frozen router", "#2a9d8f", "o-"),
        ("oracle_wer", "Oracle", "#a8adb4", "o--"),
    ):
        wer_axis.plot(labels, [row[key] for row in rows], style, color=color, label=label)
    wer_axis.set_ylabel("Word error rate (%)")
    wer_axis.set_title("Frozen policy under DEMAND noise")
    wer_axis.grid(axis="y", alpha=0.3)
    wer_axis.legend(frameon=False)

    selected = [row["avec_route_rate"] for row in rows]
    false = [100 * row["false_avec_routes"] / row["utterances"] for row in rows]
    route_axis.bar(labels, [value - error for value, error in zip(selected, false)], color="#2a9d8f", label="AVEC better/tied")
    route_axis.bar(labels, false, bottom=[value - error for value, error in zip(selected, false)], color="#c44e52", hatch="//", label="AVEC worse")
    route_axis.set_ylabel("Utterances routed to AVEC (%)")
    route_axis.set_title("Frozen routing decisions")
    route_axis.grid(axis="y", alpha=0.3)
    route_axis.set_axisbelow(True)
    route_axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"frozen_demand_noise_benchmark.{suffix}", dpi=300, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-final", type=Path, required=True)
    parser.add_argument("--condition", action="append", nargs=3, metavar=("NAME", "SNR_DB", "FINAL_DIR"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provenance-json", default="{}")
    args = parser.parse_args()
    provenance = json.loads(args.provenance_json)
    if not isinstance(provenance, dict):
        raise ValueError("--provenance-json must encode an object")
    rows = [_summary("clean", None, args.clean_final, seed=1701)]
    for index, (name, snr, final_dir) in enumerate(args.condition, 1):
        rows.append(_summary(name, float(snr), Path(final_dir), seed=1701 + index))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "frozen_noise_metrics.csv", rows)
    report = {"benchmark": "frozen_policy_synthetic_demand_noise", "conditions": rows, "provenance": provenance}
    (args.output_dir / "frozen_noise_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _plot(rows, args.output_dir)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
