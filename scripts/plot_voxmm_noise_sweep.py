"""Create presentation-ready figures from VoxMM DEMAND sweep metrics."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

_CACHE_ROOT = Path(tempfile.gettempdir()) / "team1_asr_plot_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with args.metrics.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: _snr_sort_value(row["snr_db"]), reverse=True)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_wer(rows, output_dir)
    _plot_routing_signal(rows, output_dir)


def _plot_wer(rows: list[dict[str, str]], output_dir: Path) -> None:
    labels = [_label(row) for row in rows]
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    for key, label, color in (
        ("whisper_wer", "Whisper base.en", "#1f77b4"),
        ("ao_wer", "AVEC AO fine-tuned", "#7f8c8d"),
        ("av_wer", "AVEC AV fine-tuned", "#2a9d8f"),
    ):
        axis.plot(labels, [100 * float(row[key]) for row in rows], "o-", label=label, color=color)
    axis.set_xlabel("Audio condition")
    axis.set_ylabel("Word error rate (%)")
    axis.set_title("VoxMM-200: DEMAND background-noise sweep")
    axis.grid(axis="y", color="#d9dde2", linewidth=0.8)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save(figure, output_dir / "voxmm_demand_wer")


def _plot_routing_signal(rows: list[dict[str, str]], output_dir: Path) -> None:
    labels = [_label(row) for row in rows]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    left.bar(labels, [100 * float(row["av_wins_over_whisper_rate"]) for row in rows], color="#2a9d8f")
    left.set_ylabel("AVEC-AV wins over Whisper (%)")
    left.set_title("Potential AVEC routing signal")
    left.grid(axis="y", color="#d9dde2", linewidth=0.8)
    right.bar(labels, [100 * float(row["oracle_gap_from_whisper"]) for row in rows], color="#b0b7be")
    right.set_ylabel("WER reduction versus Whisper (points)")
    right.set_title("Oracle headroom")
    right.grid(axis="y", color="#d9dde2", linewidth=0.8)
    figure.tight_layout()
    _save(figure, output_dir / "voxmm_demand_routing_signal")


def _save(figure, stem: Path) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=300)
    print(f"Wrote {stem.with_suffix('.png')}")
    print(f"Wrote {stem.with_suffix('.pdf')}")


def _label(row: dict[str, str]) -> str:
    if not row["snr_db"]:
        return "clean"
    value = float(row["snr_db"])
    return f"{int(value) if value.is_integer() else value} dB"


def _snr_sort_value(value: str) -> float:
    return 1000.0 if not value else float(value)


if __name__ == "__main__":
    main()
