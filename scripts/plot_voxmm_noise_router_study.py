"""Create presentation figures for a noise-aware routing study."""
from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with args.study.open(encoding="utf-8") as handle:
        study = json.load(handle)

    configurations = study["configurations"]
    baseline_name = str(study.get("baseline_configuration", "combined_noise"))
    baseline = configurations[baseline_name]
    benefit_names = [
        str(config["name"])
        for config in study.get("benefit_router_configurations", [])
        if str(config["name"]) in configurations
    ]
    best_benefit_name = min(
        benefit_names,
        key=lambda name: float(configurations[name]["overall"]["routed_wer"]),
        default=study["best_configuration"],
    )
    best_benefit = configurations[best_benefit_name]
    conditions = study["conditions"]
    baseline_rows = [baseline["by_condition"][condition["name"]] for condition in conditions]
    benefit_rows = [best_benefit["by_condition"][condition["name"]] for condition in conditions]
    labels = [_condition_label(condition) for condition in conditions]
    output_dir = args.output_dir or args.study.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )
    figure, (wer_axis, decisions_axis) = plt.subplots(1, 2, figsize=(12.2, 4.8))
    _plot_wer(wer_axis, labels, baseline_rows, benefit_rows, best_benefit_name)
    _plot_decisions(decisions_axis, labels, benefit_rows, best_benefit_name)
    figure.suptitle(
        "VoxMM-600: noise-aware router comparison",
        y=1.02,
        fontsize=14,
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        path = output_dir / f"voxmm_noise_router_study.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")


def _plot_wer(
    axis: plt.Axes,
    labels: list[str],
    baseline_rows: list[dict[str, object]],
    benefit_rows: list[dict[str, object]],
    benefit_name: str,
) -> None:
    audio = [100 * float(row["audio_wer"]) for row in baseline_rows]
    logistic = [100 * float(row["routed_wer"]) for row in baseline_rows]
    benefit = [100 * float(row["routed_wer"]) for row in benefit_rows]
    oracle = [100 * float(row["oracle_wer"]) for row in baseline_rows]
    axis.plot(labels, audio, "o-", color="#1f77b4", label="Whisper base.en")
    axis.plot(labels, logistic, "o-", color="#7f8c8d", label="Current logistic router")
    axis.plot(
        labels,
        benefit,
        "o-",
        color="#2a9d8f",
        label=f"Best new router ({benefit_name.replace('benefit_', '')})",
    )
    axis.plot(labels, oracle, "o--", color="#b0b7be", label="Oracle (analysis only)")
    for index, (label, baseline, result) in enumerate(zip(labels, audio, benefit)):
        gain = baseline - result
        offset = 12 if index < 2 else -15
        axis.annotate(
            f"-{gain:.2f}",
            (label, result),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#2a9d8f",
        )
    axis.set_ylabel("Word error rate (%)")
    axis.set_title("Does benefit-aware routing close the oracle gap?")
    axis.grid(axis="y", color="#d9dde2", linewidth=0.8)
    axis.legend(frameon=False, loc="upper left")
    axis.text(
        0.99,
        -0.17,
        "Labels show the best new router's WER reduction versus Whisper.",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#4f5964",
    )


def _plot_decisions(
    axis: plt.Axes,
    labels: list[str],
    rows: list[dict[str, object]],
    benefit_name: str,
) -> None:
    selected = [100 * float(row["routed_to_audio_visual_rate"]) for row in rows]
    false = [
        100 * int(row["false_audio_visual_routes"]) / int(row["utterances_scored"])
        for row in rows
    ]
    not_false = [selection - error for selection, error in zip(selected, false)]
    axis.bar(labels, not_false, color="#2a9d8f", label="AVEC route: better or tied")
    axis.bar(
        labels,
        false,
        bottom=not_false,
        color="#c44e52",
        hatch="//",
        label="AVEC route: worse",
    )
    for label, value in zip(labels, selected):
        axis.annotate(
            f"{value:.1f}%",
            (label, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axis.set_ylabel("Share of clips routed to AVEC (%)")
    axis.set_title(f"{benefit_name.replace('_', ' ')} routing decisions")
    axis.grid(axis="y", color="#d9dde2", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left")


def _condition_label(condition: dict[str, object]) -> str:
    snr = condition.get("snr_db")
    if snr is None:
        return "clean"
    return f"{int(float(snr))} dB"


if __name__ == "__main__":
    main()
