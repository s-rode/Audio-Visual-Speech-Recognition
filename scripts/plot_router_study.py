"""Create a presentation-ready summary of the VoxMM routing experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

# Avoid requiring writable user-level Matplotlib or fontconfig cache directories.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "team1_asr_plot_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        type=Path,
        required=True,
        help="router_study.json written by team1-reliability study",
    )
    parser.add_argument(
        "--full-learned",
        type=Path,
        help="cross_validation_metrics.json written by team1-reliability learn",
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        help="metrics.json from the matched routing run; used only as an upper bound",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to a figures directory beside the study output.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    study = _read_json(args.study)
    output_dir = args.output_dir or args.study.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        (
            "Whisper\n(audio-only)",
            _percent(study["baseline"]["audio_wer"]),
            "#1f77b4",
        ),
        (
            "Threshold router\n(grouped CV)",
            _percent(study["baseline"]["routed_wer"]),
            "#7f8c8d",
        ),
        (
            "Compact router study\n(best nested CV)",
            _percent(study["best_configuration"]["routed_wer"]),
            "#2a9d8f",
        ),
    ]
    points = [
        (
            "Threshold router",
            100 * study["baseline"]["routed_to_audio_visual_rate"],
            _percent(study["baseline"]["routed_wer"]),
            "#7f8c8d",
        ),
        (
            "Compact router",
            100 * study["best_configuration"]["routed_to_audio_visual_rate"],
            _percent(study["best_configuration"]["routed_wer"]),
            "#2a9d8f",
        ),
    ]
    if args.full_learned:
        learned = _read_json(args.full_learned)["aggregate"]
        methods.append(
            (
                "Full logistic router\n(grouped CV)",
                _percent(learned["learned"]["wer"]),
                "#c44e52",
            )
        )
        points.append(
            (
                "Full logistic router",
                100 * learned["learned_audio_visual_rate"],
                _percent(learned["learned"]["wer"]),
                "#c44e52",
            )
        )
    if args.oracle:
        oracle = _read_json(args.oracle)["oracle"]
        methods.insert(
            0,
            ("Oracle\n(analysis only)", _percent(oracle["wer"]), "#b0b7be"),
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )
    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.3),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )
    _plot_wer_comparison(left, methods)
    _plot_route_tradeoff(right, points, _percent(study["baseline"]["audio_wer"]))
    figure.tight_layout(pad=1.2)
    for extension in ("png", "pdf"):
        figure.savefig(output_dir / f"voxmm_router_study.{extension}", dpi=300)
    print(f"Wrote {output_dir / 'voxmm_router_study.png'}")
    print(f"Wrote {output_dir / 'voxmm_router_study.pdf'}")


def _plot_wer_comparison(axis: plt.Axes, methods: list[tuple[str, float, str]]) -> None:
    labels, values, colors = zip(*methods)
    positions = list(range(len(methods)))
    bars = axis.barh(positions, values, color=colors, height=0.62)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, max(values) + 2.2)
    axis.set_xlabel("Word error rate (%)")
    axis.set_title("VoxMM dev: honest routing comparison")
    axis.xaxis.grid(True, color="#d9dde2", linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values):
        axis.text(
            value + 0.18,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            fontsize=10,
        )
    axis.text(
        0.99,
        -0.17,
        "Lower is better. Oracle uses references and is not a deployable router.",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#4f5964",
    )


def _plot_route_tradeoff(
    axis: plt.Axes,
    points: list[tuple[str, float, float, str]],
    whisper_wer: float,
) -> None:
    for label, route_rate, wer, color in points:
        axis.scatter(route_rate, wer, s=90, color=color, zorder=3)
        offset = (8, 11) if label == "Threshold router" else (8, -10)
        if label == "Full logistic router":
            offset = (8, 6)
        axis.annotate(
            label,
            (route_rate, wer),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
        )
    axis.axhline(whisper_wer, color="#1f77b4", linewidth=1.2, linestyle="--")
    axis.text(
        0.98,
        whisper_wer + 0.08,
        "Whisper-only WER",
        transform=axis.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9,
        color="#1f77b4",
    )
    axis.set_xlim(-1.5, max(point[1] for point in points) + 5)
    values = [point[2] for point in points] + [whisper_wer]
    axis.set_ylim(min(values) - 0.8, max(values) + 0.9)
    axis.set_xlabel("Clips routed to AVEC (%)")
    axis.set_ylabel("Word error rate (%)")
    axis.set_title("More AVEC routing did not help")
    axis.grid(True, color="#d9dde2", linewidth=0.8)
    axis.set_axisbelow(True)


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _percent(value: object) -> float:
    return 100 * float(value)


if __name__ == "__main__":
    main()
