"""Plot WER & CER vs SNR curves, one curve per model.

Usage:
    python plot_snr_curves.py
    python plot_snr_curves.py --metric wer          # WER only
    python plot_snr_curves.py --metric cer          # CER only
    python plot_snr_curves.py --metric both         # default
    python plot_snr_curves.py --out outputs/curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURE YOUR MODELS HERE
# Each model maps SNR (int dB, or float("inf") for clean) -> output directory.
# ---------------------------------------------------------------------------
MODELS: dict[str, dict[float, str]] = {
    "Whisper tiny.en": {
        float("inf"): "outputs/lrs3_test_tiny_clean",
        30: "outputs/lrs3-test-tiny-whisper-advanced-30db",
        20: "outputs/lrs3-test-tiny-whisper-advanced-20db",
        15: "outputs/lrs3-test-tiny-whisper-advanced-15db",
        10: "outputs/lrs3-test-tiny-whisper-advanced-10db",
        5:  "outputs/lrs3-test-tiny-whisper-advanced-5db",
        0:  "outputs/lrs3-test-tiny-whisper-advanced-0db",
        -5: "outputs/lrs3-test-tiny-whisper-advanced-minus5db",
    },
    "Whisper small.en": {
        float("inf"): "outputs/lrs3_test_small_clean",
        30: "outputs/lrs3-test-small-whisper-advanced-30db",
        20: "outputs/lrs3-test-small-whisper-advanced-20db",
        15: "outputs/lrs3-test-small-whisper-advanced-15db",
        10: "outputs/lrs3-test-small-whisper-advanced-10db",
        5:  "outputs/lrs3-test-small-whisper-advanced-5db",
        0:  "outputs/lrs3-test-small-whisper-advanced-0db",
        -5: "outputs/lrs3-test-small-whisper-advanced-minus5db",
    },
    "Whisper medium.en": {
        float("inf"): "outputs/lrs3_test_medium_clean",
        30: "outputs/lrs3-test-medium-whisper-advanced-30db",
        20: "outputs/lrs3-test-medium-whisper-advanced-20db",
        15: "outputs/lrs3-test-medium-whisper-advanced-15db",
        10: "outputs/lrs3-test-medium-whisper-advanced-10db",
        5:  "outputs/lrs3-test-medium-whisper-advanced-5db",
        0:  "outputs/lrs3-test-medium-whisper-advanced-0db",
        -5: "outputs/lrs3-test-medium-whisper-advanced-minus5db",
    },
    "Whisper large": {
        float("inf"): "outputs/lrs3_test_large_clean",
        30: "outputs/lrs3-test-large-whisper-advanced-30db",
        20: "outputs/lrs3-test-large-whisper-advanced-20db",
        15: "outputs/lrs3-test-large-whisper-advanced-15db",
        10: "outputs/lrs3-test-large-whisper-advanced-10db",
        5:  "outputs/lrs3-test-large-whisper-advanced-5db",
        0:  "outputs/lrs3-test-large-whisper-advanced-0db",
        -5: "outputs/lrs3-test-large-whisper-advanced-minus5db",
    },
    "Granite": {
        float("inf"): "outputs/lrs3_test_granite_speech_clean",
        30: "outputs/lrs3_test_granite_speech_advanced_30db",
        20: "outputs/lrs3_test_granite_speech_advanced_20db",
        15: "outputs/lrs3_test_granite_speech_advanced_15db",
        10: "outputs/lrs3_test_granite_speech_advanced_10db",
        5:  "outputs/lrs3_test_granite_speech_advanced_5db",
        0:  "outputs/lrs3_test_granite_speech_advanced_0db",
        -5: "outputs/lrs3_test_granite_speech_advanced_minus5db",
    },
    "Parakeet": {
        float("inf"): "outputs/lrs3-test-parakeet-coreml-clean",
        30: "outputs/lrs3-test-parakeet-coreml-advanced-30db",
        20: "outputs/lrs3-test-parakeet-coreml-advanced-20db",
        15: "outputs/lrs3-test-parakeet-coreml-advanced-15db",
        10: "outputs/lrs3-test-parakeet-coreml-advanced-10db",
        5:  "outputs/lrs3-test-parakeet-coreml-advanced-5db",
        0:  "outputs/lrs3-test-parakeet-coreml-advanced-0db",
        -5: "outputs/lrs3-test-parakeet-coreml-advanced-minus5db",
    },

}

# One colour per model (extend if needed)
MODEL_COLORS = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#0891b2",  # cyan
]

# WER = solid, CER = dashed
LINESTYLES = {"wer": ("-",  "o"), "cer": ("--", "s")}

# X position for the "clean" point – placed just right of the highest finite SNR
CLEAN_X_OFFSET = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metrics(directory: Path) -> dict | None:
    f = directory / "metrics.json"
    if not f.exists():
        return None
    with f.open() as fh:
        return json.load(fh)


def collect() -> dict[str, list[dict]]:
    """
    Returns { model_name: [ {snr, wer, cer}, ... ] } sorted by SNR ascending.
    Missing SNR points are simply omitted (the line will have a gap).
    """
    all_data: dict[str, list[dict]] = {}

    for model_name, snr_map in MODELS.items():
        rows = []
        print(f"\n{model_name}")
        for snr, directory in snr_map.items():
            metrics = load_metrics(Path(directory))
            if metrics is None:
                print(f"  [missing]  {directory}/metrics.json")
                continue
            wer = metrics.get("wer")
            cer = None #metrics.get("cer")
            if wer is None:
                print(f"  [no WER]   {directory}")
                continue
            rows.append({"snr": snr, "wer": wer * 100, "cer": cer * 100 if cer is not None else None})
            cer_str = f"CER={cer:.3f}" if cer is not None else "CER=n/a"
            label = "Clean" if snr == float("inf") else f"SNR={int(snr):>4} dB"
            print(f"  ✓  {label}   WER={wer:.3f}  {cer_str}")

        all_data[model_name] = sorted(rows, key=lambda r: (r["snr"] == float("inf"), r["snr"]))

    return all_data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_x_axis(all_data: dict[str, list[dict]]) -> tuple[list[float], list[float], list[str]]:
    """
    Build a shared numeric x axis from every SNR value seen across all models.
    Returns (all_snrs_sorted, x_numeric, x_labels).
    """
    snr_set: set[float] = set()
    for rows in all_data.values():
        for r in rows:
            snr_set.add(r["snr"])

    finite_snrs = sorted(s for s in snr_set if s != float("inf"))
    max_finite  = finite_snrs[-1] if finite_snrs else 30
    clean_x     = max_finite + CLEAN_X_OFFSET

    all_snrs_sorted = finite_snrs + ([float("inf")] if float("inf") in snr_set else [])

    x_numeric: list[float] = []
    x_labels:  list[str]   = []
    for snr in all_snrs_sorted:
        if snr == float("inf"):
            x_numeric.append(clean_x)
            x_labels.append("Clean\n(∞ dB)")
        else:
            x_numeric.append(float(snr))
            x_labels.append(f"{int(snr)} dB")

    return all_snrs_sorted, x_numeric, x_labels


def plot(
    all_data: dict[str, list[dict]],
    metric: str,        # "wer", "cer", or "both"
    out_path: Path,
) -> None:
    if not all_data:
        print("\nNo data to plot.")
        return

    all_snrs, x_numeric, x_labels = make_x_axis(all_data)
    snr_to_x = dict(zip(all_snrs, x_numeric))

    metrics_to_plot = ["wer", "cer"] if metric == "both" else [metric]
    has_cer = any(
        r["cer"] is not None
        for rows in all_data.values()
        for r in rows
    )
    if "cer" in metrics_to_plot and not has_cer:
        print("  [warn] No CER data found; plotting WER only.")
        metrics_to_plot = ["wer"]

    fig, ax = plt.subplots(figsize=(11, 5))

    for (model_name, rows), color in zip(all_data.items(), MODEL_COLORS):
        # Build arrays aligned to the shared x axis (None = missing)
        snr_lookup = {r["snr"]: r for r in rows}

        for met in metrics_to_plot:
            ls, marker = LINESTYLES[met]
            xs, ys = [], []
            for snr, xv in zip(all_snrs, x_numeric):
                r = snr_lookup.get(snr)
                if r is not None and r[met] is not None:
                    xs.append(xv)
                    ys.append(r[met])

            if not xs:
                continue

            label = model_name if len(metrics_to_plot) == 1 else f"{model_name} ({met.upper()})"
            ax.plot(
                xs, ys,
                linestyle=ls, marker=marker,
                color=color,
                linewidth=2, markersize=7,
                label=label,
                alpha=1.0 if met == "wer" else 0.7,
            )

            # Annotate each point
            offset_y = 8 if met == "wer" else -14
            for xv, yv in zip(xs, ys):
                ax.annotate(
                    f"{yv:.1f}%", (xv, yv),
                    textcoords="offset points",
                    xytext=(0, offset_y),
                    ha="center", fontsize=7.5, color=color,
                    alpha=1.0 if met == "wer" else 0.8,
                )

    ax.set_xticks(x_numeric)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel("SNR (dB)  ←  more noise", fontsize=11)
    ax.set_ylabel("Error Rate (%)", fontsize=11)

    metric_str = {"wer": "WER", "cer": "CER", "both": "WER & CER"}[metric]
    ax.set_title(f"Model comparison — WER vs SNR\n(LRS3)", fontsize=13)

    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_ylim(bottom=0)

    # Separator between finite-SNR region and clean
    if float("inf") in snr_to_x:
        sep_x = (max(v for s, v in snr_to_x.items() if s != float("inf"))
                 + snr_to_x[float("inf")]) / 2
        ax.axvline(sep_x, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metric",
        choices=["wer", "cer", "both"],
        default="both",
        help="Which metric(s) to plot (default: both).",
    )
    parser.add_argument(
        "--out",
        default="outputs/wer_cer_vs_snr_by_model.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    print("Loading results...")
    all_data = collect()

    # Drop models with no data at all
    all_data = {m: rows for m, rows in all_data.items() if rows}

    if not all_data:
        print("\nNo complete results found – nothing to plot.")
        return

    plot(all_data, args.metric, Path(args.out))


if __name__ == "__main__":
    main()