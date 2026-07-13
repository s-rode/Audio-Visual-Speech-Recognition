"""Compare WER & CER across models at a specific SNR level.

Usage:
    python plot_models_at_snr.py --snr 10
    python plot_models_at_snr.py --snr clean        # ∞ dB / no noise
    python plot_models_at_snr.py --snr 10 --out outputs/compare_10db.png
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

# Colours assigned to models in insertion order (extend if you have more models)
MODEL_COLORS = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#0891b2",  # cyan
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_snr(raw: str) -> float:
    """Accept 'clean', 'inf', or an integer string like '10' or '-5'."""
    if raw.lower() in ("clean", "inf"):
        return float("inf")
    return float(int(raw))  # keeps -5 etc. working


def snr_label(snr: float) -> str:
    return "Clean (∞ dB)" if snr == float("inf") else f"{int(snr)} dB"


def load_metrics(path: Path) -> dict | None:
    """Return metrics dict or None on any error."""
    metrics_file = path / "metrics.json"
    if not metrics_file.exists():
        return None
    with metrics_file.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def collect(snr: float) -> list[dict]:
    """
    Returns a list of dicts:
        {"model": str, "wer": float, "cer": float | None}
    Only includes models that have a valid metrics.json for the given SNR.
    """
    rows = []
    print(f"\nLoading results for SNR = {snr_label(snr)}\n")

    for model_name, snr_map in MODELS.items():
        if snr not in snr_map:
            print(f"  [no entry]  {model_name} has no path for SNR={snr_label(snr)}")
            continue

        output_dir = Path(snr_map[snr])
        metrics = load_metrics(output_dir)

        if metrics is None:
            print(f"  [missing]   {output_dir / 'metrics.json'}")
            continue

        wer = metrics.get("wer")
        cer = metrics.get("cer")

        if wer is None:
            print(f"  [no WER]    {output_dir}")
            continue

        rows.append({"model": model_name, "wer": wer * 100, "cer": cer * 100 if cer is not None else None})
        cer_str = f"CER={cer:.3f}" if cer is not None else "CER=n/a"
        print(f"  ✓  {model_name:<25}  WER={wer:.3f}  {cer_str}")

    return rows


def plot(rows: list[dict], snr: float, out_path: Path) -> None:
    if not rows:
        print("\nNo data to plot.")
        return

    model_names = [r["model"] for r in rows]
    wers        = [r["wer"]   for r in rows]
    cers        = [r["cer"]   for r in rows]
    has_cer     = any(c is not None for c in cers)

    n      = len(rows)
    x      = np.arange(n)
    colors = MODEL_COLORS[:n]

    # --- layout: one bar group per model (WER solid, CER hatched) ----------
    if has_cer:
        bar_w = 0.35
        fig, ax = plt.subplots(figsize=(max(8, n * 1.8 + 2), 5))

        wer_bars = ax.bar(x - bar_w / 2, wers, bar_w,
                          color=colors, label="WER", edgecolor="white", linewidth=0.8)
        cer_bars = ax.bar(x + bar_w / 2,
                          [c if c is not None else 0 for c in cers],
                          bar_w, color=colors, label="CER",
                          alpha=0.55, hatch="///", edgecolor="white", linewidth=0.8)

        for bar, val in zip(wer_bars, wers):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        for bar, val in zip(cer_bars, cers):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

        # Custom legend: show WER/CER pattern + per-model colours
        from matplotlib.patches import Patch
        legend_handles = (
            [Patch(facecolor="grey", label="WER (solid)"),
             Patch(facecolor="grey", alpha=0.55, hatch="///", label="CER (hatched)")]
            + [Patch(facecolor=c, label=m) for c, m in zip(colors, model_names)]
        )
        ax.legend(handles=legend_handles, fontsize=9,
                  loc="upper right", framealpha=0.85)

    else:
        # CER not available – just WER bars
        bar_w = 0.55
        fig, ax = plt.subplots(figsize=(max(8, n * 1.8 + 2), 5))

        wer_bars = ax.bar(x, wers, bar_w, color=colors,
                          edgecolor="white", linewidth=0.8, label="WER")
        for bar, val in zip(wer_bars, wers):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=c, label=m) for c, m in zip(colors, model_names)],
                  fontsize=9, loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel("Error Rate (%)", fontsize=11)
    ax.set_title(f"Model comparison — {snr_label(snr)}\n(WER & CER, LRS3)", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--snr", required=True,
        help="SNR level to compare models at. Use an integer (e.g. 10, -5, 0) "
             "or 'clean' / 'inf' for the noise-free condition.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output PNG path. Defaults to outputs/compare_<snr>.png",
    )
    args = parser.parse_args()

    snr = parse_snr(args.snr)
    out_path = Path(args.out) if args.out else Path(f"outputs/compare_{args.snr}db.png")

    rows = collect(snr)

    if len(rows) < 1:
        print("\nNo complete results found – nothing to plot.")
        return

    if len(rows) == 1:
        print(f"\nOnly one model found ({rows[0]['model']}). "
              "Add more models to MODELS to get a comparison chart.")

    plot(rows, snr, out_path)


if __name__ == "__main__":
    main()