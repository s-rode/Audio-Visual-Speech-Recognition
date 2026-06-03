"""Plot WER vs SNR from experiment metrics.json files."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Map output dir -> SNR label (clean = inf dB)
EXPERIMENTS = {
    "outputs/lrs3_300_clean":         float("inf"),
    "outputs/lrs3_300_white_30db":    30,
    "outputs/lrs3_300_white_20db":    20,
    "outputs/lrs3_300_white_15db":    15,
    "outputs/lrs3_300_white_10db":    10,
    "outputs/lrs3_300_white_5db":      5,
    "outputs/lrs3_300_white_0db":      0,
    "outputs/lrs3_300_white_-5db":    -5,
}


def load_results():
    results = []
    for output_dir, snr in EXPERIMENTS.items():
        metrics_path = Path(output_dir) / "metrics.json"
        if not metrics_path.exists():
            print(f"  [missing] {metrics_path}")
            continue
        with metrics_path.open() as f:
            data = json.load(f)
        wer = data.get("wer")
        cer = data.get("cer")
        if wer is None:
            print(f"  [no WER]  {metrics_path}")
            continue
        results.append({"snr": snr, "wer": wer, "cer": cer})
        print(f"  SNR={snr:>6} dB  WER={wer:.3f}  CER={cer:.3f}")
    return sorted(results, key=lambda x: x["snr"])


def plot(results):
    snr_labels = []
    snr_numeric = []
    wers = []
    cers = []

    for r in results:
        wers.append(r["wer"] * 100)
        cers.append(r["cer"] * 100)
        if r["snr"] == float("inf"):
            snr_labels.append("Clean\n(∞ dB)")
            snr_numeric.append(35)   # plot clean slightly right of 30
        else:
            snr_labels.append(f"{int(r['snr'])} dB")
            snr_numeric.append(r["snr"])

    x = np.array(snr_numeric)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, wers, "o-", color="#2563eb", linewidth=2, markersize=7, label="WER")
    ax.plot(x, cers, "s--", color="#dc2626", linewidth=2, markersize=7, label="CER")

    # Annotate points
    for xi, wer, cer in zip(x, wers, cers):
        ax.annotate(f"{wer:.1f}%", (xi, wer), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#2563eb")
        ax.annotate(f"{cer:.1f}%", (xi, cer), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="#dc2626")

    ax.set_xticks(x)
    ax.set_xticklabels(snr_labels)
    ax.set_xlabel("SNR (dB)  ←  more noise", fontsize=11)
    ax.set_ylabel("Error Rate (%)", fontsize=11)
    ax.set_title("Whisper tiny.en — WER & CER vs White Noise SNR\n(LRS3, 300 utterances)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = Path("outputs/wer_vs_snr.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\nSaved plot to {out}")
    plt.show()


if __name__ == "__main__":
    print("Loading results...")
    results = load_results()
    if len(results) < 2:
        print("Not enough completed runs to plot yet.")
    else:
        plot(results)

