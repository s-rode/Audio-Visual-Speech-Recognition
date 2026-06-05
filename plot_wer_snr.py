"""Plot WER vs SNR for all models from W&B runs."""

import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

try:
    import wandb
except ImportError:
    raise SystemExit("wandb not installed. Run: uv add wandb")

WANDB_PROJECT = "your-wandb-entity/team1-asr"

MODELS = ["tiny", "small", "medium", "large", "turbo"]

COLORS = {
    "tiny":   "#94a3b8",
    "small":  "#2563eb",
    "medium": "#16a34a",
    "large":  "#dc2626",
    "turbo":  "#9333ea",
}

SNR_TO_X = {-5: -5, 0: 0, 10: 10, 20: 20, 30: 30, "clean": 35}
SNR_LABELS = {-5: "-5 dB", 0: "0 dB", 10: "10 dB", 20: "20 dB", 30: "30 dB", "clean": "Clean\n(∞ dB)"}


def parse_snr(run_name: str):
    """Extract SNR from run name like lrs3-test-tiny-clean or lrs3-test-tiny-white_10db."""
    if run_name.endswith("clean"):
        return "clean"
    m = re.search(r"white_(-?\d+)db", run_name)
    if m:
        return int(m.group(1))
    return None


def fetch_results():
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT)

    results = {model: {} for model in MODELS}

    for run in runs:
        name = run.name or ""
        if not name.startswith("lrs3-test-"):
            continue
        if run.state != "finished":
            print(f"  [skipping {run.state}] {name}")
            continue

        model = next((m for m in MODELS if f"-{m}-" in name or name.endswith(f"-{m}")), None)
        if model is None:
            continue

        snr = parse_snr(name)
        if snr is None:
            continue

        wer = run.summary.get("wer")
        cer = run.summary.get("cer")
        if wer is None:
            print(f"  [no WER] {name}")
            continue

        results[model][snr] = {"wer": wer * 100, "cer": cer * 100}
        print(f"  {model:8s} SNR={str(snr):>6}  WER={wer:.3f}  CER={cer:.3f}")

    return results


def plot(results):
    fig, ax = plt.subplots(figsize=(12, 6))

    for model in MODELS:
        model_results = results.get(model, {})
        if not model_results:
            continue

        snrs = sorted(model_results.keys(), key=lambda s: SNR_TO_X[s])
        x = np.array([SNR_TO_X[s] for s in snrs])
        wers = [model_results[s]["wer"] for s in snrs]

        ax.plot(x, wers, "o-", color=COLORS[model], linewidth=2,
                markersize=7, label=model)

        for xi, wer in zip(x, wers):
            ax.annotate(f"{wer:.1f}%", (xi, wer), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color=COLORS[model])

    all_x = sorted(SNR_TO_X.values())
    ax.set_xticks(all_x)
    ax.set_xticklabels([SNR_LABELS[s] for s in sorted(SNR_TO_X.keys(), key=lambda s: SNR_TO_X[s])])
    ax.set_xlabel("SNR (dB)  ←  more noise", fontsize=11)
    ax.set_ylabel("WER (%)", fontsize=11)
    ax.set_title("Whisper WER vs White Noise SNR — LRS3 Test Set (1321 utterances)", fontsize=13)
    ax.legend(fontsize=10, title="Model")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    import os
    os.makedirs("outputs", exist_ok=True)
    out = "outputs/wer_vs_snr.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved plot to {out}")
    plt.show()


if __name__ == "__main__":
    print(f"Fetching runs from {WANDB_PROJECT}...\n")
    results = fetch_results()
    total = sum(len(v) for v in results.values())
    if total < 2:
        print("Not enough finished runs to plot yet.")
    else:
        plot(results)
