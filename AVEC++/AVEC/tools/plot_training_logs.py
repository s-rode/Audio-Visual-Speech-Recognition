"""
Turn TensorBoard event files (written by nnet.model.Model.fit via SummaryWriter,
see nnet/model.py:696) into CSVs + matplotlib PNGs, offline, after a training run
finishes. No wandb / internet needed: SummaryWriter already logs everything
locally to <callback_path>/logs on CLAIX, this script just reads those event
files back.

Usage (run locally after scp/rsync-ing the "logs" dir back from CLAIX, or
directly on CLAIX since it never touches the network):
    uv run tools/plot_training_logs.py \
        --logdir callbacks/EffConfInterCTC_pretrained/logs \
        --out callbacks/EffConfInterCTC_pretrained/report

Output:
    <out>/scalars.csv                 long-format dump of every scalar tag
    <out>/plots/<sanitized_tag>.png    one PNG per scalar tag
    <out>/plots/overview.png          train vs eval loss/metric grid, if both present

Requires only `tensorboard` (already a dependency, since SummaryWriter comes
from torch.utils.tensorboard) and `matplotlib`/`pandas`.
"""

import argparse
import os
import re

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(logdir):
    """Load every scalar tag from a TensorBoard logdir into a long-format DataFrame."""
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()

    rows = []
    for tag in ea.Tags().get("scalars", []):
        for event in ea.Scalars(tag):
            rows.append({"tag": tag, "step": event.step, "wall_time": event.wall_time, "value": event.value})

    if not rows:
        raise RuntimeError(
            "No scalar tags found in '{}'. Check that this is the 'logs' subfolder "
            "under callback_path (see writer = SummaryWriter(...) in nnet/model.py).".format(logdir)
        )

    return pd.DataFrame(rows)


def sanitize(tag):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", tag)


def plot_each_tag(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for tag, group in df.groupby("tag"):
        group = group.sort_values("step")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(group["step"], group["value"])
        ax.set_title(tag)
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, sanitize(tag) + ".png"), dpi=150)
        plt.close(fig)


def plot_overview(df, out_dir):
    """Best-effort grid comparing Training-* vs Evaluation-* tags that share a metric name."""
    train_tags = [t for t in df["tag"].unique() if t.startswith("Training-")]
    eval_tags = [t for t in df["tag"].unique() if t.startswith("Evaluation")]

    def metric_name(tag):
        return tag.split("/")[-1]

    train_by_metric = {}
    for t in train_tags:
        train_by_metric.setdefault(metric_name(t), []).append(t)
    eval_by_metric = {}
    for t in eval_tags:
        eval_by_metric.setdefault(metric_name(t), []).append(t)

    shared = sorted(set(train_by_metric) & set(eval_by_metric))
    if not shared:
        return

    n = len(shared)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, metric in enumerate(shared):
        ax = axes[i // ncols][i % ncols]
        for t in train_by_metric[metric]:
            g = df[df["tag"] == t].sort_values("step")
            ax.plot(g["step"], g["value"], label="train: " + t, alpha=0.8)
        for t in eval_by_metric[metric]:
            g = df[df["tag"] == t].sort_values("step")
            ax.plot(g["step"], g["value"], label="eval: " + t, alpha=0.8, linestyle="--")
        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "overview.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True, help="TensorBoard 'logs' dir (callback_path/logs)")
    parser.add_argument("--out", type=str, default=None, help="Output dir (default: <logdir>/../report)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(args.logdir, "..", "report")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    df = load_scalars(args.logdir)
    df.to_csv(os.path.join(out_dir, "scalars.csv"), index=False)
    print("Loaded {} tags, {} points -> {}".format(df["tag"].nunique(), len(df), os.path.join(out_dir, "scalars.csv")))

    plot_each_tag(df, plots_dir)
    plot_overview(df, plots_dir)
    print("Plots written to {}".format(plots_dir))
