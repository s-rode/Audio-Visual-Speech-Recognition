"""
Split a flat AVEC manifest.csv (video_path, audio_path, utterance_id, reference,
language, video_preprocessing) into an 80/20 train/val pair, so
build_lrs2_from_manifest.py can be run once per resulting file
(--split train / --split val).

Grouped by source video (the parent directory of audio_path) before
splitting, so every utterance from the same video/speaker lands on the same
side -- a plain row-level shuffle can't guarantee that, which is exactly what
the original version of this script did (~8 utterances/video on average for
LRS3's 31,982 utterances / 4,004 videos, so a flat shuffle leaked a
same-video sibling into val for most val rows). That leak was enough on its
own to make val WER converge implausibly fast and track train loss almost
exactly -- see the training-run investigation this fixes.

Uses audio_path, not video_path, to derive the grouping key: this manifest's
video_path is already flattened to a single "video/" directory by the
mouth-crop extraction step (video/<id>_mouth.mp4, no per-speaker structure
left), but audio_path still points at the original per-video layout
(../LRS3-TED/trainval/<video_id>/<clip>.wav) -- dirname of that gives the
real video id directly. Confirmed against a real sample of this manifest;
don't assume video_path works for a manifest built a different way.

Whole video-groups are shuffled and assigned to val until the accumulated row
count reaches --val-fraction of the total, so the achieved split won't be
exactly 80/20 (group sizes vary) -- the script prints what it actually landed
on, and asserts zero group overlap between the two output files as a
self-check.

Usage (lightweight, safe to run directly on a login node -- just CSV
read/write, no video/audio processing):
    python tools/split_manifest_train_val.py \
        --manifest /rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/lrs3/avec_roi/manifest.csv \
        --out-dir /rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/lrs3/avec_roi \
        --val-fraction 0.2 \
        --seed 0

Writes <out-dir>/manifest_train.csv and <out-dir>/manifest_val.csv. After
regenerating these, rebuild BOTH splits with build_lrs2_from_manifest.py
--no-skip-existing (train.txt/val.txt get fully rewritten from whichever
manifest was passed in, so a rebuild of both is what actually applies the new
split -- utterances that moved sides won't fix themselves otherwise).
"""

import argparse
import csv
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to the full manifest.csv")
    parser.add_argument("--out-dir", required=True, help="Where to write manifest_train.csv / manifest_val.csv")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed, for a reproducible split")
    parser.add_argument("--group-by", default="audio_path", help="Manifest column whose parent directory identifies the source video/speaker (default: audio_path -- video_path is flattened in this manifest, see module docstring)")
    args = parser.parse_args()

    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Group rows by source video so a random split can't separate utterances
    # from the same speaker/video across train and val.
    groups = {}
    for row in rows:
        key = os.path.dirname(row[args.group_by])
        groups.setdefault(key, []).append(row)

    group_keys = list(groups.keys())

    # Fail loudly instead of silently writing a garbage split: if --group-by's
    # dirname collapses to (almost) one group, the manifest doesn't encode
    # speaker/video via directory structure the way this script assumes, and
    # every row would end up on one side.
    if len(group_keys) < max(2, 0.01 * len(rows)):
        raise SystemExit(
            "Only {} group(s) found across {} rows using os.path.dirname(row['{}']) as the key -- "
            "that's not a usable per-video grouping. Sample a few rows (video_path, utterance_id) "
            "and check how this manifest actually encodes speaker/video identity, then adjust "
            "--group-by or the grouping logic accordingly.".format(len(group_keys), len(rows), args.group_by)
        )

    rng = random.Random(args.seed)
    rng.shuffle(group_keys)

    n_val_target = round(len(rows) * args.val_fraction)
    val_rows, train_rows = [], []
    val_keys, train_keys = set(), set()
    val_row_count = 0
    for key in group_keys:
        if val_row_count < n_val_target:
            val_rows.extend(groups[key])
            val_keys.add(key)
            val_row_count += len(groups[key])
        else:
            train_rows.extend(groups[key])
            train_keys.add(key)

    # Self-check: this should be structurally impossible given the loop above,
    # but it's cheap insurance against ever silently reintroducing the leak.
    assert not (val_keys & train_keys), "video group(s) ended up on both sides of the split: {}".format(val_keys & train_keys)
    assert len(val_rows) + len(train_rows) == len(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "manifest_train.csv")
    val_path = os.path.join(args.out_dir, "manifest_val.csv")

    for path, subset in [(train_path, train_rows), (val_path, val_rows)]:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(subset)

    print("{} videos, {} rows -> {}".format(len(train_keys), len(train_rows), train_path))
    print("{} videos, {} rows -> {}".format(len(val_keys), len(val_rows), val_path))
    print("Achieved val fraction: {:.1%} (target was {:.1%})".format(len(val_rows) / len(rows), args.val_fraction))


if __name__ == "__main__":
    main()
