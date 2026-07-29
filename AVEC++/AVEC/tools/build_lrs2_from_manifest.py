"""
Build an LRS2-shaped dataset (datasets/LRS2/...) from a flat manifest.csv,
so it can be loaded by nnet.datasets.LRS(version="LRS2", ...) unchanged.

Manifest columns expected (header row required):
    video_path, audio_path, utterance_id, reference, language, video_preprocessing

- video_path / audio_path are resolved relative to the manifest's own directory
  (matches the "video/..." and "../lrs-voxmm/..." style paths already in use).
- video_path is assumed to already be a mouth-cropped clip (matches AVEC's
  crop_mouth=True convention, i.e. LRS expects a "<base>_mouth.mp4" file).
- audio_path can be any format torchaudio can read (wav, flac, ...); it is
  resampled to --sample-rate mono and re-saved as "<base>.flac".

Output layout, per split (pretrain/train/val/test):
    datasets/LRS2/<split>.txt                                  (filelist, one utterance id per line)
    datasets/LRS2/mvlrs_v1/<pretrain|main>/<utterance_id>_mouth.mp4   (symlink or copy)
    datasets/LRS2/mvlrs_v1/<pretrain|main>/<utterance_id>.flac        (resampled audio)
    datasets/LRS2/mvlrs_v1/<pretrain|main>/<utterance_id>.pt          (label + lengths)
    datasets/LRS2/mvlrs_v1/<pretrain|main>/<utterance_id>.txt         (raw transcript, "Text:  ..." -
                                                                        only used by create_corpus(), not
                                                                        required by __getitem__)

Usage (run from the AVEC/ project root, inside your uv env):
    uv run tools/build_lrs2_from_manifest.py \
        --manifest /Users/alexiswisniewski/Desktop/AVSR/data/voxmm_and_lrs-voxmm/lrs-voxmm_avec_roi/manifest.csv \
        --split train \
        --tokenizer ../tokenizer/tokenizerbpe256.model

Re-run any time with --skip-existing (default) to resume / add more rows without
recomputing utterances that already have a .pt file.
"""

import argparse
import csv
import os
import re
import sys

import sentencepiece as spm
import torch
import torchaudio
import torchvision

# LRS2 uses "pretrain" for the pretrain split and "main" for train/val/test
SPLIT_TO_SUBDIR = {
    "pretrain": "pretrain",
    "train": "main",
    "val": "main",
    "test": "main",
}


def clean_reference(text: str) -> str:
    """Strip manifest annotation markup down to plain lowercase transcript text.

    {huh}        -> huh      (uncertain word)
    (1/one)      -> one      (alt transcription: prefer the alphabetic form)
    !tv!         -> tv       (entity tag)
    """
    text = text.lower()

    # !word! -> word
    text = re.sub(r"!([^!]+)!", r"\1", text)

    # {word} -> word
    text = re.sub(r"\{([^}]+)\}", r"\1", text)

    # (a/b/...) -> prefer the alphabetic variant, else the last one
    def pick_alt(m):
        parts = m.group(1).split("/")
        alpha = [p for p in parts if re.search("[a-z]", p)]
        return alpha[-1] if alpha else parts[-1]

    text = re.sub(r"\(([^)]+)\)", pick_alt, text)

    # Drop anything outside the LRS2/3 vocab (a-z, apostrophe, 0-9, space)
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def link_or_copy(src, dst, copy=False):
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    if copy:
        import shutil
        shutil.copyfile(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="Path to manifest.csv")
    parser.add_argument("--root", default="datasets", help="Dataset root (matches LRS(root=...), default 'datasets')")
    parser.add_argument("--split", required=True, choices=["pretrain", "train", "val", "test"], help="Which LRS2 split to write these utterances into")
    parser.add_argument("--tokenizer", default="../tokenizer/tokenizerbpe256.model", help="SentencePiece model path")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--copy", action="store_true", help="Copy video files instead of symlinking")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip utterances that already have a .pt file (default on)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--touch-empty-splits", action="store_true", default=True, help="Create empty filelists for the other 3 splits if missing, since LRS(mode='pretrain+train+val') opens all of them")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (for a quick smoke test)")
    args = parser.parse_args()

    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
    subdir = SPLIT_TO_SUBDIR[args.split]
    out_dir = os.path.join(args.root, "LRS2", "mvlrs_v1", subdir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.root, "LRS2"), exist_ok=True)

    if args.touch_empty_splits:
        for s in ["pretrain", "train", "val", "test"]:
            p = os.path.join(args.root, "LRS2", "{}.txt".format(s))
            if not os.path.isfile(p):
                open(p, "w").close()

    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    filelist_path = os.path.join(args.root, "LRS2", "{}.txt".format(args.split))
    written_ids = []
    n_skipped_existing = 0
    n_skipped_error = 0

    for row in rows:
        utt_id = row["utterance_id"]
        base = os.path.join(out_dir, utt_id)
        pt_path = base + ".pt"

        if args.skip_existing and os.path.isfile(pt_path):
            written_ids.append(utt_id)
            n_skipped_existing += 1
            continue

        try:
            video_src = os.path.normpath(os.path.join(manifest_dir, row["video_path"]))
            audio_src = os.path.normpath(os.path.join(manifest_dir, row["audio_path"]))
            if not os.path.isfile(video_src):
                raise FileNotFoundError(video_src)
            if not os.path.isfile(audio_src):
                raise FileNotFoundError(audio_src)

            # Video: already mouth-cropped -> just link/copy it as "<base>_mouth.mp4"
            link_or_copy(video_src, base + "_mouth.mp4", copy=args.copy)
            video, _, _ = torchvision.io.read_video(base + "_mouth.mp4", pts_unit="sec")
            video_len = video.shape[0]

            # Audio: resample to target sample rate, mono, save as flac
            waveform, sr = torchaudio.load(audio_src)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != args.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, args.sample_rate)
            torchaudio.save(base + ".flac", waveform, args.sample_rate, format="flac")
            audio_len = waveform.shape[1]

            # Label
            text = clean_reference(row["reference"])
            label = torch.LongTensor(tokenizer.encode(text))
            label_len = label.shape[0]

            # Raw transcript file (only needed for create_corpus(), not __getitem__)
            with open(base + ".txt", "w") as tf:
                tf.write("Text:  {}\n".format(text.upper()))

            # Infos
            torch.save(
                {"label": label, "video_len": video_len, "audio_len": audio_len, "label_len": label_len},
                pt_path,
            )

            written_ids.append(utt_id)

        except Exception as e:
            print("SKIP {}: {}".format(utt_id, e), file=sys.stderr)
            n_skipped_error += 1
            continue

    with open(filelist_path, "w") as f:
        for utt_id in written_ids:
            f.write(utt_id + "\n")

    print("Wrote {} utterances to {}".format(len(written_ids), filelist_path))
    print("  ({} already existed and were kept, {} failed and were skipped)".format(n_skipped_existing, n_skipped_error))


if __name__ == "__main__":
    main()
