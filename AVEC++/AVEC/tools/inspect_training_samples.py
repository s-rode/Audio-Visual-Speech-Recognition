"""
Pipeline sanity check: pull a handful of samples from the TRAINING set (with
HardArtificialNoise already applied -- same noise pipeline the model actually
trains on, see nnet/transforms.py) through the current checkpoint, and print
ground truth vs. greedy-decoded prediction for each. Also dumps a few of the
noisy waveforms as .wav so you can literally listen and confirm the noise is
there.

This is NOT a real metric. It deliberately reuses training_dataset, not
evaluation_dataset -- these are seen samples, and (per the ongoing leakage
investigation) even evaluation_dataset's WER isn't currently trustworthy as a
generalization measure. The only thing this script answers is "does the
noise -> frozen encoders -> conformer/interCTC/fusion -> greedy CTC decode
pipeline run end to end and produce something plausible," nothing more.

Internally this is exactly main.py's `--mode evaluation` path
(functions.load_model / functions.load_datasets / Model._evaluate), just
pointed at config.training_dataset instead of config.evaluation_dataset, so
it reuses the same checkpoint loading, device transfer, and CTCGreedySearchDecoder
already exercised by every real eval epoch -- nothing reimplemented here.

Note: the printed predictions and the saved .wav files are two separate passes
over the dataset, each drawing its own fresh noise realization (by design,
HardArtificialNoise reseeds unseeded on every __getitem__ call -- see its
docstring). They won't be the exact same waveform instance for a given sample
index, which is fine for this purpose: one pass demonstrates the model handles
noisy input sanely, the other demonstrates the noise itself is actually being
applied and is audible. Not meant to be paired 1:1.

Usage (run from AVEC/ project root, inside your uv env):
    python tools/inspect_training_samples.py \
        --config_file configs/EffConfInterCTC_pretrained_claix.py \
        --load_last \
        --num_samples 10 \
        --out_dir callbacks/LRS3_CLAIX/EffConfInterCTC_Pretrained/sample_check

Or point at a specific checkpoint instead of --load_last:
    --checkpoint checkpoints_epoch_44_step_17600.ckpt
"""

import argparse
import importlib
import os
import sys

import torch
import torchaudio

# functions.py and nnet/ live in AVEC/ (this script's parent dir), not in tools/.
# Unlike main.py (which gets its own directory added to sys.path[0] automatically
# when run as `python main.py` from AVEC/), running `python tools/inspect_training_samples.py`
# puts tools/ on sys.path[0] instead -- add AVEC/ explicitly so `import functions`
# and the config's own `import nnet` resolve regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functions


def build_args(config_file, checkpoint, load_last, cpu, num_workers):
    """ Minimal stand-in for main.py's parsed argparse Namespace -- every
    attribute functions.load_model reads (checked against its full body, not
    just the checkpoint-loading part -- it also prints a model summary and
    checks distributed/parallel strategy even when we're not using either). """
    return argparse.Namespace(
        rank=0,
        distributed=False,
        parallel=False,
        cpu=cpu,
        config_file=config_file,
        checkpoint=checkpoint,
        load_last=load_last,
        world_size=1,
        num_workers=num_workers,
        show_dict=False,
        show_modules=False,
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Same config the real training job is using")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint filename inside callback_path, e.g. checkpoints_epoch_44_step_17600.ckpt")
    parser.add_argument("--load_last", action="store_true", help="Auto-pick the newest checkpoint in callback_path instead of naming one")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2, help="Batching for the prediction pass only, doesn't affect --num_samples")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="torch.manual_seed for which samples get shuffled in -- omit for a different slice each run")
    parser.add_argument("--out_dir", default=None, help="Where to write noisy .wav files (default: <callback_path>/sample_check)")
    cli_args = parser.parse_args()

    if cli_args.checkpoint is None and not cli_args.load_last:
        parser.error("Pass --checkpoint <name> or --load_last")

    if cli_args.seed is not None:
        torch.manual_seed(cli_args.seed)

    # Load model + config exactly like main.py does
    args = build_args(cli_args.config_file, cli_args.checkpoint, cli_args.load_last, cli_args.cpu, num_workers=0)
    args.config = importlib.import_module(args.config_file.replace(".py", "").replace("/", "."))
    model = functions.load_model(args)
    model.eval()

    out_dir = cli_args.out_dir or os.path.join(args.config.callback_path, "sample_check")
    os.makedirs(out_dir, exist_ok=True)

    # training_dataset already has HardArtificialNoise wired in via audio_transform
    # (see configs/EffConfInterCTC_pretrained_claix.py) -- iterating it here applies
    # the exact same noise the real training job is using, nothing extra to wire up.
    train_dataset = args.config.training_dataset

    loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cli_args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=0,
    )

    eval_steps = -(-cli_args.num_samples // cli_args.batch_size)  # ceil

    print("Running {} batch(es) (~{} samples) from the TRAINING set (noise applied) through "
          "the loaded checkpoint. Prediction pass -- ground truth / greedy decode below.\n".format(
              eval_steps, eval_steps * cli_args.batch_size))

    model._evaluate(loader, writer=None, step=None, eval_steps=eval_steps, verbose=1, recompute_metrics=False)

    # Separate pass: dump raw noisy audio so you can listen and confirm the
    # noise is actually there (reverb_prob=0.5 / demand_prob=0.8 / bandwidth_prob=0.3
    # by default, so not every sample gets every stage -- a few samples give you a
    # realistic mix rather than a cherry-picked "loudest" one).
    print("\nSaving {} noisy .wav files to {} ...".format(cli_args.num_samples, out_dir))
    for i in range(cli_args.num_samples):
        _, audio, _, _, audio_len, _ = train_dataset[i]
        audio = audio[:int(audio_len)].reshape(1, -1)
        torchaudio.save(os.path.join(out_dir, "sample_{}_noisy.wav".format(i)), audio, 16000)
    print("Done -- listen to a few of these to confirm the noise is audible.")


if __name__ == "__main__":
    main()
