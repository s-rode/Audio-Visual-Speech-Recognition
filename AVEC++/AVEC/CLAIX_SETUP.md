# Running AVEC++ on CLAIX

Runbook for getting the AVEC++ folder onto CLAIX (RWTH Aachen) and starting a
real training run. Facts about CLAIX below are sourced from RWTH's own docs
(linked inline) as of 2026-07-26 -- double check anything time-sensitive
(partition names, node limits) if this file is more than a few months old.

## 0. What you need before starting

- Your HPC project account id for `#SBATCH --account` -- likely `lect0176`
  going by your hpcwork path (`.../hpcwork/lect0176/team-1`), not your
  personal username. Confirm with `sacctmgr show assoc user=$USER`.
- SSH access to `login23-*.hpc.itc.rwth-aachen.de` set up already.
- The absolute path to the LRS3-content-in-LRS2-layout data that's already on CLAIX.

`AVEC++/` itself is already on CLAIX at
`/rwthfs/rz/cluster/hpcwork/lect0176/team-1/AVEC++/`. Everything below
happens inside `.../AVEC++/AVEC/` and is self-contained -- it does **not**
need the rest of your local `AVSR/` folder (pyproject.toml/uv.lock/src/ for
the wider team1-asr project, other experiments, data, etc.). All we need
from that project is the resolved package list.

## 1. Export just the package list (no folder transfer needed)

On your Mac, in `AVSR/` (wherever `uv.lock` lives), turn the lock file into a
plain pinned requirements file -- this captures the exact resolved versions
(`torch==2.6.0` etc.) without dragging along any of team1-asr's other files:

```bash
cd /Users/alexiswisniewski/Desktop/AVSR
uv export --no-hashes --no-emit-project --format requirements-txt -o AVEC++/AVEC/requirements-claix.txt
```

`--no-emit-project` matters: without it, `uv export` adds an `-e .` line
that installs team1-asr's own local package in editable mode, which needs
`pyproject.toml`/`setup.py` in the install directory -- CLAIX's
`AVEC++/AVEC/` doesn't have one, so `uv pip install -r requirements-claix.txt`
fails with "does not appear to be a Python project" if you leave that flag
off. We only want the third-party dependencies here, not team1-asr itself.

That writes the file straight into `AVEC++/AVEC/`, i.e. into the folder
you've already transferred, so the only thing left to send over is this one
small text file. It'll still list every dependency in `pyproject.toml`
(faster-whisper, librispeech tooling, wandb, etc.) -- that's fine, harmless
extra installs, not worth trimming by hand.

Send it over (a copy node isn't really necessary for one small file, but
doesn't hurt) -- this will overwrite the version that's already there
without `--no-emit-project`:

```bash
rsync -avz "AVEC++/AVEC/requirements-claix.txt" \
    mmy27930@copy23-1.hpc.itc.rwth-aachen.de:/rwthfs/rz/cluster/hpcwork/lect0176/team-1/AVEC++/AVEC/
```

## 2. Build a plain venv on CLAIX from that file

SSH into a regular login node (not a copy node -- those can't run anything):

```bash
ssh mmy27930@login23-1.hpc.itc.rwth-aachen.de
cd /rwthfs/rz/cluster/hpcwork/lect0176/team-1/AVEC++/AVEC
```

**First check whether the login node can reach the internet at all** -- this
determines which path below you take:

```bash
curl -sI https://pypi.org | head -1
```

- **If that returns an HTTP response** (even a redirect): login nodes have
  outbound access even though compute nodes don't (common HPC setup -- you
  set up envs on login nodes, then submit jobs that run offline on compute
  nodes). In that case:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv isn't already available
  uv venv .venv --python 3.11                        # match the python version your local venv used if you can check it (`uv run python -V` locally)
  source .venv/bin/activate
  uv pip install -r requirements-claix.txt
  uv pip install pyroomacoustics scipy   # not yet in the AVSR-level lock this file is exported
                                          # from, so `uv export` will keep dropping it until it's
                                          # added as a real dependency there -- needed for the
                                          # reverb stage of nnet/transforms.py's HardArtificialNoise
  ```

  You don't need a separate CUDA toolkit module for this -- modern `torch`
  wheels bundle their own CUDA runtime via `nvidia-*` pip packages and only
  need the node's NVIDIA driver, which is already there. `requirements-claix.txt`
  carries platform markers from the universal lock resolution, so `pip` will
  correctly pick the Linux/CUDA `torch` wheel here even though it was
  exported from macOS.

- **If that hangs/fails**: login nodes are also offline. Build the venv on a
  Linux x86_64 machine that does have internet instead (your Mac won't work --
  wrong OS/arch for the CUDA wheels): copy `requirements-claix.txt` there,
  run the same `uv venv` + `uv pip install` commands to produce a wheel
  cache, then either rsync the resulting `.venv/` directly to CLAIX (works
  as long as that machine's OS/glibc is close enough to CLAIX's Rocky Linux
  9), or run `pip download -r requirements-claix.txt -d wheelhouse` there and
  `pip install --no-index --find-links=wheelhouse -r requirements-claix.txt`
  on CLAIX.

[Module System](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/450d33cc19fd4e50b1dd07027e9b55bd/) (if you end up needing a `module load` for anything, e.g. a system Python)

## 3. Fix the config for CLAIX's data layout

New config: `configs/EffConfInterCTC_pretrained_claix.py` (already created,
mirrors `EffConfInterCTC_pretrained.py` with these changes):

1. `wav2vec2_name_or_path` points at the localized checkpoint
   (`../wav2vec2/wav2vec2-base-960h`) instead of hitting the HF Hub.
2. The dataset block loads `version="LRS2"` instead of `"LRS3"`, with
   `data_root = "datasets"` -- because the raw LRS3 data on CLAIX is a flat
   manifest.csv, and we're building it into the LRS2-shaped convention (see
   step 3b below) at the tool's default `--root=datasets`, same as
   `datasets_smoke/LRS2` locally. The original `EffConfInterCTC_pretrained.py`'s
   `version="LRS3"` + `mode="train"` combination would actually fail an
   assertion in `nnet/datasets.py` (`"train"` isn't a valid mode for
   `version="LRS3"`) -- this was likely leftover from before this conversion.
3. `evaluation_dataset` loads `mode="val"`, not `"test"` -- the manifest only
   covers LRS3's `trainval` split (no test-split data), so we're carving an
   80/20 train/val split out of it ourselves (step 3b) rather than evaluating
   against real held-out test data. Worth revisiting once/if you get access
   to LRS3's actual test-split manifest.
4. `training_audio_transform` applies `nnet.HardArtificialNoise` (reverb ->
   DEMAND -> bandwidth filter, reproducing team-1's LRS-VoxMM-style noise
   recipe from `AVEC++/noise.py`, but in-memory and re-sampled every epoch --
   see the docstring in `nnet/transforms.py`) to the training split only.
   `demand_noise_path` must point at wherever the DEMAND corpus lives on
   CLAIX -- see step 3c below, which you need to do before this config will
   even load. Reverb needs `pyroomacoustics` installed (step 2 above); it's
   intentionally left **on** by default (`reverb_prob=0.5`) since it's a big
   part of what makes LRS-VoxMM-style noise hard.

No placeholder left to fill in this time -- `data_root`/`version`/`mode` are
already set correctly for the layout step 3b below produces.

> Heads up separately: `configs/EffConfInterCTC_pretrained_smoke.py` currently
> sets `wav2vec2_name_or_path = "../wav2vec2"`, but the actual checkpoint
> files (`config.json`, `model.safetensors`, ...) live one level deeper at
> `../wav2vec2/wav2vec2-base-960h`. Worth fixing there too before you rely on
> that smoke config again.

## 3b. Build the dataset from the raw manifest

The raw manifest (`/rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/lrs3/avec_roi/manifest.csv`)
covers the entire LRS3 `trainval` split as one flat CSV with relative
`video_path`/`audio_path` columns. Two steps: split it 80/20, then build each
half into the LRS2-shaped layout `nnet/datasets.py` expects.

**Split (lightweight -- fine to run directly on a login node):**

```bash
cd /rwthfs/rz/cluster/hpcwork/lect0176/team-1/AVEC++/AVEC
source .venv/bin/activate
python tools/split_manifest_train_val.py \
    --manifest /rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/lrs3/avec_roi/manifest.csv \
    --out-dir /rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/lrs3/avec_roi \
    --val-fraction 0.2 --seed 0
```

Writes `manifest_train.csv` / `manifest_val.csv` next to the original,
grouped by source video so no speaker/video appears on both sides of the
split (see the script's docstring -- an earlier row-level version of this
script leaked same-video utterances into val, which was enough on its own to
make val WER converge implausibly fast).

**Build (heavy -- CPU-only `sbatch` job, not a login node):**
`build_lrs2_from_manifest.py` is single-threaded (one video-read +
audio-resample + tokenize per row, sequentially), so for ~25k/6k rows this is
realistically hours, not minutes. `tools/claix_build_dataset.sbatch` wraps it,
sized for `c23ms` (no GPU needed for this step):

```bash
mkdir -p logs
sbatch tools/claix_build_dataset.sbatch \
    /rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/lrs3/avec_roi/manifest_train.csv train
sbatch tools/claix_build_dataset.sbatch \
    /rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/lrs3/avec_roi/manifest_val.csv val
```

> **Rebuilding after regenerating the tokenizer and/or the split** (as we did
> to fix the `<unk>`-collapse / leaky-split issues): add `--no-skip-existing`
> to both submissions above. Without it, `--skip-existing` (on by default)
> reuses whatever `.pt` label files already exist under `mvlrs_v1/main/` --
> which for a tokenizer change means keeping the *old, broken* label
> encoding, and for a split change means an utterance that moved sides still
> won't get rebuilt (its `.pt` already exists from before, even though it's
> now supposed to be on the other side). The `<split>.txt` filelist itself
> always gets regenerated correctly from whichever manifest you pass in --
> it's specifically the cached `.pt` contents that need the flag to actually
> refresh.

Safe to submit both at once -- they write disjoint utterance files and
separate `train.txt`/`val.txt` filelists, no shared state to race on. Also
safe to resubmit either one if it times out: `--skip-existing` (on by
default) means a re-run resumes instead of redoing finished utterances, and
still ends with a correct/complete filelist since it re-reads the full input
manifest each time.

Before committing to the full run, sanity-check on a login node with a tiny
slice (well within the 20 min / 4-core limits):

```bash
python tools/build_lrs2_from_manifest.py \
    --manifest /rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/lrs3/avec_roi/manifest_train.csv \
    --split train --limit 20 \
    --tokenizer ../tokenizer/tokenizerbpe256.model
```

Once both jobs finish, you should have `datasets/LRS2/{train,val}.txt` and
`datasets/LRS2/mvlrs_v1/main/<id>_mouth.mp4|.flac|.pt` under `AVEC++/AVEC/`.

## 3c. Transfer the DEMAND noise corpus

`HardArtificialNoise` (step 3, item 4) validates `demand_noise_path` at
**config-import time** -- `configs/EffConfInterCTC_pretrained_claix.py` will
fail to even load (let alone the smoke test in step 4) until DEMAND is
actually on CLAIX and the path is pointed at it correctly. Do this before
step 4, not after.

**Transfer it from your Mac, through the copy node** (DEMAND is ~1.3GB --
small enough for `login23-*` in a pinch, but RWTH's own guidance is to keep
bulk data transfers off login nodes, so use `copy23-1` like the rest of this
runbook does):

```bash
rsync -avz --info=progress2 /path/to/your/local/DEMAND/ \
    mmy27930@copy23-1.hpc.itc.rwth-aachen.de:/rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/DEMAND/
```

Replace `/path/to/your/local/DEMAND/` with wherever it actually sits on your
Mac. Keep the trailing slashes on both sides -- that copies the *contents* of
your local DEMAND folder into the remote one, rather than nesting a `DEMAND/`
folder inside `.../data/DEMAND/`.

**Check the layout lines up with what the loader expects**: `HardArtificialNoise`
globs `<demand_noise_path>/*/*.wav` -- i.e. it wants environment folders
directly under the root (`DEMAND/DKITCHEN/ch01.wav`, `DEMAND/NPARK/ch01.wav`,
...), each containing plain `.wav` files. Some DEMAND downloads nest an extra
resolution folder first (e.g. `DEMAND/48k/DKITCHEN/...` or a separate
`16kHz/` vs `48kHz/` split) -- if yours does, point `demand_noise_path` one
level deeper rather than restructuring the files. Sample rate doesn't matter
either way: anything that isn't 16kHz gets resampled automatically
(`torchaudio.functional.resample`) the first time it's loaded.

Once it's over, update the config:

```python
# configs/EffConfInterCTC_pretrained_claix.py
demand_noise_path = "/rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/DEMAND/16k"
```

> **If rsync fails with `Permission denied` stat-ing destination files**: check
> whether a teammate already dropped a copy of DEMAND under `data/DEMAND/` --
> in this project that turned out to be the case (`data/DEMAND/16k/<ENV>/`
> already existed, copied in by mm315273). The giveaway is `ls -ld` on each
> path component: `data/` and `data/DEMAND/` were `rws` for group `lect0176`
> (setgid, group-writable -- correct), but `data/DEMAND/16k/` had lost setgid
> and group-write, and the environment folders underneath it (e.g. `DLIVING/`)
> were owned by the *other person's own personal group*, not `lect0176` --
> locking every other team member out entirely. If you hit this, don't fight
> it with your own rsync: get the owner to grant read access instead
> (`setfacl -R -m g:lect0176:rX <path>` plus `-d` for the default ACL so new
> files inherit it too), verify the file count looks complete
> (`find <path> -name '*.wav' | wc -l` -- 288 for the standard 18-environment
> x 16-channel 16kHz release), and skip the transfer altogether if it already
> matches.

**Verify both stages actually work** before burning a smoke-test/Slurm slot on
a config mistake -- on the login node, with the venv from step 2 active:

```bash
# DEMAND path + glob resolve, and files load (cheap, no pyroomacoustics needed)
python -c "
import nnet
t = nnet.HardArtificialNoise(demand_noise_path='/rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/DEMAND', reverb_prob=0, bandwidth_prob=0, demand_prob=1)
print(len(t._tracks), 'DEMAND tracks loaded')
"

# Reverb stage runs end-to-end (exercises pyroomacoustics on a dummy 1s clip)
python -c "
import torch, nnet
t = nnet.HardArtificialNoise(demand_noise_path='/rwthfs/rz/cluster/hpcwork/lect0176/team-1/data/DEMAND', reverb_prob=1, demand_prob=0, bandwidth_prob=0)
print('reverb output shape:', t(torch.randn(1, 16000)).shape)
"
```

The first should print a track count in the hundreds; the second should
print `reverb output shape: torch.Size([1, 16000])` with no traceback. If
the second one errors on `import pyroomacoustics`, go back to step 2.

## 4. Smoke test on the login node

Per RWTH's own recommended workflow ("test your application whether it
starts as expected on a login node" -- see
[Submitting Jobs](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/13ace46cfbb84e92a64c1361e0e4c104/)),
this is exactly the right instinct. One catch: **only `login23-g-1` has
GPUs** (4x H100, shared/MIG) -- the other login nodes are CPU-only. Login
nodes also enforce (see
[Login Nodes](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/0a23d513f31b4cf1849986aaed475789/)):

- max 4 CPU cores per user
- GPU use capped at **30 minutes per process**, auto-killed after
- heavy CPU use for >20 min may be killed without warning
- your processes capped at 25% of node memory

```bash
ssh mmy27930@login23-g-1.hpc.itc.rwth-aachen.de
cd /rwthfs/rz/cluster/hpcwork/lect0176/team-1/AVEC++/AVEC
./tools/claix_login_smoke_test.sh
```

That script runs the real `EffConfInterCTC_pretrained_claix.py` config capped
to 20 steps with a 20-minute hard timeout (via `--steps_per_epoch`), so it
validates config loading, checkpoint loading, data paths, and a few real
training steps + a checkpoint save -- without tripping the 30-min GPU cutoff.
Ctrl-C once you've seen a few logged steps if you don't want to wait it out.

## 5. Submit the real run via Slurm

`tools/claix_train.sbatch` is ready with `--account=lect0176` filled in --
double check that's actually the right project id for your team before
submitting (see step 0).

```bash
cd /rwthfs/rz/cluster/hpcwork/lect0176/team-1/AVEC++/AVEC
mkdir -p logs      # Slurm won't create this for you -- #SBATCH --output needs it to exist
sbatch tools/claix_train.sbatch
squeue --me
```

Also following RWTH's own best-practice checklist: the first time you submit,
temporarily drop `#SBATCH --time=1-00:00:00` down to `00:05:00` in the script
so a config mistake fails in minutes instead of after you've waited hours in
the queue. Once that goes through cleanly, bump the time back up and
resubmit for the real run.

[Job Parameters reference](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/3d20a87835db4569ad9094d91874e2b4/) --
useful if you need to tweak `--cpus-per-task`, `--gres=gpu:N`, `--time`, etc.

## 6. Monitor / manage the job

```bash
squeue --me                        # PD = pending, R = running
tail -f logs/avecpp-train_<jobid>.out
scancel <jobid>                    # if you need to kill it
```

## 7a. Partially unfreezing wav2vec2/SwinLip (discriminative LR)

Added 2026-07-28 once the "hard" evaluation stream (`Evaluation-epoch/1`, see
step 3c) showed WER flat around 80% even after `joint_prob` was raised to 0.8 --
consistent with the frozen wav2vec2 features themselves being a ceiling, since
`HardArtificialNoise` only distorts the audio channel and wav2vec2 never saw that
distortion in its own pretraining.

`configs/EffConfInterCTC_pretrained_claix.py` now has:

```python
wav2vec2_unfreeze_last_n_layers = 2   # re-enables requires_grad on the LAST 2
                                       # layers of wav2vec2.encoder.layers, rest of
                                       # the backbone (conv feature extractor +
                                       # earlier transformer layers) stays frozen
swinlip_unfreeze_modules = []         # empty for now -- see note below
```

These newly-unfrozen params get their OWN Noam schedule
(`encoder_finetune_warmup_steps=1000`, `encoder_finetune_val_factor=0.1`, set on
the `model.compile(...)` call) -- a short warmup and low peak LR (~1.5e-4 vs the
main schedule's ~1.05e-3), since they're entering the optimizer for the first
time. The MAIN schedule (`warmup_steps=10000`, everything else -- adapters,
conformer back-ends, fusion, head) is untouched and keeps decaying exactly as it
has been; only the newly-unfrozen layers get a fresh ramp.

This is a config-shape change (new param group), but checkpoint loading is
backward compatible: `--load_last` on a checkpoint saved BEFORE this change
restores the main schedule's step count correctly and starts the new
finetune-layer schedule fresh at step 0 (see `nnet/optimizers.py`'s
`_MultiSchedulerOptimizerMixin` docstring for exactly how). No need to restart
training from scratch for this.

SwinLip is left fully frozen (`swinlip_unfreeze_modules = []`) because, unlike
wav2vec2 (HF's documented `.encoder.layers` ModuleList), this repo's SwinLip class
has no submodule naming this codebase can assume, and the noise recipe doesn't
touch video anyway. To unfreeze part of it later: inspect the real module tree
first --

```python
model.show_modules()  # or: print(model.encoder.video_encoder.front_end.model)
```

then set `swinlip_unfreeze_modules` to the correct dotted path(s) relative to
`front_end` (e.g. `["model.trunk.layers.3"]`), resolved via a plain
`getattr`-chain -- see `VisualSwinLipConformerEncoder`'s docstring in
`nnet/networks.py`.

Don't forget to rsync the updated `configs/EffConfInterCTC_pretrained_claix.py`,
`nnet/networks.py`, `nnet/models_zoo.py`, and `nnet/optimizers.py` before
resubmitting -- same as any other local edit (see the recurring stale-deploy
issue earlier in this doc).

## 7. After it finishes: plot what happened

TensorBoard event files land under
`callback_path/logs` (`callbacks/LRS3_CLAIX/EffConfInterCTC_Pretrained/logs`
per the CLAIX config) automatically -- no wandb, no internet needed, see
`nnet/model.py`'s `SummaryWriter`. Once the run is done:

```bash
source .venv/bin/activate
python tools/plot_training_logs.py \
    --logdir callbacks/LRS3_CLAIX/EffConfInterCTC_Pretrained/logs \
    --out callbacks/LRS3_CLAIX/EffConfInterCTC_Pretrained/report
```

Either run that directly on CLAIX (no internet needed for it either), or
rsync the small `logs/` folder back to your Mac first and run it there.
