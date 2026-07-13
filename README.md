# Team 1 ASR Pipeline

This repository contains a minimal, modular ASR experiment scaffold:

- Whisper-based batch transcription via `faster-whisper`
- Parakeet v3 CoreML transcription through a small Swift bridge
- AVEC audio-only, visual-only, and audio-visual transcription through an
  editable external checkout
- a swappable backend interface for comparing ASR models
- manifest-driven experiment runs
- optional white-noise and DEMAND/reverb/bandwidth noise injection
- WER and CER evaluation against references
- Hugging Face export tooling for LibriSpeech and LRS3 audio subsets

## Project layout

```text
configs/
  lrs3_test_parakeet_coreml_clean.toml
  lrs3_test_parakeet_coreml_advanced_30db.toml
  lrs3_test_parakeet_coreml_advanced_20db.toml
  lrs3_test_parakeet_coreml_advanced_10db.toml
  lrs3_test_parakeet_coreml_advanced_5db.toml
  lrs3_test_parakeet_coreml_advanced_0db.toml
  lrs3_test_parakeet_coreml_advanced_-5db.toml
  lrs3_avec_audio_clean.toml
  lrs3_avec_visual_clean.toml
  lrs3_avec_av_clean.toml
  lrs3_hf_tiny.toml
  lrs3_hf_tiny_clean.toml
  lrs3_hf_tiny_white_20db.toml
  lrs3_hf_tiny_white_10db.toml
  lrs3_hf_tiny_white_0db.toml
  librispeech_demo_tiny.toml
  librispeech_validation_whisper.toml
  noop_smoke.toml
  whisper_baseline.toml
data/
  example_manifest.csv
src/team1_asr/
  backends/
  cli.py
  config.py
  evaluation.py
  manifest.py
  noise.py
  pipeline.py
  prepare_lrs3_hf.py
  prepare_lrs3_test_hf.py
  prepare_librispeech.py
  types.py
```

## Install

Create the virtual environment and install the project with `uv`:

```bash
uv sync
```

To enable Weights & Biases logging, install the optional tracking dependencies:

```bash
uv sync --extra tracking
```

To run DEMAND/reverb/bandwidth noise experiments, install the optional noise
dependencies:

```bash
uv sync --extra noise --extra tracking
```

You also need `ffmpeg` available on the system for Whisper audio decoding.
Parakeet CoreML experiments additionally need Swift/Xcode command line tools.

This project pins `uv`'s cache to a local `.uv-cache/` directory so `uv run` and `uv sync`
do not depend on the machine's global cache configuration.

## Manifest format

The pipeline expects a CSV manifest. The default columns are:

```csv
utterance_id,audio_path,video_path,reference,language
utt-00001,data/audio/file1.wav,data/video/file1.mp4,ground truth transcript,en
utt-00002,data/audio/file2.wav,data/video/file2.mp4,another reference,en
```

- `utterance_id`: optional, auto-generated if missing
- `audio_path`: required
- `video_path`: optional, required by visual and audio-visual backends
- `reference`: optional, but required for WER/CER
- `language`: optional, falls back to config

## Run

Run a smoke test without model dependencies:

```bash
uv run team1-asr configs/noop_smoke.toml
```

Run the actual Whisper baseline:

```bash
uv run team1-asr configs/whisper_baseline.toml
```

Prepare a LibriSpeech subset from Hugging Face:

```bash
uv run team1-prepare-librispeech data/librispeech/validation --config clean --split validation --limit 500 --force
```

Run Whisper on that LibriSpeech export:

```bash
uv run team1-asr configs/librispeech_validation_whisper.toml
```

For a quick one-utterance demo with a smaller model:

```bash
uv run team1-prepare-librispeech data/librispeech/validation --config clean --split validation --limit 1 --force
uv run team1-download-model tiny.en
uv run team1-asr configs/librispeech_demo_tiny.toml
uv run team1-show-transcripts outputs/librispeech_demo_tiny/transcripts.jsonl --limit 1
```

`team1-download-model` shows Hugging Face download progress and caches the model
before `team1-asr` loads it. The experiment runner also prints progress messages
while loading the backend and transcribing utterances.

Outputs are written to the configured output directory as:

- `transcripts.jsonl`: one JSON record per utterance
- `metrics.json`: experiment summary plus normalized `wer`/`cer` and raw `raw_wer`/`raw_cer`

## Weights & Biases Tracking

Experiment configs can opt into W&B logging with a `[wandb]` section:

```toml
[wandb]
enabled = true
project = "team1-asr"
run_name = "lrs3-tiny-white-10db"
mode = "offline"
log_artifacts = true
log_transcripts = true
transcript_table_limit = 100
```

When enabled, `team1-asr` logs numeric metrics and a transcript table.
`log_artifacts = true` also uploads the run output directory as a W&B artifact.
`mode = "offline"` keeps laptop runs local. `mode = "online"` syncs directly to
W&B.

## LRS3 Test Experiment

The current setup uses LRS3 test audio, Parakeet CoreML, and seeded advanced
noise from DEMAND/reverb/bandwidth transforms.

Small smoke export using the same manifest path as the checked-in configs:

```bash
uv run team1-prepare-lrs3-test data/lrs3/hf_test_full --limit 2 --force
uv run team1-asr configs/lrs3_test_parakeet_coreml_clean.toml
```

Full test export and Parakeet SNR sweep:

```bash
uv run team1-prepare-lrs3-test data/lrs3/hf_test_full --force
uv run --extra tracking team1-asr configs/lrs3_test_parakeet_coreml_clean.toml
uv run --extra noise --extra tracking team1-asr configs/lrs3_test_parakeet_coreml_advanced_30db.toml
uv run --extra noise --extra tracking team1-asr configs/lrs3_test_parakeet_coreml_advanced_20db.toml
uv run --extra noise --extra tracking team1-asr configs/lrs3_test_parakeet_coreml_advanced_10db.toml
uv run --extra noise --extra tracking team1-asr configs/lrs3_test_parakeet_coreml_advanced_5db.toml
uv run --extra noise --extra tracking team1-asr configs/lrs3_test_parakeet_coreml_advanced_0db.toml
uv run --extra noise --extra tracking team1-asr configs/lrs3_test_parakeet_coreml_advanced_-5db.toml
```

The local DEMAND dataset is expected at `DEMAND/16k`. Dataset exports, generated
audio, W&B runs, and build outputs are ignored by Git.

## Noise configuration

Legacy flat noise configs still work:

```toml
[noise]
enabled = true
kind = "white"
snr_db = 10.0
seed = 0
```

Advanced noise uses nested sections and a deterministic seed:

```toml
[noise]
seed = 0

[noise.demand]
enabled = true
snr_db = 10.0
noise_path = "DEMAND/16k"

[noise.reverb]
enabled = true

[noise.bandwidth]
enabled = true
```

Place the local DEMAND dataset at `DEMAND/` in the repository root. The pipeline
uses `DEMAND/16k` and ignores the whole folder in Git.

For advanced noise, the pipeline applies reverb, then DEMAND background mixing,
then bandwidth limitation. Random choices are derived from `seed + utterance_id`,
so SNR sweeps remain reproducible per utterance.

## Parakeet CoreML

LRS3 test export:

```bash
uv run team1-prepare-lrs3-test data/lrs3/hf_test_full --force
```

FluidAudio/CoreML bridge build:

```bash
swift build -c release --package-path tools/parakeet_coreml_bridge
```

Clean Parakeet run:

```bash
uv run team1-asr configs/lrs3_test_parakeet_coreml_clean.toml
```

Seeded advanced-noise run:

```bash
uv run --extra noise --extra tracking team1-asr configs/lrs3_test_parakeet_coreml_advanced_10db.toml
```

The Parakeet backend uses `FluidInference/parakeet-tdt-0.6b-v3-coreml` through
FluidAudio and labels runs as `parakeet-coreml`.

## Design choice

The project uses a small local pipeline instead of wrapping everything in one
external CLI. The important parts stay explicit:

- backend and model
- noise type
- evaluation metrics
- logging and output format

## LibriSpeech Workflow

LibriSpeech uses Hugging Face `openslr/librispeech_asr` and exports a local
manifest plus WAV files.

The `team1-prepare-librispeech` command:

- loads a LibriSpeech split directly from Hugging Face
- exports the first `--limit` utterances in streaming mode for quick local tests
- writes one local `.wav` file per utterance plus a `manifest.csv`
- can also save a Hugging Face subset with `--save-dataset`, which may require downloading the full split before applying `--limit`

Useful split options are `--config clean --split validation` and
`--config clean --split test`. `train.100` is available when training data is
needed.

## LRS3 Hugging Face Workflow

For now we use the Hugging Face mirror `Ainncy/LRS3`, because the official LRS3
download is not available to us. The helper streams the dataset, downloads one
MP4 clip at a time, extracts mono 16 kHz WAV audio with `ffmpeg`, and writes a
normal `manifest.csv`.

```bash
uv run team1-prepare-lrs3-hf data/lrs3/hf_train_100 --limit 100 --force
uv run team1-download-model tiny.en
uv run team1-asr configs/lrs3_hf_tiny.toml
uv run team1-show-transcripts outputs/lrs3_hf_tiny/transcripts.jsonl --limit 5
```

Noise variants reuse the same exported audio manifest:

```bash
uv run team1-asr configs/lrs3_hf_tiny_clean.toml
uv run team1-asr configs/lrs3_hf_tiny_white_20db.toml
uv run team1-asr configs/lrs3_hf_tiny_white_10db.toml
uv run team1-asr configs/lrs3_hf_tiny_white_0db.toml
```

The mirror currently exposes one `train` split. So runs from this source are
useful for development and first baselines, but they should not yet be described
as an exact official LRS3 test benchmark.

For AVSR experiments, keep the MP4 clips as well. This adds `video_path` to the
manifest:

```bash
uv run team1-prepare-lrs3-hf data/lrs3/hf_av_tiny --limit 10 --keep-video --force
```

For visual-only and audio-visual AVEC, the raw videos should first be converted
to mouth ROI clips. This is closer to what AVEC does than just using the full
video or a simple center crop:

```bash
uv run team1-preprocess-avec-video \
  data/lrs3/hf_av_tiny/manifest.csv \
  data/lrs3/hf_av_tiny_avec_roi \
  --force
```

This command calls `external/AVEC/.venv/bin/python3` internally and reuses
AVEC's `LipDetectCrop` with `media/20words_mean_face.npy`. The AVEC environment
also needs the `ibug.face_detection` and `ibug.face_alignment` packages, using
the repositories linked from the AVEC README:

```bash
cd external/AVEC
source .venv/bin/activate

mkdir -p external_deps
git clone https://github.com/hhj1897/face_detection.git external_deps/face_detection
git clone https://github.com/hhj1897/face_alignment.git external_deps/face_alignment

.venv/bin/python3 -m pip install -e external_deps/face_detection
.venv/bin/python3 -m pip install -e external_deps/face_alignment
```

The face detection repo uses Git LFS for one model file, so the clone can fail
if the upstream LFS quota is used up.

If that happens, use the RetinaFace weights linked by the original
`biubug6/Pytorch_Retinaface` repository instead:
`https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1`.
Download `Resnet50_Final.pth` and place it at:

```text
external/AVEC/external_deps/face_detection/ibug/face_detection/retina_face/weights/Resnet50_Final.pth
```

The preprocessing command uses these RetinaFace weights to detect and align the
face before cropping the mouth ROI.

## AVEC Workflow

AVEC is integrated through an external checkout. The experiment pipeline uses
the same manifests, noise transforms, transcript format, and WER/CER evaluation
as the other backends. The checkout can also be modified or fine-tuned directly
when needed.

Initial setup:

```bash
git clone https://github.com/burchim/AVEC external/AVEC
cd external/AVEC
pip install -r requirements.txt
python download_checkpoints_and_tokenizers.py
```

The official AVEC configs expect checkpoints and tokenizers in AVEC's own folder
layout. These commands are useful to verify that the original AVEC setup works
before running it through this repository:

```bash
cd external/AVEC
python main.py -c configs/LRS23/AO/EffConfCTC.py --mode evaluation --checkpoint checkpoints_swa-equal-200-210.ckpt --eval_steps 1
python main.py -c configs/LRS23/VO/EffConfInterCTC.py --mode evaluation --checkpoint checkpoints_swa-equal-90-100.ckpt --eval_steps 1
python main.py -c configs/LRS23/AV/EffConfInterCTC.py --mode evaluation --checkpoint checkpoints_swa-equal-60-70.ckpt --eval_steps 1
```

Once the AVEC checkout is set up, run the same models through our experiment
pipeline:

```bash
uv run team1-asr configs/lrs3_avec_audio_clean.toml
uv run team1-asr configs/lrs3_avec_visual_clean.toml
uv run team1-asr configs/lrs3_avec_av_clean.toml
```

After mouth ROI preprocessing, use these configs:

```bash
uv run team1-asr configs/lrs3_avec_audio_preprocessed_clean.toml
uv run team1-asr configs/lrs3_avec_visual_preprocessed_clean.toml
uv run team1-asr configs/lrs3_avec_av_preprocessed_clean.toml
```

The `avec` backend starts `scripts/run_avec_manifest.py` as a subprocess. The
runner loads the editable AVEC checkout, processes one manifest row at a time,
and writes the same `transcripts.jsonl` / `metrics.json` format as the other
backends. For visual-only and audio-visual runs, `video_path` should point to
the mouth ROI video.

The Hugging Face LRS3 mirror currently provides a practical development source.
It is not yet an exact replacement for the official LRS3 benchmark split.
## VoxMM Finetuning (AVEC)

AVEC was fine-tuned on VoxMM train data and evaluated on a small LRS-VoxMM dev
subset. This uses the normal AVEC setup above, including the checkpoint,
tokenizer, and face-processing dependencies.

### Getting the preprocessed data

VoxMM requires downloading the original YouTube videos and running face/lip
detection, which is slow even on a GPU. To skip this, a preprocessed copy of
the train data and a small LRS-VoxMM dev eval subset are available:

- `voxmm_preprocessed.tar.gz` (~2.3G) - VoxMM train set, already converted to
  88x88 grayscale mouth ROI clips + sliced WAVs, with `manifest.csv`
- `voxmm_dev_eval.tar.gz` (~71M) - 500-utterance LRS-VoxMM dev subset,
  preprocessed the same way, used for periodic evaluation during finetuning.
  Self-contained (both WAVs and mouth ROI videos included) with relative
  manifest paths.

Download both from the shared Drive link (see team channel), then extract:

```bash
mkdir -p data/voxmm_preprocessed
tar -xzf voxmm_preprocessed.tar.gz -C data/voxmm_preprocessed --strip-components=1

mkdir -p data/voxmm_dev_eval
tar -xzf voxmm_dev_eval.tar.gz -C data/voxmm_dev_eval --strip-components=1
```

The `manifest.csv` files inside the tarballs may still contain the original
cluster paths (`/hpcwork/...`). Update the audio and video paths after
extracting the archives if they do not match the local directory.

### Applying the AVEC patch

The AVEC patch adds a VoxMM CSV-manifest dataset and fixes
`align_video_to_audio` for clips where the video is slightly longer than the
audio. Apply it from the AVEC checkout root:

```bash
cd external/AVEC
git apply ../../patches/avec_voxmm_dataset.patch
```

This adds `nnet.datasets.VoxMM` (a CSV-manifest-driven dataset that reads our
preprocessed mouth ROI clips and WAVs) and fixes `align_video_to_audio` to
trim rather than crash when the video is longer than the audio.

### Finetuning config

The finetuning config is in `configs/VoxMM/EffConfInterCTC_finetune.py`. Copy
it into the AVEC checkout before training:

```bash
mkdir -p external/AVEC/configs/VoxMM
cp configs/VoxMM/EffConfInterCTC_finetune.py external/AVEC/configs/VoxMM/
```

The config:

- loads the pretrained AV checkpoint
  (`callbacks/LRS23/AV/EffConfInterCTC/checkpoints_swa-equal-60-70.ckpt`)
  with `strict=True`
- freezes the visual front-end (ResNet-18 + 3D conv stem) and visual back-end
  stage 2 (~23% of params), leaving the rest trainable
- uses `torch.float32` precision (float16 caused NaN losses on this data)
- uses a `CosineAnnealingScheduler` (300 warmup steps, peak LR 1e-4, 9000
  total steps, 15 epochs)
- logs to the `team1-asr` Weights & Biases project

Set `manifest_train` and `manifest_eval` in the config to the extracted train
and dev manifests.

### Running finetuning

```bash
cd external/AVEC
.venv/bin/python3 main.py \
    -c configs/VoxMM/EffConfInterCTC_finetune.py \
    --mode training \
    --eval_period_epoch 1 \
    --saving_period_epoch 1 \
    --batch_size_eval 8 \
    --step_log_period 50
```

On a single H100 this takes roughly 5-6 hours for the full 15 epochs. Smaller
or slower GPUs will take longer; you can shorten the run for a smoke test with
`--steps_per_epoch 10 --eval_steps 5`.

Each epoch produces a checkpoint in
`external/AVEC/callbacks/VoxMM/AV/EffConfInterCTC_finetune/`.

## Visual reliability routing

For the routing experiments we compare Whisper `base.en` with our fine-tuned
AVEC models. Whisper is still a different model trained on different data, so
this is not a perfectly controlled comparison. It is still a more useful local
baseline than only comparing AVEC with the much larger Parakeet model.

The AV config expects the fine-tuned checkpoint at:

```text
external/AVEC/callbacks/VoxMM/AV/EffConfInterCTC_finetune/checkpoints_epoch_12_step_7201.ckpt
```

The matching audio-only config expects its checkpoint at:

```text
external/AVEC/callbacks/VoxMM/AO/EffConfInterCTC_finetune/checkpoints_epoch_6_step_3600.ckpt
```

To start from the original LRS-VoxMM dev clips, build a manifest and create the
mouth crops once:

```bash
uv run python scripts/build_lrs_voxmm_dev_manifest.py \
  /path/to/lrs-voxmm/original/dev \
  data/lrs_voxmm/manifest_dev.csv

uv run team1-preprocess-avec-video \
  data/lrs_voxmm/manifest_dev.csv \
  data/lrs_voxmm/dev_avec_roi \
  --force
```

This writes the mouth videos, a normal `manifest.csv`, and
`visual_reliability.csv`. The last file contains face/landmark confidence,
missing-frame statistics, and whether preprocessing had to fall back to the
full frame.

For the clean baseline, run Whisper and AVEC-AV:

```bash
uv run team1-download-model base.en
uv run team1-asr configs/lrs_voxmm_whisper_base_dev.toml
uv run team1-asr configs/lrs_voxmm_avec_av_finetuned_dev.toml
```

The reliability scripts only need the transcript files and join them using
`utterance_id`. Parakeet is still available through
`configs/lrs_voxmm_parakeet_dev.toml` as a larger secondary baseline.

### Clean routing check

The first router tried to decide whether Whisper or AVEC-AV should be used for
each clip. The simple version uses one threshold. The learned version uses the
preprocessing statistics together with Whisper confidence values.

For the simple threshold check:

```bash
uv run team1-reliability calibrate \
  --manifest data/lrs_voxmm/manifest_dev.csv \
  --reliability data/lrs_voxmm/dev_avec_roi/visual_reliability.csv \
  --audio-transcripts outputs/lrs_voxmm_whisper_base_dev/transcripts.jsonl \
  --audio-visual-transcripts outputs/lrs_voxmm_avec_av_finetuned_dev/transcripts.jsonl \
  --audio-system-name whisper_base_en \
  --av-system-name avec_voxmm_finetuned_av \
  --output-dir outputs/lrs_voxmm_reliability_dev
```

This writes the selected rule, threshold sweep, routing decisions, and WER/CER
results.

The learned router uses the same visual features together with Whisper's own
confidence values. It uses grouped cross-validation, so clips from one source
video never appear in both training and validation. It only reads the completed
ASR outputs, so it does not redo preprocessing or transcription:

```bash
uv run team1-reliability learn \
  --manifest data/lrs_voxmm/manifest_dev_200_balanced.csv \
  --reliability data/lrs_voxmm/dev_avec_roi_200_balanced/visual_reliability.csv \
  --audio-transcripts outputs/lrs_voxmm_whisper_base_200_balanced/transcripts.jsonl \
  --audio-visual-transcripts outputs/lrs_voxmm_avec_av_finetuned_200_balanced/transcripts.jsonl \
  --audio-system-name whisper_base_en \
  --av-system-name avec_voxmm_finetuned_av \
  --output-dir outputs/lrs_voxmm_reliability_200_balanced_learned
```

`cross_validation_metrics.json` contains the relevant out-of-fold result. The
command also writes the out-of-fold decisions and model coefficients.

We also compared smaller audio-only, visual-only, and combined routers. This
removes the almost constant face-detector values, uses stronger regularization,
and chooses the threshold inside grouped validation:

```bash
uv run team1-reliability study \
  --manifest data/lrs_voxmm/manifest_dev_200_balanced.csv \
  --reliability data/lrs_voxmm/dev_avec_roi_200_balanced/visual_reliability.csv \
  --audio-transcripts outputs/lrs_voxmm_whisper_base_200_balanced/transcripts.jsonl \
  --audio-visual-transcripts outputs/lrs_voxmm_avec_av_finetuned_200_balanced/transcripts.jsonl \
  --audio-system-name whisper_base_en \
  --av-system-name avec_voxmm_finetuned_av \
  --output-dir outputs/lrs_voxmm_reliability_200_balanced_router_study
```

`router_study.csv` contains the out-of-fold WER values. On the current clean
VoxMM-200 subset, none of these router variants beat Whisper. To create the
summary figure:

```bash
uv run python scripts/plot_router_study.py \
  --study outputs/lrs_voxmm_reliability_200_balanced_router_study/router_study.json \
  --full-learned outputs/lrs_voxmm_reliability_200_balanced_learned/cross_validation_metrics.json \
  --oracle outputs/lrs_voxmm_reliability_200_balanced_audio_uncertainty/metrics.json \
  --output-dir outputs/lrs_voxmm_reliability_200_balanced_router_study/figures
```

The script writes both PNG and PDF versions of the figure.

### DEMAND noise sweep

The noise sweep tests whether AVEC becomes more useful once the audio is harder,
while keeping the same mouth crops. It reuses the balanced VoxMM-200 ROI clips.

First run clean fine-tuned AO once:

```bash
uv run team1-asr configs/lrs_voxmm_avec_ao_finetuned_200_balanced.toml
```

Then run DEMAND-only background noise at 10, 0, and -5 dB. All configs use the
same seed, so an utterance gets the same DEMAND track for every model and SNR.

```bash
uv run --extra noise team1-asr configs/lrs_voxmm_whisper_base_200_balanced_demand_10db.toml
uv run --extra noise team1-asr configs/lrs_voxmm_avec_ao_finetuned_200_balanced_demand_10db.toml
uv run --extra noise team1-asr configs/lrs_voxmm_avec_av_finetuned_200_balanced_demand_10db.toml

uv run --extra noise team1-asr configs/lrs_voxmm_whisper_base_200_balanced_demand_0db.toml
uv run --extra noise team1-asr configs/lrs_voxmm_avec_ao_finetuned_200_balanced_demand_0db.toml
uv run --extra noise team1-asr configs/lrs_voxmm_avec_av_finetuned_200_balanced_demand_0db.toml

uv run --extra noise team1-asr configs/lrs_voxmm_whisper_base_200_balanced_demand_-5db.toml
uv run --extra noise team1-asr configs/lrs_voxmm_avec_ao_finetuned_200_balanced_demand_-5db.toml
uv run --extra noise team1-asr configs/lrs_voxmm_avec_av_finetuned_200_balanced_demand_-5db.toml
```

After the runs finish, summarize them and create the figures:

```bash
uv run python scripts/analyze_voxmm_noise_sweep.py \
  --manifest data/lrs_voxmm/manifest_dev_200_balanced.csv \
  --clean-whisper outputs/lrs_voxmm_whisper_base_200_balanced/transcripts.jsonl \
  --clean-ao outputs/lrs_voxmm_noise_200/avec_ao_clean/transcripts.jsonl \
  --clean-av outputs/lrs_voxmm_avec_av_finetuned_200_balanced/transcripts.jsonl \
  --noise-root outputs/lrs_voxmm_noise_200 \
  --output-dir outputs/lrs_voxmm_noise_200/analysis

uv run python scripts/plot_voxmm_noise_sweep.py \
  --metrics outputs/lrs_voxmm_noise_200/analysis/noise_sweep_metrics.csv \
  --output-dir outputs/lrs_voxmm_noise_200/analysis/figures
```

The analysis reports WER curves, AVEC-AV wins over Whisper and AO, AVEC
coverage, and the Whisper/AV oracle. These results provide the input for the
next routing experiment under noise.

### Noise-aware routing study

The noise-aware study trains small logistic routers from the completed Whisper
and AVEC-AV transcripts. It compares Whisper confidence features, waveform
features from the noisy audio, and their combination with mouth-landmark
confidence. The nominal DEMAND SNR is only used for reporting; it is not a
router feature. Grouped cross-validation keeps every condition from the same
source video in the same fold.

```bash
uv run team1-reliability noise-study \
  --manifest data/lrs_voxmm/manifest_dev_200_balanced.csv \
  --reliability data/lrs_voxmm/dev_avec_roi_200_balanced/visual_reliability.csv \
  --condition clean \
    outputs/lrs_voxmm_whisper_base_200_balanced/transcripts.jsonl \
    outputs/lrs_voxmm_avec_av_finetuned_200_balanced/transcripts.jsonl \
  --condition demand_10db \
    outputs/lrs_voxmm_noise_200/whisper_demand_10db/transcripts.jsonl \
    outputs/lrs_voxmm_noise_200/avec_av_demand_10db/transcripts.jsonl \
  --condition demand_0db \
    outputs/lrs_voxmm_noise_200/whisper_demand_0db/transcripts.jsonl \
    outputs/lrs_voxmm_noise_200/avec_av_demand_0db/transcripts.jsonl \
  --condition demand_-5db \
    outputs/lrs_voxmm_noise_200/whisper_demand_-5db/transcripts.jsonl \
    outputs/lrs_voxmm_noise_200/avec_av_demand_-5db/transcripts.jsonl \
  --output-dir outputs/lrs_voxmm_noise_200/noise_router_study
```

The study writes `noise_router_metrics.csv` with an out-of-fold WER breakdown
for every SNR, `noise_router_decisions.csv`, and `noise_router_study.json` with
the feature-set comparison.

For a new split, use that frozen policy to build the routed manifests:

```bash
uv run team1-reliability split \
  --source-manifest data/lrs_voxmm/manifest_test.csv \
  --roi-manifest data/lrs_voxmm/test_avec_roi/manifest.csv \
  --reliability data/lrs_voxmm/test_avec_roi/visual_reliability.csv \
  --policy outputs/lrs_voxmm_reliability_dev/policy.json \
  --output-dir data/lrs_voxmm/test_routed
```

For a learned policy, first transcribe the complete test audio with Whisper and
pass its transcript file to `split`. The router uses Whisper confidence features
when selecting clips for AVEC:

```bash
uv run team1-reliability split \
  --source-manifest data/lrs_voxmm/manifest_test.csv \
  --roi-manifest data/lrs_voxmm/test_avec_roi/manifest.csv \
  --reliability data/lrs_voxmm/test_avec_roi/visual_reliability.csv \
  --audio-transcripts outputs/lrs_voxmm_test_whisper/transcripts.jsonl \
  --policy outputs/lrs_voxmm_reliability_dev_learned/policy.json \
  --output-dir data/lrs_voxmm/test_routed
```

This writes `audio_manifest.csv` for the complete audio-only baseline,
`audio_fallback_manifest.csv` for only the audio-routed rows, and
`audio_visual_manifest.csv` for AVEC.

For a proper test report, run Whisper on the complete audio manifest and AVEC
on its routed manifest. Then evaluate the combined predictions with:

```bash
uv run team1-reliability evaluate \
  --manifest data/lrs_voxmm/manifest_test.csv \
  --reliability data/lrs_voxmm/test_avec_roi/visual_reliability.csv \
  --audio-transcripts outputs/lrs_voxmm_test_routed_whisper/transcripts.jsonl \
  --audio-visual-transcripts outputs/lrs_voxmm_test_routed_avec/transcripts.jsonl \
  --policy outputs/lrs_voxmm_reliability_dev/policy.json \
  --output-dir outputs/lrs_voxmm_reliability_test
```

The oracle is the upper bound obtained by selecting the lower-error prediction
per utterance using the reference transcript. It is reported only for analysis.

For a quick local check without VoxMM, use the two LRS3 clips that are already
part of the local setup:

```bash
uv run team1-asr configs/lrs3_reliability_whisper_base_smoke.toml
uv run team1-asr configs/lrs3_reliability_avec_finetuned_smoke.toml

uv run team1-reliability calibrate \
  --manifest data/lrs3/hf_av_tiny/manifest.csv \
  --reliability data/lrs3/hf_av_tiny_reliability/visual_reliability.csv \
  --audio-transcripts outputs/lrs3_reliability_whisper_base_smoke/transcripts.jsonl \
  --audio-visual-transcripts outputs/lrs3_reliability_avec_finetuned_smoke/transcripts.jsonl \
  --audio-system-name whisper_base_en \
  --av-system-name avec_voxmm_finetuned_av \
  --output-dir outputs/lrs3_reliability_whisper_smoke
```
