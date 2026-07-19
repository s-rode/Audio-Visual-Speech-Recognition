# Team 1 ASR Pipeline

This repository contains the speech recognition experiments for the lab. It
supports Whisper, Parakeet CoreML, and AVEC audio-only, visual-only, and
audio-visual recognition. All backends use CSV manifests and write transcripts,
WER, and CER in the same format.

## Setup

The project uses uv:

~~~bash
uv sync
~~~

The DEMAND noise experiments need the optional noise dependencies:

~~~bash
uv sync --extra noise
~~~

ffmpeg is needed for audio and video decoding. Parakeet CoreML also needs
Swift/Xcode command-line tools.

## Project layout

~~~text
configs/           experiment configurations
data/              manifests and local datasets
external/AVEC/     editable AVEC checkout
patches/           patch for the AVEC VoxMM dataset loader
scripts/           manifest builders and analysis/plot scripts
src/team1_asr/     experiment pipeline and backends
~~~

## Manifest format

The pipeline uses CSV manifests. The standard columns are:

~~~csv
utterance_id,audio_path,video_path,reference,language
utt-00001,data/audio/file1.wav,data/video/file1.mp4,ground truth transcript,en
~~~

audio_path is required. AVEC visual and audio-visual runs also need video_path.
reference is used to calculate WER and CER.

## Basic runs

The no-op config checks that the pipeline itself is installed:

~~~bash
uv run team1-asr configs/noop_smoke.toml
~~~

Whisper can then be run with the normal experiment command:

~~~bash
uv run team1-asr configs/whisper_baseline.toml
~~~

Each run writes transcripts.jsonl and metrics.json to the configured output
directory. transcripts.jsonl has one record per utterance and includes the
actual noisy-audio path when noise is enabled.

## Hugging Face datasets

### LibriSpeech

~~~bash
uv run team1-prepare-librispeech data/librispeech/validation \
  --config clean --split validation --limit 500 --force
uv run team1-asr configs/librispeech_validation_whisper.toml
~~~

### LRS3

The official LRS3 download is currently unavailable, so development runs use
the Hugging Face mirror Ainncy/LRS3. The helper streams the dataset, exports
WAV audio, and writes a normal manifest.

~~~bash
uv run team1-prepare-lrs3-hf data/lrs3/hf_train_100 --limit 100 --force
uv run team1-asr configs/lrs3_hf_tiny.toml
~~~

Keeping the video adds video_path to the manifest and makes the export usable
for AVEC:

~~~bash
uv run team1-prepare-lrs3-hf data/lrs3/hf_av_tiny \
  --limit 10 --keep-video --force
~~~

The mirror is useful for development, but it is not an exact reproduction of
the official LRS3 test benchmark.

## Noise

The pipeline supports white noise as well as DEMAND background noise, reverb,
and bandwidth changes. A simple white-noise config looks like this:

~~~toml
[noise]
enabled = true
kind = "white"
snr_db = 10.0
seed = 0
~~~

DEMAND uses a local copy of the data at DEMAND/16k:

~~~toml
[noise]
seed = 0

[noise.demand]
enabled = true
snr_db = 10.0
noise_path = "DEMAND/16k"
~~~

Using the same seed makes the noise selection deterministic for a given
utterance, which is useful when comparing models at several SNRs.

## AVEC and VoxMM

AVEC is kept as an editable checkout in external/AVEC. The repository wraps its
audio-only, visual-only, and audio-visual models so they use the same manifests,
transcripts, noise pipeline, and WER/CER evaluation as the other backends.
Fine-tuning still happens inside the AVEC checkout.

### AVEC setup

~~~bash
git clone https://github.com/burchim/AVEC external/AVEC
cd external/AVEC
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 download_checkpoints_and_tokenizers.py
~~~

Mouth preprocessing uses AVEC's LipDetectCrop together with ibug face detection
and alignment:

~~~bash
mkdir -p external_deps
git clone https://github.com/hhj1897/face_detection.git external_deps/face_detection
git clone https://github.com/hhj1897/face_alignment.git external_deps/face_alignment
python3 -m pip install -e external_deps/face_detection
python3 -m pip install -e external_deps/face_alignment
~~~

The face-detection repository uses Git LFS for RetinaFace weights. When that
download is unavailable, the original RetinaFace repository links the same
Resnet50_Final.pth weight file. Place it here:

~~~text
external/AVEC/external_deps/face_detection/ibug/face_detection/retina_face/weights/Resnet50_Final.pth
~~~

The VoxMM patch adds a CSV-manifest dataset loader and fixes video/audio
alignment for clips where the video is slightly longer than the audio:

~~~bash
cd external/AVEC
git apply ../../patches/avec_voxmm_dataset.patch
~~~

### Preprocessing VoxMM

The router experiments start from original VoxMM clips, not already-cropped face
videos. The manifest builder finds the audio, video, and transcripts in a split
directory. AVEC preprocessing then creates the mouth ROI clips and the metadata
needed for routing.

~~~bash
uv run python scripts/build_lrs_voxmm_dev_manifest.py \
  /path/to/voxmm/dev \
  data/lrs_voxmm/manifest_dev.csv

uv run team1-preprocess-avec-video \
  data/lrs_voxmm/manifest_dev.csv \
  data/lrs_voxmm/dev_avec_roi \
  --force
~~~

The output directory contains:

- manifest.csv: audio paths and generated mouth ROI video paths for AVEC
- visual_reliability.csv: face/landmark confidence, missing-frame and
  interpolation statistics, multiple-face rate, ROI status, and failure reason
- video/: generated 88x88 mouth ROI clips

The same workflow applies to train data. The generated manifest is used for
AVEC training and inference; visual_reliability.csv is used by the router.

### AVEC inference and fine-tuning

The preprocessed LRS3 configs are a quick AVEC smoke test:

~~~bash
uv run team1-asr configs/lrs3_avec_audio_preprocessed_clean.toml
uv run team1-asr configs/lrs3_avec_visual_preprocessed_clean.toml
uv run team1-asr configs/lrs3_avec_av_preprocessed_clean.toml
~~~

The VoxMM AV fine-tuning config is
configs/VoxMM/EffConfInterCTC_finetune.py. It starts from the LRS23 AV model,
freezes part of the visual branch, and uses float32 with cosine learning-rate
decay. Copy it into AVEC, set manifest_train and manifest_eval, then run:

~~~bash
mkdir -p external/AVEC/configs/VoxMM
cp configs/VoxMM/EffConfInterCTC_finetune.py external/AVEC/configs/VoxMM/

cd external/AVEC
.venv/bin/python3 main.py \
  -c configs/VoxMM/EffConfInterCTC_finetune.py \
  --mode training \
  --eval_period_epoch 1 \
  --saving_period_epoch 1 \
  --batch_size_eval 8 \
  --step_log_period 50
~~~

The AV checkpoint is expected at:

~~~text
external/AVEC/callbacks/VoxMM/AV/EffConfInterCTC_finetune/checkpoints_epoch_12_step_7201.ckpt
~~~

The matching audio-only checkpoint is expected at:

~~~text
external/AVEC/callbacks/VoxMM/AO/EffConfInterCTC_finetune/checkpoints_epoch_6_step_3600.ckpt
~~~

## Live router demo

The browser demo streams the sequential decision as it happens: Whisper
transcribes first, the frozen policy scores pre-AVEC features, and AVEC runs only
for requests selected by the router. Presentation mode is self-contained and
does not load either model:

~~~bash
uv run team1-router-demo
~~~

Open http://127.0.0.1:7860, allow camera and microphone access, and record a
five-second synchronized window. The server extracts 16 kHz mono audio, checks
the video stream, runs Whisper, and shows whether the frozen router keeps the
Whisper result or invokes AVEC. The inspector exposes the captured clip, the exact
video passed to AVEC, extracted-audio playback and waveform, plus routing feature
telemetry. Whisper and AVEC transcripts update independently with timestamped
segments and the selected result is highlighted. The Samples tab retains the clear-room and
street-noise controls for deterministic presentations.

In real mode, live browser captures also pass through the AVEC mouth-ROI
preprocessor. That requires the AVEC repository, its configured Python
environment, and face-detection dependencies to be available.

Real mode reads samples from an AVEC mouth-ROI manifest and its preprocessing
reliability CSV. It lazily loads both backends and serves only media paths listed
in that manifest:

~~~bash
uv run team1-router-demo \
  --manifest data/lrs_voxmm/dev_avec_roi_600_balanced/manifest.csv \
  --reliability data/lrs_voxmm/dev_avec_roi_600_balanced/visual_reliability.csv \
  --policy outputs/lrs_voxmm_noise_600_enriched/router_study/best_policy.json \
  --whisper-config configs/lrs_voxmm_whisper_base_600_enriched.toml \
  --avec-config configs/lrs_voxmm_avec_av_finetuned_600_balanced.toml
~~~

## Reliability routing

The router is a sequential setup. Whisper base.en always transcribes the audio
first. Based on transcript statistics, noisy-audio features, mouth ROI quality
and motion, audio-video synchronization, and preprocessing reliability values,
it decides whether AVEC-AV should also run. It does not use AVEC predictions or
AVEC confidence for this decision.

The current development experiment is a balanced 600-clip VoxMM subset across
source videos. It uses clean audio and three DEMAND conditions: 10 dB, 0 dB,
and -5 dB.

~~~bash
uv run team1-asr configs/lrs_voxmm_whisper_base_600_enriched.toml
uv run --extra noise team1-asr configs/lrs_voxmm_whisper_base_600_enriched_demand_10db.toml
uv run --extra noise team1-asr configs/lrs_voxmm_whisper_base_600_enriched_demand_0db.toml
uv run --extra noise team1-asr configs/lrs_voxmm_whisper_base_600_enriched_demand_-5db.toml
~~~

The resulting Whisper transcripts, noisy waveforms, and existing mouth ROI
clips are turned into router features with:

~~~bash
uv run team1-reliability extract-features \
  --roi-manifest data/lrs_voxmm/dev_avec_roi_600_balanced/manifest.csv \
  --reliability data/lrs_voxmm/dev_avec_roi_600_balanced/visual_reliability.csv \
  --condition clean outputs/lrs_voxmm_noise_600_enriched/whisper_clean/transcripts.jsonl \
  --condition demand_10db outputs/lrs_voxmm_noise_600_enriched/whisper_demand_10db/transcripts.jsonl \
  --condition demand_0db outputs/lrs_voxmm_noise_600_enriched/whisper_demand_0db/transcripts.jsonl \
  --condition demand_-5db outputs/lrs_voxmm_noise_600_enriched/whisper_demand_-5db/transcripts.jsonl \
  --output-dir outputs/lrs_voxmm_noise_600_enriched/features
~~~

routing_features.csv contains one row per condition and utterance. The summary
CSV and JSON files describe missingness and the p10, median, and p90 of each
feature for every condition.

The noise study compares the existing logistic baseline with enriched logistic
and benefit-aware tree routers. It uses grouped nested cross-validation, so all
conditions from a source video remain in the same fold. Nominal SNR is only a
reporting condition, not a router input.

~~~bash
uv run team1-reliability noise-study \
  --manifest data/lrs_voxmm/manifest_dev_600_balanced.csv \
  --reliability data/lrs_voxmm/dev_avec_roi_600_balanced/visual_reliability.csv \
  --features outputs/lrs_voxmm_noise_600_enriched/features/routing_features.csv \
  --condition clean \
    outputs/lrs_voxmm_noise_600_enriched/whisper_clean/transcripts.jsonl \
    outputs/lrs_voxmm_noise_600/avec_av_clean/transcripts.jsonl \
  --condition demand_10db \
    outputs/lrs_voxmm_noise_600_enriched/whisper_demand_10db/transcripts.jsonl \
    outputs/lrs_voxmm_noise_600/avec_av_demand_10db/transcripts.jsonl \
  --condition demand_0db \
    outputs/lrs_voxmm_noise_600_enriched/whisper_demand_0db/transcripts.jsonl \
    outputs/lrs_voxmm_noise_600/avec_av_demand_0db/transcripts.jsonl \
  --condition demand_-5db \
    outputs/lrs_voxmm_noise_600_enriched/whisper_demand_-5db/transcripts.jsonl \
    outputs/lrs_voxmm_noise_600/avec_av_demand_-5db/transcripts.jsonl \
  --output-dir outputs/lrs_voxmm_noise_600_enriched/noise_router_study
~~~

The study writes out-of-fold WER values, routing decisions, oracle WER, the
remaining router-to-oracle gap, bootstrap comparisons, and fitted coefficients
or feature importances. The presentation figure can be generated with:

~~~bash
uv run python scripts/plot_voxmm_noise_router_study.py \
  --study outputs/lrs_voxmm_noise_600_enriched/noise_router_study/noise_router_study.json \
  --output-dir outputs/lrs_voxmm_noise_600_enriched/figures
~~~
