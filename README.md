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

## Reliability routing

The router is a sequential setup. Whisper base.en always transcribes the audio
first. It then decides whether AVEC-AV is likely to reduce word errors for the
utterance. AVEC predictions, AVEC confidence, transcript disagreement, and the
reference are not available to the decision.

The full VoxMM study uses source-video-disjoint train, development, and test
splits. Every utterance from one source video stays in the same split. Candidate
models are selected on development data, while the test split stays frozen for
the final evaluation.

Router features are extracted from the waveform, aligned mouth ROI, visual
preprocessing metadata, and Whisper output. They cover Whisper uncertainty,
audio quality, visual reliability, mouth motion and image quality, and
audio-video synchronization. The enriched feature set has 33 inputs, all
available before AVEC runs.

~~~bash
uv run team1-reliability extract-features --help
uv run team1-reliability train-dev-study --help
~~~

`extract-features` writes one feature row per utterance. The train/dev study
fits router candidates on training videos, calibrates each routing threshold
from grouped out-of-fold training predictions, and evaluates the fitted policy
once on development videos. Its output includes decisions, WER/CER summaries,
oracle WER, route rates, grouped-bootstrap comparisons, and policy metadata.

The staged commands `train-dev-candidate`, `train-dev-oof-fold`,
`select-train-dev-candidate`, and `finalize-train-dev-study` support the same
workflow in separate cluster jobs.

The candidates include logistic baselines, benefit-aware ridge regression,
HistGradientBoosting regression, and a small MLP regression ablation. All
benefit models predict Whisper word errors minus AVEC word errors, so positive
values mean that AVEC is expected to help. The current full study selected the
enriched HistGradientBoosting router; the MLP was kept as a controlled
comparison and did not replace it.

After development selection, `split` applies the frozen policy to new
utterances and writes separate Whisper and AVEC manifests. `evaluate` combines
the resulting transcripts into one routed result. The same policy can then be
reported under controlled DEMAND noise conditions. Nominal SNR is a reporting
condition only and is never included as a router feature.
