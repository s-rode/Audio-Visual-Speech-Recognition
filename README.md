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

For audio-only ASR experiments on LRS3, the project can stream the third-party
Hugging Face mirror `Ainncy/LRS3` without storing the full dataset locally. The
command downloads one MP4 clip at a time, extracts mono 16 kHz WAV audio with
`ffmpeg`, writes `manifest.csv`, and deletes the temporary MP4.

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

The Hugging Face mirror currently exposes one `train` split. Runs from this
mirror are exploratory ASR/noise runs, not an official LRS3 test benchmark.

For AVSR experiments, keep the paired MP4 clips and write `video_path` into the
manifest:

```bash
uv run team1-prepare-lrs3-hf data/lrs3/hf_av_tiny --limit 10 --keep-video --force
```

To run visual or audio-visual AVEC closer to the authors' preprocessing, convert
the raw video manifest to AVEC-style mouth ROI clips using the editable AVEC
environment:

```bash
uv run team1-preprocess-avec-video \
  data/lrs3/hf_av_tiny/manifest.csv \
  data/lrs3/hf_av_tiny_avec_roi \
  --force
```

This command delegates to `external/AVEC/.venv/bin/python3` and reuses AVEC's
`LipDetectCrop` with `media/20words_mean_face.npy`. It requires the AVEC
environment to include `ibug.face_detection` and `ibug.face_alignment`.
AVEC's README installs those from `https://github.com/hhj1897/face_detection`
and `https://github.com/hhj1897/face_alignment`; the detector repository uses
Git LFS for model weights, so preprocessing may be blocked if the upstream LFS
quota is unavailable.

## AVEC Workflow

AVEC is used as an editable external source checkout so that later experiments
can fine-tune checkpoints or modify the model code without rewriting the local
ASR/noise/metrics pipeline.

Initial setup:

```bash
git clone https://github.com/burchim/AVEC external/AVEC
cd external/AVEC
pip install -r requirements.txt
python download_checkpoints_and_tokenizers.py
```

The official AVEC configs expect pretrained checkpoints under their own
`callbacks/...` folders and tokenizers under `datasets/LRS3`. Verify the
official checkout first:

```bash
cd external/AVEC
python main.py -c configs/LRS23/AO/EffConfCTC.py --mode evaluation --checkpoint checkpoints_swa-equal-200-210.ckpt --eval_steps 1
python main.py -c configs/LRS23/VO/EffConfInterCTC.py --mode evaluation --checkpoint checkpoints_swa-equal-90-100.ckpt --eval_steps 1
python main.py -c configs/LRS23/AV/EffConfInterCTC.py --mode evaluation --checkpoint checkpoints_swa-equal-60-70.ckpt --eval_steps 1
```

Once the official checkout works, run it through this repository's experiment
pipeline:

```bash
uv run team1-asr configs/lrs3_avec_audio_clean.toml
uv run team1-asr configs/lrs3_avec_visual_clean.toml
uv run team1-asr configs/lrs3_avec_av_clean.toml
```

After mouth ROI preprocessing, use:

```bash
uv run team1-asr configs/lrs3_avec_audio_preprocessed_clean.toml
uv run team1-asr configs/lrs3_avec_visual_preprocessed_clean.toml
uv run team1-asr configs/lrs3_avec_av_preprocessed_clean.toml
```

The `avec` backend starts `scripts/run_avec_manifest.py` as a subprocess.
That runner loads the editable AVEC checkout, decodes one manifest row at a time,
and returns predictions in the same `transcripts.jsonl` / `metrics.json` format
as Whisper and Parakeet. Visual and audio-visual runs require the manifest's
`video_path` to point at suitable mouth-region video input. The Hugging Face
mirror is the current data source; treat full-score reproduction as an
equivalent-source attempt until split identity and preprocessing parity are
verified.
