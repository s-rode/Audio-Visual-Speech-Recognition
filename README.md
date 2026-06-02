# Team 1 ASR Pipeline

This repository contains a minimal, modular ASR experiment scaffold:

- Whisper-based batch transcription via `faster-whisper`
- a swappable backend interface so other ASR models can be added later
- manifest-driven experiment runs
- optional noise injection before transcription
- WER and CER evaluation against references
- Hugging Face LibriSpeech export tooling for a first real benchmark

## Project layout

```text
configs/
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

You also need `ffmpeg` available on the system for Whisper audio decoding.

This project pins `uv`'s cache to a local `.uv-cache/` directory so `uv run` and `uv sync`
do not depend on the machine's global cache configuration.

Additionally, you need to install the DEMAND dataset for background noise inside of the data/ folder, according to the following structure:

```text
data/.
|____DEMAND
| |____48k
| | |____NRIVER
| | |____PRESTO
| | |____NPARK
| | |____SCAFE
| | |____OMEETING
| | |____NFIELD
| | |____OOFFICE
| | |____PCAFETER
| | |____DWASHING
| | |____TMETRO
| | |____TCAR
| | |____DLIVING
| | |____PSTATION
| | |____STRAFFIC
| | |____SPSQUARE
| | |____DKITCHEN
| | |____TBUS
| | |____OHALLWAY
| |____16k
| | |____NRIVER
| | |____PRESTO
| | |____NPARK
| | |____OMEETING
| | |____NFIELD
| | |____OOFFICE
| | |____PCAFETER
| | |____DWASHING
| | |____TMETRO
| | |____TCAR
| | |____DLIVING
| | |____PSTATION
| | |____STRAFFIC
| | |____SPSQUARE
| | |____DKITCHEN
| | |____TBUS
| | |____OHALLWAY
```
## Manifest format

The pipeline expects a CSV manifest. The default columns are:

```csv
utterance_id,audio_path,reference,language
utt-00001,data/audio/file1.wav,ground truth transcript,en
utt-00002,data/audio/file2.wav,another reference,en
```

- `utterance_id`: optional, auto-generated if missing
- `audio_path`: required
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

When enabled, `team1-asr` logs numeric metrics and a transcript table. Set
`log_artifacts = true` when you also want the run output directory uploaded as a
W&B artifact. Use `mode = "offline"` for laptop runs that should not upload
immediately; remove it or set `mode = "online"` when you want to sync to W&B.

## Recommended next steps

1. Replace `data/example_manifest.csv` with a real dataset manifest.
2. Start with LibriSpeech `--config clean --split validation` as the first real benchmark.
3. Verify clean-audio WER first, then run noise sweeps.
4. Add more backends or models behind the same interface.
5. Add the visual correction stage later on top of the audio baseline.

## Noise configuration

Two noise modes are scaffolded:

- `white`: synthetic white noise
- `external`: mix a provided noise clip from `noise.noise_path`

Example:

```toml
[noise]
enabled = true
kind = "white"
snr_db = 10.0
seed = 0
```

## Design choice

The scaffold is intentionally centered on a thin local pipeline rather than a third-party CLI wrapper. That keeps the experiment loop explicit and makes it easier to swap:

- backend and model
- noise type
- evaluation metrics
- logging and output format

## LibriSpeech Workflow

The recommended first real benchmark is Hugging Face `openslr/librispeech_asr`, then a frozen local export for experiments.

The `team1-prepare-librispeech` command:

- loads a LibriSpeech split directly from Hugging Face
- exports the first `--limit` utterances in streaming mode for quick local tests
- writes one local `.wav` file per utterance plus a `manifest.csv`
- can also save a Hugging Face subset with `--save-dataset`, which may require downloading the full split before applying `--limit`

For laptop-friendly evaluation, start with `--config clean --split validation` or
`--config clean --split test`, then move to `train.100` only if needed.

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

The Hugging Face mirror currently exposes one `train` split. Treat these runs as
exploratory ASR/noise experiments rather than an official LRS3 test benchmark
unless the official test split provenance is confirmed.
