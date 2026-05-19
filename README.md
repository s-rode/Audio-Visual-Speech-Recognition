# Team 1 ASR Pipeline

This repository contains a minimal, modular ASR experiment scaffold for week 1:

- Whisper-based batch transcription via `faster-whisper`
- a swappable backend interface so other ASR models can be added later
- manifest-driven experiment runs
- optional noise injection before transcription
- WER and CER evaluation against references
- TED-LIUM export tooling for a first real benchmark

## Project layout

```text
configs/
  noop_smoke.toml
  tedlium_dev_whisper.toml
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
  prepare_tedlium.py
  types.py
```

## Install

Create the virtual environment and install the project with `uv`:

```bash
uv sync
```

You also need `ffmpeg` available on the system for Whisper audio decoding.

This project pins `uv`'s cache to a local `.uv-cache/` directory so `uv run` and `uv sync`
do not depend on the machine's global cache configuration.

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

Prepare a TED-LIUM subset from a local archive:

```bash
uv run team1-prepare-tedlium data/tedlium/dev --archive /path/to/dev.tar.gz --split validation --limit 500 --save-dataset --force
```

Run Whisper on that TED-LIUM export:

```bash
uv run team1-asr configs/tedlium_dev_whisper.toml
```

Outputs are written to the configured output directory as:

- `transcripts.jsonl`: one JSON record per utterance
- `metrics.json`: experiment summary and aggregate metrics

## Recommended next steps

1. Replace `data/example_manifest.csv` with a real dataset manifest.
2. Start with TED-LIUM validation as the first real benchmark.
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

## TED-LIUM Workflow

The recommended first real benchmark is TED-LIUM, then a frozen local export for experiments.

The `team1-prepare-tedlium` command:

- accepts either a local TED-LIUM archive or downloads a full release from OpenSLR
- optionally samples a reproducible subset with `--limit` and `--seed`
- writes one local `.wav` file per utterance plus a `manifest.csv`
- can also save metadata about the source archive

For laptop-friendly evaluation, prefer a small split archive such as `dev.tar.gz` or `test.tar.gz`
instead of the full TED-LIUM release archive.
