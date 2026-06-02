from __future__ import annotations

from dataclasses import asdict
import json
import sys
from typing import TextIO

from team1_asr.backends.base import TranscriptionRequest
from team1_asr.backends.factory import create_backend
from team1_asr.config import ExperimentConfig
from team1_asr.evaluation import compute_metrics
from team1_asr.manifest import load_manifest
from team1_asr.noise import apply_noise
from team1_asr.tracking import WandbTracker
from team1_asr.types import Transcript

def run_experiment(
    config: ExperimentConfig,
    progress: bool = False,
    progress_stream: TextIO | None = None,
) -> dict[str, object]:
    progress_handle = progress_stream or sys.stderr
    tracker = WandbTracker(config)
    tracker.start()
    try:
        summary = _run_experiment(config, progress, progress_handle)
        tracker.log_summary(summary, config.output_dir)
        return summary
    finally:
        tracker.finish()


def _run_experiment(
    config: ExperimentConfig,
    progress: bool,
    progress_handle: TextIO,
) -> dict[str, object]:
    _show_progress(progress, progress_handle, f"Loading manifest: {config.input.manifest_path}")
    rows = load_manifest(config.input)
    _show_progress(progress, progress_handle, f"Loaded {len(rows)} utterance(s)")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear noisy_audio folder to avoid cluttering
    try:
        import shutil
        noisy_audio_dir = output_dir / "noisy_audio"
        if noisy_audio_dir.exists():
            _show_progress(progress, progress_handle, "Clearing old noisy audio artifacts...")
            shutil.rmtree(noisy_audio_dir)
    except:
        print("Failed to clear noisy_audio folder.")
        
    _show_progress(
        progress,
        progress_handle,
        f"Loading backend '{config.backend.kind}' with model '{config.backend.model}'",
    )
    backend = create_backend(config.backend)
    _show_progress(progress, progress_handle, "Backend ready")
    transcript_path = output_dir / config.output.transcript_name

    references: list[str] = []
    predictions: list[str] = []
    with transcript_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            _show_progress(
                progress,
                progress_handle,
                f"Transcribing {index}/{len(rows)}: {row.utterance_id}",
            )
            prepared_audio_path = apply_noise(row.audio_path, config.noise, output_dir)
            transcript = backend.transcribe(
                TranscriptionRequest(
                    audio_path=prepared_audio_path,
                    language=row.language,
                )
            )
            handle.write(
                json.dumps(
                    _serialize_transcript(
                        row.utterance_id,
                        row.audio_path,
                        row.reference,
                        prepared_audio_path,
                        transcript,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )

            if row.reference is not None:
                references.append(row.reference)
                predictions.append(transcript.text)

    metrics = compute_metrics(references, predictions)
    summary = {
        "experiment_name": config.experiment_name,
        "num_utterances": len(rows),
        "backend": config.backend.kind,
        "model": config.backend.model,
        "demand_noise_enabled": config.noise.demand.enabled,
        "demand_snr_db": config.noise.demand.snr_db if config.noise.demand.enabled else None,
        "reverberation_enabled": config.noise.reverb.enabled,
        "utterances_scored": metrics.utterances_scored,
        "wer": metrics.wer,
        "cer": metrics.cer,
        "raw_wer": metrics.raw_wer,
        "raw_cer": metrics.raw_cer,
        "transcript_path": str(transcript_path),
    }
    metrics_path = output_dir / config.output.metrics_name
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _show_progress(progress, progress_handle, f"Wrote transcripts: {transcript_path}")
    _show_progress(progress, progress_handle, f"Wrote metrics: {metrics_path}")
    return summary


def _show_progress(enabled: bool, stream: TextIO, message: str) -> None:
    if enabled:
        print(message, file=stream, flush=True)


def _serialize_transcript(
    utterance_id: str,
    clean_audio_path: str,
    reference: str | None,
    used_audio_path: str,
    transcript: Transcript,
) -> dict[str, object]:
    return {
        "utterance_id": utterance_id,
        "clean_audio_path": clean_audio_path,
        "used_audio_path": used_audio_path,
        "reference": reference,
        "prediction": transcript.text,
        "language": transcript.language,
        "duration": transcript.duration,
        "segments": [asdict(segment) for segment in transcript.segments],
        "metadata": transcript.metadata,
    }
