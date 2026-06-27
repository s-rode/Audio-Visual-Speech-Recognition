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
from team1_asr.noise import maybe_apply_noise
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
    try:
        with transcript_path.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows, start=1):
                _show_progress(
                    progress,
                    progress_handle,
                    f"Transcribing {index}/{len(rows)}: {row.utterance_id}",
                )
                prepared_audio_path = maybe_apply_noise(
                    row.audio_path,
                    config.noise,
                    output_dir,
                    utterance_id=row.utterance_id,
                )
                transcript = backend.transcribe(
                    TranscriptionRequest(
                        audio_path=prepared_audio_path,
                        language=row.language,
                        utterance_id=row.utterance_id,
                        video_path=row.video_path,
                        reference=row.reference,
                        video_preprocessing=(
                            str(row.metadata["video_preprocessing"])
                            if row.metadata and "video_preprocessing" in row.metadata
                            else None
                        ),
                    )
                )
                handle.write(
                    json.dumps(
                        _serialize_transcript(
                            row.utterance_id,
                            row.audio_path,
                            row.reference,
                            prepared_audio_path,
                            row.video_path,
                            transcript,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                if row.reference is not None:
                    references.append(row.reference)
                    predictions.append(transcript.text)
    finally:
        close = getattr(backend, "close", None)
        if close is not None:
            close()

    metrics = compute_metrics(references, predictions)
    summary = {
        "experiment_name": config.experiment_name,
        "num_utterances": len(rows),
        "backend": config.backend.kind,
        "model": config.backend.model,
        "noise_enabled": config.noise.enabled or config.noise.advanced_enabled,
        "noise_kind": _describe_noise(config.noise),
        "snr_db": _summary_snr(config.noise),
        "noise_seed": config.noise.seed,
        "demand_noise_enabled": config.noise.demand.enabled,
        "demand_snr_db": config.noise.demand.snr_db if config.noise.demand.enabled else None,
        "reverb_enabled": config.noise.reverb.enabled,
        "bandwidth_enabled": config.noise.bandwidth.enabled,
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


def _describe_noise(config) -> str | None:
    if config.advanced_enabled:
        parts = []
        if config.reverb.enabled:
            parts.append("reverb")
        if config.demand.enabled:
            parts.append("demand")
        if config.bandwidth.enabled:
            parts.append("bandwidth")
        return "+".join(parts)
    if config.enabled:
        return config.kind
    return None


def _summary_snr(config) -> float | None:
    if config.demand.enabled:
        return config.demand.snr_db
    if config.enabled:
        return config.snr_db
    return None


def _serialize_transcript(
    utterance_id: str,
    clean_audio_path: str,
    reference: str | None,
    used_audio_path: str,
    video_path: str | None,
    transcript: Transcript,
) -> dict[str, object]:
    return {
        "utterance_id": utterance_id,
        "clean_audio_path": clean_audio_path,
        "used_audio_path": used_audio_path,
        "video_path": video_path,
        "reference": reference,
        "prediction": transcript.text,
        "language": transcript.language,
        "duration": transcript.duration,
        "segments": [asdict(segment) for segment in transcript.segments],
        "metadata": transcript.metadata,
    }
