from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

from team1_asr.backends.base import ASRBackend, TranscriptionRequest
from team1_asr.router_features import extract_live_features
from team1_asr.routing_policy import evaluate_policy, validate_live_policy
from team1_asr.types import Transcript
from team1_asr.visual_reliability import is_real_mouth_roi


EventSink = Callable[[dict[str, object]], None]
FeatureExtractor = Callable[[Path, Path | None, Transcript], dict[str, object]]


@dataclass(frozen=True)
class RouteResult:
    selected_system: str
    transcript: Transcript
    whisper_transcript: Transcript
    avec_transcript: Transcript | None
    policy_score: float | None
    policy_threshold: float | None
    feature_row: dict[str, object]
    elapsed_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_system": self.selected_system,
            "transcript": _serialize_transcript(self.transcript),
            "whisper_transcript": _serialize_transcript(self.whisper_transcript),
            "avec_transcript": (
                _serialize_transcript(self.avec_transcript)
                if self.avec_transcript is not None
                else None
            ),
            "policy_score": self.policy_score,
            "policy_threshold": self.policy_threshold,
            "features": self.feature_row,
            "elapsed_seconds": self.elapsed_seconds,
        }


class LiveRouter:
    """Run Whisper first and invoke AVEC only when a frozen policy selects it."""

    def __init__(
        self,
        policy: dict[str, object],
        whisper_backend: ASRBackend,
        avec_backend: ASRBackend,
        *,
        feature_extractor: FeatureExtractor = extract_live_features,
    ) -> None:
        validate_live_policy(policy)
        self._policy = dict(policy)
        self._whisper = whisper_backend
        self._avec = avec_backend
        self._feature_extractor = feature_extractor

    @property
    def policy(self) -> dict[str, object]:
        return dict(self._policy)

    def route(
        self,
        request: TranscriptionRequest,
        reliability: dict[str, object],
        *,
        feature_overrides: dict[str, object] | None = None,
        emit: EventSink | None = None,
    ) -> RouteResult:
        started = perf_counter()
        sink = emit or (lambda event: None)

        sink(_event("whisper", "running", "Whisper is transcribing the audio."))
        whisper_started = perf_counter()
        whisper = self._whisper.transcribe(request)
        sink(
            _event(
                "whisper",
                "complete",
                "Whisper transcript is ready.",
                elapsed_seconds=perf_counter() - whisper_started,
                transcript=whisper.text,
            )
        )

        sink(_event("features", "running", "Extracting routing features."))
        features = dict(reliability)
        features.update(
            self._feature_extractor(
                Path(request.audio_path),
                Path(request.video_path)
                if request.video_path and is_real_mouth_roi(reliability)
                else None,
                whisper,
            )
        )
        if feature_overrides:
            features.update(feature_overrides)
        sink(
            _event(
                "features",
                "complete",
                "Audio and visual reliability signals are ready.",
                features=_display_features(features),
            )
        )

        decision = evaluate_policy(self._policy, features)
        score, threshold = decision.score, decision.threshold
        selects_av = decision.selects_avec
        audio_system = str(self._policy.get("audio_system") or "whisper")
        av_system = str(self._policy.get("av_system") or "avec")
        selected_system = av_system if selects_av else audio_system
        sink(
            _event(
                "router",
                "complete",
                _decision_message(self._policy, features, selects_av),
                selected_system=selected_system,
                selects_avec=selects_av,
                score=score,
                threshold=threshold,
            )
        )

        avec: Transcript | None = None
        selected = whisper
        if selects_av:
            sink(_event("avec", "running", "AVEC is fusing audio and mouth motion."))
            avec_started = perf_counter()
            transcribe_with_events = getattr(
                self._avec,
                "transcribe_with_events",
                None,
            )
            if callable(transcribe_with_events):
                avec = transcribe_with_events(request, sink)
            else:
                avec = self._avec.transcribe(request)
            selected = avec
            decoder = str(avec.metadata.get("decoder") or "greedy")
            message = (
                "AVEC fusion, beam search, and RoBERTa reranking are complete."
                if decoder == "flashlight_kenlm_roberta"
                else "AVEC transcript is ready."
            )
            sink(
                _event(
                    "avec",
                    "complete",
                    message,
                    elapsed_seconds=perf_counter() - avec_started,
                    transcript=avec.text,
                    decoder=decoder,
                )
            )
        else:
            sink(
                _event(
                    "avec",
                    "skipped",
                    "AVEC was not needed for this request.",
                )
            )

        result = RouteResult(
            selected_system=selected_system,
            transcript=selected,
            whisper_transcript=whisper,
            avec_transcript=avec,
            policy_score=score,
            policy_threshold=threshold,
            feature_row=features,
            elapsed_seconds=perf_counter() - started,
        )
        sink(_event("result", "complete", "Routing complete.", result=result.as_dict()))
        return result

    def preload(self) -> None:
        for backend in (self._whisper, self._avec):
            preload = getattr(backend, "preload", None)
            if callable(preload):
                preload()

    def close(self) -> None:
        for backend in (self._whisper, self._avec):
            close = getattr(backend, "close", None)
            if close is not None:
                close()


def load_policy(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        policy = json.load(handle)
    if not isinstance(policy, dict):
        raise ValueError(f"Expected a JSON policy object in {path}")
    validate_live_policy(policy)
    return policy


def _decision_message(
    policy: dict[str, object],
    features: dict[str, object],
    selects_av: bool,
) -> str:
    if not is_real_mouth_roi(features):
        return "Mouth ROI is unavailable or unreliable; keep the Whisper result."
    policy_type = str(policy.get("policy_type") or "")
    if policy_type == "audio_only":
        return "The frozen policy is audio-only; keep the Whisper result."
    if policy_type == "roi_success":
        return "A reliable mouth ROI is available; route to AVEC."
    target = "AVEC" if selects_av else "Whisper"
    return f"The frozen policy selected {target} from the live reliability signals."


def _display_features(features: dict[str, object]) -> dict[str, object]:
    names = (
        "audio_avg_logprob_mean",
        "audio_no_speech_prob_max",
        "audio_spectral_flatness",
        "mouth_landmark_confidence_mean",
        "face_detection_rate",
        "video_mouth_motion_mean",
        "audio_video_sync_correlation",
        "roi_status",
    )
    return {name: features.get(name) for name in names if name in features}


def _event(stage: str, status: str, message: str, **data: object) -> dict[str, object]:
    return {"stage": stage, "status": status, "message": message, **data}


def _serialize_transcript(transcript: Transcript) -> dict[str, object]:
    return {
        "text": transcript.text,
        "language": transcript.language,
        "duration": transcript.duration,
        "segments": [asdict(segment) for segment in transcript.segments],
        "metadata": transcript.metadata,
    }
