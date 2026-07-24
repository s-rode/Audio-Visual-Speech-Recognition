from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import mimetypes
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import urlparse
from uuid import uuid4

import soundfile as sf

from team1_asr.backends.base import ASRBackend, TranscriptionRequest
from team1_asr.backends.factory import create_backend
from team1_asr.config import BackendConfig, load_config
from team1_asr.live_capture import BrowserCapturePreparer, capture_suffix
from team1_asr.live_router import LiveRouter, load_policy
from team1_asr.router_features import extract_live_features
from team1_asr.types import Segment, Transcript


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".tmp/hf"))
mimetypes.knownfiles.clear()
mimetypes.init()

_ASSET_DIR = Path(__file__).with_name("live_demo_assets")
_DEFAULT_REAL_MANIFEST = "data/voxmm_dev_eval/manifest.local.csv"
_DEFAULT_REAL_POLICY = "configs/live_router/voxmm_policy_dev.json"
_DEFAULT_REAL_WHISPER_CONFIG = "configs/lrs_voxmm_whisper_base_600_enriched.toml"
_DEFAULT_REAL_AVEC_CONFIG = "configs/live_router/avec_beam_roberta.toml"
_MAX_CAPTURE_BYTES = 32 * 1024 * 1024
_OVERRIDE_RANGES = {
    "audio_avg_logprob_mean": (-2.0, 0.0),
    "mouth_landmark_confidence_mean": (0.0, 1.0),
}


@dataclass(frozen=True)
class DemoSample:
    utterance_id: str
    label: str
    condition: str
    audio_path: str
    video_path: str | None
    language: str | None
    reference: str | None
    reliability: dict[str, object]

    def public(self, has_media: bool) -> dict[str, object]:
        return {
            "id": self.utterance_id,
            "label": self.label,
            "condition": self.condition,
            "reference": self.reference,
            "has_audio": has_media and bool(self.audio_path),
            "has_video": has_media and bool(self.video_path),
            "features": {
                "audio_avg_logprob_mean": self.reliability.get(
                    "audio_avg_logprob_mean"
                ),
                "mouth_landmark_confidence_mean": self.reliability.get(
                    "mouth_landmark_confidence_mean"
                ),
                "roi_ready": (
                    str(self.reliability.get("roi_status") or "").lower() == "success"
                    and not _parse_bool(self.reliability.get("fallback_used"))
                ),
            },
        }


class DemoApplication:
    def __init__(
        self,
        router: LiveRouter,
        samples: list[DemoSample],
        capture_preparer: BrowserCapturePreparer,
        *,
        mode: str,
        live_transcription: str = "mock",
    ) -> None:
        if not samples:
            raise ValueError("The live demo requires at least one sample.")
        self.router = router
        self.samples = {sample.utterance_id: sample for sample in samples}
        self.capture_preparer = capture_preparer
        self.mode = mode
        self.live_transcription = live_transcription
        self.lock = threading.Lock()
        capture_root = _PROJECT_ROOT / ".tmp" / "live_demo"
        capture_root.mkdir(parents=True, exist_ok=True)
        self.capture_directory = TemporaryDirectory(
            prefix="team1-live-media-",
            dir=capture_root,
        )
        self.capture_media: dict[str, dict[str, Path]] = {}
        self.capture_order: list[str] = []

    def state(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "capture_enabled": True,
            "live_transcription": self.live_transcription,
            "policy": {
                "type": self.router.policy.get("policy_type"),
                "threshold": self.router.policy.get("threshold"),
                "audio_system": self.router.policy.get("audio_system"),
                "av_system": self.router.policy.get("av_system"),
            },
            "samples": [
                sample.public(has_media=self.mode == "real")
                for sample in self.samples.values()
            ],
        }

    def run_route(
        self,
        sample_id: str,
        overrides: dict[str, object],
        emit: Callable[[dict[str, object]], None],
    ) -> None:
        sample = self.samples.get(sample_id)
        if sample is None:
            raise ValueError(f"Unknown sample: {sample_id}")
        clean_overrides = _validate_overrides(overrides) if self.mode == "mock" else {}
        reliability = dict(sample.reliability)
        roi_ready = clean_overrides.pop("roi_ready", None)
        if roi_ready is not None:
            reliability["roi_status"] = "success" if roi_ready else "fallback"
            reliability["fallback_used"] = not roi_ready

        request = TranscriptionRequest(
            audio_path=sample.audio_path,
            video_path=sample.video_path,
            language=sample.language,
            utterance_id=sample.utterance_id,
            reference=sample.reference,
            video_preprocessing="preprocessed",
        )
        with self.lock:
            self.router.route(
                request,
                reliability,
                feature_overrides=clean_overrides,
                emit=emit,
            )

    def run_capture(
        self,
        payload: bytes,
        content_type: str,
        emit: Callable[[dict[str, object]], None],
    ) -> None:
        if not payload:
            raise ValueError("Camera capture is empty.")
        utterance_id = f"live-{uuid4().hex[:12]}"
        work_dir = Path(self.capture_directory.name) / utterance_id
        work_dir.mkdir()
        with self.lock:
            try:
                artifacts = self.capture_preparer.prepare(
                    payload,
                    content_type,
                    work_dir,
                    utterance_id,
                    emit,
                )
            except Exception:
                shutil.rmtree(work_dir, ignore_errors=True)
                raise
            self.capture_media[utterance_id] = {
                "source": artifacts.source_path,
                "audio": artifacts.audio_path,
                "processed": artifacts.video_path,
            }
            self.capture_order.append(utterance_id)
            self._prune_captures()
            preview = _audio_preview(artifacts.audio_path)
            processed_distinct = (
                artifacts.video_path.resolve() != artifacts.source_path.resolve()
            )
            emit(
                {
                    "stage": "media",
                    "status": "complete",
                    "message": "Capture artifacts are ready to inspect.",
                    "capture_id": utterance_id,
                    "media": {
                        "source_url": f"/api/capture/{utterance_id}/source",
                        "audio_url": f"/api/capture/{utterance_id}/audio",
                        "processed_url": f"/api/capture/{utterance_id}/processed",
                        "processed_label": (
                            "Mouth ROI" if processed_distinct else "Router video"
                        ),
                        "processed_distinct": processed_distinct,
                        "visual_metrics_mode": (
                            "measured" if processed_distinct else "presentation"
                        ),
                        "bytes": len(payload),
                        **preview,
                    },
                }
            )
            request = TranscriptionRequest(
                audio_path=str(artifacts.audio_path),
                video_path=str(artifacts.video_path),
                language="en",
                utterance_id=utterance_id,
                video_preprocessing="preprocessed",
            )
            self.router.route(
                request,
                artifacts.reliability,
                emit=emit,
            )

    def capture_path(self, capture_id: str, kind: str) -> Path | None:
        path = self.capture_media.get(capture_id, {}).get(kind)
        return path if path is not None and path.is_file() else None

    def _prune_captures(self) -> None:
        while len(self.capture_order) > 6:
            capture_id = self.capture_order.pop(0)
            paths = self.capture_media.pop(capture_id, {})
            source = paths.get("source")
            if source is not None:
                shutil.rmtree(source.parent, ignore_errors=True)

    def preload(self) -> None:
        self.router.preload()

    def close(self) -> None:
        self.router.close()
        self.capture_directory.cleanup()

    def media_path(self, sample_id: str, kind: str) -> Path | None:
        sample = self.samples.get(sample_id)
        if sample is None or self.mode != "real":
            return None
        value = sample.audio_path if kind == "audio" else sample.video_path
        path = Path(value).resolve() if value else None
        return path if path is not None and path.is_file() else None


class _LazyBackend:
    def __init__(self, factory: Callable[[], ASRBackend]) -> None:
        self._factory = factory
        self._backend: ASRBackend | None = None
        self._lock = threading.Lock()

    def preload(self) -> ASRBackend:
        with self._lock:
            if self._backend is None:
                self._backend = self._factory()
            return self._backend

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        return self.preload().transcribe(request)

    def transcribe_with_events(
        self,
        request: TranscriptionRequest,
        emit: Callable[[dict[str, object]], None],
    ) -> Transcript:
        backend = self.preload()
        method = getattr(backend, "transcribe_with_events", None)
        if callable(method):
            return method(request, emit)
        return backend.transcribe(request)

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if close is not None:
            close()


class _LiveInputBackend:
    def __init__(self, sample_backend: ASRBackend, live_backend: ASRBackend) -> None:
        self.sample_backend = sample_backend
        self.live_backend = live_backend

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        live = bool(request.utterance_id and request.utterance_id.startswith("live-"))
        backend = self.live_backend if live else self.sample_backend
        return backend.transcribe(request)

    def close(self) -> None:
        for backend in (self.sample_backend, self.live_backend):
            close = getattr(backend, "close", None)
            if close is not None:
                close()


class _MockBackend:
    def __init__(self, kind: str, delay: float = 0.18) -> None:
        self.kind = kind
        self.delay = delay

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        time.sleep(self.delay)
        live = bool(request.utterance_id and request.utterance_id.startswith("live-"))
        if live:
            samples, sample_rate = sf.read(
                request.audio_path,
                dtype="float64",
                always_2d=True,
            )
            mono = samples.mean(axis=1)
            rms = math.sqrt(float((mono**2).mean())) if mono.size else 0.0
            dbfs = 20.0 * math.log10(max(rms, 1e-12))
            confidence = max(0.0, min(1.0, (dbfs + 55.0) / 35.0))
            live_logprob = -1.6 + 1.35 * confidence
            noisy = live_logprob <= -0.8
            duration = len(mono) / sample_rate if sample_rate > 0 else 0.0
        else:
            noisy = request.utterance_id != "clear-room"
            live_logprob = -1.24 if noisy else -0.34
            duration = 3.2

        if self.kind == "whisper":
            if live:
                text = (
                    "Live microphone signal is clear."
                    if not noisy
                    else "Live microphone signal needs visual context."
                )
            else:
                text = (
                    "Route the clean signal through the audio path."
                    if not noisy
                    else "Move the meeting to the main room."
                )
            avg_logprob = live_logprob
        else:
            text = (
                "Live audio visual signal received."
                if live
                else (
                    "Route the clean signal through the audio path."
                    if not noisy
                    else "Move the meeting to the quiet room."
                )
            )
            avg_logprob = None
        return Transcript(
            text=text,
            language="en",
            duration=duration,
            segments=[
                Segment(
                    start=0.0,
                    end=duration,
                    text=text,
                    avg_logprob=avg_logprob,
                    compression_ratio=1.08,
                    no_speech_prob=0.7 if live and noisy else 0.02,
                    temperature=0.0,
                )
            ],
            metadata={"backend": self.kind, "model": "presentation-runtime"},
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a live Whisper-to-AVEC reliability-routing demo."
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Run the complete Whisper/router/AVEC/beam/RoBERTa pipeline.",
    )
    parser.add_argument("--manifest")
    parser.add_argument(
        "--reliability",
        help=(
            "Optional visual-reliability CSV. Mouth-ROI manifests with "
            "*_mouth.mp4 paths are recognized automatically."
        ),
    )
    parser.add_argument("--policy")
    parser.add_argument("--whisper-config")
    parser.add_argument("--avec-config")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--live-whisper-model", default="base.en")
    parser.add_argument("--live-whisper-device", default="cpu")
    parser.add_argument("--live-whisper-compute-type", default="int8")
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load all selected models before starting the HTTP server.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate real-mode assets and imports, then exit.",
    )
    parser.add_argument(
        "--force-avec",
        action="store_true",
        help="Route every valid mouth ROI through AVEC (showcase/debug mode).",
    )
    parser.add_argument(
        "--mock-live-transcripts",
        action="store_true",
        help="Keep hard-coded presentation transcripts for camera captures.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _apply_real_defaults(args)
    real_args = (
        args.manifest,
        args.policy,
        args.whisper_config,
        args.avec_config,
    )
    if any(real_args) and not all(real_args):
        raise SystemExit(
            "Real mode requires --manifest, --policy, --whisper-config, "
            "and --avec-config. Use --real for the repository defaults."
        )
    real_mode = all(real_args)
    if args.reliability and not real_mode:
        raise SystemExit("--reliability is only valid in real mode.")
    if args.force_avec and not real_mode:
        raise SystemExit("--force-avec is only valid in real mode.")
    if args.check_only:
        if not real_mode:
            raise SystemExit("--check-only requires real mode.")
        _preflight_real(args)
        print("Real AVEC live-demo preflight passed.", flush=True)
        return

    application = (
        _build_real_application(args)
        if real_mode
        else build_mock_application(
            live_whisper_backend=(
                None if args.mock_live_transcripts else _build_live_whisper(args)
            )
        )
    )
    try:
        if args.preload:
            print("Preloading live-demo backends...", flush=True)
            application.preload()
            print("Backends ready.", flush=True)
        server = create_server(application, args.host, args.port)
        address, port = server.server_address[:2]
        print(f"Live router demo: http://{address}:{port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        application.close()


def create_server(
    application: DemoApplication,
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    class Handler(_DemoHandler):
        app = application

    return ThreadingHTTPServer((host, port), Handler)


def _build_live_whisper(args: argparse.Namespace) -> ASRBackend:
    config = BackendConfig(
        kind="faster-whisper",
        model=args.live_whisper_model,
        device=args.live_whisper_device,
        compute_type=args.live_whisper_compute_type,
        beam_size=1,
        best_of=1,
        language="en",
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return _LazyBackend(lambda: create_backend(config))


def build_mock_application(
    *,
    delay: float = 0.18,
    live_whisper_backend: ASRBackend | None = None,
) -> DemoApplication:
    policy = {
        "version": 1,
        "policy_type": "learned_logistic",
        "feature_family": "combined",
        "feature": None,
        "direction": "higher",
        "threshold": 0.5,
        "feature_names": [
            "audio_avg_logprob_mean",
            "mouth_landmark_confidence_mean",
        ],
        "feature_medians": {
            "audio_avg_logprob_mean": -0.8,
            "mouth_landmark_confidence_mean": 0.8,
        },
        "feature_means": {
            "audio_avg_logprob_mean": -0.8,
            "mouth_landmark_confidence_mean": 0.8,
        },
        "feature_scales": {
            "audio_avg_logprob_mean": 0.4,
            "mouth_landmark_confidence_mean": 0.1,
        },
        "coefficients": {
            "audio_avg_logprob_mean": -1.7,
            "mouth_landmark_confidence_mean": 0.6,
        },
        "intercept": 0.0,
        "audio_system": "Whisper",
        "av_system": "AVEC",
    }
    samples = [
        DemoSample(
            utterance_id="clear-room",
            label="Clear room",
            condition="High audio confidence",
            audio_path="mock-clear.wav",
            video_path="mock-clear.mp4",
            language="en",
            reference="Route the clean signal through the audio path.",
            reliability={
                "roi_status": "success",
                "fallback_used": False,
                "face_detection_rate": 0.99,
                "mouth_landmark_confidence_mean": 0.92,
                "audio_avg_logprob_mean": -0.34,
            },
        ),
        DemoSample(
            utterance_id="street-noise",
            label="Street noise",
            condition="Low audio confidence, clear mouth ROI",
            audio_path="mock-noisy.wav",
            video_path="mock-noisy.mp4",
            language="en",
            reference="Move the meeting to the quiet room.",
            reliability={
                "roi_status": "success",
                "fallback_used": False,
                "face_detection_rate": 0.97,
                "mouth_landmark_confidence_mean": 0.88,
                "audio_avg_logprob_mean": -1.24,
            },
        ),
    ]

    def mock_features(
        audio_path: Path,
        video_path: Path | None,
        transcript: Transcript,
    ) -> dict[str, object]:
        if audio_path.exists() and video_path is not None and video_path.exists():
            return extract_live_features(audio_path, video_path, transcript)
        segment = transcript.segments[0]
        return {
            "audio_avg_logprob_mean": segment.avg_logprob,
            "audio_no_speech_prob_max": segment.no_speech_prob,
            "audio_spectral_flatness": 0.08
            if "clear" in audio_path.name
            else 0.61,
            "video_mouth_motion_mean": 0.12,
            "audio_video_sync_correlation": 0.76,
        }

    sample_whisper = _MockBackend("whisper", delay)
    whisper_backend = (
        _LiveInputBackend(sample_whisper, live_whisper_backend)
        if live_whisper_backend is not None
        else sample_whisper
    )
    router = LiveRouter(
        policy,
        whisper_backend,
        _MockBackend("avec", delay),
        feature_extractor=mock_features,
    )
    return DemoApplication(
        router,
        samples,
        BrowserCapturePreparer(real_preprocessing=False),
        mode="mock",
        live_transcription=(
            "faster-whisper" if live_whisper_backend is not None else "mock"
        ),
    )


def _build_real_application(args: argparse.Namespace) -> DemoApplication:
    _preflight_real(args)
    manifest_path = _project_path(args.manifest)
    reliability = (
        _read_csv_by_id(_project_path(args.reliability))
        if args.reliability
        else {}
    )
    samples = []
    for row in _read_csv(manifest_path)[: max(1, args.limit)]:
        utterance_id = str(row.get("utterance_id") or "")
        if not utterance_id:
            continue
        sample_reliability = reliability.get(utterance_id)
        if sample_reliability is None:
            sample_reliability = _manifest_mouth_roi_reliability(row)
        samples.append(
            DemoSample(
                utterance_id=utterance_id,
                label=utterance_id,
                condition=str(row.get("condition") or "VoxMM development sample"),
                audio_path=str(_resolve_path(manifest_path, row.get("audio_path"))),
                video_path=(
                    str(_resolve_path(manifest_path, row.get("video_path")))
                    if row.get("video_path")
                    else None
                ),
                language=str(row.get("language") or "en"),
                reference=str(row.get("reference") or "") or None,
                reliability=dict(sample_reliability),
            )
        )
    whisper_config = load_config(_project_path(args.whisper_config)).backend
    avec_config = load_config(_project_path(args.avec_config)).backend
    policy = load_policy(_project_path(args.policy))
    if args.force_avec:
        policy = _force_avec_policy(policy)
    router = LiveRouter(
        policy,
        _LazyBackend(lambda: create_backend(whisper_config)),
        _LazyBackend(lambda: create_backend(avec_config)),
    )
    capture_preparer = BrowserCapturePreparer(
        real_preprocessing=True,
        repo_path=avec_config.repo_path,
        python_executable=avec_config.python_executable,
    )
    return DemoApplication(
        router,
        samples,
        capture_preparer,
        mode="real",
        live_transcription="faster-whisper",
    )


def _apply_real_defaults(args: argparse.Namespace) -> None:
    if not (args.real or args.check_only):
        return
    defaults = {
        "manifest": _DEFAULT_REAL_MANIFEST,
        "policy": _DEFAULT_REAL_POLICY,
        "whisper_config": _DEFAULT_REAL_WHISPER_CONFIG,
        "avec_config": _DEFAULT_REAL_AVEC_CONFIG,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def _preflight_real(args: argparse.Namespace) -> None:
    manifest_path = _require_file(args.manifest, "VoxMM manifest")
    policy_path = _require_file(args.policy, "router policy")
    whisper_path = _require_file(args.whisper_config, "Whisper config")
    avec_path = _require_file(args.avec_config, "AVEC config")
    if args.reliability:
        _require_file(args.reliability, "visual reliability CSV")

    rows = _read_csv(manifest_path)
    if not rows:
        raise ValueError(f"Real-mode manifest is empty: {manifest_path}")
    for row in rows[: max(1, args.limit)]:
        utterance_id = str(row.get("utterance_id") or "<missing id>")
        for column in ("audio_path", "video_path"):
            media_path = _resolve_path(manifest_path, row.get(column))
            if not media_path.is_file():
                raise FileNotFoundError(
                    f"{column} for {utterance_id} does not exist: {media_path}"
                )

    load_policy(policy_path)
    whisper_config = load_config(whisper_path).backend
    avec_config = load_config(avec_path).backend
    if whisper_config.kind != "faster-whisper":
        raise ValueError("--whisper-config must use the faster-whisper backend.")
    if avec_config.kind != "avec":
        raise ValueError("--avec-config must use the avec backend.")

    repo_path = _project_path(avec_config.repo_path)
    if not repo_path.is_dir():
        raise FileNotFoundError(f"AVEC repository does not exist: {repo_path}")
    python_path = _require_file(avec_config.python_executable, "AVEC Python")
    config_path = _path_under(repo_path, avec_config.config_file)
    if not config_path.is_file():
        raise FileNotFoundError(f"AVEC model config does not exist: {config_path}")
    callback_value = avec_config.callback_path
    if callback_value is None:
        callback_value = str(Path(avec_config.config_file).with_suffix("")).replace(
            "configs",
            "callbacks",
            1,
        )
    callback_path = _path_under(repo_path, callback_value)
    checkpoint_path = _path_under(callback_path, avec_config.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"AVEC checkpoint does not exist: {checkpoint_path}")

    for option in ("--lm-manifest", "--kenlm-path", "--lexicon-unigrams"):
        value = _extra_arg_value(avec_config.extra_args, option)
        if value is not None:
            _require_file(value, option)

    imports = [
        "import av, sentencepiece, torch, torchaudio, torchvision",
        "from torchaudio.models.decoder import ctc_decoder",
        "from ibug.face_detection import RetinaFacePredictor",
        "from ibug.face_alignment import FANPredictor",
    ]
    if "flashlight_kenlm_roberta" in avec_config.extra_args:
        imports.append("import transformers")
    environment = os.environ.copy()
    environment.setdefault("HF_HOME", str(_PROJECT_ROOT / ".tmp/hf"))
    matplotlib_cache = _PROJECT_ROOT / ".tmp" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    environment.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    completed = subprocess.run(
        [str(python_path), "-c", "; ".join(imports)],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "AVEC environment dependency check failed. Run the decoder setup "
            f"from README.md. Original error:\n{details}"
        )


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.absolute() if path.is_absolute() else (_PROJECT_ROOT / path).absolute()


def _require_file(value: str | Path, label: str) -> Path:
    path = _project_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _path_under(parent: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _extra_arg_value(values: list[str], option: str) -> str | None:
    try:
        index = values.index(option)
    except ValueError:
        return None
    if index + 1 >= len(values):
        raise ValueError(f"{option} requires a value in the AVEC config.")
    return values[index + 1]


def _manifest_mouth_roi_reliability(
    row: dict[str, str],
) -> dict[str, object]:
    preprocessing = str(row.get("video_preprocessing") or "").lower()
    video_name = Path(str(row.get("video_path") or "")).stem.lower()
    is_mouth_roi = (
        preprocessing in {"preprocessed", "mouth_roi", "mouth-roi"}
        or video_name.endswith("_mouth")
    )
    if not is_mouth_roi:
        return {}
    return {
        "roi_status": "success",
        "fallback_used": False,
        "reliability_source": "mouth_roi_manifest",
    }


def _force_avec_policy(policy: dict[str, object]) -> dict[str, object]:
    return {
        "version": 1,
        "policy_type": "roi_success",
        "feature_family": "visual",
        "feature": None,
        "direction": "higher",
        "threshold": None,
        "audio_system": policy.get("audio_system") or "whisper_base",
        "av_system": policy.get("av_system") or "avec_av_beam_roberta",
    }


class _DemoHandler(BaseHTTPRequestHandler):
    app: DemoApplication

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(self.app.state())
            return
        if path.startswith("/api/capture/"):
            self._send_capture_media(path)
            return
        if path.startswith("/api/media/"):
            self._send_media(path)
            return
        asset = "index.html" if path == "/" else path.lstrip("/")
        if asset not in {"index.html", "app.js", "styles.css"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(_ASSET_DIR / asset)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/live-route":
            self._handle_live_capture()
            return
        if path != "/api/route":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            sample_id = str(payload.get("sample_id") or "")
            overrides = payload.get("overrides") or {}
            if not isinstance(overrides, dict):
                raise ValueError("overrides must be an object.")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._stream(lambda emit: self.app.run_route(sample_id, overrides, emit))

    def _handle_live_capture(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                raise ValueError("Camera capture is empty.")
            if length > _MAX_CAPTURE_BYTES:
                self.send_error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "Camera capture exceeds the 32 MB limit.",
                )
                return
            content_type = self.headers.get("Content-Type") or ""
            capture_suffix(content_type)
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise ValueError("Camera capture ended before all bytes were received.")
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._stream(
            lambda emit: self.app.run_capture(payload, content_type, emit)
        )

    def _stream(
        self,
        action: Callable[[Callable[[dict[str, object]], None]], None],
    ) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def emit(event: dict[str, object]) -> None:
            self.wfile.write((json.dumps(event) + "\n").encode("utf-8"))
            self.wfile.flush()

        try:
            action(emit)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                emit({"stage": "error", "status": "error", "message": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_capture_media(self, route: str) -> None:
        parts = route.strip("/").split("/")
        if len(parts) != 4 or parts[3] not in {"source", "audio", "processed"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = self.app.capture_path(parts[2], parts[3])
        if path is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(path)

    def _send_json(
        self,
        value: object,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        start, end = 0, max(0, size - 1)
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header and size:
            try:
                start, end = _parse_byte_range(range_header, size)
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1 if size else 0
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            self.wfile.write(handle.read(length))

    def _send_media(self, route: str) -> None:
        parts = route.strip("/").split("/")
        if len(parts) != 4 or parts[3] not in {"audio", "video"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = self.app.media_path(parts[2], parts[3])
        if path is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(path)


def _parse_byte_range(value: str, size: int) -> tuple[int, int]:
    unit, separator, bounds = value.partition("=")
    if unit.strip().lower() != "bytes" or not separator or "," in bounds:
        raise ValueError("Invalid byte range.")
    start_text, separator, end_text = bounds.strip().partition("-")
    if not separator or (not start_text and not end_text):
        raise ValueError("Invalid byte range.")
    if not start_text:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise ValueError("Invalid byte range.")
        return max(0, size - suffix_length), size - 1
    start = int(start_text)
    end = int(end_text) if end_text else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("Invalid byte range.")
    return start, min(end, size - 1)


def _validate_overrides(values: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in values.items():
        if name == "roi_ready":
            result[name] = bool(value)
            continue
        if name not in _OVERRIDE_RANGES:
            continue
        number = float(value)
        minimum, maximum = _OVERRIDE_RANGES[name]
        result[name] = max(minimum, min(maximum, number))
    return result


def _audio_preview(path: Path) -> dict[str, object]:
    samples, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    mono = samples.mean(axis=1)
    if mono.size:
        bin_width = max(1, math.ceil(len(mono) / 180))
        peaks = [
            float(abs(mono[start : start + bin_width]).max())
            for start in range(0, len(mono), bin_width)
        ]
        peak = max(peaks)
        waveform = [value / peak for value in peaks] if peak > 0 else peaks
        rms = math.sqrt(float((mono**2).mean()))
    else:
        waveform = []
        rms = 0.0
    return {
        "sample_rate": sample_rate,
        "duration_seconds": len(mono) / sample_rate if sample_rate > 0 else 0.0,
        "waveform": waveform,
        "rms_dbfs": 20.0 * math.log10(max(rms, 1e-12)),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    return {
        str(row.get("utterance_id") or ""): row
        for row in _read_csv(path)
        if row.get("utterance_id")
    }


def _resolve_path(manifest_path: Path, value: object) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
