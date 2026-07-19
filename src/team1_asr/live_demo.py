from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import mimetypes
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
from team1_asr.config import load_config
from team1_asr.live_capture import BrowserCapturePreparer, capture_suffix
from team1_asr.live_router import LiveRouter, load_policy
from team1_asr.router_features import extract_live_features
from team1_asr.types import Segment, Transcript


_ASSET_DIR = Path(__file__).with_name("live_demo_assets")
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
    ) -> None:
        if not samples:
            raise ValueError("The live demo requires at least one sample.")
        self.router = router
        self.samples = {sample.utterance_id: sample for sample in samples}
        self.capture_preparer = capture_preparer
        self.mode = mode
        self.lock = threading.Lock()

    def state(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "capture_enabled": True,
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
        with TemporaryDirectory(prefix="team1-live-capture-") as temp_dir:
            with self.lock:
                artifacts = self.capture_preparer.prepare(
                    payload,
                    content_type,
                    Path(temp_dir),
                    utterance_id,
                    emit,
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

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        with self._lock:
            if self._backend is None:
                self._backend = self._factory()
        return self._backend.transcribe(request)

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
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
    parser.add_argument("--manifest")
    parser.add_argument("--reliability")
    parser.add_argument("--policy")
    parser.add_argument("--whisper-config")
    parser.add_argument("--avec-config")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    real_args = (
        args.manifest,
        args.reliability,
        args.policy,
        args.whisper_config,
        args.avec_config,
    )
    if any(real_args) and not all(real_args):
        raise SystemExit(
            "Real mode requires --manifest, --reliability, --policy, "
            "--whisper-config, and --avec-config."
        )
    application = (
        _build_real_application(args)
        if all(real_args)
        else build_mock_application()
    )
    server = create_server(application, args.host, args.port)
    address, port = server.server_address[:2]
    print(f"Live router demo: http://{address}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        application.router.close()


def create_server(
    application: DemoApplication,
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    class Handler(_DemoHandler):
        app = application

    return ThreadingHTTPServer((host, port), Handler)


def build_mock_application(*, delay: float = 0.18) -> DemoApplication:
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

    router = LiveRouter(
        policy,
        _MockBackend("whisper", delay),
        _MockBackend("avec", delay),
        feature_extractor=mock_features,
    )
    return DemoApplication(
        router,
        samples,
        BrowserCapturePreparer(real_preprocessing=False),
        mode="mock",
    )


def _build_real_application(args: argparse.Namespace) -> DemoApplication:
    manifest_path = Path(args.manifest)
    reliability_path = Path(args.reliability)
    reliability = _read_csv_by_id(reliability_path)
    samples = []
    for row in _read_csv(manifest_path)[: max(1, args.limit)]:
        utterance_id = str(row.get("utterance_id") or "")
        if not utterance_id:
            continue
        samples.append(
            DemoSample(
                utterance_id=utterance_id,
                label=utterance_id,
                condition=str(row.get("condition") or "Live sample"),
                audio_path=str(_resolve_path(manifest_path, row.get("audio_path"))),
                video_path=(
                    str(_resolve_path(manifest_path, row.get("video_path")))
                    if row.get("video_path")
                    else None
                ),
                language=str(row.get("language") or "en"),
                reference=str(row.get("reference") or "") or None,
                reliability=reliability.get(utterance_id, {}),
            )
        )
    whisper_config = load_config(args.whisper_config).backend
    avec_config = load_config(args.avec_config).backend
    if whisper_config.kind != "faster-whisper":
        raise ValueError("--whisper-config must use the faster-whisper backend.")
    if avec_config.kind != "avec":
        raise ValueError("--avec-config must use the avec backend.")
    router = LiveRouter(
        load_policy(args.policy),
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
    )


class _DemoHandler(BaseHTTPRequestHandler):
    app: DemoApplication

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(self.app.state())
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
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
