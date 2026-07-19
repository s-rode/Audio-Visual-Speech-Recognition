from __future__ import annotations

import json
import threading
import unittest
import wave
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from team1_asr.backends.base import TranscriptionRequest
from team1_asr.live_capture import CaptureArtifacts
from team1_asr.live_demo import (
    _LiveInputBackend,
    build_mock_application,
    create_server,
)
from team1_asr.types import Transcript


class LiveDemoServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.application = build_mock_application(delay=0.0)
        self.server = create_server(self.application, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.application.close()

    def test_state_exposes_both_presentation_samples(self) -> None:
        with urlopen(f"{self.base_url}/api/state") as response:
            state = json.load(response)

        self.assertEqual(state["mode"], "mock")
        self.assertTrue(state["capture_enabled"])
        self.assertEqual(state["live_transcription"], "mock")
        self.assertEqual([sample["id"] for sample in state["samples"]], [
            "clear-room",
            "street-noise",
        ])

    def test_live_backend_dispatches_camera_requests_to_real_whisper(self) -> None:
        class RecordingBackend:
            def __init__(self, text: str) -> None:
                self.text = text
                self.requests: list[str | None] = []

            def transcribe(self, request: TranscriptionRequest) -> Transcript:
                self.requests.append(request.utterance_id)
                return Transcript(self.text, "en", 1.0, [])

        sample_backend = RecordingBackend("sample transcript")
        live_backend = RecordingBackend("actual microphone words")
        backend = _LiveInputBackend(sample_backend, live_backend)

        sample = backend.transcribe(TranscriptionRequest("sample.wav", "en", "sample"))
        live = backend.transcribe(TranscriptionRequest("live.wav", "en", "live-123"))

        self.assertEqual(sample.text, "sample transcript")
        self.assertEqual(live.text, "actual microphone words")
        self.assertEqual(sample_backend.requests, ["sample"])
        self.assertEqual(live_backend.requests, ["live-123"])

    def test_stream_shows_whisper_to_avec_switch(self) -> None:
        events = self._route(
            "street-noise",
            {
                "audio_avg_logprob_mean": -1.3,
                "mouth_landmark_confidence_mean": 0.9,
                "roi_ready": True,
            },
        )

        self.assertEqual(events[0]["stage"], "whisper")
        decision = next(event for event in events if event["stage"] == "router")
        self.assertTrue(decision["selects_avec"])
        self.assertTrue(any(event["stage"] == "avec" for event in events))
        self.assertEqual(events[-1]["result"]["selected_system"], "AVEC")

    def test_stream_skips_avec_for_reliable_audio(self) -> None:
        events = self._route(
            "clear-room",
            {
                "audio_avg_logprob_mean": -0.2,
                "mouth_landmark_confidence_mean": 0.9,
                "roi_ready": True,
            },
        )

        avec = next(event for event in events if event["stage"] == "avec")
        self.assertEqual(avec["status"], "skipped")
        self.assertEqual(events[-1]["result"]["selected_system"], "Whisper")

    def test_live_capture_rejects_unsupported_media(self) -> None:
        request = Request(
            f"{self.base_url}/api/live-route",
            data=b"not a recording",
            headers={"Content-Type": "audio/wav"},
            method="POST",
        )

        with self.assertRaises(HTTPError) as caught:
            urlopen(request)

        self.assertEqual(caught.exception.code, 400)
        error = json.loads(caught.exception.read())
        self.assertIn("Unsupported capture media type", error["error"])

    def test_capture_retains_inspection_artifacts(self) -> None:
        events: list[dict[str, object]] = []

        def prepare(payload, content_type, work_dir, utterance_id, emit):
            source = work_dir / f"{utterance_id}.webm"
            audio = work_dir / f"{utterance_id}.wav"
            processed = work_dir / "mouth-roi.mp4"
            source.write_bytes(payload)
            processed.write_bytes(b"processed")
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 160)
            return CaptureArtifacts(
                source,
                audio,
                processed,
                {"roi_status": "success"},
            )

        with (
            patch.object(
                self.application.capture_preparer,
                "prepare",
                side_effect=prepare,
            ),
            patch.object(self.application.router, "route"),
        ):
            self.application.run_capture(
                b"source capture",
                "video/webm",
                events.append,
            )

        event = next(item for item in events if item["stage"] == "media")
        self.assertTrue(event["media"]["processed_distinct"])
        self.assertEqual(event["media"]["processed_label"], "Mouth ROI")
        with urlopen(f"{self.base_url}{event['media']['source_url']}") as response:
            self.assertEqual(response.read(), b"source capture")

    def test_page_exposes_both_live_transcript_panels(self) -> None:
        with urlopen(f"{self.base_url}/") as response:
            page = response.read().decode("utf-8")

        self.assertIn("whisper-live-transcript", page)
        self.assertIn("avec-live-transcript", page)
        self.assertIn("whisper-segments", page)
        self.assertIn("avec-segments", page)

    def test_static_media_supports_byte_ranges(self) -> None:
        request = Request(
            f"{self.base_url}/app.js",
            headers={"Range": "bytes=0-9"},
        )

        with urlopen(request) as response:
            body = response.read()
            content_range = response.headers["Content-Range"]

        self.assertEqual(response.status, 206)
        self.assertEqual(len(body), 10)
        self.assertTrue(content_range.startswith("bytes 0-9/"))

    def _route(
        self,
        sample_id: str,
        overrides: dict[str, object],
    ) -> list[dict[str, object]]:
        request = Request(
            f"{self.base_url}/api/route",
            data=json.dumps(
                {"sample_id": sample_id, "overrides": overrides}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            return [
                json.loads(line)
                for line in response.read().decode("utf-8").splitlines()
                if line
            ]


if __name__ == "__main__":
    unittest.main()
