from __future__ import annotations

import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from team1_asr.live_demo import build_mock_application, create_server


class LiveDemoServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = create_server(build_mock_application(delay=0.0), "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_state_exposes_both_presentation_samples(self) -> None:
        with urlopen(f"{self.base_url}/api/state") as response:
            state = json.load(response)

        self.assertEqual(state["mode"], "mock")
        self.assertTrue(state["capture_enabled"])
        self.assertEqual([sample["id"] for sample in state["samples"]], [
            "clear-room",
            "street-noise",
        ])

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
