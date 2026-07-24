from __future__ import annotations

import unittest

from team1_asr.backends.base import TranscriptionRequest
from team1_asr.live_router import LiveRouter
from team1_asr.types import Segment, Transcript


class _Backend:
    def __init__(self, name: str, text: str, avg_logprob: float = -0.4) -> None:
        self.name = name
        self.text = text
        self.avg_logprob = avg_logprob
        self.calls = 0

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        self.calls += 1
        return Transcript(
            text=self.text,
            language="en",
            duration=2.0,
            segments=[
                Segment(
                    start=0.0,
                    end=2.0,
                    text=self.text,
                    avg_logprob=self.avg_logprob,
                    no_speech_prob=0.01,
                )
            ],
            metadata={"backend": self.name},
        )


class _StreamingBackend(_Backend):
    def transcribe_with_events(self, request, emit):
        emit(
            {
                "stage": "beam",
                "status": "complete",
                "message": "Generated hypotheses.",
                "transcript": "beam result",
            }
        )
        emit(
            {
                "stage": "rescore",
                "status": "complete",
                "message": "Selected a hypothesis.",
                "transcript": self.text,
            }
        )
        transcript = super().transcribe(request)
        transcript.metadata["decoder"] = "flashlight_kenlm_roberta"
        return transcript


class LiveRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = {
            "version": 1,
            "policy_type": "threshold",
            "feature_family": "audio_uncertainty",
            "feature": "audio_avg_logprob_mean",
            "threshold": -0.8,
            "direction": "lower",
            "audio_system": "whisper",
            "av_system": "avec",
        }
        self.whisper = _Backend("whisper", "audio result")
        self.avec = _Backend("avec", "audio visual result")
        self.features = {"audio_avg_logprob_mean": -0.4}
        self.router = LiveRouter(
            self.policy,
            self.whisper,
            self.avec,
            feature_extractor=lambda audio, video, transcript: dict(self.features),
        )
        self.request = TranscriptionRequest(
            audio_path="audio.wav",
            video_path="mouth.mp4",
            language="en",
            utterance_id="u1",
        )

    def test_keeps_whisper_when_audio_is_reliable(self) -> None:
        events = []
        result = self.router.route(
            self.request,
            {"roi_status": "success", "fallback_used": False},
            emit=events.append,
        )

        self.assertEqual(result.selected_system, "whisper")
        self.assertEqual(result.transcript.text, "audio result")
        self.assertEqual(self.whisper.calls, 1)
        self.assertEqual(self.avec.calls, 0)
        self.assertEqual(events[-1]["stage"], "result")

    def test_runs_avec_when_audio_is_uncertain_and_roi_is_real(self) -> None:
        result = self.router.route(
            self.request,
            {"roi_status": "success", "fallback_used": False},
            feature_overrides={"audio_avg_logprob_mean": -1.3},
        )

        self.assertEqual(result.selected_system, "avec")
        self.assertEqual(result.transcript.text, "audio visual result")
        self.assertEqual(self.avec.calls, 1)

    def test_never_runs_avec_for_a_fallback_roi(self) -> None:
        result = self.router.route(
            self.request,
            {"roi_status": "fallback", "fallback_used": True},
            feature_overrides={"audio_avg_logprob_mean": -1.3},
        )

        self.assertEqual(result.selected_system, "whisper")
        self.assertEqual(self.avec.calls, 0)

    def test_forwards_online_decoder_progress(self) -> None:
        events = []
        router = LiveRouter(
            self.policy,
            self.whisper,
            _StreamingBackend("avec", "reranked result"),
            feature_extractor=lambda audio, video, transcript: dict(self.features),
        )

        result = router.route(
            self.request,
            {"roi_status": "success", "fallback_used": False},
            feature_overrides={"audio_avg_logprob_mean": -1.3},
            emit=events.append,
        )

        stages = [event["stage"] for event in events]
        self.assertLess(stages.index("beam"), stages.index("rescore"))
        self.assertLess(stages.index("rescore"), stages.index("result"))
        self.assertEqual(result.transcript.text, "reranked result")
        avec_complete = [
            event
            for event in events
            if event["stage"] == "avec" and event["status"] == "complete"
        ]
        self.assertIn("RoBERTa", avec_complete[0]["message"])


if __name__ == "__main__":
    unittest.main()
