from __future__ import annotations

import unittest

import numpy as np

from team1_asr.avec_preprocess_runner import _detect_landmarks
from team1_asr.visual_reliability import (
    FrameReliability,
    is_real_mouth_roi,
    summarize_visual_reliability,
)


class VisualReliabilityTest(unittest.TestCase):
    def test_extracts_retinaface_and_mouth_landmark_scores(self) -> None:
        landmarks, observations = _detect_landmarks(
            [_FakeFrame(), _FakeFrame()],
            _FakeCropper(),
        )

        self.assertEqual(len(landmarks), 2)
        self.assertEqual(observations[0].face_count, 2)
        self.assertAlmostEqual(observations[0].face_confidence, 0.9)
        self.assertAlmostEqual(observations[0].mouth_landmark_confidence, 0.75)

    def test_summarizes_detections_before_interpolation(self) -> None:
        row = summarize_visual_reliability(
            "utt-1",
            [
                FrameReliability(1, 0.9, 0.8),
                FrameReliability(2, 0.7, None),
                FrameReliability(0, None, None),
                FrameReliability(1, 0.8, 0.6),
            ],
            roi_status="success",
            fallback_used=False,
            detector_threshold=0.3,
        )

        self.assertEqual(row["total_frames"], 4)
        self.assertEqual(row["face_detected_frames"], 3)
        self.assertEqual(row["landmark_detected_frames"], 2)
        self.assertEqual(row["interpolated_frames"], 2)
        self.assertEqual(row["longest_missing_run"], 2)
        self.assertEqual(row["multiple_face_frames"], 1)
        self.assertAlmostEqual(row["face_detection_rate"], 0.75)
        self.assertAlmostEqual(row["retinaface_confidence_mean"], 0.8)
        self.assertAlmostEqual(row["retinaface_confidence_p10"], 0.7)
        self.assertAlmostEqual(row["mouth_landmark_confidence_mean"], 0.7)
        self.assertTrue(is_real_mouth_roi(row))

    def test_fallback_and_error_are_not_real_mouth_rois(self) -> None:
        fallback = summarize_visual_reliability(
            "utt-fallback",
            [],
            roi_status="fallback_full_frame",
            fallback_used=True,
            failure_reason="no landmarks detected",
        )
        error = summarize_visual_reliability(
            "utt-error",
            [],
            roi_status="error",
            fallback_used=False,
            failure_reason="bad video",
        )

        self.assertFalse(is_real_mouth_roi(fallback))
        self.assertFalse(is_real_mouth_roi(error))
        self.assertEqual(fallback["face_detection_rate"], 0.0)
        self.assertEqual(error["failure_reason"], "bad video")


class _FakeFrame:
    def numpy(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


class _FakeDetector:
    def __call__(self, frame, rgb=True):
        return np.array(
            [
                [0, 0, 4, 4, 0.9],
                [1, 1, 5, 5, 0.7],
            ],
            dtype=np.float32,
        )


class _FakeLandmarkDetector:
    def __call__(self, frame, faces, rgb=True):
        landmarks = np.zeros((2, 68, 2), dtype=np.float32)
        scores = np.full((2, 68), 0.5, dtype=np.float32)
        scores[0, 48:68] = 0.75
        return landmarks, scores


class _FakeCropper:
    start_idx = 48
    stop_idx = 68
    face_detector = _FakeDetector()
    landmark_detector = _FakeLandmarkDetector()


if __name__ == "__main__":
    unittest.main()
