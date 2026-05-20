from __future__ import annotations

import unittest

from team1_asr.evaluation import compute_metrics


class EvaluationTest(unittest.TestCase):
    def test_normalized_metrics_ignore_case_and_punctuation(self) -> None:
        metrics = compute_metrics(
            ["HE WAS IN A FEVERED STATE"],
            ["He was in a fevered state."],
        )

        self.assertEqual(metrics.utterances_scored, 1)
        self.assertEqual(metrics.wer, 0.0)
        self.assertEqual(metrics.cer, 0.0)
        self.assertGreater(metrics.raw_wer or 0.0, 0.0)
        self.assertGreater(metrics.raw_cer or 0.0, 0.0)

    def test_empty_references_have_no_metrics(self) -> None:
        metrics = compute_metrics([], [])

        self.assertEqual(metrics.utterances_scored, 0)
        self.assertIsNone(metrics.wer)
        self.assertIsNone(metrics.cer)
        self.assertIsNone(metrics.raw_wer)
        self.assertIsNone(metrics.raw_cer)


if __name__ == "__main__":
    unittest.main()
