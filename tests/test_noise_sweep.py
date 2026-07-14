from __future__ import annotations

import unittest

from team1_asr.noise_sweep import analyze_condition, flatten_condition


class NoiseSweepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.references = {"u1": "hello world", "u2": "good day"}
        self.whisper = {"u1": "wrong words", "u2": "good day"}
        self.ao = {"u1": "hello world", "u2": "wrong words"}
        self.av = {"u1": "hello world", "u2": "good day"}

    def test_reports_system_pairwise_and_oracle_metrics(self) -> None:
        result = analyze_condition(
            name="demand_0db",
            snr_db=0.0,
            references=self.references,
            whisper=self.whisper,
            avec_ao=self.ao,
            avec_av=self.av,
        )
        row = flatten_condition(result)

        self.assertEqual(row["condition"], "demand_0db")
        self.assertEqual(row["whisper_wer"], 0.5)
        self.assertEqual(row["ao_wer"], 0.5)
        self.assertEqual(row["av_wer"], 0.0)
        self.assertEqual(row["av_wins_over_whisper"], 1)
        self.assertEqual(row["av_wins_over_ao"], 1)
        self.assertEqual(row["oracle_wer"], 0.0)

    def test_reports_av_coverage_without_hiding_missing_predictions(self) -> None:
        result = analyze_condition(
            name="demand_10db",
            snr_db=10.0,
            references=self.references,
            whisper=self.whisper,
            avec_ao=self.ao,
            avec_av={"u1": "hello world"},
        )
        row = flatten_condition(result)

        self.assertEqual(row["av_coverage"], 0.5)
        self.assertEqual(row["av_wins_over_whisper"], 1)
        self.assertEqual(row["oracle_wer"], 0.0)

    def test_requires_complete_whisper_predictions(self) -> None:
        with self.assertRaisesRegex(ValueError, "Whisper is missing"):
            analyze_condition(
                name="clean",
                snr_db=None,
                references=self.references,
                whisper={"u1": "hello world"},
                avec_ao=self.ao,
                avec_av=self.av,
            )


if __name__ == "__main__":
    unittest.main()
