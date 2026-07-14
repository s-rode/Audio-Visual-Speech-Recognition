import argparse
import unittest

from scripts.experimental.voxmm_transformer_lm import (
    FILLED_PAUSE_WORDS,
    dropped_words_from_args,
    format_text,
    parse_setting,
    select_prediction,
    word_errors,
)
from scripts.experimental.tune_voxmm_transformer_lm import (
    parse_float_range,
    parse_weights,
)


class VoxMMTransformerLMTest(unittest.TestCase):
    def test_formats_sentence_style(self):
        self.assertEqual(format_text("hello world", "raw"), "hello world")
        self.assertEqual(format_text("hello world", "sentence"), "Hello world.")

    def test_filled_pause_preset_extends_explicit_words(self):
        args = argparse.Namespace(
            drop_word=["Custom"],
            drop_filled_pauses=True,
        )

        words = dropped_words_from_args(args)

        self.assertIn("custom", words)
        self.assertTrue(FILLED_PAUSE_WORDS.issubset(words))

    def test_parses_all_rescoring_weights(self):
        setting = parse_setting("best:0.15:1.35:1.0:0.25:0.4:0.01:0.02")

        self.assertEqual(setting["name"], "best")
        self.assertEqual(setting["transformer_weight"], 0.15)
        self.assertEqual(setting["word_bonus"], 1.35)
        self.assertEqual(setting["base_weight"], 1.0)
        self.assertEqual(setting["drop_penalty"], 0.25)
        self.assertEqual(setting["posterior_temperature"], 0.4)
        self.assertEqual(setting["forward_lm_weight"], 0.01)
        self.assertEqual(setting["reverse_lm_weight"], 0.02)

    def test_posterior_groups_normalized_equivalent_hypotheses(self):
        candidates = [
            {"text": "uh hello", "score": 0.0, "transformer_lm_score": 0.0},
            {"text": "hello", "score": -0.1, "transformer_lm_score": 0.0},
            {"text": "yellow", "score": 0.05, "transformer_lm_score": 0.0},
        ]
        direct = parse_setting("direct:0:0")
        posterior = parse_setting("posterior:0:0:1:0:1")

        self.assertEqual(select_prediction(candidates, direct, {"uh"}), "yellow")
        self.assertEqual(
            select_prediction(candidates, posterior, {"uh"}),
            "hello",
        )


    def test_word_errors_counts_edits(self):
        self.assertEqual(word_errors("a b c", "a x c d"), (2, 3))

    def test_parses_tuning_ranges_and_weights(self):
        self.assertEqual(parse_float_range("-0.5:1.5"), (-0.5, 1.5))
        self.assertEqual(
            parse_weights("0.15:1.35:0.01:0.0:-0.2"),
            [0.15, 1.35, 0.01, 0.0, -0.2],
        )


if __name__ == "__main__":
    unittest.main()
