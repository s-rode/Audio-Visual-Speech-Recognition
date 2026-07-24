from __future__ import annotations

import unittest

from team1_asr.transformer_rescoring import (
    RerankSettings,
    normalize_hypothesis,
    rerank_scored_candidates,
)


class TransformerRescoringTest(unittest.TestCase):
    def test_normalizes_filled_pauses(self) -> None:
        text, dropped = normalize_hypothesis("UH this is um correct")

        self.assertEqual(text, "this is correct")
        self.assertEqual(dropped, 2)

    def test_selects_candidate_with_offline_recipe_formula(self) -> None:
        result = rerank_scored_candidates(
            [
                {
                    "text": "plain first pass",
                    "score": 10.0,
                    "transformer_lm_score": -10.0,
                },
                {
                    "text": "uh better language",
                    "score": 9.0,
                    "transformer_lm_score": 0.0,
                },
            ],
            RerankSettings(
                top_k=2,
                transformer_weight=0.5,
                word_bonus=0.0,
                drop_penalty=-1.0,
            ),
        )

        self.assertEqual(result.first_pass_text, "plain first pass")
        self.assertEqual(result.text, "better language")
        self.assertEqual(result.selected_rank, 2)
        self.assertEqual(result.candidates[1]["dropped_word_count"], 1)


if __name__ == "__main__":
    unittest.main()
