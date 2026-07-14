import math
import tempfile
import unittest
from pathlib import Path

from scripts.experimental.build_voxmm_ngram_arpa import AbsoluteDiscountLM, words


class VoxMMNGramTest(unittest.TestCase):
    def test_spoken_alternative_is_used_once(self):
        self.assertEqual(
            words("WE HAVE (23/TWENTY THREE) CASES"),
            ["we", "have", "twenty", "three", "cases"],
        )

    def test_interpolated_and_backoff_modes_normalize(self):
        sentences = [["a", "b"], ["a", "c"]]
        for mode in ("interpolated", "backoff"):
            with self.subTest(mode=mode):
                lm = AbsoluteDiscountLM(
                    order=2,
                    discount=0.5,
                    mode=mode,
                ).fit(sentences)
                probability_mass = sum(
                    lm.probability(("a", token))
                    for token in lm.vocab
                )
                self.assertTrue(
                    math.isclose(probability_mass, 1.0, abs_tol=1e-9),
                    (mode, probability_mass),
                )

    def test_writes_loadable_arpa_structure(self):
        lm = AbsoluteDiscountLM(order=2).fit([["hello", "world"]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.arpa"
            lm.write_arpa(path)
            contents = path.read_text(encoding="utf-8")

        self.assertIn("\\data\\", contents)
        self.assertIn("\\1-grams:", contents)
        self.assertIn("\\2-grams:", contents)
        self.assertTrue(contents.rstrip().endswith("\\end\\"))


if __name__ == "__main__":
    unittest.main()
