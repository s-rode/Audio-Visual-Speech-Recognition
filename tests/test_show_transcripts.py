from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest

from team1_asr.show_transcripts import print_transcripts


class ShowTranscriptsTest(unittest.TestCase):
    def test_prints_reference_and_prediction(self) -> None:
        row = {
            "utterance_id": "utt-1",
            "reference": "HELLO WORLD",
            "prediction": "hello world",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            transcript_path = Path(temp_dir) / "transcripts.jsonl"
            transcript_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = io.StringIO()

            print_transcripts(transcript_path, limit=1, output=output)

        rendered = output.getvalue()
        self.assertIn("[utt-1]", rendered)
        self.assertIn("Reference : HELLO WORLD", rendered)
        self.assertIn("Prediction: hello world", rendered)


if __name__ == "__main__":
    unittest.main()
