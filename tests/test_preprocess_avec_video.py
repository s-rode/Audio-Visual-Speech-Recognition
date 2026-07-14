from __future__ import annotations

import unittest

from team1_asr.preprocess_avec_video import build_parser, _runner_command


class PreprocessAvecVideoTest(unittest.TestCase):
    def test_runner_command_uses_avec_python_and_manifest_args(self) -> None:
        args = build_parser().parse_args(
            [
                "data/lrs3/hf_av_tiny/manifest.csv",
                "data/lrs3/hf_av_tiny_avec_roi",
                "--repo-path",
                "external/AVEC",
                "--python-executable",
                "external/AVEC/.venv/bin/python3",
                "--force",
            ]
        )

        command = _runner_command(args)

        self.assertEqual(command[0], "external/AVEC/.venv/bin/python3")
        self.assertIn("run_avec_preprocess_manifest.py", command[1])
        self.assertIn("data/lrs3/hf_av_tiny/manifest.csv", command)
        self.assertIn("data/lrs3/hf_av_tiny_avec_roi", command)
        self.assertIn("--force", command)


if __name__ == "__main__":
    unittest.main()
