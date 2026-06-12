from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from team1_asr import prepare_lrs3_hf


class LRS3ParsingTest(unittest.TestCase):
    def test_parses_text_before_confidence_block(self) -> None:
        raw = (
            "Text:  HELLO   WORLD FROM LRS3  \n"
            "Conf:  4\n"
            "WORD START END ASDSCORE\n"
        )

        self.assertEqual(prepare_lrs3_hf._parse_transcript(raw), "HELLO WORLD FROM LRS3")

    def test_sanitizes_keys_for_file_names(self) -> None:
        self.assertEqual(
            prepare_lrs3_hf._sanitize_key("pretrain/6ORDQFh0Byw/00001"),
            "pretrain_6ORDQFh0Byw_00001",
        )


class LRS3ExportTest(unittest.TestCase):
    def test_exports_manifest_and_deletes_temp_mp4(self) -> None:
        sample = _sample()
        seen_video_paths: list[Path] = []

        def fake_ffmpeg(command, capture_output, text):
            video_path = Path(command[command.index("-i") + 1])
            output_path = Path(command[-1])
            seen_video_paths.append(video_path)
            output_path.write_bytes(b"wav")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "lrs3"
            with (
                patch.object(prepare_lrs3_hf, "_load_dataset", return_value=iter([sample])),
                patch.object(prepare_lrs3_hf.subprocess, "run", side_effect=fake_ffmpeg),
            ):
                prepare_lrs3_hf.export_lrs3_hf_subset(
                    output_dir=output_dir,
                    split="train",
                    limit=1,
                    offset=0,
                    seed=None,
                    language="en",
                    cache_dir=None,
                    skip_invalid=False,
                    force=False,
                )

            manifest_path = output_dir / "manifest.csv"
            self.assertTrue(manifest_path.exists())
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["utterance_id"], "pretrain_abc_00001")
            self.assertEqual(rows[0]["audio_path"], "audio/pretrain_abc_00001.wav")
            self.assertEqual(rows[0]["reference"], "HELLO WORLD")
            self.assertEqual(rows[0]["source_key"], "pretrain/abc/00001")
            self.assertTrue((output_dir / rows[0]["audio_path"]).exists())
            self.assertEqual(len(seen_video_paths), 1)
            self.assertFalse(seen_video_paths[0].exists())

    def test_keep_video_writes_video_manifest_column(self) -> None:
        sample = _sample()

        def fake_ffmpeg(command, capture_output, text):
            Path(command[-1]).write_bytes(b"wav")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "lrs3"
            with (
                patch.object(prepare_lrs3_hf, "_load_dataset", return_value=iter([sample])),
                patch.object(prepare_lrs3_hf.subprocess, "run", side_effect=fake_ffmpeg),
            ):
                prepare_lrs3_hf.export_lrs3_hf_subset(
                    output_dir=output_dir,
                    split="train",
                    limit=1,
                    offset=0,
                    seed=None,
                    language="en",
                    cache_dir=None,
                    skip_invalid=False,
                    force=False,
                    keep_video=True,
                )

            with (output_dir / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["video_path"], "video/pretrain_abc_00001.mp4")
            self.assertTrue((output_dir / rows[0]["video_path"]).exists())

    def test_skips_invalid_sample_when_requested(self) -> None:
        invalid = _sample(txt="Conf: 4\nWORD START END ASDSCORE")
        valid = _sample(key="trainval/xyz/00002")

        def fake_ffmpeg(command, capture_output, text):
            Path(command[-1]).write_bytes(b"wav")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "lrs3"
            with (
                patch.object(prepare_lrs3_hf, "_load_dataset", return_value=iter([invalid, valid])),
                patch.object(prepare_lrs3_hf.subprocess, "run", side_effect=fake_ffmpeg),
            ):
                prepare_lrs3_hf.export_lrs3_hf_subset(
                    output_dir=output_dir,
                    split="train",
                    limit=1,
                    offset=0,
                    seed=None,
                    language="en",
                    cache_dir=None,
                    skip_invalid=True,
                    force=False,
                )

            with (output_dir / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_key"], "trainval/xyz/00002")

    def test_invalid_sample_fails_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                prepare_lrs3_hf,
                "_load_dataset",
                return_value=iter([_sample(txt="Conf: 4")]),
            ):
                with self.assertRaises(ValueError):
                    prepare_lrs3_hf.export_lrs3_hf_subset(
                        output_dir=Path(temp_dir) / "lrs3",
                        split="train",
                        limit=1,
                        offset=0,
                        seed=None,
                        language="en",
                        cache_dir=None,
                        skip_invalid=False,
                        force=False,
                    )


def _sample(
    key: str = "pretrain/abc/00001",
    txt: str = "Text: HELLO WORLD\nConf: 4\nWORD START END ASDSCORE",
) -> dict[str, object]:
    return {
        "mp4": {"bytes": b"fake-mp4"},
        "txt": txt,
        "__key__": key,
        "__url__": "hf://datasets/Ainncy/LRS3/pretrain.00.tar",
    }


if __name__ == "__main__":
    unittest.main()
