from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import wave

from scripts import build_lrs_voxmm_dev_manifest as builder


class BuildLrsVoxmmDevManifestTest(unittest.TestCase):
    def test_sample_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_video_group(root, "video-a", count=5)
            _write_video_group(root, "video-b", count=5)
            rows = builder.build_rows(root)

            first = builder.select_rows(rows, sample=4, seed=7)
            second = builder.select_rows(rows, sample=4, seed=7)
            other = builder.select_rows(rows, sample=4, seed=8)

        self.assertEqual(_ids(first), _ids(second))
        self.assertNotEqual(_ids(first), _ids(other))

    def test_balanced_sample_spreads_across_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_video_group(root, "video-a", count=5)
            _write_video_group(root, "video-b", count=5)
            _write_video_group(root, "video-c", count=5)
            rows = builder.build_rows(root)

            selected = builder.select_rows(
                rows,
                sample=6,
                seed=0,
                balance_by_video=True,
            )

        counts = {}
        for row in selected:
            counts[row["_video_id"]] = counts.get(row["_video_id"], 0) + 1
        self.assertEqual(counts, {"video-a": 2, "video-b": 2, "video-c": 2})

    def test_min_duration_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_dir = root / "video-a"
            video_dir.mkdir()
            _write_utterance(video_dir, "00001", duration_sec=1.0)
            _write_utterance(video_dir, "00002", duration_sec=3.0)

            rows = builder.build_rows(root, min_duration_sec=2.5)

        self.assertEqual(_ids(rows), ["video-a_00002"])


def _write_video_group(root: Path, name: str, *, count: int) -> None:
    video_dir = root / name
    video_dir.mkdir()
    for index in range(count):
        _write_utterance(video_dir, f"{index:05d}", duration_sec=3.0)


def _write_utterance(video_dir: Path, stem: str, *, duration_sec: float) -> None:
    (video_dir / f"{stem}.txt").write_text(f"Text: example {stem}\n", encoding="utf-8")
    (video_dir / f"{stem}.mp4").write_bytes(b"fake video")
    _write_wav(video_dir / f"{stem}.wav", duration_sec=duration_sec)


def _write_wav(path: Path, *, duration_sec: float) -> None:
    sample_rate = 16000
    frame_count = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\0\0" * frame_count)


def _ids(rows: list[dict[str, str]]) -> list[str]:
    return [row["utterance_id"] for row in rows]


if __name__ == "__main__":
    unittest.main()
