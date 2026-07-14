from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess manifest videos into AVEC-style mouth ROI clips."
    )
    parser.add_argument("manifest", help="Input manifest.csv with video_path.")
    parser.add_argument("output_dir", help="Output directory for ROI videos and manifest.csv.")
    parser.add_argument("--repo-path", default="external/AVEC", help="Editable AVEC checkout.")
    parser.add_argument(
        "--python-executable",
        default="external/AVEC/.venv/bin/python3",
        help="Python executable from the AVEC environment.",
    )
    parser.add_argument("--video-column", default="video_path")
    parser.add_argument("--audio-column", default="audio_path")
    parser.add_argument("--id-column", default="utterance_id")
    parser.add_argument("--reference-column", default="reference")
    parser.add_argument("--language-column", default="language")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing mouth ROI clips when present.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = _runner_command(args)
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _runner_command(args: argparse.Namespace) -> list[str]:
    runner = Path(__file__).resolve().parents[2] / "scripts" / "run_avec_preprocess_manifest.py"
    command = [
        args.python_executable,
        str(runner),
        str(args.manifest),
        str(args.output_dir),
        "--repo-path",
        args.repo_path,
        "--video-column",
        args.video_column,
        "--audio-column",
        args.audio_column,
        "--id-column",
        args.id_column,
        "--reference-column",
        args.reference_column,
        "--language-column",
        args.language_column,
    ]
    if args.force:
        command.append("--force")
    if args.skip_existing:
        command.append("--skip-existing")
    return command


if __name__ == "__main__":
    main()
