from __future__ import annotations

import argparse
import json

from team1_asr.config import load_config
from team1_asr.pipeline import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a modular Whisper-based ASR experiment from a TOML config."
    )
    parser.add_argument("config", help="Path to an experiment TOML file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
