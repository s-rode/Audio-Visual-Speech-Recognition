from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a faster-whisper model with visible progress."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="tiny.en",
        help="Model size such as tiny.en, small.en, or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to save the model into instead of the HF cache.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only check for an already cached model; do not download.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        model_path = download_model(
            args.model,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None

    print(f"Model ready: {model_path}")


def download_model(
    model: str,
    output_dir: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> str:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from faster_whisper.utils import _MODELS
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        from team1_asr.backends.base import require_dependency

        require_dependency("faster-whisper and huggingface_hub", exc)

    repo_id = model if "/" in model else _MODELS.get(model)
    if repo_id is None:
        valid_models = ", ".join(sorted(_MODELS))
        raise ValueError(f"Unknown faster-whisper model '{model}'. Valid models: {valid_models}.")

    destination = Path(output_dir) if output_dir else None
    print(f"Downloading {repo_id}...", file=sys.stderr)
    if destination:
        print(f"Saving model to {destination}", file=sys.stderr)
    else:
        print("Saving model to the Hugging Face cache", file=sys.stderr)

    kwargs = {
        "allow_patterns": _ALLOW_PATTERNS,
        "local_files_only": local_files_only,
    }
    if destination:
        kwargs["local_dir"] = str(destination)
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    return snapshot_download(repo_id, **kwargs)


if __name__ == "__main__":
    main()
