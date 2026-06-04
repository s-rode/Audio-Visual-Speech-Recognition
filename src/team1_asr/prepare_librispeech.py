from __future__ import annotations

import argparse
import csv
from itertools import islice
import io
import os
from pathlib import Path
import shutil
import sys

import soundfile as sf

_VALID_SPLITS = {
    "clean": ("train.100", "train.360", "validation", "test"),
    "other": ("train.500", "validation", "test"),
    "all": (
        "train.clean.100",
        "train.clean.360",
        "train.other.500",
        "validation.clean",
        "validation.other",
        "test.clean",
        "test.other",
    ),
}

_SPLIT_ALIASES = {
    ("clean", "train.clean.100"): "train.100",
    ("clean", "train.clean.360"): "train.360",
    ("clean", "validation.clean"): "validation",
    ("clean", "test.clean"): "test",
    ("other", "train.other.500"): "train.500",
    ("other", "validation.other"): "validation",
    ("other", "test.other"): "test",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load LibriSpeech from Hugging Face and export a stable local "
            "WAV+manifest subset for ASR experiments."
        )
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write segmented WAV files and manifest.csv into.",
    )
    parser.add_argument(
        "--config",
        default="clean",
        choices=["all", "clean", "other"],
        help="LibriSpeech configuration from openslr/librispeech_asr.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help=(
            "Dataset split to export, for example validation, test, or train.100 "
            "for --config clean."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of utterances to export.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --limit is set for non-streaming exports.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language tag to write into the manifest.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--save-dataset",
        action="store_true",
        help=(
            "Also save the selected Hugging Face dataset subset with save_to_disk(). "
            "This may require downloading the full split before --limit is applied."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifest/audio files in the output directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        used_streaming = export_librispeech_split(
            output_dir=Path(args.output_dir),
            config_name=args.config,
            split=args.split,
            limit=args.limit,
            seed=args.seed,
            language=args.language,
            cache_dir=args.cache_dir,
            save_dataset=args.save_dataset,
            force=args.force,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None

    if used_streaming:
        # HF streaming can leave shutdown hooks waiting after a limited export.
        # The CLI has already written all files, so flush and exit immediately.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def export_librispeech_split(
    output_dir: Path,
    config_name: str,
    split: str,
    limit: int | None,
    seed: int,
    language: str,
    cache_dir: str | None,
    save_dataset: bool,
    force: bool,
) -> bool:
    config_name, split, warning = _normalize_config_and_split(config_name, split)
    if warning:
        print(f"Warning: {warning}", file=sys.stderr)
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1.")

    manifest_path = output_dir / "manifest.csv"
    audio_dir = output_dir / "audio"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --force to overwrite."
            )

    use_streaming = limit is not None and not save_dataset
    mode = "streaming" if use_streaming else "standard"
    print(
        f"Loading openslr/librispeech_asr ({config_name}, split={split}) "
        f"in {mode} mode..."
    )
    dataset = _load_dataset(
        config_name=config_name,
        split=split,
        cache_dir=cache_dir,
        streaming=use_streaming,
    )
    if not use_streaming:
        dataset = _select_subset(dataset, limit=limit, seed=seed)

    if output_dir.exists() and any(output_dir.iterdir()):
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    sample_iterable = islice(dataset, limit) if use_streaming else dataset
    for index, sample in enumerate(sample_iterable):
        utterance_id = sample.get("id") or f"{config_name}-{split}-{index:06d}"
        utterance_path = audio_dir / f"{_sanitize_filename(str(utterance_id))}.wav"
        audio_array, sample_rate = _read_audio(sample["audio"])
        sf.write(utterance_path, audio_array, sample_rate)
        if index == 0:
            print(f"Exporting first sample: {utterance_id}")
        rows.append(
            {
                "utterance_id": utterance_id,
                "audio_path": str(Path("audio") / utterance_path.name),
                "reference": sample["text"],
                "language": language,
                "speaker_id": sample.get("speaker_id"),
                "chapter_id": sample.get("chapter_id"),
                "source_file": _audio_source_path(sample["audio"]) or sample.get("file"),
                "hf_dataset": "openslr/librispeech_asr",
                "hf_config": config_name,
                "hf_split": split,
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "utterance_id",
                "audio_path",
                "reference",
                "language",
                "speaker_id",
                "chapter_id",
                "source_file",
                "hf_dataset",
                "hf_config",
                "hf_split",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if save_dataset:
        dataset.save_to_disk(str(output_dir / "hf_dataset"))

    print(
        f"Exported {len(rows)} utterances from openslr/librispeech_asr "
        f"({config_name}, split={split}) to {output_dir}."
    )
    print(f"Manifest: {manifest_path}")
    return use_streaming


def _load_dataset(
    config_name: str,
    split: str,
    cache_dir: str | None,
    streaming: bool,
):
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Run 'uv sync' to install it."
        ) from exc

    dataset = load_dataset(
        "openslr/librispeech_asr",
        config_name,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )
    return dataset.cast_column("audio", Audio(decode=False))


def _select_subset(dataset, limit: int | None, seed: int):
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(limit))


def _sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _normalize_config_and_split(config_name: str, split: str) -> tuple[str, str, str | None]:
    if config_name not in _VALID_SPLITS:
        valid_configs = ", ".join(sorted(_VALID_SPLITS))
        raise ValueError(
            f"Unsupported LibriSpeech config '{config_name}'. Valid configs: {valid_configs}."
        )

    if split in _VALID_SPLITS[config_name]:
        return config_name, split, None

    alias = _SPLIT_ALIASES.get((config_name, split))
    if alias:
        warning = (
            f"normalized split '{split}' to '{alias}' for config '{config_name}'. "
            f"Use --config {config_name} --split {alias} next time."
        )
        return config_name, alias, warning

    valid_splits = ", ".join(_VALID_SPLITS[config_name])
    hint = _split_hint(config_name, split)
    raise ValueError(
        f"Unsupported LibriSpeech split '{split}' for config '{config_name}'. "
        f"Valid splits: {valid_splits}.{hint}"
    )


def _split_hint(config_name: str, split: str) -> str:
    if "." in split and config_name != "all":
        return (
            " Dotted split names such as validation.clean belong to "
            "--config all; for --config clean use validation or test."
        )
    return ""


def _read_audio(audio: dict[str, object]):
    if "array" in audio and "sampling_rate" in audio:
        return audio["array"], int(audio["sampling_rate"])

    audio_bytes = audio.get("bytes")
    if audio_bytes is not None:
        return sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)

    audio_path = _audio_source_path(audio)
    if audio_path:
        return sf.read(audio_path, dtype="float32", always_2d=False)

    raise ValueError("Audio sample does not contain decoded data, bytes, or a path.")


def _audio_source_path(audio: dict[str, object]) -> str | None:
    path = audio.get("path")
    return str(path) if path else None
