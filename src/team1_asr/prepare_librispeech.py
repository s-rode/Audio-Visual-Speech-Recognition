from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

import soundfile as sf


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
        default="validation.clean",
        help=(
            "Dataset split to export, for example validation.clean, test.clean, "
            "or train.clean.100."
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
        help="Random seed used when --limit is set.",
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
        help="Also save the selected Hugging Face dataset subset with save_to_disk().",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifest/audio files in the output directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_librispeech_split(
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
) -> None:
    dataset = _load_dataset(config_name=config_name, split=split, cache_dir=cache_dir)
    dataset = _select_subset(dataset, limit=limit, seed=seed)

    manifest_path = output_dir / "manifest.csv"
    audio_dir = output_dir / "audio"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --force to overwrite."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index, sample in enumerate(dataset):
        utterance_id = sample.get("id") or f"{config_name}-{split}-{index:06d}"
        utterance_path = audio_dir / f"{_sanitize_filename(str(utterance_id))}.wav"
        audio = sample["audio"]
        sf.write(utterance_path, audio["array"], audio["sampling_rate"])
        rows.append(
            {
                "utterance_id": utterance_id,
                "audio_path": str(Path("audio") / utterance_path.name),
                "reference": sample["text"],
                "language": language,
                "speaker_id": sample.get("speaker_id"),
                "chapter_id": sample.get("chapter_id"),
                "source_file": sample.get("file"),
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
        f"({config_name}, split={split}) to {output_dir}. Manifest: {manifest_path}"
    )


def _load_dataset(config_name: str, split: str, cache_dir: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Run 'uv sync' to install it."
        ) from exc

    return load_dataset(
        "openslr/librispeech_asr",
        config_name,
        split=split,
        cache_dir=cache_dir,
    )


def _select_subset(dataset, limit: int | None, seed: int):
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(limit))


def _sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
