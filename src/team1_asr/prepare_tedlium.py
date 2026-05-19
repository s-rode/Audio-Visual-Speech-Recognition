from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import re
import shutil
import tarfile
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import numpy as np
import soundfile as sf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use a local or downloaded TED-LIUM archive and "
            "export a stable local WAV+manifest subset for ASR experiments."
        )
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write segmented WAV files and manifest.csv into.",
    )
    parser.add_argument(
        "--release",
        default="release3",
        choices=["release1", "release2", "release3"],
        help="TED-LIUM release to download from OpenSLR.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to export.",
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
        "--archive",
        default=None,
        help=(
            "Optional path to a local TED-LIUM archive. This can be a full release "
            "archive such as TEDLIUM_release-3.tgz or a smaller split archive such "
            "as dev.tar.gz or test.tar.gz."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional local cache directory for downloaded TED-LIUM archives.",
    )
    parser.add_argument(
        "--save-dataset",
        action="store_true",
        help="Also save metadata about the downloaded TED-LIUM source archive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifest/audio files in the output directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_tedlium_split(
        output_dir=Path(args.output_dir),
        release=args.release,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        language=args.language,
        archive=Path(args.archive) if args.archive else None,
        cache_dir=args.cache_dir,
        save_dataset=args.save_dataset,
        force=args.force,
    )


def export_tedlium_split(
    output_dir: Path,
    release: str,
    split: str,
    limit: int | None,
    seed: int,
    language: str,
    archive: Path | None,
    cache_dir: str | None,
    save_dataset: bool,
    force: bool,
) -> None:
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

    archive_path = _resolve_archive_path(
        release=release,
        archive=archive,
        cache_dir=cache_dir,
    )
    extracted_root = _extract_archive(archive_path)
    subset_root = _subset_root(
        extracted_root,
        release=release,
        split=split,
        archive_name=archive_path.name,
    )
    segments = _collect_segments(subset_root)
    segments = _select_subset(segments, limit=limit, seed=seed)

    rows: list[dict[str, object]] = []
    by_audio: dict[Path, list[dict[str, object]]] = {}
    for segment in segments:
        by_audio.setdefault(segment["audio_file"], []).append(segment)

    for audio_path, items in sorted(by_audio.items(), key=lambda item: str(item[0])):
        samples, sample_rate = sf.read(audio_path, dtype="int16")
        for segment in items:
            utterance_id = str(segment["id"])
            utterance_path = audio_dir / f"{_sanitize_filename(utterance_id)}.wav"
            utterance_audio = _extract_audio_segment(
                samples,
                sample_rate,
                float(segment["start_sec"]),
                float(segment["end_sec"]),
            )
            sf.write(utterance_path, utterance_audio, sample_rate)
            rows.append(
                {
                    "utterance_id": utterance_id,
                    "audio_path": str(Path("audio") / utterance_path.name),
                    "reference": segment["text"],
                    "language": language,
                    "speaker_id": segment["speaker_id"],
                    "gender": segment["gender"],
                    "source_file": str(audio_path),
                    "hf_release": release,
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
                "gender",
                "source_file",
                "hf_release",
                "hf_split",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if save_dataset:
        snapshot_metadata = {
            "source": "local-archive" if archive else "OpenSLR",
            "release": release,
            "split": split,
            "archive_path": str(archive_path),
            "num_utterances": len(rows),
        }
        if not archive:
            snapshot_metadata["download_url"] = _archive_url(release)
        (output_dir / "hf_snapshot_metadata.json").write_text(
            json.dumps(snapshot_metadata, indent=2),
            encoding="utf-8",
        )

    print(
        f"Exported {len(rows)} utterances from TED-LIUM "
        f"({release}, split={split}) to {output_dir}. Manifest: {manifest_path}"
    )


def _resolve_archive_path(
    release: str,
    archive: Path | None,
    cache_dir: str | None,
) -> Path:
    if archive is not None:
        archive = archive.expanduser().resolve()
        if not archive.exists():
            raise FileNotFoundError(f"TED-LIUM archive does not exist: {archive}")
        return archive
    return _download_archive(release=release, cache_dir=cache_dir)


def _download_archive(release: str, cache_dir: str | None) -> Path:
    root_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "team1_tedlium"
    download_dir = root_dir / release
    download_dir.mkdir(parents=True, exist_ok=True)

    url = _archive_url(release)
    destination = download_dir / Path(url).name
    if destination.exists():
        return destination
    try:
        urlretrieve(url, destination)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(
            f"Failed to download TED-LIUM archive from {url}."
        ) from exc
    return destination


def _release_base_dir(release: str) -> str:
    if release == "release1":
        return "TEDLIUM_release1"
    if release == "release2":
        return "TEDLIUM_release2"
    if release == "release3":
        return "TEDLIUM_release-3"
    raise ValueError(f"Unsupported TED-LIUM release: {release}")


def _archive_url(release: str) -> str:
    if release == "release1":
        return "http://www.openslr.org/resources/7/TEDLIUM_release1.tar.gz"
    if release == "release2":
        return "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz"
    if release == "release3":
        return "http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz"
    raise ValueError(f"Unsupported TED-LIUM release: {release}")


def _extract_archive(archive_path: Path) -> Path:
    extracted_root = Path(tempfile.mkdtemp(prefix="tedlium_extract_"))
    with tarfile.open(archive_path, "r:*") as tar_handle:
        tar_handle.extractall(extracted_root)
    return extracted_root


def _subset_root(
    extracted_root: Path,
    release: str,
    split: str,
    archive_name: str,
) -> Path:
    direct_root = _detect_direct_subset_root(extracted_root, split=split, archive_name=archive_name)
    if direct_root is not None:
        return direct_root

    base_dir = extracted_root / _release_base_dir(release)
    if release == "release3":
        if split == "train":
            return base_dir / "data" / "train"
        if split == "validation":
            return base_dir / "legacy" / "dev"
        if split == "test":
            return base_dir / "legacy" / "test"
    split_name = "dev" if split == "validation" else split
    return base_dir / split_name


def _detect_direct_subset_root(
    extracted_root: Path,
    split: str,
    archive_name: str,
) -> Path | None:
    split_name = "dev" if split == "validation" else split

    candidates = [
        extracted_root / split_name,
        extracted_root / archive_name.removesuffix(".tar.gz").removesuffix(".tgz"),
    ]
    for candidate in candidates:
        if (candidate / "stm").is_dir() and (candidate / "sph").is_dir():
            return candidate

    if (extracted_root / "stm").is_dir() and (extracted_root / "sph").is_dir():
        return extracted_root
    return None


def _collect_segments(subset_root: Path) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    stm_dir = subset_root / "stm"
    sph_dir = subset_root / "sph"
    for stm_path in sorted(stm_dir.glob("*.stm")):
        speaker_file = stm_path.stem
        audio_file = sph_dir / f"{speaker_file}.sph"
        with stm_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith(";;"):
                    continue
                fn, channel, speaker, start, end, label, transcript = line.split(" ", 6)
                transcript = _maybe_trim_suffix(transcript)
                if speaker_file != fn:
                    speaker_file = fn
                    audio_file = sph_dir / f"{speaker_file}.sph"
                segments.append(
                    {
                        "id": "-".join([speaker, start, end, label]),
                        "audio_file": audio_file,
                        "text": transcript,
                        "speaker_id": speaker,
                        "gender": _parse_gender(label),
                        "start_sec": float(start),
                        "end_sec": float(end),
                    }
                )
    if not segments:
        raise RuntimeError(f"No TED-LIUM segments found under {subset_root}")
    return segments


def _select_subset(segments: list[dict[str, object]], limit: int | None, seed: int):
    if limit is None or limit >= len(segments):
        return segments
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(segments)), limit))
    return [segments[index] for index in indices]


def _extract_audio_segment(
    samples: np.ndarray, sample_rate: int, start_sec: float, end_sec: float
):
    start_sample = int(start_sec * sample_rate)
    end_sample = min(int(end_sec * sample_rate), samples.shape[0])
    return samples[start_sample:end_sample]


def _maybe_trim_suffix(transcript: str) -> str:
    parts = transcript.rsplit(" ", 1)
    transcript = parts[0]
    if len(parts) > 1 and not parts[-1].startswith("("):
        transcript += " " + parts[-1]
    return transcript.strip()


def _parse_gender(label: str) -> str:
    gender = re.split(",|_", label)[-1][:-1]
    if gender == "M":
        return "male"
    if gender == "F":
        return "female"
    return "unknown"


def _sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
