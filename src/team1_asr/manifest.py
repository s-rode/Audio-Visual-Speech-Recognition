from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from team1_asr.config import InputConfig


@dataclass
class ManifestRow:
    utterance_id: str
    audio_path: str
    reference: str | None
    language: str | None
    video_path: str | None = None


def load_manifest(config: InputConfig) -> list[ManifestRow]:
    manifest_path = Path(config.manifest_path)
    manifest_dir = manifest_path.parent
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, row in enumerate(reader):
            audio_path = _resolve_path(row[config.audio_column], manifest_dir)
            video_path = None
            if config.video_column and config.video_column in row and row[config.video_column]:
                video_path = _resolve_path(row[config.video_column], manifest_dir)
            utterance_id = row.get(config.id_column) or f"utt-{index:05d}"
            reference = row.get(config.reference_column) or None
            language = row.get(config.language_column) or None
            rows.append(
                ManifestRow(
                    utterance_id=utterance_id,
                    audio_path=audio_path,
                    reference=reference,
                    language=language,
                    video_path=video_path,
                )
            )
    return rows


def _resolve_path(path: str, manifest_dir: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((manifest_dir / candidate).resolve())
