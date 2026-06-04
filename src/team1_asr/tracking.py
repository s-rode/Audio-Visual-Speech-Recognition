from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from team1_asr.config import ExperimentConfig


class WandbTracker:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._settings = config.wandb
        self._run = None
        self._wandb = None

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def start(self) -> None:
        if not self.enabled:
            return

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb tracking is enabled, but wandb is not installed. "
                "Run 'uv sync --extra tracking' or install wandb."
            ) from exc

        self._wandb = wandb
        self._run = wandb.init(
            project=self._settings.project,
            entity=self._settings.entity,
            name=self._settings.run_name or self._config.experiment_name,
            group=self._settings.group,
            mode=self._settings.mode,
            config=_wandb_config(self._config),
        )

    def log_summary(self, summary: dict[str, object], output_dir: Path) -> None:
        if not self.enabled or self._wandb is None:
            return

        self._wandb.log(_numeric_summary(summary))
        if self._settings.log_transcripts:
            transcript_path = Path(str(summary["transcript_path"]))
            self._log_transcript_table(transcript_path)
        if self._settings.log_artifacts:
            self._log_artifact(output_dir)

    def finish(self) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.finish()

    def _log_transcript_table(self, transcript_path: Path) -> None:
        if self._wandb is None or not transcript_path.exists():
            return

        table = self._wandb.Table(
            columns=["utterance_id", "reference", "prediction", "used_audio_path"]
        )
        with transcript_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= self._settings.transcript_table_limit:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                table.add_data(
                    row.get("utterance_id"),
                    row.get("reference"),
                    row.get("prediction"),
                    row.get("used_audio_path"),
                )
        self._wandb.log({"transcripts": table})

    def _log_artifact(self, output_dir: Path) -> None:
        if self._wandb is None:
            return

        artifact = self._wandb.Artifact(
            name=_artifact_name(self._config.experiment_name),
            type="asr-run",
        )
        artifact.add_dir(str(output_dir))
        self._wandb.log_artifact(artifact)


def _numeric_summary(summary: dict[str, object]) -> dict[str, float | int]:
    return {
        key: value
        for key, value in summary.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None
    }


def _wandb_config(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "backend": vars(config.backend),
        "noise": vars(config.noise),
        "input": vars(config.input),
        "output": vars(config.output),
    }


def _artifact_name(experiment_name: str) -> str:
    safe_name = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "-"
        for ch in experiment_name.lower()
    )
    return f"{safe_name}-outputs"
