from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BackendConfig:
    kind: str = "faster-whisper"
    model: str = "small.en"
    device: str = "auto"
    compute_type: str = "default"
    beam_size: int = 5
    best_of: int = 5
    batch_size: int = 1
    language: str | None = "en"
    task: str = "transcribe"
    word_timestamps: bool = True
    vad_filter: bool = True
    condition_on_previous_text: bool = True
    repo_path: str = "external/AVEC"
    config_file: str = "configs/LRS23/AV/EffConfInterCTC.py"
    checkpoint: str = "checkpoints_swa-equal-60-70.ckpt"
    callback_path: str | None = None
    modality: str = "audio_visual"
    python_executable: str = "python"
    runner_module: str = "team1_asr.avec_manifest_runner"
    extra_args: list[str] = field(default_factory=list)


@dataclass
class DemandNoiseConfig:
    enabled: bool = False
    snr_db: float = 0.0
    noise_path: str = ""


@dataclass
class ReverbNoiseConfig:
    enabled: bool = False
    length_range: list[float] = field(default_factory=lambda: [5.0, 15.0])
    width_range: list[float] = field(default_factory=lambda: [5.0, 15.0])
    height_range: list[float] = field(default_factory=lambda: [2.0, 6.0])
    rt60_range: list[float] = field(default_factory=lambda: [0.4, 1.0])


@dataclass
class BandwidthNoiseConfig:
    enabled: bool = False
    available_filters: list[str] = field(
        default_factory=lambda: ["butter", "bessel", "cheby1"]
    )
    available_cutoffs_hz: list[int] = field(default_factory=lambda: [2000, 4000, 8000])
    filter_order: int = 5


@dataclass
class NoiseConfig:
    enabled: bool = False
    kind: str = "white"
    snr_db: float = 20.0
    noise_path: str | None = None
    seed: int = 0
    demand: DemandNoiseConfig = field(default_factory=DemandNoiseConfig)
    reverb: ReverbNoiseConfig = field(default_factory=ReverbNoiseConfig)
    bandwidth: BandwidthNoiseConfig = field(default_factory=BandwidthNoiseConfig)

    def __post_init__(self) -> None:
        if isinstance(self.demand, dict):
            self.demand = _load_dataclass(DemandNoiseConfig, self.demand)
        if isinstance(self.reverb, dict):
            self.reverb = _load_dataclass(ReverbNoiseConfig, self.reverb)
        if isinstance(self.bandwidth, dict):
            self.bandwidth = _load_dataclass(BandwidthNoiseConfig, self.bandwidth)

    @property
    def advanced_enabled(self) -> bool:
        return self.demand.enabled or self.reverb.enabled or self.bandwidth.enabled


@dataclass
class InputConfig:
    manifest_path: str = "data/manifest.csv"
    audio_column: str = "audio_path"
    video_column: str | None = "video_path"
    reference_column: str = "reference"
    id_column: str = "utterance_id"
    language_column: str = "language"
    video_preprocessing_column: str | None = "video_preprocessing"


@dataclass
class OutputConfig:
    output_dir: str = "outputs/latest"
    transcript_name: str = "transcripts.jsonl"
    metrics_name: str = "metrics.json"


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "team1-asr"
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    mode: str | None = None
    log_artifacts: bool = False
    log_transcripts: bool = True
    transcript_table_limit: int = 100


@dataclass
class ExperimentConfig:
    experiment_name: str = "whisper-baseline"
    backend: BackendConfig = field(default_factory=BackendConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @property
    def output_dir(self) -> Path:
        return Path(self.output.output_dir)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = _load_toml(path)
    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", "whisper-baseline"),
        backend=_load_dataclass(BackendConfig, raw.get("backend", {})),
        noise=_load_dataclass(NoiseConfig, raw.get("noise", {})),
        input=_load_dataclass(InputConfig, raw.get("input", {})),
        output=_load_dataclass(OutputConfig, raw.get("output", {})),
        wandb=_load_dataclass(WandbConfig, raw.get("wandb", {})),
    )


def _load_dataclass(cls: type[Any], data: dict[str, Any]) -> Any:
    allowed = {field.name for field in cls.__dataclass_fields__.values()}
    filtered = {key: value for key, value in data.items() if key in allowed}
    return cls(**filtered)


def _load_toml(path: str | Path) -> dict[str, Any]:
    try:
        import tomllib

        with Path(path).open("rb") as handle:
            return tomllib.load(handle)
    except ImportError:
        return _load_simple_toml(path)


def _load_simple_toml(path: str | Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current = root

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section_name = line[1:-1].strip()
            current = root
            for section_part in section_name.split("."):
                current = current.setdefault(section_part.strip(), {})
            continue

        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"Invalid config line: {raw_line}")
        current[key.strip()] = _parse_simple_value(value.strip())

    return root


def _parse_simple_value(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_value(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
