from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import re
import subprocess
import wave

from team1_asr.backends.base import ASRBackend, TranscriptionRequest
from team1_asr.config import BackendConfig
from team1_asr.types import Segment, Transcript


class MmsLlamaBackend(ASRBackend):
    """Run MMS-LLaMA as an external Fairseq/Hydra evaluation process."""

    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        self._repo_path = Path(config.repo_path).resolve()
        self._src_path = self._repo_path / "src"
        self._fairseq_path = self._repo_path / "fairseq"
        self._noise_wav = Path(config.noise_wav).resolve()
        self._python_executable = _resolve_executable(config.python_executable)
        if not self._repo_path.exists():
            raise FileNotFoundError(
                f"MMS-LLaMA repo not found: {self._repo_path}. Clone it or set backend.repo_path."
            )
        if not (self._src_path / "eval.py").exists():
            raise FileNotFoundError(f"MMS-LLaMA eval.py not found under {self._src_path}.")

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        if not request.video_path:
            raise ValueError("MMS-LLaMA backend requires a manifest video_path for each row.")
        if not request.reference:
            raise ValueError("MMS-LLaMA backend requires reference text to write test.wrd.")

        utterance_id = _safe_id(request.utterance_id or Path(request.audio_path).stem)
        run_dir = self._run_dir(utterance_id)
        manifest_dir = run_dir / "manifest"
        results_dir = run_dir / "results"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        audio_path = Path(request.audio_path).resolve()
        video_path = Path(request.video_path).resolve()
        frame_count = _video_frame_count(video_path)
        sample_count = _audio_sample_count(audio_path)
        speech_rate = _speech_rate(request.reference, sample_count)
        _write_mms_manifest(
            manifest_dir=manifest_dir,
            utterance_id=utterance_id,
            video_path=video_path,
            audio_path=audio_path,
            frame_count=frame_count,
            sample_count=sample_count,
            speech_rate=speech_rate,
            reference=request.reference,
        )

        command = self._command(manifest_dir, results_dir)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self._fairseq_path)
        if self._config.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self._config.cuda_visible_devices

        completed = subprocess.run(
            command,
            cwd=self._repo_path,
            env=env,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"MMS-LLaMA eval failed for {utterance_id}: "
                f"{_format_eval_error(details, self._python_executable)}"
            )

        prediction = _read_prediction(results_dir, utterance_id)
        return Transcript(
            text=prediction,
            language=request.language or self._config.language,
            duration=sample_count / 16000,
            segments=[Segment(start=0.0, end=sample_count / 16000, text=prediction)],
            metadata={
                "backend": "mms-llama",
                "model": self._config.model,
                "repo_path": str(self._repo_path),
                "checkpoint_path": self._config.checkpoint_path,
                "llm_path": self._config.llm_path,
                "manifest_dir": str(manifest_dir),
                "results_dir": str(results_dir),
                "modalities": self._config.modalities,
            },
        )

    def _run_dir(self, utterance_id: str) -> Path:
        root = Path(self._config.results_dir or "outputs/mms_llama").resolve()
        return root / utterance_id

    def _command(self, manifest_dir: Path, results_dir: Path) -> list[str]:
        modalities = "[" + ",".join(f"'{item}'" for item in self._config.modalities) + "]"
        command = [
            self._python_executable,
            "-B",
            str(self._src_path / "eval.py"),
            "--config-dir",
            str(self._src_path / "conf"),
            "--config-name",
            "s2s_decode",
            "dataset.gen_subset=test",
            "common.user_dir=" + str(self._src_path),
            "generation.beam=" + str(self._config.beam_size),
            "generation.temperature=0.3",
            "override.llm_path=" + self._config.llm_path,
            "override.modalities=" + modalities,
            "common_eval.path=" + str(Path(self._config.checkpoint_path).resolve()),
            "common_eval.results_path=" + str(results_dir),
            "override.label_dir=" + str(manifest_dir),
            "override.data=" + str(manifest_dir),
            "override.noise_wav=" + str(self._noise_wav),
            "override.noise_prob=0",
        ]
        command.extend(self._config.extra_args)
        return command


def _resolve_executable(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if len(path.parts) > 1:
        return str(Path.cwd() / path)
    return value


def _format_eval_error(details: str, python_executable: str) -> str:
    missing = re.search(r"ModuleNotFoundError: No module named '([^']+)'", details)
    if missing:
        package_name = missing.group(1)
        return (
            f"missing Python module '{package_name}' in the MMS-LLaMA runtime. "
            f"The backend is using python_executable={python_executable!r}; set "
            "[backend].python_executable to the Python inside the MMS-LLaMA environment "
            "or install MMS-LLaMA's dependencies there. Original traceback follows:\n"
            f"{details}"
        )
    if "Cannot access gated repo" in details or "GatedRepoError" in details:
        return (
            "MMS-LLaMA could not access the gated LLaMA model on Hugging Face. "
            "Request access to meta-llama/Llama-3.2-3B, authenticate with "
            "`external/MMS-LLaMA/.venv/bin/huggingface-cli login`, or set "
            "[backend].llm_path to a local downloaded LLaMA-3.2-3B directory. "
            "Original traceback follows:\n"
            f"{details}"
        )
    return details


def _write_mms_manifest(
    manifest_dir: Path,
    utterance_id: str,
    video_path: Path,
    audio_path: Path,
    frame_count: int,
    sample_count: int,
    speech_rate: float,
    reference: str,
) -> None:
    (manifest_dir / "test.tsv").write_text(
        "/\n"
        + "\t".join(
            [
                utterance_id,
                str(video_path),
                str(audio_path),
                str(frame_count),
                str(sample_count),
                f"{speech_rate:.2f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (manifest_dir / "test.wrd").write_text(reference.strip() + "\n", encoding="utf-8")


def _read_prediction(results_dir: Path, utterance_id: str) -> str:
    candidates = sorted(
        glob.glob(str(results_dir / "hypo-*.json")),
        key=lambda item: Path(item).stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"MMS-LLaMA did not write a hypo-*.json file in {results_dir}.")
    payload = json.loads(Path(candidates[0]).read_text(encoding="utf-8"))
    utt_ids = payload.get("utt_id", [])
    hypos = payload.get("hypo", [])
    for item_id, hypo in zip(utt_ids, hypos):
        if str(item_id) == utterance_id:
            return str(hypo).strip()
    if hypos:
        return str(hypos[0]).strip()
    raise RuntimeError(f"MMS-LLaMA hypothesis file contains no predictions: {candidates[0]}")


def _audio_sample_count(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != 16000:
            raise ValueError(f"MMS-LLaMA expects 16 kHz WAV audio, got {handle.getframerate()} Hz: {path}")
        if handle.getnchannels() != 1:
            raise ValueError(f"MMS-LLaMA expects mono WAV audio, got {handle.getnchannels()} channels: {path}")
        return handle.getnframes()


def _video_frame_count(path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode == 0:
        value = completed.stdout.strip().splitlines()
        if value and value[-1].isdigit():
            return int(value[-1])
    raise RuntimeError(
        f"Could not determine frame count for {path}. Install ffprobe or provide valid video."
    )


def _speech_rate(reference: str, sample_count: int) -> float:
    duration = max(sample_count / 16000, 1e-6)
    return len(reference.split()) / duration


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "utt"
