from __future__ import annotations

from typing import Any

from team1_asr.backends.base import (
    ASRBackend,
    TranscriptionRequest,
    require_dependency,
)
from team1_asr.config import BackendConfig
from team1_asr.types import Segment, Transcript

_GRANITE_PROMPT = "<|audio|>transcribe the speech with proper punctuation and capitalization."
_SAMPLE_RATE = 16_000


class GraniteSpeechBackend(ASRBackend):
    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        (
            self._torch,
            self._torchaudio,
            self._soundfile,
            auto_processor,
            auto_model,
        ) = _load_runtime()
        self._device = _resolve_device(config.device, self._torch)
        self._torch_dtype = _dtype_for_device(self._device, self._torch)
        self._processor = auto_processor.from_pretrained(config.model)
        self._tokenizer = self._processor.tokenizer
        self._model = auto_model.from_pretrained(
            config.model,
            torch_dtype=self._torch_dtype,
        ).eval()
        self._model.to(self._device)

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        wav = self._load_audio(request.audio_path)
        prompt = self._build_prompt()
        model_inputs = self._processor(
            prompt,
            wav,
            device=self._device,
            return_tensors="pt",
        )
        model_inputs = _move_inputs_to_device(model_inputs, self._device)

        with self._torch.no_grad():
            model_outputs = self._model.generate(
                **model_inputs,
                max_new_tokens=200,
                do_sample=False,
                num_beams=1,
            )

        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
        text = self._tokenizer.batch_decode(
            new_tokens,
            add_special_tokens=False,
            skip_special_tokens=True,
        )[0].strip()

        return Transcript(
            text=text,
            language=request.language or self._config.language,
            duration=_duration_seconds(wav, self._torch),
            segments=[Segment(start=0.0, end=_duration_seconds(wav, self._torch), text=text)],
            metadata={
                "backend": "granite-speech",
                "model": self._config.model,
                "runtime": "transformers",
                "quantized": False,
                "device": self._device,
                "torch_dtype": str(self._torch_dtype),
            },
        )

    def _build_prompt(self) -> str:
        chat = [{"role": "user", "content": _GRANITE_PROMPT}]
        return self._tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _load_audio(self, audio_path: str) -> Any:
        wav, sample_rate = self._soundfile.read(audio_path, dtype="float32", always_2d=True)
        wav = self._torch.from_numpy(wav).mean(dim=1)
        wav = wav.unsqueeze(0)
        if sample_rate != _SAMPLE_RATE:
            wav = self._torchaudio.functional.resample(wav, sample_rate, _SAMPLE_RATE)
        return wav.squeeze(0)


def _load_runtime():
    try:
        import torch
        import torchaudio
        import soundfile
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    except ImportError as exc:
        require_dependency("transformers, torch, torchaudio and soundfile", exc)

    return torch, torchaudio, soundfile, AutoProcessor, AutoModelForSpeechSeq2Seq


def _resolve_device(configured_device: str, torch_module) -> str:
    if configured_device != "auto":
        return configured_device
    if torch_module.cuda.is_available():
        return "cuda"
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _dtype_for_device(device: str, torch_module):
    if device == "cuda":
        return torch_module.bfloat16
    return torch_module.float32


def _move_inputs_to_device(model_inputs, device: str):
    move = getattr(model_inputs, "to", None)
    if callable(move):
        return move(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }


def _duration_seconds(wav, torch_module) -> float | None:
    try:
        return float(wav.shape[-1]) / _SAMPLE_RATE
    except (AttributeError, TypeError, ValueError):
        return None
