from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, TextIO
import wave


_MODALITIES = {"audio", "visual", "audio_visual"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve AVEC manifest transcription requests over JSON lines."
    )
    parser.add_argument("--repo-path", default="external/AVEC")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--modality", choices=sorted(_MODALITIES), required=True)
    parser.add_argument("--model-name", default="AVEC")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--mock-transcript",
        default=None,
        help="Return this transcript without loading AVEC. Intended for tests only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        transcriber: _BaseTranscriber
        if args.mock_transcript is not None:
            transcriber = _MockTranscriber(args)
        else:
            transcriber = _AvecTranscriber(args)
    except Exception as exc:
        _write({"type": "error", "message": str(exc)})
        raise SystemExit(1) from None

    _write(
        {
            "type": "ready",
            "backend": "avec",
            "model": args.model_name,
            "modality": args.modality,
        }
    )
    _serve(transcriber, sys.stdin)


def _serve(transcriber: "_BaseTranscriber", stream: TextIO) -> None:
    for raw_line in stream:
        if not raw_line.strip():
            continue
        try:
            message = json.loads(raw_line)
            message_type = message.get("type")
            if message_type == "shutdown":
                return
            if message_type != "transcribe":
                raise ValueError(f"Unsupported message type: {message_type}")

            result = transcriber.transcribe(
                audio_path=str(message.get("audio_path") or ""),
                video_path=message.get("video_path"),
                language=message.get("language"),
                utterance_id=message.get("utterance_id"),
                video_preprocessing=message.get("video_preprocessing"),
            )
            _write({"type": "transcript", **result})
        except Exception as exc:
            _write({"type": "error", "message": str(exc)})


class _BaseTranscriber:
    def transcribe(
        self,
        audio_path: str,
        video_path: object,
        language: object,
        utterance_id: object,
        video_preprocessing: object = None,
    ) -> dict[str, object]:
        raise NotImplementedError


class _MockTranscriber(_BaseTranscriber):
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args

    def transcribe(
        self,
        audio_path: str,
        video_path: object,
        language: object,
        utterance_id: object,
        video_preprocessing: object = None,
    ) -> dict[str, object]:
        return {
            "text": self._args.mock_transcript,
            "language": language,
            "duration": None,
            "metadata": {
                "runtime": "mock",
                "audio_path": audio_path,
                "video_path": video_path,
                "utterance_id": utterance_id,
                "video_preprocessing": video_preprocessing,
            },
        }


class _AvecTranscriber(_BaseTranscriber):
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._repo_path = Path(args.repo_path).resolve()
        if not self._repo_path.exists():
            raise FileNotFoundError(
                f"AVEC repo not found at {self._repo_path}. Clone https://github.com/burchim/AVEC there first."
            )
        self._previous_cwd = Path.cwd()
        sys.path.insert(0, str(self._repo_path))
        os.chdir(self._repo_path)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                self._torch = importlib.import_module("torch")
                self._torchvision = importlib.import_module("torchvision")
                self._functions = importlib.import_module("functions")
                self._config = self._build_manifest_config()
                self._model = self._functions.load_model(self._load_args())
            self._model.eval()
        except Exception:
            os.chdir(self._previous_cwd)
            raise

    def transcribe(
        self,
        audio_path: str,
        video_path: object,
        language: object,
        utterance_id: object,
        video_preprocessing: object = None,
    ) -> dict[str, object]:
        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            raise FileNotFoundError(f"Audio path does not exist: {audio_path}")
        video_path_obj = Path(str(video_path)) if video_path else None
        if self._args.modality in {"visual", "audio_visual"}:
            if video_path_obj is None:
                raise ValueError("AVEC visual/audio_visual modes require video_path in the manifest.")
            if not video_path_obj.exists():
                raise FileNotFoundError(f"Video path does not exist: {video_path_obj}")

        sample, duration = self._build_sample(audio_path_obj, video_path_obj)
        batch = self._config.collate_fn([sample])
        inputs = self._model.transfer_to_device(batch["inputs"])

        with self._torch.no_grad(), contextlib.redirect_stdout(sys.stderr):
            outputs = self._model.forward(inputs)
            if not isinstance(outputs, dict) or "outputs" not in outputs:
                raise RuntimeError("AVEC model did not return an 'outputs' prediction.")
            if not getattr(self._model, "built", False):
                self._model.build(outputs)
            decoded = self._model.decoders["outputs"](outputs["outputs"])

        return {
            "text": _first_decoded_text(decoded),
            "language": language,
            "duration": duration,
            "metadata": {
                "runtime": "AVEC",
                "modality": self._args.modality,
                "utterance_id": utterance_id,
                "video_preprocessing": video_preprocessing or "center_crop",
            },
        }

    def _load_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            rank=0,
            config=self._config,
            config_file=self._args.config_file,
            checkpoint=self._args.checkpoint,
            load_last=False,
            cpu=self._args.cpu,
            distributed=False,
            parallel=False,
            show_dict=False,
            show_modules=False,
            world_size=1,
        )

    def _build_manifest_config(self) -> SimpleNamespace:
        nnet = importlib.import_module("nnet")
        torch = self._torch
        torchvision = self._torchvision
        tokenizer_path = "datasets/LRS3/tokenizerbpe256.model"
        checkpoint_parent = str(Path(self._args.config_file).with_suffix("")).replace(
            "configs",
            "callbacks",
            1,
        )

        if self._args.modality == "audio":
            model = nnet.AudioEfficientConformerInterCTC(
                vocab_size=256,
                att_type="patch",
                interctc_blocks=[],
            )
            model.compile(
                losses=nnet.CTCLoss(zero_infinity=True, assert_shorter=False),
                metrics=nnet.WordErrorRate(),
                decoders=nnet.CTCGreedySearchDecoder(tokenizer_path=tokenizer_path),
                loss_weights=None,
            )
            collate_fn = nnet.CollateFn(
                inputs_params=[{"axis": 1, "padding": True}, {"axis": 4}],
                targets_params=({"axis": 2, "padding": True}, {"axis": 5}),
            )
            preprocessing = SimpleNamespace(
                audio_preprocessing=torch.nn.Identity(),
                video_preprocessing=torch.nn.Identity(),
                align=False,
            )
        elif self._args.modality == "visual":
            model = nnet.VisualEfficientConformerInterCTC(
                vocab_size=256,
                interctc_blocks=[3, 6, 9],
                test_augments=None,
            )
            model.compile(
                losses=nnet.CTCLoss(zero_infinity=True, assert_shorter=False),
                decoders={"outputs": nnet.CTCGreedySearchDecoder(tokenizer_path=tokenizer_path)},
                metrics={"outputs": nnet.WordErrorRate()},
                loss_weights=[0.5 / 3, 0.5 / 3, 0.5 / 3, 0.5],
            )
            collate_fn = nnet.CollateFn(
                inputs_params=[{"axis": 0, "padding": True}, {"axis": 3}],
                targets_params=({"axis": 2, "padding": True}, {"axis": 5}),
            )
            preprocessing = SimpleNamespace(
                audio_preprocessing=torch.nn.Identity(),
                video_preprocessing=_video_preprocessing(torchvision, nnet),
                align=False,
            )
        else:
            model = nnet.AudioVisualEfficientConformerInterCTC(
                vocab_size=256,
                v_interctc_blocks=[3, 6],
                a_interctc_blocks=[8, 11],
                f_interctc_blocks=[2],
            )
            model.compile(
                losses=nnet.CTCLoss(zero_infinity=True, assert_shorter=False),
                decoders={"outputs": nnet.CTCGreedySearchDecoder(tokenizer_path=tokenizer_path)},
                metrics={"outputs": nnet.WordErrorRate()},
                loss_weights={
                    "v_ctc_2": 0.5 / 3,
                    "v_ctc_5": 0.5 / 3,
                    "a_ctc_7": 0.5 / 3,
                    "a_ctc_10": 0.5 / 3,
                    "f_ctc_1": 0.5 / 3,
                    "outputs": 0.5,
                },
            )
            collate_fn = nnet.CollateFn(
                inputs_params=[
                    {"axis": 0, "padding": True},
                    {"axis": 3},
                    {"axis": 1, "padding": True},
                    {"axis": 4},
                ],
                targets_params=({"axis": 2, "padding": True}, {"axis": 5}),
            )
            preprocessing = SimpleNamespace(
                audio_preprocessing=torch.nn.Identity(),
                video_preprocessing=_video_preprocessing(torchvision, nnet),
                align=True,
            )

        self._eval_dataset = preprocessing
        return SimpleNamespace(
            model=model,
            callback_path=checkpoint_parent,
            collate_fn=collate_fn,
        )


    def _build_sample(
        self,
        audio_path: Path,
        video_path: Path | None,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        torch = self._torch
        video = torch.empty(0)
        video_len = torch.tensor(0, dtype=torch.long)
        audio = torch.empty(0)
        audio_len = torch.tensor(0, dtype=torch.long)
        duration = None

        if self._args.modality in {"audio", "audio_visual"}:
            audio, sample_rate = _load_wav(audio_path, torch)
            duration = float(audio.shape[1] / sample_rate) if sample_rate else None
            audio_len = torch.tensor(audio.shape[1], dtype=torch.long)
            audio = self._eval_dataset.audio_preprocessing(audio).squeeze(dim=0)

        if self._args.modality in {"visual", "audio_visual"}:
            assert video_path is not None
            video = _load_video(video_path, torch)
            video = video.permute(3, 0, 1, 2)
            video = self._eval_dataset.video_preprocessing(video)
            if getattr(self._eval_dataset, "align", False):
                transforms = importlib.import_module("nnet.transforms")
                video = transforms.align_video_to_audio(video.permute(1, 2, 3, 0), audio)
            else:
                video = video.permute(1, 2, 3, 0)
            video_len = torch.tensor(video.shape[0], dtype=torch.long)

        label = torch.empty(0, dtype=torch.long)
        label_len = torch.tensor(0, dtype=torch.long)
        return (video, audio, label, video_len, audio_len, label_len), duration


def _first_decoded_text(decoded: object) -> str:
    if isinstance(decoded, str):
        return decoded.strip()
    if isinstance(decoded, list) and decoded:
        return str(decoded[0]).strip()
    return str(decoded or "").strip()


def _video_preprocessing(torchvision, nnet):
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ConvertImageDtype(dtype=importlib.import_module("torch").float32),
            nnet.Permute(dims=(1, 0, 2, 3)),
            torchvision.transforms.Grayscale(),
            nnet.Permute(dims=(1, 0, 2, 3)),
            nnet.NormalizeVideo(mean=(0.5,), std=(0.5,)),
            torchvision.transforms.CenterCrop((88, 88)),
        ]
    )


def _load_wav(path: Path, torch):
    np = importlib.import_module("numpy")
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype("float32") / 2147483648.0
    elif sample_width == 3:
        raw_bytes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        audio = (raw_bytes[:, 0].astype(np.int32) |
                 (raw_bytes[:, 1].astype(np.int32) << 8) |
                 (raw_bytes[:, 2].astype(np.int32) << 16))
        audio = np.where(audio >= 0x800000, audio - 0x1000000, audio)
        audio = audio.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sample_width} bytes: {path}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return torch.from_numpy(audio.copy()).unsqueeze(0), sample_rate


def _load_video(path: Path, torch):
    np = importlib.import_module("numpy")
    av = importlib.import_module("av")
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"Video contains no frames: {path}")
    return torch.from_numpy(np.stack(frames, axis=0))


def _write(message: dict[str, object]) -> None:
    print(json.dumps(message), flush=True)


if __name__ == "__main__":
    main()
