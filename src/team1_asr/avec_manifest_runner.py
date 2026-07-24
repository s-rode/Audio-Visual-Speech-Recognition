from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, TextIO
import wave

from team1_asr.transformer_rescoring import (
    RerankSettings,
    RobertaPseudoLikelihoodRescorer,
)


_MODALITIES = {"audio", "visual", "audio_visual"}
_DECODERS = {"greedy", "flashlight_kenlm", "flashlight_kenlm_roberta"}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROTOCOL_OUTPUT = sys.stdout
ProgressSink = Callable[[dict[str, object]], None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve AVEC manifest transcription requests over JSON lines."
    )
    parser.add_argument("--repo-path", default="external/AVEC")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--callback-path",
        default=None,
        help="Override the AVEC callback directory used to resolve the checkpoint.",
    )
    parser.add_argument("--modality", choices=sorted(_MODALITIES), required=True)
    parser.add_argument("--model-name", default="AVEC")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--decoder", choices=sorted(_DECODERS), default="greedy")
    parser.add_argument("--lm-manifest")
    parser.add_argument("--kenlm-path")
    parser.add_argument("--lexicon-unigrams")
    parser.add_argument("--decoder-asset-dir", default=".tmp/avec_live_decoder")
    parser.add_argument("--lm-order", type=int, default=3)
    parser.add_argument("--lm-discount", type=float, default=0.60)
    parser.add_argument("--lm-weight", type=float, default=1.15)
    parser.add_argument("--word-bonus", type=float, default=1.10)
    parser.add_argument("--beam-size", type=int, default=500)
    parser.add_argument("--beam-token-cutoff", type=int, default=50)
    parser.add_argument("--beam-threshold", type=float, default=50.0)
    parser.add_argument("--nbest", type=int, default=100)
    parser.add_argument("--max-words", type=int, default=200000)
    parser.add_argument("--rerank-top-k", type=int, default=50)
    parser.add_argument("--masked-lm", default="FacebookAI/roberta-base")
    parser.add_argument("--masked-lm-device", default="auto")
    parser.add_argument("--masked-lm-batch-size", type=int, default=32)
    parser.add_argument("--masked-lm-cache-dir", default=".tmp/hf")
    parser.add_argument("--transformer-weight", type=float, default=0.16)
    parser.add_argument("--rescore-word-bonus", type=float, default=1.30)
    parser.add_argument("--drop-penalty", type=float, default=-1.0)
    parser.add_argument("--drop-filled-pauses", action="store_true")
    parser.add_argument("--candidate-preview", type=int, default=5)
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
                emit=lambda event: _write({"type": "progress", "event": event}),
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
        emit: ProgressSink | None = None,
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
        emit: ProgressSink | None = None,
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
        self._lm_manifest: Path | None = None
        self._kenlm_path: Path | None = None
        self._lexicon_unigrams: Path | None = None
        self._decoder_asset_dir: Path | None = None
        self._masked_lm_cache_dir: Path | None = None
        if args.decoder != "greedy":
            self._lm_manifest = _required_project_file(
                args.lm_manifest,
                "--lm-manifest",
            )
            self._kenlm_path = _required_project_file(
                args.kenlm_path,
                "--kenlm-path",
            )
            self._lexicon_unigrams = _required_project_file(
                args.lexicon_unigrams,
                "--lexicon-unigrams",
            )
            self._decoder_asset_dir = _project_path(args.decoder_asset_dir)
            self._masked_lm_cache_dir = _project_path(args.masked_lm_cache_dir)
        if not self._repo_path.exists():
            raise FileNotFoundError(
                f"AVEC repo not found at {self._repo_path}. "
                "Clone https://github.com/burchim/AVEC there first."
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
                self._decoder, self._rescorer = self._build_decoding_stack()
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
        emit: ProgressSink | None = None,
    ) -> dict[str, object]:
        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            raise FileNotFoundError(f"Audio path does not exist: {audio_path}")
        video_path_obj = Path(str(video_path)) if video_path else None
        if self._args.modality in {"visual", "audio_visual"}:
            if video_path_obj is None:
                raise ValueError(
                    "AVEC visual/audio_visual modes require video_path in the manifest."
                )
            if not video_path_obj.exists():
                raise FileNotFoundError(f"Video path does not exist: {video_path_obj}")

        sink = emit or (lambda event: None)
        sample, duration = self._build_sample(audio_path_obj, video_path_obj)
        batch = self._config.collate_fn([sample])
        inputs = self._model.transfer_to_device(batch["inputs"])

        with self._torch.no_grad(), contextlib.redirect_stdout(sys.stderr):
            outputs = self._model.forward(inputs)
            if not isinstance(outputs, dict) or "outputs" not in outputs:
                raise RuntimeError("AVEC model did not return an 'outputs' prediction.")
            if not getattr(self._model, "built", False):
                self._model.build(outputs)
            text, decoder_metadata = self._decode(outputs["outputs"], sink)

        return {
            "text": text,
            "language": language,
            "duration": duration,
            "metadata": {
                "runtime": "AVEC",
                "modality": self._args.modality,
                "utterance_id": utterance_id,
                "video_preprocessing": video_preprocessing or "center_crop",
                **decoder_metadata,
            },
        }

    def _build_decoding_stack(self):
        if self._args.decoder == "greedy":
            return None, None
        assert self._lm_manifest is not None
        assert self._kenlm_path is not None
        assert self._lexicon_unigrams is not None
        assert self._decoder_asset_dir is not None
        ctc_decoding = _load_ctc_decoding_module()
        word_lm = ctc_decoding.KneserNeyWordNGramLM(
            manifest_path=str(self._lm_manifest),
            order=self._args.lm_order,
            discount=self._args.lm_discount,
        )
        decoder = ctc_decoding.FlashlightCTCDecoder(
            tokenizer_path=str(
                self._repo_path / "datasets/LRS3/tokenizerbpe256.model"
            ),
            lm=word_lm,
            beam_size=self._args.beam_size,
            cutoff_top_n=self._args.beam_token_cutoff,
            temperature=1.0,
            blank_bias=0.0,
            lm_weight=self._args.lm_weight,
            word_bonus=self._args.word_bonus,
            asset_dir=str(self._decoder_asset_dir),
            kenlm_path=str(self._kenlm_path),
            lexicon_unigram_path=str(self._lexicon_unigrams),
            max_words=self._args.max_words,
            beam_threshold=self._args.beam_threshold,
            log_add=False,
            nbest=self._args.nbest,
        )
        rescorer = None
        if self._args.decoder == "flashlight_kenlm_roberta":
            assert self._masked_lm_cache_dir is not None
            rescorer = RobertaPseudoLikelihoodRescorer(
                model_name=self._args.masked_lm,
                device=self._args.masked_lm_device,
                batch_size=self._args.masked_lm_batch_size,
                cache_dir=self._masked_lm_cache_dir,
            )
        return decoder, rescorer

    def _decode(
        self,
        outputs: object,
        emit: ProgressSink,
    ) -> tuple[str, dict[str, object]]:
        if self._decoder is None:
            decoded = self._model.decoders["outputs"](outputs)
            return _first_decoded_text(decoded), {"decoder": "greedy"}
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise RuntimeError("AVEC CTC outputs must contain logits and lengths.")

        emit(
            _progress(
                "beam",
                "running",
                f"Decoding a {self._args.nbest}-best list with the VoxMM 3-gram LM.",
            )
        )
        beam_started = time.perf_counter()
        candidate_batches = self._decoder.decode_candidates(outputs[0], outputs[1])
        if len(candidate_batches) != 1 or not candidate_batches[0]:
            raise RuntimeError("Flashlight returned no hypotheses for the live sample.")
        candidates = candidate_batches[0]
        beam_seconds = time.perf_counter() - beam_started
        first_pass_text = str(candidates[0].get("text") or "").strip()
        emit(
            _progress(
                "beam",
                "complete",
                f"Generated {len(candidates)} CTC hypotheses.",
                transcript=first_pass_text,
                nbest_size=len(candidates),
                elapsed_seconds=beam_seconds,
            )
        )

        metadata: dict[str, object] = {
            "decoder": self._args.decoder,
            "first_pass_transcript": first_pass_text,
            "nbest_size": len(candidates),
            "beam_seconds": beam_seconds,
            "lm_weight": self._args.lm_weight,
            "first_pass_word_bonus": self._args.word_bonus,
        }
        if self._rescorer is None:
            metadata["top_candidates"] = _candidate_preview(
                candidates,
                self._args.candidate_preview,
            )
            return first_pass_text, metadata

        emit(
            _progress(
                "rescore",
                "running",
                f"Reranking the top {self._args.rerank_top_k} hypotheses with RoBERTa.",
            )
        )
        rerank_started = time.perf_counter()
        settings = RerankSettings(
            top_k=self._args.rerank_top_k,
            transformer_weight=self._args.transformer_weight,
            word_bonus=self._args.rescore_word_bonus,
            drop_penalty=self._args.drop_penalty,
        )
        dropped_words = None if self._args.drop_filled_pauses else ()
        result = self._rescorer.rerank(
            candidates,
            settings,
            **({} if dropped_words is None else {"dropped_words": dropped_words}),
        )
        rerank_seconds = time.perf_counter() - rerank_started
        emit(
            _progress(
                "rescore",
                "complete",
                f"RoBERTa selected beam rank {result.selected_rank}.",
                transcript=result.text,
                first_pass_transcript=result.first_pass_text,
                selected_rank=result.selected_rank,
                elapsed_seconds=rerank_seconds,
            )
        )
        metadata.update(
            {
                "reranker": self._args.masked_lm,
                "rerank_top_k": self._args.rerank_top_k,
                "rerank_seconds": rerank_seconds,
                "transformer_weight": self._args.transformer_weight,
                "second_pass_word_bonus": self._args.rescore_word_bonus,
                "drop_penalty": self._args.drop_penalty,
                "selected_rank": result.selected_rank,
                "selected_score": result.selected_score,
                "top_candidates": _candidate_preview(
                    result.candidates,
                    self._args.candidate_preview,
                ),
            }
        )
        return result.text, metadata

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
        checkpoint_parent = self._args.callback_path or str(
            Path(self._args.config_file).with_suffix("")
        ).replace("configs", "callbacks", 1)

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


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _required_project_file(value: str | None, option: str) -> Path:
    if not value:
        raise ValueError(f"{option} is required for non-greedy AVEC decoding.")
    path = _project_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{option} does not exist: {path}")
    return path


def _load_ctc_decoding_module():
    module_name = "_team1_voxmm_ctc_decoding"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = _PROJECT_ROOT / "configs/VoxMM/ctc_decoding.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CTC decoder module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _progress(
    stage: str,
    status: str,
    message: str,
    **values: object,
) -> dict[str, object]:
    return {"stage": stage, "status": status, "message": message, **values}


def _candidate_preview(
    candidates: list[dict[str, object]],
    limit: int,
) -> list[dict[str, object]]:
    fields = (
        "rank",
        "text",
        "normalized_text",
        "score",
        "transformer_lm_score",
        "combined_score",
    )
    return [
        {name: candidate[name] for name in fields if name in candidate}
        for candidate in candidates[: max(0, limit)]
    ]


def _first_decoded_text(decoded: object) -> str:
    if isinstance(decoded, str):
        return decoded.strip()
    if isinstance(decoded, list) and decoded:
        return str(decoded[0]).strip()
    return str(decoded or "").strip()


def _video_preprocessing(torchvision, nnet):
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ConvertImageDtype(
                dtype=importlib.import_module("torch").float32
            ),
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
    _PROTOCOL_OUTPUT.write(json.dumps(message) + "\n")
    _PROTOCOL_OUTPUT.flush()


if __name__ == "__main__":
    main()
