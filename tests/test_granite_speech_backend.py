from __future__ import annotations

import unittest
from unittest.mock import patch

from team1_asr.backends.base import TranscriptionRequest
from team1_asr.backends.factory import create_backend
from team1_asr.backends.granite_speech_backend import (
    GraniteSpeechBackend,
    _resolve_device,
)
from team1_asr.config import BackendConfig, load_config


class GraniteSpeechBackendTest(unittest.TestCase):
    def test_factory_creates_granite_backend(self) -> None:
        config = BackendConfig(
            kind="granite-speech",
            model="ibm-granite/granite-speech-4.1-2b",
        )

        with patch(
            "team1_asr.backends.granite_speech_backend._load_runtime",
            return_value=(
                _FakeTorch(),
                _FakeTorchaudio(),
                _FakeSoundfile(),
                _FakeAutoProcessor,
                _FakeAutoModel,
            ),
        ):
            backend = create_backend(config)

        self.assertIsInstance(backend, GraniteSpeechBackend)

    def test_transcribes_with_mocked_transformers_runtime(self) -> None:
        config = BackendConfig(
            kind="granite-speech",
            model="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            language="en",
        )

        with patch(
            "team1_asr.backends.granite_speech_backend._load_runtime",
            return_value=(
                _FakeTorch(),
                _FakeTorchaudio(),
                _FakeSoundfile(),
                _FakeAutoProcessor,
                _FakeAutoModel,
            ),
        ):
            backend = GraniteSpeechBackend(config)
            transcript = backend.transcribe(
                TranscriptionRequest(audio_path="audio/utt.wav", language="en")
            )

        self.assertEqual(transcript.text, "hello world")
        self.assertEqual(transcript.language, "en")
        self.assertEqual(transcript.metadata["backend"], "granite-speech")
        self.assertEqual(transcript.metadata["model"], "ibm-granite/granite-speech-4.1-2b")
        self.assertEqual(transcript.metadata["runtime"], "transformers")
        self.assertFalse(transcript.metadata["quantized"])
        self.assertEqual(_FakeTokenizer.last_prompt[0]["content"], _FakeTokenizer.prompt_text)
        self.assertEqual(_FakeModel.last_generate_kwargs["do_sample"], False)
        self.assertEqual(_FakeModel.last_generate_kwargs["num_beams"], 1)
        self.assertEqual(_FakeModel.last_generate_kwargs["max_new_tokens"], 200)

    def test_auto_device_prefers_cuda_then_mps_then_cpu(self) -> None:
        self.assertEqual(_resolve_device("auto", _FakeTorch(cuda=True)), "cuda")
        self.assertEqual(_resolve_device("auto", _FakeTorch(mps=True)), "mps")
        self.assertEqual(_resolve_device("auto", _FakeTorch()), "cpu")
        self.assertEqual(_resolve_device("cpu", _FakeTorch(cuda=True)), "cpu")

    def test_granite_configs_load_expected_model_and_noise(self) -> None:
        clean = load_config("configs/lrs3_test_granite_speech_clean.toml")
        noisy = load_config("configs/lrs3_test_granite_speech_advanced_10db.toml")

        self.assertEqual(clean.backend.kind, "granite-speech")
        self.assertEqual(clean.backend.model, "ibm-granite/granite-speech-4.1-2b")
        self.assertFalse(clean.noise.advanced_enabled)
        self.assertEqual(noisy.backend.kind, "granite-speech")
        self.assertEqual(noisy.noise.demand.snr_db, 10.0)
        self.assertEqual(noisy.noise.demand.noise_path, "DEMAND/16k")
        self.assertEqual(noisy.noise.reverb.rt60_range, [0.4, 1.0])
        self.assertEqual(noisy.noise.bandwidth.available_filters, ["butter", "bessel", "cheby1"])
        self.assertEqual(noisy.noise.bandwidth.available_cutoffs_hz, [2000, 4000, 8000])


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeMps:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeBackends:
    def __init__(self, mps: bool) -> None:
        self.mps = _FakeMps(mps)


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeTorch:
    bfloat16 = "torch.bfloat16"
    float32 = "torch.float32"

    def __init__(self, cuda: bool = False, mps: bool = False) -> None:
        self.cuda = _FakeCuda(cuda)
        self.backends = _FakeBackends(mps)

    def no_grad(self):
        return _FakeNoGrad()

    def from_numpy(self, array):
        return _FakeWaveform((array.shape[0], array.shape[1]))


class _FakeWaveform:
    def __init__(self, shape=(1, 16000)) -> None:
        self.shape = shape

    def mean(self, dim=0, keepdim=True):
        if dim == 1:
            return _FakeWaveform((self.shape[0],))
        return _FakeWaveform((self.shape[-1],))

    def unsqueeze(self, dim):
        return _FakeWaveform((1, self.shape[-1]))

    def squeeze(self, dim):
        return _FakeWaveform((self.shape[-1],))


class _FakeFunctional:
    @staticmethod
    def resample(wav, sample_rate, target_rate):
        return wav


class _FakeTorchaudio:
    functional = _FakeFunctional()


class _FakeArray:
    shape = (16000, 1)


class _FakeSoundfile:
    @staticmethod
    def read(audio_path, dtype="float32", always_2d=True):
        return _FakeArray(), 16000


class _FakeTokenizer:
    prompt_text = "<|audio|>transcribe the speech with proper punctuation and capitalization."
    last_prompt = None

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
        _FakeTokenizer.last_prompt = chat
        return "formatted prompt"

    def batch_decode(self, tokens, add_special_tokens=False, skip_special_tokens=True):
        return ["hello world"]


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def __call__(self, prompt, wav, device, return_tensors):
        return _FakeInputs()


class _FakeAutoProcessor:
    @staticmethod
    def from_pretrained(model):
        return _FakeProcessor()


class _FakeModel:
    last_generate_kwargs = {}

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def generate(self, **kwargs):
        _FakeModel.last_generate_kwargs = kwargs
        return _FakeGenerated()


class _FakeAutoModel:
    @staticmethod
    def from_pretrained(model, torch_dtype):
        return _FakeModel()


class _FakeInputIds:
    shape = (1, 3)


class _FakeInputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=_FakeInputIds(), audio="fake-audio")

    def to(self, device):
        self.device = device
        return self


class _FakeTokenSlice:
    def unsqueeze(self, dim):
        return self


class _FakeGenerated:
    def __getitem__(self, item):
        return _FakeTokenSlice()


if __name__ == "__main__":
    unittest.main()
