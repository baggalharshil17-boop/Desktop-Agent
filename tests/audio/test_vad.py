import numpy as np
import torch

from financial_voice_agent.audio.vad import SileroVadScorer


class _FakeModel:
    def __init__(self, value: float):
        self._value = value
        self.calls = []

    def __call__(self, tensor, sample_rate):
        self.calls.append((tensor, sample_rate))
        return torch.tensor(self._value)


def test_score_returns_model_output_as_float():
    fake_model = _FakeModel(0.87)
    scorer = SileroVadScorer(model=fake_model)
    pcm = np.array([0, 16384, -16384, 0], dtype=np.int16).tobytes()

    result = scorer.score(pcm)

    assert isinstance(result, float)
    assert abs(result - 0.87) < 1e-6


def test_score_calls_model_with_16000_sample_rate():
    fake_model = _FakeModel(0.1)
    scorer = SileroVadScorer(model=fake_model)

    scorer.score(b"\x00\x00" * 160)

    assert fake_model.calls[0][1] == 16000


def test_score_passes_float_tensor_derived_from_pcm():
    fake_model = _FakeModel(0.1)
    scorer = SileroVadScorer(model=fake_model)
    pcm = np.array([0, 32767], dtype=np.int16).tobytes()

    scorer.score(pcm)

    tensor_arg = fake_model.calls[0][0]
    assert isinstance(tensor_arg, torch.Tensor)
    assert tensor_arg.dtype == torch.float32
    assert abs(float(tensor_arg[1]) - 1.0) < 0.001


def test_injected_model_is_reused_across_calls_without_reloading():
    fake_model = _FakeModel(0.1)
    scorer = SileroVadScorer(model=fake_model)

    scorer.score(b"\x00\x00" * 10)
    scorer.score(b"\x00\x00" * 10)

    assert len(fake_model.calls) == 2
    assert scorer._model is fake_model
