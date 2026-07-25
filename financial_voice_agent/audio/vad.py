from __future__ import annotations

import torch

from financial_voice_agent.audio import dsp


SILERO_REQUIRED_SAMPLES = 512


class SileroVadScorer:
    def __init__(self, model=None) -> None:
        self._model = model

    def _ensure_model(self):
        if self._model is None:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
        return self._model

    def score(self, pcm_chunk: bytes) -> float:
        model = self._ensure_model()
        float_arr = dsp.pcm_to_float32(pcm_chunk)
        if len(float_arr) != SILERO_REQUIRED_SAMPLES:
            raise ValueError(
                f"Silero VAD requires exactly {SILERO_REQUIRED_SAMPLES} samples at 16kHz, "
                f"got {len(float_arr)}"
            )
        tensor = torch.from_numpy(float_arr)
        with torch.no_grad():
            output = model(tensor, 16000)
        return float(output.item())
