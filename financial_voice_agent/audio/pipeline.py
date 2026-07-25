from __future__ import annotations

import asyncio
from typing import AsyncIterator, Protocol

from financial_voice_agent.audio import dsp

SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM


class VadScorer(Protocol):
    def score(self, pcm_chunk: bytes) -> float: ...


def _chunk_duration_ms(chunk: bytes, sample_rate: int) -> float:
    num_samples = len(chunk) / SAMPLE_WIDTH_BYTES
    return (num_samples / sample_rate) * 1000.0


class AudioPipeline:
    def __init__(
        self,
        queue,
        vad_scorer: VadScorer,
        *,
        sample_rate: int = 16000,
        speech_threshold: float = 0.5,
        silence_duration_ms: float = 600.0,
        min_speech_duration_ms: float = 200.0,
        apply_noise_reduction: bool = True,
    ) -> None:
        self._queue = queue
        self._vad_scorer = vad_scorer
        self._sample_rate = sample_rate
        self._speech_threshold = speech_threshold
        self._silence_duration_ms = silence_duration_ms
        self._min_speech_duration_ms = min_speech_duration_ms
        self._apply_noise_reduction = apply_noise_reduction
        self.speech_active = asyncio.Event()

    async def run(self) -> AsyncIterator[bytes]:
        buffer = bytearray()
        speech_ms = 0.0
        silence_ms = 0.0
        in_utterance = False

        while True:
            chunk = await self._queue.get()
            if chunk is None:
                if in_utterance and speech_ms >= self._min_speech_duration_ms:
                    yield bytes(buffer)
                self.speech_active.clear()
                return

            if self._apply_noise_reduction:
                chunk = dsp.reduce_noise(chunk, sample_rate=self._sample_rate)

            score = self._vad_scorer.score(chunk)
            is_speech = score >= self._speech_threshold
            duration_ms = _chunk_duration_ms(chunk, self._sample_rate)

            if is_speech:
                if not in_utterance:
                    in_utterance = True
                    buffer = bytearray()
                    speech_ms = 0.0
                    silence_ms = 0.0
                if not self.speech_active.is_set():
                    self.speech_active.set()
                buffer.extend(chunk)
                speech_ms += duration_ms
                silence_ms = 0.0
            elif in_utterance:
                buffer.extend(chunk)
                silence_ms += duration_ms
                if silence_ms > self._silence_duration_ms:
                    self.speech_active.clear()
                    if speech_ms >= self._min_speech_duration_ms:
                        yield bytes(buffer)
                    in_utterance = False
                    buffer = bytearray()
                    speech_ms = 0.0
                    silence_ms = 0.0
