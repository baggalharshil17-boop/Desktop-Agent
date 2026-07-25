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
    """Assembles raw microphone chunks into VAD-gated utterances.

    Noise reduction is applied once per finalized utterance (at the moment it
    is yielded), not per chunk. This is a deliberate deviation from the
    original plan's literal "PyAudio -> noisereduce -> VAD gate -> buffer"
    ordering: real `noisereduce` computes an STFT whose window is larger than
    a single VAD-sized chunk (512 samples at 16kHz), so calling
    `dsp.reduce_noise()` on one chunk raises
    `ValueError: noverlap must be less than nperseg` and cannot work at any
    real chunk size. Denoising also has no benefit for VAD scoring, since
    Silero is trained to handle real-world noisy audio directly. VAD
    therefore always scores the raw chunk, and denoising happens exactly
    once, on the fully-assembled utterance buffer, immediately before it is
    yielded -- which still satisfies the actual requirement that STT receives
    denoised audio.
    """

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
        # This event only reflects reality while `run()`'s async generator is
        # actively being iterated. Blocking inside the `async for` loop body
        # (e.g. awaiting a slow STT call before requesting the next
        # utterance) freezes the generator and therefore freezes
        # `speech_active` at its last value until iteration resumes. Phase 3
        # must drive `run()` from a dedicated background task that
        # continuously pulls utterances into an output queue -- not block
        # inside the loop body -- or barge-in detection will not work.
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
                    yield self._finalize(buffer)
                self.speech_active.clear()
                return

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
                        yield self._finalize(buffer)
                    in_utterance = False
                    buffer = bytearray()
                    speech_ms = 0.0
                    silence_ms = 0.0

    def _finalize(self, buffer: bytearray) -> bytes:
        raw = bytes(buffer)
        if self._apply_noise_reduction:
            return dsp.reduce_noise(raw, sample_rate=self._sample_rate)
        return raw
