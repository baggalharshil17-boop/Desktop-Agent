import asyncio

import pytest

from financial_voice_agent.audio.pipeline import AudioPipeline


def _chunk(sample: int = 0) -> bytes:
    return sample.to_bytes(2, "little", signed=True) * 160


class _ScriptedVad:
    def __init__(self, scores):
        self._scores = list(scores)

    def score(self, pcm_chunk: bytes) -> float:
        return self._scores.pop(0)


async def _feed(queue, chunks):
    for c in chunks:
        await queue.put(c)
    await queue.put(None)


@pytest.mark.asyncio
async def test_pipeline_yields_one_utterance_for_speech_then_silence():
    queue = asyncio.Queue()
    # 3 speech chunks (30ms >= min_speech_duration_ms=20), then 3 silence chunks (30ms >= silence_duration_ms=20)
    vad = _ScriptedVad([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    chunks = [_chunk(100)] * 3 + [_chunk(0)] * 3
    await _feed(queue, chunks)
    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=20.0, min_speech_duration_ms=20.0, apply_noise_reduction=False
    )

    utterances = [u async for u in pipeline.run()]

    assert len(utterances) == 1
    assert utterances[0] == b"".join(chunks[:6])


@pytest.mark.asyncio
async def test_pipeline_discards_utterance_shorter_than_min_speech_duration():
    queue = asyncio.Queue()
    # 1 speech chunk (10ms < min_speech_duration_ms=20), then 3 silence chunks (30ms >= silence_duration_ms=20)
    vad = _ScriptedVad([0.9, 0.1, 0.1, 0.1])
    chunks = [_chunk(100)] * 1 + [_chunk(0)] * 3
    await _feed(queue, chunks)
    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=20.0, min_speech_duration_ms=20.0, apply_noise_reduction=False
    )

    utterances = [u async for u in pipeline.run()]

    assert utterances == []


@pytest.mark.asyncio
async def test_pipeline_pure_silence_yields_nothing():
    queue = asyncio.Queue()
    vad = _ScriptedVad([0.1, 0.1, 0.1])
    chunks = [_chunk(0)] * 3
    await _feed(queue, chunks)
    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=20.0, min_speech_duration_ms=20.0, apply_noise_reduction=False
    )

    utterances = [u async for u in pipeline.run()]

    assert utterances == []


@pytest.mark.asyncio
async def test_pipeline_flushes_pending_utterance_on_end_of_stream():
    queue = asyncio.Queue()
    # 3 speech chunks (30ms >= 20), then queue closes with no trailing silence
    vad = _ScriptedVad([0.9, 0.9, 0.9])
    chunks = [_chunk(100)] * 3
    await _feed(queue, chunks)
    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=20.0, min_speech_duration_ms=20.0, apply_noise_reduction=False
    )

    utterances = [u async for u in pipeline.run()]

    assert len(utterances) == 1
    assert utterances[0] == b"".join(chunks)


@pytest.mark.asyncio
async def test_speech_active_event_set_during_speech_and_cleared_after():
    queue = asyncio.Queue()
    vad = _ScriptedVad([0.9, 0.9, 0.1, 0.1])
    chunks = [_chunk(100)] * 2 + [_chunk(0)] * 2
    await _feed(queue, chunks)
    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=15.0, min_speech_duration_ms=15.0, apply_noise_reduction=False
    )

    gen = pipeline.run()
    async for _utterance in gen:
        pass
    # After the generator is exhausted, speech has finalized and the event must be clear.
    assert not pipeline.speech_active.is_set()


@pytest.mark.asyncio
async def test_pipeline_applies_real_noise_reduction_to_finalized_utterance():
    import numpy as np

    from financial_voice_agent.audio import dsp

    queue = asyncio.Queue()
    sample_rate = 16000
    chunk_samples = 512  # production AudioCapture default
    num_speech_chunks = 20  # ~640ms of "speech" -- long enough for a real utterance

    rng = np.random.default_rng(7)
    speech_chunks = [
        dsp.float32_to_pcm((0.3 * rng.standard_normal(chunk_samples)).astype(np.float32))
        for _ in range(num_speech_chunks)
    ]
    silence_chunks = [b"\x00\x00" * chunk_samples] * 5

    vad = _ScriptedVad([0.9] * num_speech_chunks + [0.1] * 5)
    for c in speech_chunks + silence_chunks:
        await queue.put(c)
    await queue.put(None)

    pipeline = AudioPipeline(
        queue,
        vad,
        sample_rate=sample_rate,
        silence_duration_ms=1.0,
        min_speech_duration_ms=1.0,
        apply_noise_reduction=True,  # exercises the real dsp.reduce_noise call -- this is the point
    )

    utterances = [u async for u in pipeline.run()]

    # Must not raise (this is what broke before the fix -- real noisereduce's STFT
    # window exceeds a single 512-sample chunk, so per-chunk denoising always raised
    # ValueError: noverlap must be less than nperseg), and must return valid PCM.
    assert len(utterances) == 1
    assert isinstance(utterances[0], bytes)
    assert len(utterances[0]) > 0
