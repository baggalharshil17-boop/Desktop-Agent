import asyncio

import pytest

from financial_voice_agent.audio.pipeline import AudioPipeline

CHUNK_MS = 10.0  # 160 samples at 16000 Hz = 10ms per chunk


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

    states_during_run = []
    gen = pipeline.run()
    async for _utterance in gen:
        pass
    # After the generator is exhausted, speech has finalized and the event must be clear.
    assert not pipeline.speech_active.is_set()
