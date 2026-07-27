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


class _AlwaysEchoGate:
    def is_echo(self, pcm_chunk: bytes) -> bool:
        return True

    def diagnose(self, pcm_chunk: bytes) -> dict:
        return {"mic_rms": 0.01, "reference_rms": 0.02, "predicted_echo": 0.02, "threshold": 0.05, "is_echo": True}


class _NeverEchoGate:
    def __init__(self, *, reference_rms: float = 0.02):
        self._reference_rms = reference_rms

    def is_echo(self, pcm_chunk: bytes) -> bool:
        return False

    def diagnose(self, pcm_chunk: bytes) -> dict:
        return {
            "mic_rms": 0.5, "reference_rms": self._reference_rms,
            "predicted_echo": 0.001, "threshold": 0.002, "is_echo": False,
        }


@pytest.mark.asyncio
async def test_pipeline_suppresses_speech_the_echo_gate_attributes_to_playback():
    # Silero scores the assistant's own voice coming back through the mic as
    # speech -- because it is speech, just not the user's. Without the gate
    # veto this became both a spurious barge-in and a bogus utterance that
    # got transcribed (confirmed live: turns logged with transcripts like
    # "Thank you." that the user never said).
    queue = asyncio.Queue()
    vad = _ScriptedVad([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    chunks = [_chunk(100)] * 3 + [_chunk(0)] * 3
    await _feed(queue, chunks)
    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=20.0, min_speech_duration_ms=20.0,
        apply_noise_reduction=False, echo_gate=_AlwaysEchoGate(),
    )

    utterances = [u async for u in pipeline.run()]

    assert utterances == []
    assert not pipeline.speech_active.is_set()


@pytest.mark.asyncio
async def test_pipeline_still_captures_speech_the_echo_gate_clears():
    # The gate must only veto -- genuine speech during playback (a real
    # barge-in) has to survive, or the feature would just be a mute switch.
    queue = asyncio.Queue()
    vad = _ScriptedVad([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    chunks = [_chunk(100)] * 3 + [_chunk(0)] * 3
    await _feed(queue, chunks)
    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=20.0, min_speech_duration_ms=20.0,
        apply_noise_reduction=False, echo_gate=_NeverEchoGate(),
    )

    utterances = [u async for u in pipeline.run()]

    assert len(utterances) == 1


@pytest.mark.asyncio
async def test_pipeline_does_not_barge_in_on_speech_shorter_than_min_duration():
    # A one-or-two-chunk blip is residual echo spiking at playback onset, not
    # a person interrupting -- measured at ~30x the steady echo level on a
    # laptop array mic, so level alone can't tell them apart. Duration can.
    queue = asyncio.Queue()
    vad = _ScriptedVad([0.9, 0.1, 0.1])
    await _feed(queue, [_chunk(100)] + [_chunk(0)] * 2)  # 10ms of "speech"
    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=5.0, min_speech_duration_ms=5.0,
        apply_noise_reduction=False, min_barge_in_ms=96.0,
    )

    [u async for u in pipeline.run()]

    assert not pipeline.speech_active.is_set()


@pytest.mark.asyncio
async def test_pipeline_barges_in_once_speech_exceeds_min_duration():
    queue = asyncio.Queue()
    speech_chunks = 12  # 12 * 10ms = 120ms > 96ms threshold
    vad = _ScriptedVad([0.9] * speech_chunks)
    for c in [_chunk(100)] * speech_chunks:
        await queue.put(c)

    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=5.0, min_speech_duration_ms=5.0,
        apply_noise_reduction=False, min_barge_in_ms=96.0,
    )

    # Drive the generator far enough to consume the speech chunks, then stop
    # (no sentinel queued -- it would block otherwise).
    agen = pipeline.run()
    consume = asyncio.create_task(agen.__anext__())
    await asyncio.sleep(0.05)
    consume.cancel()

    assert pipeline.speech_active.is_set()


@pytest.mark.asyncio
async def test_pipeline_fires_echo_diagnostic_for_every_close_call():
    # Every VAD-flagged-speech chunk that overlaps recent playback is a
    # potential leak or a real barge-in, whichever way it's classified --
    # this is what gives a reported false-trigger real numbers to diagnose
    # instead of needing to be reproduced blind.
    queue = asyncio.Queue()
    vad = _ScriptedVad([0.9, 0.9, 0.1])
    await _feed(queue, [_chunk(100), _chunk(100), _chunk(0)])
    seen = []
    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=5.0, min_speech_duration_ms=5.0,
        apply_noise_reduction=False, echo_gate=_NeverEchoGate(reference_rms=0.02),
        on_echo_diagnostic=seen.append,
    )

    [u async for u in pipeline.run()]

    assert len(seen) == 2
    assert all(d["reference_rms"] == 0.02 for d in seen)


@pytest.mark.asyncio
async def test_pipeline_skips_echo_diagnostic_when_nothing_recently_played():
    # reference_rms == 0 means no playback overlap at all -- not an
    # ambiguous case, so it shouldn't be logged as one.
    class _SilentReferenceGate:
        def is_echo(self, pcm_chunk: bytes) -> bool:
            return False

        def diagnose(self, pcm_chunk: bytes) -> dict:
            return {"mic_rms": 0.5, "reference_rms": 0.0, "predicted_echo": 0.0, "threshold": 0.0, "is_echo": False}

    queue = asyncio.Queue()
    vad = _ScriptedVad([0.9, 0.1])
    await _feed(queue, [_chunk(100), _chunk(0)])
    seen = []
    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=5.0, min_speech_duration_ms=5.0,
        apply_noise_reduction=False, echo_gate=_SilentReferenceGate(),
        on_echo_diagnostic=seen.append,
    )

    [u async for u in pipeline.run()]

    assert seen == []
