import asyncio

import pytest

from financial_voice_agent.orchestrator.main_loop import (
    GENERIC_FAILURE_MESSAGES,
    RATE_LIMIT_MESSAGES,
    STT_FAILURE_MESSAGES,
    drive_pipeline,
    play_with_barge_in,
    run_voice_loop,
)
from financial_voice_agent.orchestrator.turn import LlmTurnResult
from financial_voice_agent.db import init_db


class _FakePipeline:
    """Mimics AudioPipeline: an async generator yielding scripted utterances,
    plus a speech_active event a test can set to simulate barge-in."""

    def __init__(self, utterances: list[bytes]):
        self._utterances = utterances
        self.speech_active = asyncio.Event()

    async def run(self):
        for u in self._utterances:
            yield u


class _FakePlayback:
    def __init__(self, *, interrupted_after: int | None = None):
        self.play_calls: list[bytes] = []
        self._interrupted_after = interrupted_after

    def play(self, pcm_bytes: bytes, *, chunk_size: int = 1024, interrupt_event=None) -> bool:
        self.play_calls.append(pcm_bytes)
        if self._interrupted_after is not None and len(self.play_calls) > self._interrupted_after:
            return False
        return True


@pytest.mark.asyncio
async def test_drive_pipeline_wraps_utterances_in_wav_and_forwards_to_output_queue():
    pcm_one = b"\x00\x01" * 100
    pcm_two = b"\x02\x03" * 100
    pipeline = _FakePipeline([pcm_one, pcm_two])
    output_queue: asyncio.Queue = asyncio.Queue()

    await drive_pipeline(pipeline, output_queue)

    first = output_queue.get_nowait()
    second = output_queue.get_nowait()
    assert first.startswith(b"RIFF")
    assert second.startswith(b"RIFF")

    import io
    import wave

    with wave.open(io.BytesIO(first), "rb") as wav_file:
        assert wav_file.readframes(wav_file.getnframes()) == pcm_one


@pytest.mark.asyncio
async def test_run_voice_loop_processes_each_utterance_and_plays_response(tmp_path):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    pipeline = _FakePipeline([b"utterance-1"])
    playback = _FakePlayback()

    async def stt_fn(wav):
        return "hello"

    async def llm_fn(transcript, history):
        return LlmTurnResult(response_text="hi there", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return b"audio-bytes"

    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
    )

    assert playback.play_calls == [b"audio-bytes"]


@pytest.mark.asyncio
async def test_run_voice_loop_processes_multiple_utterances_in_order(tmp_path):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    pipeline = _FakePipeline([b"utterance-1", b"utterance-2"])
    playback = _FakePlayback()
    seen_transcripts: list[str] = []
    call_count = [0]

    async def stt_fn(wav):
        call_count[0] += 1
        transcript = "first" if call_count[0] == 1 else "second"
        seen_transcripts.append(transcript)
        return transcript

    async def llm_fn(transcript, history):
        return LlmTurnResult(response_text=f"response to {transcript}", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return text.encode()

    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
    )

    assert seen_transcripts == ["first", "second"]
    assert playback.play_calls == [b"response to first", b"response to second"]


class _SlowFakePlayback:
    """Simulates real blocking, chunked PyAudio writes with a real per-chunk
    delay, so there's a genuine time window during which an external
    interrupt_event.set() call (mirrored from speech_active by
    _play_with_barge_in) can actually take effect mid-playback."""

    def __init__(self, num_chunks: int = 10, chunk_delay: float = 0.02):
        self.play_calls: list[int] = []
        self._num_chunks = num_chunks
        self._chunk_delay = chunk_delay

    def play(self, pcm_bytes: bytes, *, chunk_size: int = 1024, interrupt_event=None) -> bool:
        import time

        chunks_written = 0
        for _ in range(self._num_chunks):
            if interrupt_event is not None and interrupt_event.is_set():
                self.play_calls.append(chunks_written)
                return False
            time.sleep(self._chunk_delay)
            chunks_written += 1
        self.play_calls.append(chunks_written)
        return True


@pytest.mark.asyncio
async def test_run_voice_loop_stops_playback_mid_stream_when_speech_active_set_during_playback(
    tmp_path,
):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    pipeline = _FakePipeline([b"utterance-1"])
    playback = _SlowFakePlayback(num_chunks=15, chunk_delay=0.02)

    async def stt_fn(wav):
        return "hello"

    async def llm_fn(transcript, history):
        return LlmTurnResult(response_text="a long response", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return b"audio-bytes"

    async def set_speech_active_partway_through():
        await asyncio.sleep(0.06)  # a few chunks into a 15 * 0.02s = 0.3s playback
        pipeline.speech_active.set()

    setter_task = asyncio.create_task(set_speech_active_partway_through())
    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
    )
    await setter_task

    assert len(playback.play_calls) == 1
    # Must have stopped well before writing all 15 chunks -- proves the
    # interrupt actually propagated from pipeline.speech_active into the
    # playback's interrupt_event WHILE playback was in progress, not before
    # it started.
    assert playback.play_calls[0] < 15


@pytest.mark.asyncio
async def test_play_with_barge_in_interrupts_on_matching_keyword_from_new_arrival():
    # A new utterance lands in output_queue mid-playback; stt_fn reports it
    # as "stop" -- a deliberate redirect phrase -- so playback must be cut
    # short even though pipeline.speech_active (the VAD/echo-gate signal)
    # never fires at all.
    playback = _SlowFakePlayback(num_chunks=15, chunk_delay=0.02)
    pipeline = _FakePipeline([])  # speech_active never set
    output_queue: asyncio.Queue = asyncio.Queue()

    async def stt_fn(wav):
        return "stop" if wav == b"new-utterance" else "irrelevant"

    async def arrive_mid_playback():
        await asyncio.sleep(0.06)
        await output_queue.put(b"new-utterance")

    arrival_task = asyncio.create_task(arrive_mid_playback())
    result = await play_with_barge_in(
        playback, b"long-response-audio", pipeline,
        output_queue=output_queue, stt_fn=stt_fn,
    )
    await arrival_task

    assert result is False  # interrupted, not completed
    assert playback.play_calls[0] < 15  # stopped mid-stream, not at the end
    # Handed back for the normal turn loop to process -- not consumed.
    assert output_queue.get_nowait() == b"new-utterance"


@pytest.mark.asyncio
async def test_play_with_barge_in_does_not_interrupt_on_non_keyword_arrival():
    playback = _SlowFakePlayback(num_chunks=8, chunk_delay=0.02)
    pipeline = _FakePipeline([])
    output_queue: asyncio.Queue = asyncio.Queue()

    async def stt_fn(wav):
        return "what's the nifty level"  # a normal query, not a redirect

    async def arrive_mid_playback():
        await asyncio.sleep(0.03)
        await output_queue.put(b"new-utterance")

    arrival_task = asyncio.create_task(arrive_mid_playback())
    result = await play_with_barge_in(
        playback, b"response-audio", pipeline,
        output_queue=output_queue, stt_fn=stt_fn,
    )
    await arrival_task

    assert result is True  # completed uninterrupted
    assert playback.play_calls[0] == 8


@pytest.mark.asyncio
async def test_play_with_barge_in_never_speculatively_transcribes_utterances_queued_before_playback():
    # An utterance already sitting in the queue when playback starts is
    # simply the next turn in line, queued while this turn was still being
    # processed -- not a live interruption. Must not be touched by stt_fn
    # at all, even if it happens to look like a keyword.
    playback = _SlowFakePlayback(num_chunks=6, chunk_delay=0.02)
    pipeline = _FakePipeline([])
    output_queue: asyncio.Queue = asyncio.Queue()
    output_queue.put_nowait(b"already-queued")

    stt_calls: list[bytes] = []

    async def stt_fn(wav):
        stt_calls.append(wav)
        return "stop"  # would force an interrupt if it were ever checked

    result = await play_with_barge_in(
        playback, b"response-audio", pipeline,
        output_queue=output_queue, stt_fn=stt_fn,
    )

    assert result is True  # completed uninterrupted -- pre-existing item never checked
    assert playback.play_calls[0] == 6
    assert stt_calls == []
    assert output_queue.get_nowait() == b"already-queued"  # still there, untouched


@pytest.mark.asyncio
async def test_run_voice_loop_still_processes_the_handed_back_utterance_normally(tmp_path):
    # End-to-end: a keyword-triggered interrupt must not lose the utterance
    # that triggered it -- the normal turn loop picks it up next.
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)

    class _DelayedSecondUtterancePipeline:
        def __init__(self):
            self.speech_active = asyncio.Event()

        async def run(self):
            yield b"utterance-1"
            await asyncio.sleep(0.06)
            yield b"utterance-2"

    class _SlowFakePlaybackWithContent:
        """Like _SlowFakePlayback, but also records which audio bytes each
        call played, so a specific response's presence can be verified."""

        def __init__(self, num_chunks: int = 15, chunk_delay: float = 0.02):
            self.play_calls: list[int] = []
            self.played_audio: list[bytes] = []
            self._num_chunks = num_chunks
            self._chunk_delay = chunk_delay

        def play(self, pcm_bytes, *, chunk_size=1024, interrupt_event=None):
            import time

            self.played_audio.append(pcm_bytes)
            chunks_written = 0
            for _ in range(self._num_chunks):
                if interrupt_event is not None and interrupt_event.is_set():
                    self.play_calls.append(chunks_written)
                    return False
                time.sleep(self._chunk_delay)
                chunks_written += 1
            self.play_calls.append(chunks_written)
            return True

    pipeline = _DelayedSecondUtterancePipeline()
    playback = _SlowFakePlaybackWithContent(num_chunks=15, chunk_delay=0.02)

    async def stt_fn(wav):
        # drive_pipeline wraps each yielded utterance in a WAV container
        # (encode_wav) before it reaches output_queue/stt_fn, so the raw
        # bytes survive as a substring of the WAV data, not an exact match.
        return "stop" if b"utterance-2" in wav else "hello"

    async def llm_fn(transcript, history):
        return LlmTurnResult(response_text=f"response to {transcript}", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return text.encode()

    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
    )

    # First response got interrupted by the "stop" arrival; the utterance
    # that said "stop" was then processed as its own normal turn.
    assert playback.play_calls[0] < 15
    assert playback.play_calls[1] == 15  # second turn's response played in full
    assert b"response to stop" in playback.played_audio


@pytest.mark.asyncio
async def test_run_voice_loop_ignores_speech_active_when_barge_in_disabled(tmp_path):
    # Setups where mic/speaker aren't acoustically isolated (e.g. a headset
    # mic picking up laptop-speaker bleed) can have speech_active fire from
    # the assistant's own voice, not real user barge-in. barge_in_enabled=
    # False must play the full response regardless of speech_active.
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    pipeline = _FakePipeline([b"utterance-1"])
    playback = _SlowFakePlayback(num_chunks=15, chunk_delay=0.02)

    async def stt_fn(wav):
        return "hello"

    async def llm_fn(transcript, history):
        return LlmTurnResult(response_text="a long response", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return b"audio-bytes"

    async def set_speech_active_partway_through():
        await asyncio.sleep(0.06)
        pipeline.speech_active.set()

    setter_task = asyncio.create_task(set_speech_active_partway_through())
    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path,
        barge_in_enabled=False,
    )
    await setter_task

    assert len(playback.play_calls) == 1
    # All 15 chunks written -- speech_active being set had no effect.
    assert playback.play_calls[0] == 15


@pytest.mark.asyncio
async def test_run_voice_loop_speaks_fallback_message_on_stt_failure(tmp_path):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    pipeline = _FakePipeline([b"utterance-1"])
    playback = _FakePlayback()

    async def stt_fn(wav):
        raise RuntimeError("groq is down")

    async def llm_fn(transcript, history):
        raise AssertionError("must not be called after stt_fn fails")

    tts_calls: list[str] = []

    async def tts_fn(text):
        tts_calls.append(text)
        return f"audio-for:{text}".encode()

    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path,
        choice_fn=lambda options: options[0],
    )

    assert tts_calls == [STT_FAILURE_MESSAGES[0]]
    assert playback.play_calls == [f"audio-for:{STT_FAILURE_MESSAGES[0]}".encode()]


@pytest.mark.asyncio
async def test_run_voice_loop_speaks_rate_limit_message_on_llm_rate_limit(tmp_path):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    pipeline = _FakePipeline([b"utterance-1"])
    playback = _FakePlayback()

    async def stt_fn(wav):
        return "hello"

    async def llm_fn(transcript, history):
        exc = RuntimeError("rate limited")
        exc.rate_limited = True
        raise exc

    tts_calls: list[str] = []

    async def tts_fn(text):
        tts_calls.append(text)
        return f"audio-for:{text}".encode()

    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path,
        choice_fn=lambda options: options[0],
    )

    assert tts_calls == [RATE_LIMIT_MESSAGES[0]]


@pytest.mark.asyncio
async def test_run_voice_loop_speaks_generic_fallback_when_stt_succeeded_but_llm_failed(tmp_path):
    # STT succeeded (there's a transcript) but the LLM call failed for some
    # other reason -- "I didn't catch that" would be a lie here, since we
    # clearly did hear the user. Must use the generic-failure phrasing, not
    # the STT one.
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    pipeline = _FakePipeline([b"utterance-1"])
    playback = _FakePlayback()

    async def stt_fn(wav):
        return "what's the nifty level"

    async def llm_fn(transcript, history):
        raise RuntimeError("model provider unavailable")

    tts_calls: list[str] = []

    async def tts_fn(text):
        tts_calls.append(text)
        return f"audio-for:{text}".encode()

    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path,
        choice_fn=lambda options: options[0],
    )

    assert tts_calls == [GENERIC_FAILURE_MESSAGES[0]]


@pytest.mark.asyncio
async def test_run_voice_loop_does_not_repeat_fallback_message_within_recent_window(tmp_path):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    # Five consecutive STT failures -- with a recent_fallback_window of 5,
    # none of these should repeat, even though choice_fn always prefers the
    # first available candidate (proving the filtering, not just luck).
    pipeline = _FakePipeline([b"utterance-1"] * 5)
    playback = _FakePlayback()

    async def stt_fn(wav):
        raise RuntimeError("groq is down")

    async def llm_fn(transcript, history):
        raise AssertionError("must not be called after stt_fn fails")

    tts_calls: list[str] = []

    async def tts_fn(text):
        tts_calls.append(text)
        return f"audio-for:{text}".encode()

    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path,
        choice_fn=lambda options: options[0], recent_fallback_window=5,
    )

    assert len(tts_calls) == 5
    assert len(set(tts_calls)) == 5  # no repeats within the 5-message window
    assert set(tts_calls) == set(STT_FAILURE_MESSAGES)  # exactly 5 phrases exist for this category


@pytest.mark.asyncio
async def test_run_voice_loop_reraises_drive_pipeline_crash(tmp_path):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)

    class _CrashingPipeline:
        def __init__(self):
            self.speech_active = asyncio.Event()

        async def run(self):
            raise RuntimeError("pipeline crashed")
            yield  # pragma: no cover -- required to make this an async generator

    pipeline = _CrashingPipeline()
    playback = _FakePlayback()

    async def stt_fn(wav):
        return "x"

    async def llm_fn(transcript, history):
        return LlmTurnResult(response_text="y", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return b"z"

    with pytest.raises(RuntimeError, match="pipeline crashed"):
        await run_voice_loop(
            pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
        )


@pytest.mark.asyncio
async def test_run_voice_loop_calls_on_processing_start_and_end_around_each_turn(tmp_path):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    pipeline = _FakePipeline([b"utterance-1"])
    playback = _FakePlayback()
    events: list[str] = []

    async def stt_fn(wav):
        return "hello"

    async def llm_fn(transcript, history):
        events.append("during_turn")
        return LlmTurnResult(response_text="hi there", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return b"audio-bytes"

    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path,
        on_processing_start=lambda: events.append("start"),
        on_processing_end=lambda: events.append("end"),
    )

    assert events == ["start", "during_turn", "end"]


@pytest.mark.asyncio
async def test_run_voice_loop_works_without_processing_callbacks(tmp_path):
    # Both callbacks default to None -- must not raise when omitted.
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    pipeline = _FakePipeline([b"utterance-1"])
    playback = _FakePlayback()

    async def stt_fn(wav):
        return "hello"

    async def llm_fn(transcript, history):
        return LlmTurnResult(response_text="hi there", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return b"audio-bytes"

    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
    )

    assert playback.play_calls == [b"audio-bytes"]


class _FailingPlayback:
    """Mimics PortAudio failing at the device level mid-write, which surfaces
    as OSError out of playback.play()."""

    def __init__(self, *, fail_on_call: int = 1):
        self.play_calls: list[bytes] = []
        self._fail_on_call = fail_on_call

    def play(self, pcm_bytes: bytes, *, chunk_size: int = 1024, interrupt_event=None) -> bool:
        self.play_calls.append(pcm_bytes)
        if len(self.play_calls) >= self._fail_on_call:
            raise OSError(-9999, "Unanticipated host error")
        return True


@pytest.mark.asyncio
async def test_play_with_barge_in_survives_a_device_level_playback_failure():
    pipeline = _FakePipeline([])
    playback = _FailingPlayback()

    completed = await play_with_barge_in(playback, b"audio", pipeline, barge_in_enabled=False)

    assert completed is False  # reported as not-played, not raised


@pytest.mark.asyncio
async def test_run_voice_loop_keeps_listening_after_the_speaker_fails(tmp_path):
    # A speaker failing must not end the conversation: the user can still be
    # heard, and the next turn may play fine. Before this, the OSError
    # propagated out of run_voice_loop and killed the agent.
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    pipeline = _FakePipeline([b"utterance-1", b"utterance-2"])
    playback = _FailingPlayback()
    transcripts = []

    async def stt_fn(wav):
        transcripts.append(wav)
        return "hello"

    async def llm_fn(transcript, history):
        return LlmTurnResult(response_text="hi there", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return b"audio-bytes"

    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
    )

    # Both utterances were still transcribed despite playback failing on each.
    assert len(transcripts) == 2
    assert len(playback.play_calls) == 2
