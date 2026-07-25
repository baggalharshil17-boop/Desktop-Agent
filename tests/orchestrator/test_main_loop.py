import asyncio

import pytest

from financial_voice_agent.orchestrator.main_loop import drive_pipeline, run_voice_loop
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
async def test_drive_pipeline_forwards_utterances_to_output_queue():
    pipeline = _FakePipeline([b"utterance-1", b"utterance-2"])
    output_queue: asyncio.Queue = asyncio.Queue()

    await drive_pipeline(pipeline, output_queue)

    assert output_queue.get_nowait() == b"utterance-1"
    assert output_queue.get_nowait() == b"utterance-2"


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

    async def stt_fn(wav):
        transcript = "first" if wav == b"utterance-1" else "second"
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
