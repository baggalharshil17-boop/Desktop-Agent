import sqlite3

import pytest

from financial_voice_agent.db import init_db
from financial_voice_agent.orchestrator.turn import LlmTurnResult, TurnResult, run_turn, update_history


def _fake_monotonic(values):
    it = iter(values)
    def fn():
        import inspect
        stack = inspect.stack()
        for frame_info in stack:
            if 'run_turn' in frame_info.function:
                return next(it)
        return 0.0
    return fn


@pytest.mark.asyncio
async def test_run_turn_computes_latencies_and_logs_success(tmp_path, monkeypatch):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    # 8 time.monotonic() calls: turn_start, stt_start, stt_end, llm_start,
    # llm_end, tts_start, tts_end, total_end.
    monkeypatch.setattr(
        "financial_voice_agent.orchestrator.turn.time.monotonic",
        _fake_monotonic([0.0, 0.0, 0.1, 0.1, 0.35, 0.35, 0.40, 0.60]),
    )

    async def stt_fn(wav_bytes):
        return "hello"

    async def llm_fn(transcript, history):
        return LlmTurnResult(response_text="hi there", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return b"audio-bytes"

    result = await run_turn(
        b"wav-bytes", [], stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
    )

    assert result == TurnResult(
        transcript="hello",
        response_text="hi there",
        tts_audio=b"audio-bytes",
        latency_stt_ms=100,
        latency_llm_ms=250,
        latency_tts_ms=50,
        latency_total_ms=600,
        error=None,
    )

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT transcript, response_text, latency_stt_ms, latency_llm_ms, "
            "latency_tts_ms, latency_total_ms, error FROM turns"
        ).fetchone()
    finally:
        conn.close()
    assert row == ("hello", "hi there", 100, 250, 50, 600, None)


@pytest.mark.asyncio
async def test_run_turn_catches_stage_failure_and_logs_error_without_raising(tmp_path):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)

    async def stt_fn(wav_bytes):
        raise RuntimeError("groq is down")

    async def llm_fn(transcript, history):
        raise AssertionError("must not be called after stt_fn fails")

    async def tts_fn(text):
        raise AssertionError("must not be called after stt_fn fails")

    result = await run_turn(
        b"wav-bytes", [], stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
    )

    assert result.transcript is None
    assert result.response_text is None
    assert result.tts_audio is None
    assert result.error == "groq is down"

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT error FROM turns").fetchone()
    finally:
        conn.close()
    assert row == ("groq is down",)


@pytest.mark.asyncio
async def test_run_turn_with_real_stub_functions_end_to_end(tmp_path):
    from financial_voice_agent.orchestrator.stubs import fake_llm, fake_stt, fake_tts

    db_path = str(tmp_path / "turns.db")
    init_db(db_path)

    result = await run_turn(
        b"wav-bytes", [], stt_fn=fake_stt, llm_fn=fake_llm, tts_fn=fake_tts, db_path=db_path
    )

    assert isinstance(result.transcript, str) and result.transcript
    assert isinstance(result.response_text, str) and result.response_text
    assert isinstance(result.tts_audio, bytes)
    assert result.error is None


def test_update_history_appends_user_and_assistant_turns():
    history = update_history([], "what's the nifty level", "Nifty is at 24500")

    assert history == [
        {"role": "user", "content": "what's the nifty level"},
        {"role": "assistant", "content": "Nifty is at 24500"},
    ]


def test_update_history_trims_to_max_turns():
    history: list[dict] = []
    for i in range(10):
        history = update_history(history, f"question {i}", f"answer {i}", max_turns=8)

    assert len(history) == 16  # 8 turns * 2 messages
    assert history[0] == {"role": "user", "content": "question 2"}
    assert history[-1] == {"role": "assistant", "content": "answer 9"}


def test_update_history_does_not_mutate_input():
    original = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]

    result = update_history(original, "c", "d")

    assert len(original) == 2
    assert len(result) == 4
