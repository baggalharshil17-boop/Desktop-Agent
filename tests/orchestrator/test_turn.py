import sqlite3

import pytest

from financial_voice_agent.db import init_db
from financial_voice_agent.orchestrator.turn import (
    LlmTurnResult,
    TurnResult,
    run_turn,
    strip_markdown_for_speech,
    update_history,
)


def _fake_monotonic(values):
    it = iter(values)
    return lambda: next(it)


@pytest.mark.asyncio
async def test_run_turn_computes_latencies_and_logs_success(tmp_path):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    # 8 clock_fn() calls: turn_start, stt_start, stt_end, llm_start,
    # llm_end, tts_start, tts_end, total_end. Passed directly as an argument
    # rather than monkeypatched onto the global `time` module -- patching
    # `time.monotonic` process-wide also intercepts calls asyncio's own
    # event loop makes internally, exhausting the fake iterator early.
    clock_fn = _fake_monotonic([0.0, 0.0, 0.1, 0.1, 0.35, 0.35, 0.40, 0.60])

    async def stt_fn(wav_bytes):
        return "hello"

    async def llm_fn(transcript, history):
        return LlmTurnResult(response_text="hi there", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return b"audio-bytes"

    result = await run_turn(
        b"wav-bytes", [], stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path, clock_fn=clock_fn
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


def test_strip_markdown_for_speech_removes_bold():
    assert strip_markdown_for_speech("The P/E ratio is **8.64**.") == "The P/E ratio is 8.64."
    assert strip_markdown_for_speech("This is __also bold__.") == "This is also bold."


def test_strip_markdown_for_speech_removes_italic():
    assert strip_markdown_for_speech("This is *emphasized* text.") == "This is emphasized text."
    assert strip_markdown_for_speech("This is _also emphasized_.") == "This is also emphasized."


def test_strip_markdown_for_speech_removes_headers():
    assert strip_markdown_for_speech("# Heading\nBody text") == "Heading\nBody text"
    assert strip_markdown_for_speech("### Smaller heading") == "Smaller heading"


def test_strip_markdown_for_speech_removes_bullet_markers():
    text = "Recent news includes:\n- First point\n* Second point\n+ Third point"
    assert strip_markdown_for_speech(text) == "Recent news includes:\nFirst point\nSecond point\nThird point"


def test_strip_markdown_for_speech_removes_numbered_list_markers():
    text = "Steps:\n1. Do this\n2. Do that"
    assert strip_markdown_for_speech(text) == "Steps:\nDo this\nDo that"


def test_strip_markdown_for_speech_handles_real_world_example():
    text = (
        "Here is the update on ONGC:\n\n"
        "The **P/E ratio** is 8.64.\n\n"
        "Recent news includes:\n"
        "- ONGC plans to build a large strategic petroleum reserve.\n"
        "- Three Executive Directors recently retired."
    )
    result = strip_markdown_for_speech(text)
    assert "*" not in result
    assert "- " not in result
    assert "P/E ratio" in result
    assert "ONGC plans to build a large strategic petroleum reserve." in result


def test_strip_markdown_for_speech_leaves_plain_text_unchanged():
    assert strip_markdown_for_speech("Just a plain sentence.") == "Just a plain sentence."


@pytest.mark.asyncio
async def test_run_turn_strips_markdown_before_calling_tts_fn(tmp_path):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    captured_tts_text = []

    async def stt_fn(wav_bytes):
        return "hello"

    async def llm_fn(transcript, history):
        return LlmTurnResult(
            response_text="The **P/E ratio** is 8.64.", tool_calls_json=None, tool_results_json=None
        )

    async def tts_fn(text):
        captured_tts_text.append(text)
        return b"audio-bytes"

    await run_turn(b"wav-bytes", [], stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path)

    assert captured_tts_text == ["The P/E ratio is 8.64."]
