import sqlite3

from financial_voice_agent.db import init_db, log_turn


def test_init_db_creates_turns_table(tmp_path):
    db_path = str(tmp_path / "agent_turns.db")

    init_db(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='turns'")
        assert cursor.fetchone() is not None
    finally:
        conn.close()


def test_init_db_is_idempotent(tmp_path):
    db_path = str(tmp_path / "agent_turns.db")

    init_db(db_path)
    init_db(db_path)  # must not raise

    conn = sqlite3.connect(db_path)
    conn.close()


def test_log_turn_inserts_row_and_returns_id(tmp_path):
    db_path = str(tmp_path / "agent_turns.db")
    init_db(db_path)

    turn_id = log_turn(
        db_path,
        transcript="what's the nifty level",
        tool_calls_json='[{"tool": "get_quote", "args": {"symbol": "NIFTY 50"}}]',
        tool_results_json='{"ltp": 24500}',
        response_text="Nifty is at 24500.",
        screenshot_path=None,
        latency_stt_ms=120,
        latency_llm_ms=430,
        latency_tool_ms=90,
        latency_tts_ms=60,
        latency_total_ms=700,
        error=None,
    )

    assert isinstance(turn_id, int)
    assert turn_id >= 1

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT transcript, response_text, latency_total_ms, ts_utc FROM turns WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    transcript, response_text, latency_total_ms, ts_utc = row
    assert transcript == "what's the nifty level"
    assert response_text == "Nifty is at 24500."
    assert latency_total_ms == 700
    assert ts_utc  # non-empty ISO timestamp string


def test_log_turn_second_call_increments_turn_id(tmp_path):
    db_path = str(tmp_path / "agent_turns.db")
    init_db(db_path)

    first_id = log_turn(
        db_path,
        transcript="a",
        tool_calls_json=None,
        tool_results_json=None,
        response_text="b",
        screenshot_path=None,
        latency_stt_ms=None,
        latency_llm_ms=None,
        latency_tool_ms=None,
        latency_tts_ms=None,
        latency_total_ms=None,
        error=None,
    )
    second_id = log_turn(
        db_path,
        transcript="c",
        tool_calls_json=None,
        tool_results_json=None,
        response_text="d",
        screenshot_path=None,
        latency_stt_ms=None,
        latency_llm_ms=None,
        latency_tool_ms=None,
        latency_tts_ms=None,
        latency_total_ms=None,
        error=None,
    )

    assert second_id == first_id + 1
