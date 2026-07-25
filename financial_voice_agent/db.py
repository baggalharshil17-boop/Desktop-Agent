from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
  turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc TEXT NOT NULL,
  transcript TEXT,
  tool_calls_json TEXT,
  tool_results_json TEXT,
  response_text TEXT,
  screenshot_path TEXT,
  latency_stt_ms INTEGER,
  latency_llm_ms INTEGER,
  latency_tool_ms INTEGER,
  latency_tts_ms INTEGER,
  latency_total_ms INTEGER,
  error TEXT
);
"""


def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def log_turn(
    db_path: str,
    *,
    transcript: str | None,
    tool_calls_json: str | None,
    tool_results_json: str | None,
    response_text: str | None,
    screenshot_path: str | None,
    latency_stt_ms: int | None,
    latency_llm_ms: int | None,
    latency_tool_ms: int | None,
    latency_tts_ms: int | None,
    latency_total_ms: int | None,
    error: str | None,
) -> int:
    ts_utc = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO turns (
                ts_utc, transcript, tool_calls_json, tool_results_json,
                response_text, screenshot_path, latency_stt_ms, latency_llm_ms,
                latency_tool_ms, latency_tts_ms, latency_total_ms, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts_utc,
                transcript,
                tool_calls_json,
                tool_results_json,
                response_text,
                screenshot_path,
                latency_stt_ms,
                latency_llm_ms,
                latency_tool_ms,
                latency_tts_ms,
                latency_total_ms,
                error,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()
