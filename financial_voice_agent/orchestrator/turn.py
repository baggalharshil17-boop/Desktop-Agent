from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from financial_voice_agent import db


@dataclass(frozen=True)
class LlmTurnResult:
    response_text: str
    tool_calls_json: str | None
    tool_results_json: str | None


@dataclass(frozen=True)
class TurnResult:
    transcript: str | None
    response_text: str | None
    tts_audio: bytes | None
    latency_stt_ms: int
    latency_llm_ms: int
    latency_tts_ms: int
    latency_total_ms: int
    error: str | None


async def run_turn(
    utterance_wav: bytes,
    history: list[dict],
    *,
    stt_fn: Callable[[bytes], Awaitable[str]],
    llm_fn: Callable[[str, list[dict]], Awaitable[LlmTurnResult]],
    tts_fn: Callable[[str], Awaitable[bytes]],
    db_path: str,
) -> TurnResult:
    turn_start = time.monotonic()
    transcript: str | None = None
    llm_result: LlmTurnResult | None = None
    tts_audio: bytes | None = None
    latency_stt_ms = 0
    latency_llm_ms = 0
    latency_tts_ms = 0
    error: str | None = None

    try:
        stt_start = time.monotonic()
        transcript = await stt_fn(utterance_wav)
        latency_stt_ms = int((time.monotonic() - stt_start) * 1000)

        llm_start = time.monotonic()
        llm_result = await llm_fn(transcript, history)
        latency_llm_ms = int((time.monotonic() - llm_start) * 1000)

        tts_start = time.monotonic()
        tts_audio = await tts_fn(llm_result.response_text)
        latency_tts_ms = int((time.monotonic() - tts_start) * 1000)
    except Exception as exc:  # noqa: BLE001 -- a turn must never crash the caller
        error = str(exc)

    latency_total_ms = int((time.monotonic() - turn_start) * 1000)

    db.log_turn(
        db_path,
        transcript=transcript,
        tool_calls_json=llm_result.tool_calls_json if llm_result else None,
        tool_results_json=llm_result.tool_results_json if llm_result else None,
        response_text=llm_result.response_text if llm_result else None,
        screenshot_path=None,
        latency_stt_ms=latency_stt_ms or None,
        latency_llm_ms=latency_llm_ms or None,
        latency_tool_ms=None,
        latency_tts_ms=latency_tts_ms or None,
        latency_total_ms=latency_total_ms,
        error=error,
    )

    return TurnResult(
        transcript=transcript,
        response_text=llm_result.response_text if llm_result else None,
        tts_audio=tts_audio,
        latency_stt_ms=latency_stt_ms,
        latency_llm_ms=latency_llm_ms,
        latency_tts_ms=latency_tts_ms,
        latency_total_ms=latency_total_ms,
        error=error,
    )


def update_history(
    history: list[dict], transcript: str, response_text: str, *, max_turns: int = 8
) -> list[dict]:
    updated = [
        *history,
        {"role": "user", "content": transcript},
        {"role": "assistant", "content": response_text},
    ]
    return updated[-(max_turns * 2):]
