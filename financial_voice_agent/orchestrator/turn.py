from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from financial_voice_agent import db

# Matches markdown syntax an LLM might produce even when told not to --
# stripped before TTS so it's never spoken aloud literally (e.g. "asterisk
# asterisk"). Order matters: bold (**/__) must run before italic (*/_) so a
# **bold** span doesn't get treated as two italic markers.
_MD_HEADER_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)\*(?!\*)|(?<!_)_(?!_)(.+?)_(?!_)")
_MD_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_MD_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)


def strip_markdown_for_speech(text: str) -> str:
    text = _MD_HEADER_RE.sub("", text)
    text = _MD_BOLD_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = _MD_ITALIC_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = _MD_BULLET_RE.sub("", text)
    text = _MD_NUMBERED_RE.sub("", text)
    return text


@dataclass(frozen=True)
class LlmTurnResult:
    response_text: str
    tool_calls_json: str | None
    tool_results_json: str | None
    latency_tool_ms: int | None = None


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
    latency_tool_ms: int | None = None
    rate_limited: bool = False


async def run_turn(
    utterance_wav: bytes,
    history: list[dict],
    *,
    stt_fn: Callable[[bytes], Awaitable[str]],
    llm_fn: Callable[[str, list[dict]], Awaitable[LlmTurnResult]],
    tts_fn: Callable[[str], Awaitable[bytes]],
    db_path: str,
    clock_fn: Callable[[], float] = time.monotonic,
) -> TurnResult:
    turn_start = clock_fn()
    transcript: str | None = None
    llm_result: LlmTurnResult | None = None
    tts_audio: bytes | None = None
    latency_stt_ms = 0
    latency_llm_ms = 0
    latency_tts_ms = 0
    error: str | None = None
    rate_limited = False

    try:
        stt_start = clock_fn()
        transcript = await stt_fn(utterance_wav)
        latency_stt_ms = round((clock_fn() - stt_start) * 1000)

        llm_start = clock_fn()
        llm_result = await llm_fn(transcript, history)
        latency_llm_ms = round((clock_fn() - llm_start) * 1000)

        tts_start = clock_fn()
        tts_audio = await tts_fn(strip_markdown_for_speech(llm_result.response_text))
        latency_tts_ms = round((clock_fn() - tts_start) * 1000)
    except Exception as exc:  # noqa: BLE001 -- a turn must never crash the caller
        error = str(exc)
        rate_limited = getattr(exc, "rate_limited", False)

    latency_total_ms = round((clock_fn() - turn_start) * 1000)

    db.log_turn(
        db_path,
        transcript=transcript,
        tool_calls_json=llm_result.tool_calls_json if llm_result else None,
        tool_results_json=llm_result.tool_results_json if llm_result else None,
        response_text=llm_result.response_text if llm_result else None,
        screenshot_path=None,
        latency_stt_ms=latency_stt_ms or None,
        latency_llm_ms=latency_llm_ms or None,
        latency_tool_ms=llm_result.latency_tool_ms if llm_result else None,
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
        latency_tool_ms=llm_result.latency_tool_ms if llm_result else None,
        rate_limited=rate_limited,
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
