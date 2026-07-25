from __future__ import annotations

import asyncio

from financial_voice_agent.orchestrator.turn import LlmTurnResult


async def fake_stt(wav_bytes: bytes) -> str:
    await asyncio.sleep(0)
    return "this is a fake transcript"


async def fake_llm(transcript: str, history: list[dict]) -> LlmTurnResult:
    await asyncio.sleep(0)
    return LlmTurnResult(
        response_text=f"You said: {transcript}",
        tool_calls_json=None,
        tool_results_json=None,
    )


async def fake_tts(text: str) -> bytes:
    await asyncio.sleep(0)
    return b""
