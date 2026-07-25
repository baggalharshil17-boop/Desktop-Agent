from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Protocol

from financial_voice_agent.orchestrator.retry import RetryExhaustedError, retry_with_fixed_backoff


class SttError(Exception):
    """Raised when STT fails after all retries. The turn orchestrator (Task 5)
    catches this and speaks "I didn't catch that, one moment" per PRD
    Section 15.1, then re-listens rather than crashing."""


class GroqSttClient(Protocol):
    async def transcribe(self, wav_bytes: bytes, *, model: str) -> str: ...


async def transcribe_with_retry(
    client: GroqSttClient,
    wav_bytes: bytes,
    *,
    model: str,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> str:
    async def _attempt() -> str:
        return await client.transcribe(wav_bytes, model=model)

    try:
        return await retry_with_fixed_backoff(
            _attempt, max_attempts=3, backoff_seconds=1.0, sleep_fn=sleep_fn
        )
    except RetryExhaustedError as exc:
        raise SttError("Speech-to-text failed after 3 attempts") from exc


class RealGroqSttClient:
    """Thin adapter around groq.AsyncGroq's real transcription API.

    Verify against console.groq.com/docs/api-reference at build time --
    Groq's SDK surface can change (PRD Section 2.3's model-churn warning
    applies to the SDK, not just the model name).
    """

    def __init__(self, groq_async_client) -> None:
        self._client = groq_async_client

    async def transcribe(self, wav_bytes: bytes, *, model: str) -> str:
        response = await self._client.audio.transcriptions.create(
            file=("utterance.wav", wav_bytes),
            model=model,
            response_format="text",
        )
        return response if isinstance(response, str) else response.text
