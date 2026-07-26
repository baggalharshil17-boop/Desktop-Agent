from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Protocol

from financial_voice_agent.orchestrator.retry import RetryExhaustedError, retry_with_fixed_backoff

CARTESIA_OUTPUT_FORMAT = {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000}


class TtsError(Exception):
    pass


class TtsClient(Protocol):
    async def synthesize(self, text: str) -> bytes: ...


async def synthesize_with_fallback(
    primary: TtsClient,
    text: str,
    *,
    fallback: TtsClient | None = None,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> bytes:
    async def _primary_attempt() -> bytes:
        return await primary.synthesize(text)

    try:
        return await retry_with_fixed_backoff(
            _primary_attempt, max_attempts=2, backoff_seconds=1.0, sleep_fn=sleep_fn
        )
    except RetryExhaustedError as exc:
        if fallback is not None:
            try:
                return await fallback.synthesize(text)
            except Exception as fallback_exc:  # noqa: BLE001
                raise TtsError("Both Cartesia and Deepgram TTS failed") from fallback_exc
        raise TtsError("Cartesia TTS failed and no fallback configured") from exc


class RealCartesiaTtsClient:
    """Thin adapter around Cartesia's websocket TTS API, using the exact
    PRD Section 10.2 output format (16kHz, pcm_s16le, matching mic capture
    rate so nothing needs resampling).

    Verified against the installed cartesia==3.5.0 SDK at build time:
    `client.tts.websocket()` is itself a coroutine returning an
    `AsyncBackcompatTTSResourceConnection` that does NOT implement the async
    context manager protocol (no __aenter__/__aexit__), so it must be
    awaited and explicitly `.close()`d rather than used with `async with`.
    Its `.send(...)` call is also async and yields a stream of
    `BackcompatWebSocketTtsOutput` pydantic objects (attribute access via
    `.audio`), not dicts -- `response["audio"]` / `response.get("audio")`
    would raise, since BaseModel is not subscriptable.
    """

    def __init__(self, cartesia_async_client, *, voice_id: str, model_id: str = "sonic-3.5") -> None:
        self._client = cartesia_async_client
        self._voice_id = voice_id
        self._model_id = model_id

    async def synthesize(self, text: str) -> bytes:
        chunks: list[bytes] = []
        ws = await self._client.tts.websocket()
        try:
            response_stream = await ws.send(
                model_id=self._model_id,
                transcript=text,
                voice={"mode": "id", "id": self._voice_id},
                output_format=CARTESIA_OUTPUT_FORMAT,
            )
            async for response in response_stream:
                if response.audio:
                    chunks.append(response.audio)
        finally:
            await ws.close()
        return b"".join(chunks)
