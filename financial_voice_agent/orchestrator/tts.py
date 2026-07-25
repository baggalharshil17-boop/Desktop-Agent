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

    Verify against Cartesia's current Python SDK docs at build time -- this
    adapter's websocket call shape is the highest-uncertainty piece of this
    task.
    """

    def __init__(self, cartesia_async_client, *, voice_id: str, model_id: str = "sonic-2") -> None:
        self._client = cartesia_async_client
        self._voice_id = voice_id
        self._model_id = model_id

    async def synthesize(self, text: str) -> bytes:
        chunks: list[bytes] = []
        async with self._client.tts.websocket() as ws:
            async for response in ws.send(
                model_id=self._model_id,
                transcript=text,
                voice={"mode": "id", "id": self._voice_id},
                output_format=CARTESIA_OUTPUT_FORMAT,
            ):
                if response.get("audio"):
                    chunks.append(response["audio"])
        return b"".join(chunks)
