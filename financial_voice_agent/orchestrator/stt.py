from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Protocol

import groq

from financial_voice_agent.orchestrator.retry import RetryExhaustedError, retry_with_fixed_backoff


class SttError(Exception):
    """Raised when STT fails after all retries. The turn orchestrator (Task 5)
    catches this and speaks "I didn't catch that, one moment" per PRD
    Section 15.1, then re-listens rather than crashing."""


class SttClient(Protocol):
    async def transcribe(self, wav_bytes: bytes, *, model: str) -> str: ...


async def transcribe_with_retry(
    client: SttClient,
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


class HuggingFaceSttClient:
    """Thin adapter around Hugging Face's serverless Inference API -- usable
    as a Groq alternative while developing without Groq credits. Switch back
    via config.yaml's stt.provider key (see make_stt_client() below).

    Verify against huggingface.co/docs/api-inference at build time. Known
    quirk: a cold-started model can return a 503 with a JSON body like
    {"error": "... is currently loading", "estimated_time": ...} instead of
    a transcription -- this can take longer to clear than
    transcribe_with_retry's default 3-attempt/1s-backoff policy covers, so a
    cold HF endpoint may still surface as SttError on the first real call.
    Not verified against a live HF endpoint in this environment (no API key
    configured yet).
    """

    def __init__(self, http_client) -> None:
        self._client = http_client

    async def transcribe(self, wav_bytes: bytes, *, model: str) -> str:
        response = await self._client.post(
            f"/models/{model}",
            content=wav_bytes,
            headers={"Content-Type": "audio/wav"},
        )
        response.raise_for_status()
        data = response.json()
        if "text" not in data:
            raise SttError(f"Hugging Face inference returned an unexpected response shape: {data}")
        return data["text"]


def make_stt_client(config, http_clients) -> SttClient:
    """Picks the real STT adapter based on config.stt_provider -- the seam
    that lets stt.provider in config.yaml switch between Groq and Hugging
    Face without touching any calling code."""
    if config.stt_provider == "huggingface":
        return HuggingFaceSttClient(http_clients.huggingface)
    if config.stt_provider == "groq":
        return RealGroqSttClient(groq.AsyncGroq(api_key=config.groq_api_key))
    raise ValueError(f"Unsupported stt_provider: {config.stt_provider!r}")
