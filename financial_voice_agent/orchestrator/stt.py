from __future__ import annotations

import asyncio
import pathlib
import tempfile
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
    """Thin adapter around huggingface_hub's AsyncInferenceClient -- usable
    as a Groq alternative while developing without Groq credits. Switch back
    via config.yaml's stt.provider key (see make_stt_client() below).

    Verify against huggingface_hub's current docs/version at build time --
    this project's Groq/Cartesia adapters have both needed correction once
    against the real installed SDK. Known quirk: a cold-started model on the
    "hf-inference" provider can be slow or raise on first use while it spins
    up -- this can take longer to clear than transcribe_with_retry's default
    3-attempt/1s-backoff policy covers, so a cold HF endpoint may still
    surface as SttError on the first real call.

    Raw `bytes` alone don't carry a MIME type, and huggingface_hub can't
    guess one from bytes -- it sends Content-Type: None, which HF's server
    rejects with "Content type None not supported" (confirmed against a
    real call). A file-like object doesn't work either -- the hf-inference
    provider's ASR helper only accepts bytes, Path, or str (confirmed
    against the installed SDK's source), so a BytesIO is rejected outright.
    Writing to a temp .wav file and passing its Path is the only input shape
    that both is accepted and lets the SDK guess audio/wav from the
    extension.
    """

    def __init__(self, client) -> None:
        self._client = client  # a huggingface_hub.AsyncInferenceClient

    async def transcribe(self, wav_bytes: bytes, *, model: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            temp_path = pathlib.Path(f.name)
        try:
            output = await self._client.automatic_speech_recognition(temp_path, model=model)
        finally:
            temp_path.unlink(missing_ok=True)
        return output if isinstance(output, str) else output.text


def make_stt_client(config) -> SttClient:
    """Picks the real STT adapter based on config.stt_provider -- the seam
    that lets stt.provider in config.yaml switch between Groq and Hugging
    Face without touching any calling code. Both vendor SDKs manage their
    own HTTP transport, so no shared HTTPClients are needed here."""
    if config.stt_provider == "huggingface":
        from huggingface_hub import AsyncInferenceClient

        return HuggingFaceSttClient(
            AsyncInferenceClient(provider="hf-inference", api_key=config.huggingface_api_key)
        )
    if config.stt_provider == "groq":
        return RealGroqSttClient(groq.AsyncGroq(api_key=config.groq_api_key))
    raise ValueError(f"Unsupported stt_provider: {config.stt_provider!r}")
