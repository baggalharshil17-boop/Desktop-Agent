from __future__ import annotations

from dataclasses import dataclass

import httpx

from financial_voice_agent.config import Config

GROQ_BASE_URL = "https://api.groq.com"
CARTESIA_BASE_URL = "https://api.cartesia.ai"
DEEPGRAM_BASE_URL = "https://api.deepgram.com"
KITE_BASE_URL = "https://api.kite.trade"
TAVILY_BASE_URL = "https://api.tavily.com"

GROQ_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
REST_TIMEOUT = httpx.Timeout(15.0, connect=10.0)
TTS_TIMEOUT = httpx.Timeout(15.0, connect=10.0)


@dataclass
class HTTPClients:
    groq: httpx.AsyncClient
    tts: httpx.AsyncClient
    kite: httpx.AsyncClient
    tavily: httpx.AsyncClient


async def create_http_clients(config: Config) -> HTTPClients:
    groq = httpx.AsyncClient(
        base_url=GROQ_BASE_URL,
        headers={"Authorization": f"Bearer {config.groq_api_key}"},
        timeout=GROQ_TIMEOUT,
    )

    if config.tts_provider == "cartesia":
        tts = httpx.AsyncClient(
            base_url=CARTESIA_BASE_URL,
            headers={"X-API-Key": config.cartesia_api_key or ""},
            timeout=TTS_TIMEOUT,
        )
    elif config.tts_provider == "deepgram":
        tts = httpx.AsyncClient(
            base_url=DEEPGRAM_BASE_URL,
            headers={"Authorization": f"Token {config.deepgram_api_key or ''}"},
            timeout=TTS_TIMEOUT,
        )
    else:
        raise ValueError(f"Unsupported tts_provider: {config.tts_provider!r}")

    kite_headers = {}
    if config.kite_api_key and config.kite_access_token:
        kite_headers["Authorization"] = f"token {config.kite_api_key}:{config.kite_access_token}"
    kite = httpx.AsyncClient(
        base_url=KITE_BASE_URL,
        headers=kite_headers,
        timeout=REST_TIMEOUT,
    )

    tavily = httpx.AsyncClient(base_url=TAVILY_BASE_URL, timeout=REST_TIMEOUT)

    return HTTPClients(groq=groq, tts=tts, kite=kite, tavily=tavily)


async def close_http_clients(clients: HTTPClients) -> None:
    await clients.groq.aclose()
    await clients.tts.aclose()
    await clients.kite.aclose()
    await clients.tavily.aclose()
