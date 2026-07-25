import httpx
import pytest

from financial_voice_agent.config import Config
from financial_voice_agent.http_clients import HTTPClients, close_http_clients, create_http_clients


def _make_config(tts_provider="cartesia", kite_api_key="kite-key", kite_access_token="kite-token") -> Config:
    return Config(
        vad_speech_threshold=0.5,
        vad_silence_duration_ms=600,
        vad_min_speech_duration_ms=200,
        audio_output_device_index=None,
        input_mode="always_on",
        tts_provider=tts_provider,
        llm_model="test-model",
        storage_db_path="./agent_turns.db",
        mode="live",
        groq_api_key="groq-secret",
        cartesia_api_key="cartesia-secret",
        deepgram_api_key="deepgram-secret",
        kite_api_key=kite_api_key,
        kite_access_token=kite_access_token,
        tavily_api_key="tavily-key",
    )


@pytest.mark.asyncio
async def test_create_http_clients_returns_one_client_per_vendor():
    config = _make_config()

    clients = await create_http_clients(config)
    try:
        assert isinstance(clients, HTTPClients)
        assert isinstance(clients.groq, httpx.AsyncClient)
        assert isinstance(clients.tts, httpx.AsyncClient)
        assert isinstance(clients.kite, httpx.AsyncClient)
        assert isinstance(clients.tavily, httpx.AsyncClient)
        # Each vendor gets a distinct client instance (no accidental sharing).
        assert len({id(clients.groq), id(clients.tts), id(clients.kite), id(clients.tavily)}) == 4
    finally:
        await close_http_clients(clients)


@pytest.mark.asyncio
async def test_groq_client_carries_auth_header():
    config = _make_config()

    clients = await create_http_clients(config)
    try:
        assert clients.groq.headers["Authorization"] == "Bearer groq-secret"
    finally:
        await close_http_clients(clients)


@pytest.mark.asyncio
async def test_tts_client_uses_cartesia_key_when_provider_is_cartesia():
    config = _make_config(tts_provider="cartesia")

    clients = await create_http_clients(config)
    try:
        assert clients.tts.headers["X-API-Key"] == "cartesia-secret"
    finally:
        await close_http_clients(clients)


@pytest.mark.asyncio
async def test_tts_client_uses_deepgram_key_when_provider_is_deepgram():
    config = _make_config(tts_provider="deepgram")

    clients = await create_http_clients(config)
    try:
        assert clients.tts.headers["Authorization"] == "Token deepgram-secret"
    finally:
        await close_http_clients(clients)


@pytest.mark.asyncio
async def test_close_http_clients_closes_all():
    config = _make_config()
    clients = await create_http_clients(config)

    await close_http_clients(clients)

    assert clients.groq.is_closed
    assert clients.tts.is_closed
    assert clients.kite.is_closed
    assert clients.tavily.is_closed


@pytest.mark.asyncio
async def test_create_http_clients_rejects_unsupported_provider():
    config = _make_config(tts_provider="elevenlabs")

    with pytest.raises(ValueError):
        await create_http_clients(config)


@pytest.mark.asyncio
async def test_groq_client_has_extended_timeout():
    config = _make_config()

    clients = await create_http_clients(config)
    try:
        assert clients.groq.timeout.read == 30.0
        assert clients.groq.timeout.connect == 10.0
    finally:
        await close_http_clients(clients)


@pytest.mark.asyncio
async def test_kite_client_has_auth_header_when_credentials_present():
    config = _make_config(kite_api_key="kite-key", kite_access_token="kite-token")

    clients = await create_http_clients(config)
    try:
        assert clients.kite.headers["Authorization"] == "token kite-key:kite-token"
    finally:
        await close_http_clients(clients)


@pytest.mark.asyncio
async def test_kite_client_has_no_auth_header_when_credentials_absent():
    config = _make_config(kite_api_key=None, kite_access_token=None)

    clients = await create_http_clients(config)
    try:
        assert "Authorization" not in clients.kite.headers
    finally:
        await close_http_clients(clients)
