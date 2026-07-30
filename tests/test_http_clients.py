import httpx
import pytest

from financial_voice_agent.config import Config
from financial_voice_agent.http_clients import HTTPClients, close_http_clients, create_http_clients


def _make_config(
    tts_provider="cartesia",
    kite_api_key="kite-key",
    kite_access_token="kite-token",
) -> Config:
    return Config(
        vad_speech_threshold=0.5,
        vad_silence_duration_ms=600,
        vad_min_speech_duration_ms=200,
        barge_in_enabled=True,
        barge_in_min_speech_ms=96.0,
        audio_output_device_index=None,
        audio_input_device_index=None,
        echo_suppression_enabled=True,
        echo_margin=2.0,
        echo_gain=None,
        input_mode="always_on",
        tts_provider=tts_provider,
        cartesia_voice_id="test-voice-id",
        fish_audio_model=None,
        stt_provider="huggingface",
        stt_model="openai/whisper-large-v3",
        llm_provider="huggingface",
        llm_model="test-model",
        storage_db_path="./agent_turns.db",
        mode="live",
        processing_overlay_enabled=True,
        groq_api_key="groq-secret",
        cartesia_api_key="cartesia-secret",
        fish_audio_api_key="fish-audio-secret",
        huggingface_api_key="hf-secret",
        kite_api_key=kite_api_key,
        kite_access_token=kite_access_token,
        tavily_api_key="tavily-key",
        indian_stock_api_key="indian-stock-secret",
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
        assert isinstance(clients.indian_stock, httpx.AsyncClient)
        # Each vendor gets a distinct client instance (no accidental sharing).
        assert len({
            id(clients.groq), id(clients.tts), id(clients.kite), id(clients.tavily), id(clients.indian_stock)
        }) == 5
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
async def test_tts_client_uses_fish_audio_key_when_provider_is_fish_audio():
    config = _make_config(tts_provider="fish_audio")

    clients = await create_http_clients(config)
    try:
        assert clients.tts.headers["Authorization"] == "Bearer fish-audio-secret"
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


@pytest.mark.asyncio
async def test_kite_client_sends_x_kite_version_header():
    config = _make_config()

    clients = await create_http_clients(config)
    try:
        assert clients.kite.headers["X-Kite-Version"] == "3"
    finally:
        await close_http_clients(clients)


@pytest.mark.asyncio
async def test_indian_stock_client_carries_api_key_header():
    config = _make_config()

    clients = await create_http_clients(config)
    try:
        assert clients.indian_stock.headers["X-Api-Key"] == "indian-stock-secret"
    finally:
        await close_http_clients(clients)
