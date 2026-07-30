import httpx
import pytest

from financial_voice_agent.setup.validators import (
    validate_cartesia_key,
    validate_fish_audio_key,
    validate_groq_key,
    validate_huggingface_key,
    validate_tavily_key,
)


class _MockResponse:
    """Mock response object for groq exception."""

    request = None
    status_code = 401
    text = "Unauthorized"
    content = b"Unauthorized"


class _FakeGroqModels:
    def __init__(self, *, should_raise=False):
        self._should_raise = should_raise

    def list(self):
        if self._should_raise:
            import groq

            raise groq.AuthenticationError("bad key", response=_MockResponse(), body=None)
        return ["model-a", "model-b"]


class _FakeGroqClient:
    def __init__(self, *, api_key, should_raise=False):
        self.models = _FakeGroqModels(should_raise=should_raise)


def test_validate_groq_key_ok_on_successful_list_call():
    result = validate_groq_key(
        "test-key", client_factory=lambda api_key: _FakeGroqClient(api_key=api_key)
    )

    assert result.ok is True


def test_validate_groq_key_reports_failure_on_bad_key():
    result = validate_groq_key(
        "bad-key",
        client_factory=lambda api_key: _FakeGroqClient(api_key=api_key, should_raise=True),
    )

    assert result.ok is False
    assert "bad key" in result.message.lower() or "authentication" in result.message.lower()


def test_validate_huggingface_key_ok_on_successful_whoami():
    result = validate_huggingface_key("test-token", whoami_fn=lambda token: {"name": "someone"})

    assert result.ok is True


def test_validate_huggingface_key_reports_failure_on_bad_token():
    import huggingface_hub.errors

    def failing_whoami(token):
        raise huggingface_hub.errors.HfHubHTTPError("bad token")

    result = validate_huggingface_key("bad-token", whoami_fn=failing_whoami)

    assert result.ok is False


class _FakeVoice:
    def __init__(self, id, name):
        self.id = id
        self.name = name


class _FakeCartesiaVoices:
    def __init__(self, *, should_raise=False):
        self._should_raise = should_raise

    def list(self, *, limit):
        if self._should_raise:
            import cartesia

            raise cartesia.AuthenticationError("bad key", response=_MockResponse(), body=None)
        return [_FakeVoice("voice-1", "Friendly Guide"), _FakeVoice("voice-2", "Calm Narrator")]


class _FakeCartesiaClient:
    def __init__(self, *, api_key, should_raise=False):
        self.voices = _FakeCartesiaVoices(should_raise=should_raise)


def test_validate_cartesia_key_ok_and_returns_voices_on_success():
    result = validate_cartesia_key(
        "test-key", client_factory=lambda api_key: _FakeCartesiaClient(api_key=api_key)
    )

    assert result.ok is True
    assert result.data["voices"] == [("voice-1", "Friendly Guide"), ("voice-2", "Calm Narrator")]


def test_validate_cartesia_key_reports_failure_on_bad_key():
    result = validate_cartesia_key(
        "bad-key",
        client_factory=lambda api_key: _FakeCartesiaClient(api_key=api_key, should_raise=True),
    )

    assert result.ok is False


@pytest.mark.asyncio
async def test_validate_tavily_key_ok_on_200_response():
    def handler(request):
        return httpx.Response(200, json={"results": []})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://api.tavily.com", transport=transport)

    result = validate_tavily_key("test-key", http_client=client)

    assert result.ok is True


@pytest.mark.asyncio
async def test_validate_tavily_key_reports_failure_on_401():
    def handler(request):
        return httpx.Response(401, json={"error": "invalid api key"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://api.tavily.com", transport=transport)

    result = validate_tavily_key("bad-key", http_client=client)

    assert result.ok is False


def test_validate_fish_audio_key_ok_on_200_response():
    def handler(request):
        return httpx.Response(200, content=b"audio-bytes")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://api.fish.audio", transport=transport)

    result = validate_fish_audio_key("test-key", http_client=client)

    assert result.ok is True


def test_validate_fish_audio_key_reports_failure_on_402():
    def handler(request):
        return httpx.Response(402, json={"message": "Insufficient API credit", "status": 402})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://api.fish.audio", transport=transport)

    result = validate_fish_audio_key("test-key", http_client=client)

    assert result.ok is False
