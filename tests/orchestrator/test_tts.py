import httpx
import pytest

from financial_voice_agent.orchestrator.tts import (
    CARTESIA_OUTPUT_FORMAT,
    FISH_AUDIO_DEFAULT_MODEL,
    RealFishAudioTtsClient,
    TtsError,
    make_tts_client,
    synthesize_with_fallback,
)


class _FailNTimesClient:
    def __init__(self, fail_count: int, result: bytes = b"audio"):
        self._fail_count = fail_count
        self._result = result
        self.calls = 0

    async def synthesize(self, text: str) -> bytes:
        self.calls += 1
        if self.calls <= self._fail_count:
            raise RuntimeError("cartesia unavailable")
        return self._result


class _AlwaysFailsClient:
    def __init__(self):
        self.calls = 0

    async def synthesize(self, text: str) -> bytes:
        self.calls += 1
        raise RuntimeError("cartesia unavailable")


async def _instant_sleep(seconds: float) -> None:
    pass


@pytest.mark.asyncio
async def test_synthesize_with_fallback_succeeds_on_retry():
    primary = _FailNTimesClient(fail_count=1)

    result = await synthesize_with_fallback(primary, "hello", sleep_fn=_instant_sleep)

    assert result == b"audio"
    assert primary.calls == 2


@pytest.mark.asyncio
async def test_synthesize_with_fallback_uses_fallback_after_primary_exhausted():
    primary = _AlwaysFailsClient()
    fallback = _FailNTimesClient(fail_count=0, result=b"deepgram-audio")

    result = await synthesize_with_fallback(primary, "hello", fallback=fallback, sleep_fn=_instant_sleep)

    assert result == b"deepgram-audio"
    assert primary.calls == 2  # retry once = 2 total attempts


@pytest.mark.asyncio
async def test_synthesize_with_fallback_raises_ttserror_when_no_fallback_configured():
    primary = _AlwaysFailsClient()

    with pytest.raises(TtsError):
        await synthesize_with_fallback(primary, "hello", fallback=None, sleep_fn=_instant_sleep)


@pytest.mark.asyncio
async def test_synthesize_with_fallback_raises_ttserror_when_both_fail():
    primary = _AlwaysFailsClient()
    fallback = _AlwaysFailsClient()

    with pytest.raises(TtsError):
        await synthesize_with_fallback(primary, "hello", fallback=fallback, sleep_fn=_instant_sleep)


def test_cartesia_output_format_matches_prd_data_contract():
    assert CARTESIA_OUTPUT_FORMAT == {
        "container": "raw",
        "encoding": "pcm_s16le",
        "sample_rate": 16000,
    }


@pytest.mark.asyncio
async def test_fish_audio_client_sends_model_header_and_pcm_format():
    captured_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, content=b"raw-pcm-bytes")

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.fish.audio"
    )
    client = RealFishAudioTtsClient(http_client, model="s2.1-pro-free", reference_id="test-voice-id")

    result = await client.synthesize("Hello there")

    assert result == b"raw-pcm-bytes"
    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.headers["model"] == "s2.1-pro-free"
    import json
    body = json.loads(request.content)
    assert body == {
        "text": "Hello there",
        "format": "pcm",
        "sample_rate": 16000,
        "reference_id": "test-voice-id",
    }
    assert body["reference_id"] == "test-voice-id"


@pytest.mark.asyncio
async def test_fish_audio_client_defaults_to_free_model():
    captured_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, content=b"audio")

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.fish.audio"
    )
    client = RealFishAudioTtsClient(http_client, reference_id="test-voice-id")

    await client.synthesize("test")

    assert captured_requests[0].headers["model"] == FISH_AUDIO_DEFAULT_MODEL


@pytest.mark.asyncio
async def test_fish_audio_client_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"message": "Insufficient API credit", "status": 402})

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.fish.audio"
    )
    client = RealFishAudioTtsClient(http_client, reference_id="test-voice-id")

    with pytest.raises(httpx.HTTPStatusError):
        await client.synthesize("test")


def test_make_tts_client_returns_fish_audio_client_for_fish_audio_provider():
    class _FakeConfig:
        tts_provider = "fish_audio"
        fish_audio_model = "s2.1-pro"
        fish_audio_voice_id = "some-id"

    class _FakeHttpClients:
        tts = httpx.AsyncClient(base_url="https://api.fish.audio")

    client = make_tts_client(_FakeConfig(), _FakeHttpClients())

    assert isinstance(client, RealFishAudioTtsClient)
    assert client._model == "s2.1-pro"
    assert client._reference_id == "some-id"


def test_make_tts_client_uses_default_model_when_config_model_is_none():
    class _FakeConfig:
        tts_provider = "fish_audio"
        fish_audio_model = None
        fish_audio_voice_id = "some-id"

    class _FakeHttpClients:
        tts = httpx.AsyncClient(base_url="https://api.fish.audio")

    client = make_tts_client(_FakeConfig(), _FakeHttpClients())

    assert client._model == FISH_AUDIO_DEFAULT_MODEL
    assert client._reference_id == "some-id"


def test_make_tts_client_raises_for_unsupported_provider():
    class _FakeConfig:
        tts_provider = "elevenlabs"
        fish_audio_model = None
        fish_audio_voice_id = None

    class _FakeHttpClients:
        tts = None

    with pytest.raises(ValueError, match="elevenlabs"):
        make_tts_client(_FakeConfig(), _FakeHttpClients())
