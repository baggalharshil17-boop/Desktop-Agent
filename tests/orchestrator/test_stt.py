import pytest

from financial_voice_agent.orchestrator.stt import (
    HuggingFaceSttClient,
    RealGroqSttClient,
    SttError,
    make_stt_client,
    transcribe_with_retry,
)


class _FailNTimesClient:
    def __init__(self, fail_count: int, result: str = "transcribed text"):
        self._fail_count = fail_count
        self._result = result
        self.calls = 0

    async def transcribe(self, wav_bytes: bytes, *, model: str) -> str:
        self.calls += 1
        if self.calls <= self._fail_count:
            raise RuntimeError("groq unavailable")
        return self._result


class _AlwaysFailsClient:
    async def transcribe(self, wav_bytes: bytes, *, model: str) -> str:
        raise RuntimeError("groq unavailable")


@pytest.mark.asyncio
async def test_transcribe_with_retry_succeeds_after_transient_failures():
    async def instant_sleep(seconds):
        pass

    client = _FailNTimesClient(fail_count=2)

    result = await transcribe_with_retry(
        client, b"wav-bytes", model="whisper-large-v3-turbo", sleep_fn=instant_sleep
    )

    assert result == "transcribed text"
    assert client.calls == 3


@pytest.mark.asyncio
async def test_transcribe_with_retry_raises_sttexception_after_exhaustion():
    client = _AlwaysFailsClient()

    with pytest.raises(SttError):
        await transcribe_with_retry(client, b"wav-bytes", model="whisper-large-v3-turbo")


@pytest.mark.asyncio
async def test_real_groq_stt_client_calls_transcriptions_create_with_wav_bytes():
    class _FakeTranscriptionsResource:
        def __init__(self):
            self.received_kwargs = None

        async def create(self, **kwargs):
            self.received_kwargs = kwargs
            return "adapter transcribed text"

    class _FakeAudioNamespace:
        def __init__(self, transcriptions):
            self.transcriptions = transcriptions

    class _FakeGroqAsyncClient:
        def __init__(self, transcriptions):
            self.audio = _FakeAudioNamespace(transcriptions)

    transcriptions = _FakeTranscriptionsResource()
    fake_groq_client = _FakeGroqAsyncClient(transcriptions)
    adapter = RealGroqSttClient(fake_groq_client)

    result = await adapter.transcribe(b"wav-bytes", model="whisper-large-v3-turbo")

    assert result == "adapter transcribed text"
    assert transcriptions.received_kwargs["model"] == "whisper-large-v3-turbo"
    assert transcriptions.received_kwargs["file"][1] == b"wav-bytes"


class _FakeHfResponse:
    def __init__(self, json_data: dict):
        self._json_data = json_data
        self.raise_for_status_called = False

    def raise_for_status(self):
        self.raise_for_status_called = True

    def json(self):
        return self._json_data


class _FakeHfHttpClient:
    def __init__(self, response: _FakeHfResponse):
        self._response = response
        self.received_path = None
        self.received_content = None
        self.received_headers = None

    async def post(self, path, *, content=None, headers=None):
        self.received_path = path
        self.received_content = content
        self.received_headers = headers
        return self._response


@pytest.mark.asyncio
async def test_huggingface_stt_client_calls_models_endpoint_with_wav_bytes():
    response = _FakeHfResponse({"text": "hf transcribed text"})
    http_client = _FakeHfHttpClient(response)
    adapter = HuggingFaceSttClient(http_client)

    result = await adapter.transcribe(b"wav-bytes", model="openai/whisper-large-v3")

    assert result == "hf transcribed text"
    assert http_client.received_path == "/models/openai/whisper-large-v3"
    assert http_client.received_content == b"wav-bytes"
    assert http_client.received_headers["Content-Type"] == "audio/wav"
    assert response.raise_for_status_called is True


@pytest.mark.asyncio
async def test_huggingface_stt_client_raises_sttexception_on_unexpected_response_shape():
    # Mirrors HF's real "model is loading" cold-start response shape.
    response = _FakeHfResponse({"error": "Model is currently loading", "estimated_time": 20.0})
    adapter = HuggingFaceSttClient(_FakeHfHttpClient(response))

    with pytest.raises(SttError, match="unexpected response shape"):
        await adapter.transcribe(b"wav-bytes", model="openai/whisper-large-v3")


class _FakeConfig:
    def __init__(self, stt_provider: str, groq_api_key: str = "groq-secret"):
        self.stt_provider = stt_provider
        self.groq_api_key = groq_api_key


class _FakeHttpClients:
    def __init__(self, huggingface=None):
        self.huggingface = huggingface


def test_make_stt_client_returns_huggingface_client_for_huggingface_provider():
    config = _FakeConfig(stt_provider="huggingface")
    http_clients = _FakeHttpClients(huggingface="fake-hf-http-client")

    client = make_stt_client(config, http_clients)

    assert isinstance(client, HuggingFaceSttClient)


def test_make_stt_client_returns_groq_client_for_groq_provider():
    config = _FakeConfig(stt_provider="groq")
    http_clients = _FakeHttpClients()

    client = make_stt_client(config, http_clients)

    assert isinstance(client, RealGroqSttClient)


def test_make_stt_client_raises_for_unknown_provider():
    config = _FakeConfig(stt_provider="whisper-cpp")
    http_clients = _FakeHttpClients()

    with pytest.raises(ValueError, match="whisper-cpp"):
        make_stt_client(config, http_clients)
