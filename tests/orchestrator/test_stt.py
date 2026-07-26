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


class _FakeAsrOutput:
    def __init__(self, text: str):
        self.text = text


class _FakeHfInferenceClient:
    def __init__(self, output):
        self._output = output
        self.received_audio = None
        self.received_model = None

    async def automatic_speech_recognition(self, audio, *, model=None):
        self.received_audio = audio
        self.received_model = model
        return self._output


@pytest.mark.asyncio
async def test_huggingface_stt_client_calls_automatic_speech_recognition_with_wav_bytes():
    client = _FakeHfInferenceClient(_FakeAsrOutput("hf transcribed text"))
    adapter = HuggingFaceSttClient(client)

    result = await adapter.transcribe(b"wav-bytes", model="openai/whisper-large-v3-turbo")

    assert result == "hf transcribed text"
    assert client.received_audio == b"wav-bytes"
    assert client.received_model == "openai/whisper-large-v3-turbo"


@pytest.mark.asyncio
async def test_huggingface_stt_client_handles_plain_string_output():
    client = _FakeHfInferenceClient("hf transcribed text")
    adapter = HuggingFaceSttClient(client)

    result = await adapter.transcribe(b"wav-bytes", model="openai/whisper-large-v3-turbo")

    assert result == "hf transcribed text"


class _FakeConfig:
    def __init__(self, stt_provider: str, groq_api_key: str = "groq-secret", huggingface_api_key: str = "hf-secret"):
        self.stt_provider = stt_provider
        self.groq_api_key = groq_api_key
        self.huggingface_api_key = huggingface_api_key


def test_make_stt_client_returns_huggingface_client_for_huggingface_provider():
    config = _FakeConfig(stt_provider="huggingface")

    client = make_stt_client(config)

    assert isinstance(client, HuggingFaceSttClient)


def test_make_stt_client_returns_groq_client_for_groq_provider():
    config = _FakeConfig(stt_provider="groq")

    client = make_stt_client(config)

    assert isinstance(client, RealGroqSttClient)


def test_make_stt_client_raises_for_unknown_provider():
    config = _FakeConfig(stt_provider="whisper-cpp")

    with pytest.raises(ValueError, match="whisper-cpp"):
        make_stt_client(config)
