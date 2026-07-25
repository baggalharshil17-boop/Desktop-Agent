import pytest

from financial_voice_agent.orchestrator.tts import CARTESIA_OUTPUT_FORMAT, TtsError, synthesize_with_fallback


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
