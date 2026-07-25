import pytest

from financial_voice_agent.orchestrator.retry import (
    RetryExhaustedError,
    retry_with_exponential_backoff,
    retry_with_fixed_backoff,
)


class _FakeSleeper:
    def __init__(self):
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.mark.asyncio
async def test_retry_with_fixed_backoff_returns_on_eventual_success():
    attempts = {"count": 0}

    async def fn():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("not yet")
        return "ok"

    sleeper = _FakeSleeper()
    result = await retry_with_fixed_backoff(fn, max_attempts=3, backoff_seconds=1.0, sleep_fn=sleeper)

    assert result == "ok"
    assert attempts["count"] == 3
    assert sleeper.calls == [1.0, 1.0]


@pytest.mark.asyncio
async def test_retry_with_fixed_backoff_raises_after_exhausting_attempts():
    async def fn():
        raise RuntimeError("always fails")

    sleeper = _FakeSleeper()
    with pytest.raises(RetryExhaustedError) as exc_info:
        await retry_with_fixed_backoff(fn, max_attempts=3, backoff_seconds=1.0, sleep_fn=sleeper)

    assert isinstance(exc_info.value.last_exception, RuntimeError)
    assert sleeper.calls == [1.0, 1.0]  # sleeps between attempts, not after the last


@pytest.mark.asyncio
async def test_retry_with_exponential_backoff_doubles_delay_each_attempt():
    async def fn():
        raise RuntimeError("always fails")

    sleeper = _FakeSleeper()
    with pytest.raises(RetryExhaustedError):
        await retry_with_exponential_backoff(fn, max_attempts=4, base_delay_seconds=1.0, sleep_fn=sleeper)

    assert sleeper.calls == [1.0, 2.0, 4.0]
