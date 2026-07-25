import httpx
import pytest

from financial_voice_agent.tools.kite_client import (
    KiteRateLimitedError,
    KiteSessionExpiredError,
    KiteUnavailableError,
    kite_get,
)


class _FakeResponse:
    def __init__(self, status_code: int, json_data: dict | None = None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)


class _FakeHttpClient:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict | None]] = []

    async def get(self, path, params=None):
        self.calls.append((path, params))
        return self._responses.pop(0)


async def _instant_sleep(seconds: float) -> None:
    pass


@pytest.mark.asyncio
async def test_kite_get_returns_json_on_success():
    client = _FakeHttpClient([_FakeResponse(200, {"data": "ok"})])

    result = await kite_get(client, "/quote", params={"i": "NIFTY 50"})

    assert result == {"data": "ok"}
    assert client.calls == [("/quote", {"i": "NIFTY 50"})]


@pytest.mark.asyncio
async def test_kite_get_raises_session_expired_on_401_without_retry():
    client = _FakeHttpClient([_FakeResponse(401)])

    with pytest.raises(KiteSessionExpiredError):
        await kite_get(client, "/quote", sleep_fn=_instant_sleep)

    assert len(client.calls) == 1  # never retried


@pytest.mark.asyncio
async def test_kite_get_raises_rate_limited_on_429_without_retry():
    client = _FakeHttpClient([_FakeResponse(429)])

    with pytest.raises(KiteRateLimitedError):
        await kite_get(client, "/quote", sleep_fn=_instant_sleep)

    assert len(client.calls) == 1  # never retried


@pytest.mark.asyncio
async def test_kite_get_retries_5xx_then_succeeds():
    client = _FakeHttpClient([_FakeResponse(503), _FakeResponse(200, {"data": "ok"})])

    result = await kite_get(client, "/quote", sleep_fn=_instant_sleep)

    assert result == {"data": "ok"}
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_kite_get_raises_unavailable_after_exhausting_5xx_retries():
    client = _FakeHttpClient([_FakeResponse(503), _FakeResponse(503), _FakeResponse(503)])

    with pytest.raises(KiteUnavailableError):
        await kite_get(client, "/quote", max_attempts=3, sleep_fn=_instant_sleep)

    assert len(client.calls) == 3
