import gzip

import httpx
import pytest

from financial_voice_agent.tools.kite_client import (
    KiteRateLimitedError,
    KiteSessionExpiredError,
    KiteUnavailableError,
    kite_get,
    kite_get_csv,
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


@pytest.mark.asyncio
async def test_kite_get_raises_session_expired_on_403_token_exception_without_retry():
    # Kite's real auth failures (expired/invalid access_token) come back as
    # 403 with error_type "TokenException", not 401 -- confirmed against a
    # real expired-token call live. Missing this meant every real session
    # expiry surfaced as a raw, uncaught HTTP error instead of the intended
    # "Kite session expired" fallback.
    client = _FakeHttpClient(
        [_FakeResponse(403, {"status": "error", "error_type": "TokenException", "message": "Incorrect api_key or access_token."})]
    )

    with pytest.raises(KiteSessionExpiredError):
        await kite_get(client, "/quote", sleep_fn=_instant_sleep)

    assert len(client.calls) == 1  # never retried


@pytest.mark.asyncio
async def test_kite_get_does_not_treat_plain_403_as_session_expired():
    # A 403 without error_type "TokenException" is a different problem (e.g.
    # permission/subscription) -- must propagate as a normal HTTP error, not
    # get misreported as "log in again".
    client = _FakeHttpClient(
        [_FakeResponse(403, {"status": "error", "error_type": "PermissionException", "message": "no access"})]
    )

    with pytest.raises(httpx.HTTPStatusError):
        await kite_get(client, "/quote", sleep_fn=_instant_sleep)


class _FakeCsvResponse:
    def __init__(self, status_code: int, content: bytes = b"", json_data: dict | None = None):
        self.status_code = status_code
        self.content = content
        self._json_data = json_data or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)


@pytest.mark.asyncio
async def test_kite_get_csv_decodes_gzipped_body():
    raw_csv = b"instrument_token,tradingsymbol\r\n128031234,RELIANCE\r\n"
    client = _FakeHttpClient([_FakeCsvResponse(200, gzip.compress(raw_csv))])

    result = await kite_get_csv(client, "/instruments/NSE")

    assert result == raw_csv.decode("utf-8")


@pytest.mark.asyncio
async def test_kite_get_csv_handles_plain_uncompressed_body():
    raw_csv = b"instrument_token,tradingsymbol\r\n128031234,RELIANCE\r\n"
    client = _FakeHttpClient([_FakeCsvResponse(200, raw_csv)])

    result = await kite_get_csv(client, "/instruments/NSE")

    assert result == raw_csv.decode("utf-8")


@pytest.mark.asyncio
async def test_kite_get_csv_raises_session_expired_on_401():
    client = _FakeHttpClient([_FakeCsvResponse(401)])

    with pytest.raises(KiteSessionExpiredError):
        await kite_get_csv(client, "/instruments/NSE", sleep_fn=_instant_sleep)


@pytest.mark.asyncio
async def test_kite_get_csv_raises_session_expired_on_403_token_exception():
    client = _FakeHttpClient(
        [_FakeCsvResponse(403, json_data={"error_type": "TokenException", "message": "bad token"})]
    )

    with pytest.raises(KiteSessionExpiredError):
        await kite_get_csv(client, "/instruments/NSE", sleep_fn=_instant_sleep)
