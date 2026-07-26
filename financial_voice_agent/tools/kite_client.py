from __future__ import annotations

import asyncio
import gzip
from typing import Awaitable, Callable


class KiteSessionExpiredError(Exception):
    pass


class KiteRateLimitedError(Exception):
    pass


class KiteUnavailableError(Exception):
    pass


def _is_token_exception(response) -> bool:
    """Kite's real auth failures (expired/invalid access_token) come back
    as HTTP 403 with {"error_type": "TokenException"}, NOT 401 (confirmed
    against kite.trade/docs/connect/v3/exceptions and a real expired-token
    call -- the initial 401-only check here missed every real occurrence of
    this in live use). A plain 403 without that error_type is a different
    problem (e.g. permission/subscription) and must not be treated as
    session-expired."""
    if response.status_code not in (401, 403):
        return False
    if response.status_code == 401:
        return True
    try:
        return response.json().get("error_type") == "TokenException"
    except Exception:  # noqa: BLE001 -- a non-JSON body just means "not a TokenException"
        return False


async def kite_get(
    http_client,
    path: str,
    *,
    params: dict | None = None,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        response = await http_client.get(path, params=params)
        if _is_token_exception(response):
            raise KiteSessionExpiredError("Kite session expired")
        if response.status_code == 429:
            raise KiteRateLimitedError("Kite Connect rate limit hit")
        if response.status_code >= 500:
            last_exc = KiteUnavailableError(f"Kite Connect returned {response.status_code}")
            if attempt < max_attempts - 1:
                await sleep_fn(backoff_seconds)
            continue
        response.raise_for_status()
        return response.json()
    raise last_exc


async def kite_get_csv(
    http_client,
    path: str,
    *,
    params: dict | None = None,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> str:
    """Like kite_get, but for endpoints that return a CSV body (the
    instruments dump) instead of JSON -- e.g. GET /instruments/:exchange.

    Kite serves this as a gzip file body without a Content-Encoding header,
    so httpx won't auto-decompress it the way it would for a normal gzip
    HTTP response; decode manually when the gzip magic bytes are present.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        response = await http_client.get(path, params=params)
        if _is_token_exception(response):
            raise KiteSessionExpiredError("Kite session expired")
        if response.status_code == 429:
            raise KiteRateLimitedError("Kite Connect rate limit hit")
        if response.status_code >= 500:
            last_exc = KiteUnavailableError(f"Kite Connect returned {response.status_code}")
            if attempt < max_attempts - 1:
                await sleep_fn(backoff_seconds)
            continue
        response.raise_for_status()
        content = response.content
        if content[:2] == b"\x1f\x8b":
            content = gzip.decompress(content)
        return content.decode("utf-8")
    raise last_exc
