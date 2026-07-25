from __future__ import annotations

import asyncio
from typing import Awaitable, Callable


class KiteSessionExpiredError(Exception):
    pass


class KiteRateLimitedError(Exception):
    pass


class KiteUnavailableError(Exception):
    pass


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
        if response.status_code == 401:
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
