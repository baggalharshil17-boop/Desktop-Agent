from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class RetryExhaustedError(Exception):
    def __init__(self, message: str, *, last_exception: Exception | None = None):
        super().__init__(message)
        self.last_exception = last_exception


async def retry_with_fixed_backoff(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts - 1:
                await sleep_fn(backoff_seconds)
    raise RetryExhaustedError(f"failed after {max_attempts} attempts", last_exception=last_exc)


async def retry_with_exponential_backoff(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 4,
    base_delay_seconds: float = 1.0,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts - 1:
                await sleep_fn(base_delay_seconds * (2**attempt))
    raise RetryExhaustedError(f"failed after {max_attempts} attempts", last_exception=last_exc)
