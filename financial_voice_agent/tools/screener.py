from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

import httpx

from financial_voice_agent import mock

T = TypeVar("T")

# Small, explicit sector -> NSE/BSE symbol list. Not exhaustive; extend as
# needed. Alpha Vantage requires an exchange suffix -- .BSE used here.
SECTOR_SYMBOLS: dict[str, list[str]] = {
    "tech": ["TCS.BSE", "INFY.BSE", "WIPRO.BSE", "HCLTECH.BSE", "TECHM.BSE"],
    "finance": ["HDFCBANK.BSE", "ICICIBANK.BSE", "SBIN.BSE", "KOTAKBANK.BSE", "AXISBANK.BSE"],
    "pharma": ["SUNPHARMA.BSE", "DRREDDY.BSE", "CIPLA.BSE", "DIVISLAB.BSE", "LUPIN.BSE"],
    "auto": ["MARUTI.BSE", "TATAMOTORS.BSE", "M&M.BSE", "BAJAJ-AUTO.BSE", "EICHERMOT.BSE"],
}

# Alpha Vantage's free tier caps at 5 requests/minute (60/5=12s exactly);
# 13s adds margin. A full screen makes up to 10 calls (RSI for every symbol,
# then price only for symbols that already passed the RSI filter, capped to
# `limit` -- see screen_stocks), so a worst-case screen takes ~2 minutes.
# Slow, but staying under the vendor's rate limit matters more than latency
# for an occasional screening query.
DEFAULT_REQUEST_INTERVAL_SECONDS = 13.0

# The orchestrator dispatches every tool call in an LLM turn concurrently
# (asyncio.gather), but _rate_limited_map's pacing is per-invocation state --
# it has no idea another screen_stocks call is in flight. Two concurrent
# screen_stocks calls (e.g. "compare momentum in tech versus pharma") would
# interleave their independently-paced requests at ~2x the intended rate,
# breaking Alpha Vantage's free-tier limit. Serializing on this lock makes
# only one screen_stocks call make requests at a time, so the existing
# per-call pacer stays correct by construction.
_SCREEN_LOCK = asyncio.Lock()


async def _rate_limited_map(
    items: list[T],
    fn: Callable[[T], Awaitable],
    *,
    interval_seconds: float,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> list:
    """Runs fn(item) for each item sequentially, sleeping interval_seconds
    between dispatches.

    Running these concurrently (asyncio.gather) would blow through Alpha
    Vantage's free-tier rate limit in under a second for a 5-symbol sector
    list -- sequential + spaced dispatch is the simplest way to stay under
    it without a full token-bucket limiter.
    """
    results = []
    for i, item in enumerate(items):
        if i > 0:
            await sleep_fn(interval_seconds)
        results.append(await fn(item))
    return results


class _RateLimited(Exception):
    """Alpha Vantage's throttling signal.

    A throttled call returns HTTP 200 with a "Note" or "Information" key
    instead of the expected data key -- not an HTTP 429 -- so it can't be
    caught via raise_for_status() and must be detected from the response
    body. Raised (not returned as None like other per-symbol failures) so
    screen_stocks can surface one clear rate-limit error instead of
    silently returning an empty result when every symbol gets throttled.

    Carries Alpha Vantage's original message text (args[0]) so the caller
    can distinguish the per-minute throttle from the per-day quota, which
    share this exact same response shape.
    """

    @property
    def message(self) -> str | None:
        return self.args[0] if self.args else None


def _raise_if_rate_limited(data: dict) -> None:
    if "Note" in data or "Information" in data:
        raise _RateLimited(data.get("Note") or data.get("Information"))


# Alpha Vantage returns the identical 200-with-"Note"/"Information" shape for
# BOTH the 5-requests/minute throttle and the 25-requests/day quota. A single
# screen can burn up to 10 of the 25 daily requests, so the daily cap is the
# common failure mode (users hit it around the third screen of a day), not an
# edge case -- telling them "try again in a minute" when the day's quota is
# actually exhausted is actively misleading. Alpha Vantage's daily-cap copy
# reliably mentions "per day" / "requests per day" / "daily", so branch on that.
_DAILY_LIMIT_MARKERS = ("per day", "requests per day", "daily")


def _is_daily_limit_message(message: str | None) -> bool:
    if not message:
        return False
    lowered = message.lower()
    return any(marker in lowered for marker in _DAILY_LIMIT_MARKERS)


def _rate_limit_error(message: str | None) -> dict:
    if _is_daily_limit_message(message):
        return {"error": "Screener has hit its daily data limit, try again tomorrow"}
    return {"error": "Screener temporarily rate-limited, try again in a minute"}


async def _fetch_rsi(http_client, symbol: str) -> dict | None:
    """Fetches the latest RSI value for one symbol.

    Returns None (not an exception) on any per-symbol failure -- a single
    bad symbol must not fail the whole screen, it should just be skipped.
    Rate-limiting is the one exception to that: see _RateLimited.
    """
    try:
        response = await http_client.get(
            "/query",
            params={
                "function": "RSI",
                "symbol": symbol,
                "interval": "daily",
                "time_period": 14,
                "series_type": "close",
            },
        )
        response.raise_for_status()
        data = response.json()
        _raise_if_rate_limited(data)
        technical_analysis = data.get("Technical Analysis: RSI", {})
        if not technical_analysis:
            return None
        latest_date = max(technical_analysis.keys())
        rsi = float(technical_analysis[latest_date]["RSI"])
        return {"symbol": symbol, "rsi": rsi}
    except _RateLimited:
        raise
    except httpx.NetworkError:  # Network errors propagate to fail the whole screen
        raise
    except Exception:  # noqa: BLE001 -- any other per-symbol failure just drops that symbol
        return None


async def _fetch_price(http_client, symbol: str) -> float | None:
    """Fetches the latest close price for one symbol, for price-range filtering.

    Returns None (not an exception) on any per-symbol failure including network errors --
    a single bad symbol must not fail the whole screen, it should just be skipped.
    Rate-limiting is the one exception to that: see _RateLimited.
    """
    try:
        response = await http_client.get(
            "/query", params={"function": "GLOBAL_QUOTE", "symbol": symbol}
        )
        response.raise_for_status()
        data = response.json()
        _raise_if_rate_limited(data)
        quote = data.get("Global Quote", {})
        price = quote.get("05. price")
        return float(price) if price else None
    except _RateLimited:
        raise
    except Exception:  # noqa: BLE001 -- any failure (network, malformed data, etc.) drops this symbol
        return None


def _coerce_numeric_params(
    momentum_threshold, price_min, price_max, limit
) -> tuple[float, float, float | None, int] | dict:
    """Coerces model-supplied numeric args to their expected types up front.

    LLMs routinely emit numbers as strings (e.g. "limit": "2"). Left
    uncoerced, that TypeErrors deep inside screen_stocks (e.g. slicing
    candidates[:limit]) -- but only AFTER all RSI network calls have already
    completed, burning ~52s and 5 of the daily 25 requests for nothing.
    Coercing here fails fast and for free instead. Returns an error dict
    (not raising) if a value is genuinely unparseable, so the caller can
    still produce a clean {"error": ...} rather than an uncaught exception.
    """
    try:
        momentum_threshold = float(momentum_threshold)
        price_min = float(price_min)
        price_max = float(price_max) if price_max is not None else None
        limit = int(limit)
    except (TypeError, ValueError) as exc:
        return {"error": f"Invalid numeric parameter for screen_stocks: {exc}"}
    return momentum_threshold, price_min, price_max, limit


async def screen_stocks(
    sector: str,
    *,
    http_client,
    mode: str = "live",
    fixtures_dir: str = "fixtures",
    momentum_threshold: float = 70.0,
    price_min: float = 0.0,
    price_max: float | None = None,
    limit: int = 10,
    request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict:
    """Screens a sector's symbols by RSI momentum and price range.

    Price is fetched only for the top `limit` RSI-passing candidates (not
    every candidate), to keep total Alpha Vantage calls bounded even when
    every symbol in a sector clears the momentum threshold. This means the
    final result can have fewer than `limit` entries (if some of those top
    candidates fail the price filter) but never more.
    """
    if mode == "mock":
        return mock.load_fixture("stock_screener", fixtures_dir=fixtures_dir)

    coerced = _coerce_numeric_params(momentum_threshold, price_min, price_max, limit)
    if isinstance(coerced, dict):
        return coerced
    momentum_threshold, price_min, price_max, limit = coerced

    sector_key = sector.strip().lower()
    symbols = SECTOR_SYMBOLS.get(sector_key)
    if symbols is None:
        return {"error": f"Unknown sector: '{sector}'. Try: {', '.join(sorted(SECTOR_SYMBOLS))}"}

    # Serialize concurrent screen_stocks calls: the orchestrator dispatches
    # every tool call in an LLM turn via asyncio.gather, and _rate_limited_map's
    # pacing is per-invocation state, so two concurrent calls would interleave
    # their requests at ~2x the intended rate without this lock.
    async with _SCREEN_LOCK:
        try:
            rsi_results = await _rate_limited_map(
                symbols,
                lambda s: _fetch_rsi(http_client, s),
                interval_seconds=request_interval_seconds,
                sleep_fn=sleep_fn,
            )
        except _RateLimited as exc:
            return _rate_limit_error(exc.message)
        except httpx.NetworkError:
            return {"error": f"Could not reach the screener data provider for sector '{sector}'"}
        except Exception:  # noqa: BLE001 -- other unexpected errors must degrade gracefully
            return {"error": f"Could not reach the screener data provider for sector '{sector}'"}

        candidates = [r for r in rsi_results if r is not None and r["rsi"] >= momentum_threshold]
        if not candidates:
            return {"results": [], "count": 0}

        candidates.sort(key=lambda c: c["rsi"], reverse=True)
        candidates = candidates[:limit]

        # Ensure pacing gap between RSI phase and price phase to maintain rate limit compliance.
        # _rate_limited_map restarts its counter for each call, so there's no sleep between
        # the last RSI request and the first price request without this explicit gap.
        await sleep_fn(request_interval_seconds)

        try:
            prices = await _rate_limited_map(
                candidates,
                lambda c: _fetch_price(http_client, c["symbol"]),
                interval_seconds=request_interval_seconds,
                sleep_fn=sleep_fn,
            )
        except _RateLimited as exc:
            return _rate_limit_error(exc.message)
        for candidate, price in zip(candidates, prices):
            candidate["price"] = price

    filtered = [
        c
        for c in candidates
        if c["price"] is not None
        and c["price"] >= price_min
        and (price_max is None or c["price"] <= price_max)
    ]

    return {
        "results": [
            {"symbol": c["symbol"], "price": c["price"], "rsi": round(c["rsi"], 1)} for c in filtered
        ],
        "count": len(filtered),
    }
