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
    """


def _raise_if_rate_limited(data: dict) -> None:
    if "Note" in data or "Information" in data:
        raise _RateLimited(data.get("Note") or data.get("Information"))


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

    sector_key = sector.strip().lower()
    symbols = SECTOR_SYMBOLS.get(sector_key)
    if symbols is None:
        return {"error": f"Unknown sector: '{sector}'. Try: {', '.join(sorted(SECTOR_SYMBOLS))}"}

    try:
        rsi_results = await _rate_limited_map(
            symbols,
            lambda s: _fetch_rsi(http_client, s),
            interval_seconds=request_interval_seconds,
            sleep_fn=sleep_fn,
        )
    except _RateLimited:
        return {"error": "Screener temporarily rate-limited, try again in a minute"}
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
    except _RateLimited:
        return {"error": "Screener temporarily rate-limited, try again in a minute"}
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
