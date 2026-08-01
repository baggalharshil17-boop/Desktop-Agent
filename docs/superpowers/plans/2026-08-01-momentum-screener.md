# Momentum Stock Screener Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `screen_stocks` tool that lets the voice agent answer "show me the top N momentum stocks in [sector]" by querying Alpha Vantage for RSI/price data, ranking, and filtering — no UI automation against Kite.

**Architecture:** A new `financial_voice_agent/tools/screener.py` module (same shape as the existing `tools/fundamentals.py`) does the sector→symbol lookup and Alpha Vantage calls. It's wired into the existing tool-dispatch pattern in `tools/registry.py`, with a new `alpha_vantage` HTTP client added to `http_clients.py` alongside the existing per-vendor clients.

**Tech Stack:** `httpx.AsyncClient` (existing pattern), Alpha Vantage REST API (`https://www.alphavantage.co/query`, `apikey` query-param auth).

## Global Constraints

- Read-only. No order placement, no UI automation, no write access to Kite.
- Alpha Vantage free tier: 5 requests/minute (verify against https://www.alphavantage.co/premium/ before shipping, since vendor limits change — this plan builds in a rate limiter regardless of the exact current number).
- Sector-to-symbol mapping is a small hardcoded dict (tech, finance, pharma, auto) — not a general-purpose sector database.
- Indian Stock API as a screening data source is explicitly out of scope for this plan (deferred).
- New tool must be wired into `TOOLS_SCHEMA` and the `_dispatch` function in `financial_voice_agent/tools/registry.py`, following the exact pattern `get_stock_fundamentals` already uses there.
- Must support `mode: "mock"` via a fixture, consistent with every other live-data tool (see `mock.load_fixture` usage in `fundamentals.py`).
- `ALPHA_VANTAGE_API_KEY` is env-only (like `INDIAN_STOCK_API_KEY`) — no `scripts/setup.py` wizard integration in this plan, since the existing Indian Stock API key isn't wizard-integrated either and this plan follows that same lighter pattern.

---

### Task 1: Screener module — sector lookup, Alpha Vantage calls, filtering/ranking

**Files:**
- Create: `financial_voice_agent/tools/screener.py`
- Create: `fixtures/stock_screener.json`
- Test: `tests/tools/test_screener.py`

**Interfaces:**
- Consumes: `financial_voice_agent.mock.load_fixture(name, fixtures_dir)` (existing, used exactly as `fundamentals.py` uses it).
- Produces: `async def screen_stocks(sector: str, *, http_client, mode: str = "live", fixtures_dir: str = "fixtures", momentum_threshold: float = 70.0, price_min: float = 0.0, price_max: float | None = None, limit: int = 10, request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS, sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep) -> dict`. Also exports `SECTOR_SYMBOLS: dict[str, list[str]]`. Task 3 imports `screen_stocks` from this module.

- [ ] **Step 1: Write the failing tests**

Create `tests/tools/test_screener.py`:

```python
import httpx
import pytest

from financial_voice_agent.tools.screener import SECTOR_SYMBOLS, screen_stocks


def _rsi_response(rsi_value: float, date: str = "2026-08-01") -> dict:
    return {"Technical Analysis: RSI": {date: {"RSI": str(rsi_value)}}}


def _quote_response(price: float) -> dict:
    return {"Global Quote": {"05. price": str(price)}}


def _client_with_responses(responses_by_symbol_and_function: dict) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        key = (params.get("symbol"), params.get("function"))
        body = responses_by_symbol_and_function.get(key, {})
        return httpx.Response(200, json=body)
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://www.alphavantage.co"
    )


@pytest.mark.asyncio
async def test_screen_stocks_mock_mode_returns_fixture():
    result = await screen_stocks("tech", http_client=None, mode="mock", fixtures_dir="fixtures")

    assert "results" in result
    assert "error" not in result


@pytest.mark.asyncio
async def test_screen_stocks_filters_by_momentum_threshold():
    symbols = SECTOR_SYMBOLS["tech"]
    responses = {}
    for i, symbol in enumerate(symbols):
        rsi = 75.0 if i == 0 else 40.0
        responses[(symbol, "RSI")] = _rsi_response(rsi)
    responses[(symbols[0], "GLOBAL_QUOTE")] = _quote_response(1000.0)
    client = _client_with_responses(responses)

    result = await screen_stocks(
        "tech", http_client=client, mode="live", momentum_threshold=70.0,
        request_interval_seconds=0.0,
    )

    assert result["count"] == 1
    assert result["results"][0]["symbol"] == symbols[0]
    assert result["results"][0]["rsi"] == 75.0


@pytest.mark.asyncio
async def test_screen_stocks_filters_by_price_range():
    symbols = SECTOR_SYMBOLS["tech"]
    responses = {symbol: None for symbol in symbols}
    responses = {(symbol, "RSI"): _rsi_response(80.0) for symbol in symbols}
    responses[(symbols[0], "GLOBAL_QUOTE")] = _quote_response(500.0)
    for symbol in symbols[1:]:
        responses[(symbol, "GLOBAL_QUOTE")] = _quote_response(5000.0)
    client = _client_with_responses(responses)

    result = await screen_stocks(
        "tech", http_client=client, mode="live", momentum_threshold=70.0,
        price_max=1000.0, request_interval_seconds=0.0,
    )

    assert result["count"] == 1
    assert result["results"][0]["symbol"] == symbols[0]


@pytest.mark.asyncio
async def test_screen_stocks_respects_limit_and_ranks_by_rsi_descending():
    symbols = SECTOR_SYMBOLS["tech"]
    responses = {}
    for i, symbol in enumerate(symbols):
        responses[(symbol, "RSI")] = _rsi_response(70.0 + i)
    for symbol in symbols:
        responses[(symbol, "GLOBAL_QUOTE")] = _quote_response(100.0)
    client = _client_with_responses(responses)

    result = await screen_stocks(
        "tech", http_client=client, mode="live", momentum_threshold=70.0,
        limit=2, request_interval_seconds=0.0,
    )

    assert result["count"] == 2
    returned_symbols = {r["symbol"] for r in result["results"]}
    assert returned_symbols == {symbols[-1], symbols[-2]}


@pytest.mark.asyncio
async def test_screen_stocks_returns_error_for_unknown_sector():
    result = await screen_stocks("crypto", http_client=None, mode="live")

    assert "error" in result
    assert "crypto" in result["error"]


@pytest.mark.asyncio
async def test_screen_stocks_returns_empty_results_when_no_candidates_pass():
    symbols = SECTOR_SYMBOLS["tech"]
    responses = {(symbol, "RSI"): _rsi_response(10.0) for symbol in symbols}
    client = _client_with_responses(responses)

    result = await screen_stocks(
        "tech", http_client=client, mode="live", momentum_threshold=70.0,
        request_interval_seconds=0.0,
    )

    assert result == {"results": [], "count": 0}


@pytest.mark.asyncio
async def test_screen_stocks_drops_symbol_on_malformed_rsi_response():
    symbols = SECTOR_SYMBOLS["tech"]
    responses = {(symbols[0], "RSI"): {"unexpected": "shape"}}
    for symbol in symbols[1:]:
        responses[(symbol, "RSI")] = _rsi_response(75.0)
        responses[(symbol, "GLOBAL_QUOTE")] = _quote_response(100.0)
    client = _client_with_responses(responses)

    result = await screen_stocks(
        "tech", http_client=client, mode="live", momentum_threshold=70.0,
        request_interval_seconds=0.0,
    )

    returned_symbols = {r["symbol"] for r in result["results"]}
    assert symbols[0] not in returned_symbols


@pytest.mark.asyncio
async def test_screen_stocks_returns_error_on_network_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://www.alphavantage.co")

    result = await screen_stocks("tech", http_client=client, mode="live", request_interval_seconds=0.0)

    assert "error" in result


@pytest.mark.asyncio
async def test_screen_stocks_returns_rate_limit_error_on_alpha_vantage_throttle_note():
    # Verified against Alpha Vantage's documented behavior: a throttled call
    # returns HTTP 200 with a "Note" key instead of the expected data key --
    # not an HTTP 429 -- so this must be detected from the response body.
    symbols = SECTOR_SYMBOLS["tech"]
    responses = {
        (symbol, "RSI"): {"Note": "Thank you for using Alpha Vantage! Our standard API rate limit is..."}
        for symbol in symbols
    }
    client = _client_with_responses(responses)

    result = await screen_stocks("tech", http_client=client, mode="live", request_interval_seconds=0.0)

    assert result == {"error": "Screener temporarily rate-limited, try again in a minute"}


@pytest.mark.asyncio
async def test_screen_stocks_paces_requests_by_configured_interval():
    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    symbols = SECTOR_SYMBOLS["tech"]
    # All fail the momentum threshold, so no GLOBAL_QUOTE calls happen --
    # this isolates the test to just the 5 RSI calls (4 sleeps between them).
    responses = {(symbol, "RSI"): _rsi_response(10.0) for symbol in symbols}
    client = _client_with_responses(responses)

    await screen_stocks(
        "tech", http_client=client, mode="live", momentum_threshold=70.0,
        request_interval_seconds=5.0, sleep_fn=fake_sleep,
    )

    assert sleep_calls == [5.0, 5.0, 5.0, 5.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tools/test_screener.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.tools.screener'`

- [ ] **Step 3: Create the fixture**

Create `fixtures/stock_screener.json`:

```json
{
  "results": [
    {"symbol": "TCS.BSE", "price": 3850.0, "rsi": 72.5},
    {"symbol": "INFY.BSE", "price": 1920.0, "rsi": 68.3}
  ],
  "count": 2
}
```

- [ ] **Step 4: Implement the screener module**

Create `financial_voice_agent/tools/screener.py`:

```python
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

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
    except Exception:  # noqa: BLE001 -- any other per-symbol failure just drops that symbol
        return None


async def _fetch_price(http_client, symbol: str) -> float | None:
    """Fetches the latest close price for one symbol, for price-range filtering."""
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
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001 -- network/timeout errors must degrade gracefully
        return {"error": f"Could not reach the screener data provider for sector '{sector}'"}

    candidates = [r for r in rsi_results if r is not None and r["rsi"] >= momentum_threshold]
    if not candidates:
        return {"results": [], "count": 0}

    candidates.sort(key=lambda c: c["rsi"], reverse=True)
    candidates = candidates[:limit]

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/tools/test_screener.py -v`
Expected: PASS (10 tests)

- [ ] **Step 6: Commit**

```bash
git add financial_voice_agent/tools/screener.py fixtures/stock_screener.json tests/tools/test_screener.py
git commit -m "feat: add momentum stock screener module (Alpha Vantage RSI)"
```

---

### Task 2: Alpha Vantage HTTP client + config plumbing

**Files:**
- Modify: `financial_voice_agent/http_clients.py`
- Modify: `financial_voice_agent/config.py`
- Modify: `.env.example`
- Test: `tests/test_http_clients.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `HTTPClients.alpha_vantage: httpx.AsyncClient` (Task 3 dispatches `screen_stocks` against this). `Config.alpha_vantage_api_key: str | None` (consumed by `create_http_clients`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_http_clients.py`, inside `_make_config`'s call (add the new kwarg) and as a new test at the end of the file:

Modify the `_make_config` helper's `Config(...)` call (around line 43) to add the new field:

```python
        tavily_api_key="tavily-key",
        indian_stock_api_key="indian-stock-secret",
        alpha_vantage_api_key="alpha-vantage-secret",
    )
```

Add this test at the end of the file:

```python
@pytest.mark.asyncio
async def test_alpha_vantage_client_carries_api_key_as_query_param():
    config = _make_config()

    clients = await create_http_clients(config)
    try:
        assert clients.alpha_vantage.params["apikey"] == "alpha-vantage-secret"
    finally:
        await close_http_clients(clients)
```

Also update `test_create_http_clients_returns_one_client_per_vendor` (existing test) to include the new client:

```python
@pytest.mark.asyncio
async def test_create_http_clients_returns_one_client_per_vendor():
    config = _make_config()

    clients = await create_http_clients(config)
    try:
        assert isinstance(clients, HTTPClients)
        assert isinstance(clients.groq, httpx.AsyncClient)
        assert isinstance(clients.tts, httpx.AsyncClient)
        assert isinstance(clients.kite, httpx.AsyncClient)
        assert isinstance(clients.tavily, httpx.AsyncClient)
        assert isinstance(clients.indian_stock, httpx.AsyncClient)
        assert isinstance(clients.alpha_vantage, httpx.AsyncClient)
        # Each vendor gets a distinct client instance (no accidental sharing).
        assert len({
            id(clients.groq), id(clients.tts), id(clients.kite), id(clients.tavily),
            id(clients.indian_stock), id(clients.alpha_vantage),
        }) == 6
    finally:
        await close_http_clients(clients)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_http_clients.py -v`
Expected: FAIL — `TypeError: Config.__init__() got an unexpected keyword argument 'alpha_vantage_api_key'`

- [ ] **Step 3: Add the config field**

In `financial_voice_agent/config.py`, add to the `Config` dataclass (after `indian_stock_api_key: str | None` at line 57):

```python
    indian_stock_api_key: str | None
    alpha_vantage_api_key: str | None
```

Add `"ALPHA_VANTAGE_API_KEY"` to the tuple in `_load_env` (after `"INDIAN_STOCK_API_KEY"` at line 71):

```python
        "TAVILY_API_KEY",
        "INDIAN_STOCK_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
    ):
```

Add to the `Config(...)` construction in `load_config` (after `indian_stock_api_key=env.get("INDIAN_STOCK_API_KEY"),` at line 178):

```python
            indian_stock_api_key=env.get("INDIAN_STOCK_API_KEY"),
            alpha_vantage_api_key=env.get("ALPHA_VANTAGE_API_KEY"),
        )
```

- [ ] **Step 4: Add the HTTP client**

In `financial_voice_agent/http_clients.py`, add the base URL constant (after `INDIAN_STOCK_BASE_URL` at line 14):

```python
INDIAN_STOCK_BASE_URL = "https://stock.indianapi.in"
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co"
```

Add the field to `HTTPClients` (after `indian_stock: httpx.AsyncClient` at line 27):

```python
    indian_stock: httpx.AsyncClient
    alpha_vantage: httpx.AsyncClient
```

In `create_http_clients`, add construction (after the `indian_stock = ...` block, before the `return`):

```python
    # Alpha Vantage authenticates via an `apikey` query parameter on every
    # request, not a header -- setting it as a client-level default param
    # means every call in screener.py gets it merged in automatically
    # without needing to pass it explicitly each time.
    alpha_vantage = httpx.AsyncClient(
        base_url=ALPHA_VANTAGE_BASE_URL,
        params={"apikey": config.alpha_vantage_api_key or ""},
        timeout=REST_TIMEOUT,
    )

    return HTTPClients(
        groq=groq, tts=tts, kite=kite, tavily=tavily,
        indian_stock=indian_stock, alpha_vantage=alpha_vantage,
    )
```

Update `close_http_clients`:

```python
async def close_http_clients(clients: HTTPClients) -> None:
    await clients.groq.aclose()
    await clients.tts.aclose()
    await clients.kite.aclose()
    await clients.tavily.aclose()
    await clients.indian_stock.aclose()
    await clients.alpha_vantage.aclose()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_http_clients.py -v`
Expected: PASS

- [ ] **Step 6: Add the env var to `.env.example`**

Add to `.env.example` (after `INDIAN_STOCK_API_KEY=`):

```
INDIAN_STOCK_API_KEY=
ALPHA_VANTAGE_API_KEY=
```

- [ ] **Step 7: Run the full test suite to check for regressions**

Run: `python -m pytest -q`
Expected: PASS, all tests (no regressions from the new required dataclass field)

- [ ] **Step 8: Commit**

```bash
git add financial_voice_agent/http_clients.py financial_voice_agent/config.py .env.example tests/test_http_clients.py
git commit -m "feat: add Alpha Vantage HTTP client and config plumbing"
```

---

### Task 3: Wire screen_stocks into the tool registry

**Files:**
- Modify: `financial_voice_agent/tools/registry.py`
- Test: `tests/tools/test_registry.py`

**Interfaces:**
- Consumes: `screen_stocks` from Task 1 (`financial_voice_agent.tools.screener`), `http_clients.alpha_vantage` from Task 2.
- Produces: `"screen_stocks"` entry in `TOOLS_SCHEMA`, dispatched by `make_tool_executor`.

- [ ] **Step 1: Write the failing tests**

In `tests/tools/test_registry.py`, update `_FakeHttpClients` (around line 18-21) to add the new attribute:

```python
class _FakeHttpClients:
    kite = None  # unused in mock mode
    tavily = None  # overridden per-test where get_news is exercised
    indian_stock = None  # unused in mock mode
    alpha_vantage = None  # unused in mock mode
```

Update `test_tools_schema_names_include_all_expected_tools` (around line 24-35) to add the new tool name:

```python
def test_tools_schema_names_include_all_expected_tools():
    names = {tool["function"]["name"] for tool in TOOLS_SCHEMA}
    assert names == {
        "get_quote",
        "get_ohlc_history",
        "compute_indicator",
        "get_positions_holdings",
        "get_news",
        "capture_screen",
        "show_chart",
        "get_stock_fundamentals",
        "screen_stocks",
    }
```

Add a new test after `test_executor_dispatches_get_stock_fundamentals_in_mock_mode` (around line 238):

```python
@pytest.mark.asyncio
async def test_executor_dispatches_screen_stocks_in_mock_mode():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    result = await executor(ToolCall(id="1", name="screen_stocks", arguments={"sector": "tech"}))

    assert "results" in result
    assert "error" not in result
```

Add a test confirming argument filtering covers the new tool too, alongside `test_filter_tool_arguments_keeps_only_schema_declared_keys` (around line 241):

```python
def test_filter_tool_arguments_drops_undeclared_keys_for_screen_stocks():
    assert filter_tool_arguments(
        "screen_stocks", {"sector": "tech", "request_interval_seconds": 0.0}
    ) == {"sector": "tech"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tools/test_registry.py -v`
Expected: FAIL — `AssertionError` on the schema-names set (missing `"screen_stocks"`), and `UnknownToolError` on the dispatch test.

- [ ] **Step 3: Add the import**

In `financial_voice_agent/tools/registry.py`, add the import (after `from financial_voice_agent.tools.quotes import get_positions_holdings, get_quote` at line 24):

```python
from financial_voice_agent.tools.quotes import get_positions_holdings, get_quote
from financial_voice_agent.tools.screener import screen_stocks
```

- [ ] **Step 4: Add the schema entry**

Add to `TOOLS_SCHEMA` (after the `get_stock_fundamentals` entry, before the closing `]` at line 166):

```python
    {
        "type": "function",
        "function": {
            "name": "screen_stocks",
            "description": (
                "Screen stocks in a sector by momentum (RSI) and price range, returning "
                "the top matches ranked by momentum strength. Use for requests like "
                "'show me momentum stocks in tech' or 'top 10 pharma stocks under 2000'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {
                        "type": "string",
                        "description": "Sector name, e.g. 'tech', 'finance', 'pharma', 'auto'",
                    },
                    "momentum_threshold": {
                        "type": "number",
                        "description": "Minimum RSI to count as high-momentum. Default 70.",
                    },
                    "price_min": {"type": "number", "description": "Minimum price filter. Default 0."},
                    "price_max": {"type": "number", "description": "Maximum price filter. Optional."},
                    "limit": {"type": "integer", "description": "Max results to return. Default 10."},
                },
                "required": ["sector"],
            },
        },
    },
]
```

- [ ] **Step 5: Add the dispatch branch**

In `_dispatch`, add the branch (after the `get_stock_fundamentals` branch, before `raise UnknownToolError` at line 242):

```python
    if call.name == "get_stock_fundamentals":
        return await get_stock_fundamentals(
            **args, http_client=http_clients.indian_stock, mode=config.mode,
            fixtures_dir=_FIXTURES_DIR,
        )
    if call.name == "screen_stocks":
        return await screen_stocks(
            **args, http_client=http_clients.alpha_vantage, mode=config.mode,
            fixtures_dir=_FIXTURES_DIR,
        )
    raise UnknownToolError(f"Unknown tool: {call.name}")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/tools/test_registry.py -v`
Expected: PASS

- [ ] **Step 7: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS, no regressions

- [ ] **Step 8: Commit**

```bash
git add financial_voice_agent/tools/registry.py tests/tools/test_registry.py
git commit -m "feat: wire screen_stocks into the tool registry"
```

---

### Task 4: README documentation + manual verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing new (documentation-only task).
- Produces: nothing consumed by other tasks (final task).

- [ ] **Step 1: Add the provider row to README's table**

In `README.md`, add a row to the "What each provider needs" table (after the Indian Stock API row):

```
| Indian Stock API (fundamentals -- P/E, market cap, etc.) | `INDIAN_STOCK_API_KEY` | indianapi.in (dashboard's API/Manage Keys section) | Free plan available |
| Alpha Vantage (momentum stock screening) | `ALPHA_VANTAGE_API_KEY` | alphavantage.co/support/#api-key | Free tier: 5 requests/min, 25/day -- a single screen can take up to ~2 minutes due to built-in rate-limit pacing |
| Zerodha Kite Connect (live trading data) | `KITE_API_KEY` / `KITE_API_SECRET` / `KITE_ACCESS_TOKEN` | developers.kite.trade | Paid (~₹500/month), only needed for `mode: "live"` -- skip entirely with `mode: "mock"` |
```

- [ ] **Step 2: Commit the README change**

```bash
git add README.md
git commit -m "docs: document ALPHA_VANTAGE_API_KEY for the momentum screener"
```

- [ ] **Step 3: Manual live verification (not automated — confirms real Alpha Vantage response shape)**

This step validates the assumption baked into `_fetch_rsi`/`_fetch_price` (that Alpha Vantage's real response matches the documented shape) against a live call, since the spec flagged this as a documented risk area. Get a free API key from alphavantage.co, set `ALPHA_VANTAGE_API_KEY` in `.env`, then run:

```bash
python -c "
import asyncio
from financial_voice_agent.config import load_config
from financial_voice_agent.http_clients import create_http_clients, close_http_clients
from financial_voice_agent.tools.screener import screen_stocks

async def main():
    config = load_config()
    clients = await create_http_clients(config)
    try:
        result = await screen_stocks('tech', http_client=clients.alpha_vantage, mode='live', limit=3)
        print(result)
    finally:
        await close_http_clients(clients)

asyncio.run(main())
"
```

Expected: either a `{"results": [...], "count": N}` dict with real RSI/price values for tech-sector symbols, or a clear `{"error": ...}` (e.g. rate-limited). If the shape doesn't match what `_fetch_rsi`/`_fetch_price` expect (e.g. Alpha Vantage returns a `"Note"` or `"Information"` key instead of `"Technical Analysis: RSI"` when rate-limited — a known Alpha Vantage behavior), note the actual response and adjust `_fetch_rsi`/`_fetch_price` in Task 1's file accordingly before considering this plan complete.

No commit for this step — it's verification only. If it surfaces a real shape mismatch, fix it as a follow-up commit to `financial_voice_agent/tools/screener.py` with an accompanying test in `tests/tools/test_screener.py`.
