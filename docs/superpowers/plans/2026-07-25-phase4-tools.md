# Financial Voice Agent — Phase 4: Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the assistant its six read-only tools (PRD Section 6): `get_quote`, `get_ohlc_history`, `compute_indicator`, `get_positions_holdings`, `get_news`, `capture_screen`. Register them with Phase 3's `run_llm_turn` so the assistant can actually decide to use them instead of the Phase 3 stubs. No tool in this system will ever write, place, modify, or cancel anything — enforced structurally (every Kite call in this phase is a `GET`; no write endpoint is ever imported or called).

**Architecture:** Six small modules under `financial_voice_agent/tools/`, following the same pattern established in Phases 1-3: business logic (error mapping, retry policy, indicator math, response normalization) is fully unit-tested via dependency injection, and the few genuinely uncertain pieces (the real Kite Connect response shape, Tavily's request/response shape, mss/pygetwindow's real window-finding behavior) are isolated into small, explicitly-flagged adapter functions — the same containment strategy that caught three real vendor-API mismatches in Phase 3. Per your decision to build mock-first: every Kite-backed tool is fully testable and usable today via Phase 1's `mode: "mock"` fixtures, with zero Kite account required; the real (live) Kite adapter is written and structurally isolated now, but you don't need to verify it against the real API until you have a Kite Connect subscription.

**Tech Stack:** Python 3.11+, httpx (Phase 1, reused), pandas + numpy (indicator math), mss + pygetwindow (screen capture, Windows), pytest, pytest-asyncio.

## Global Constraints

- Python 3.11+ only (PRD Section 9).
- **Every tool is read-only.** No tool in this phase calls a Kite Connect write/order endpoint, ever. This is not just a policy — it is structural: no HTTP method other than `GET` (Kite) or `POST /search` (Tavily, a search call, not an order) is used anywhere in this phase's code.
- **Per-tool error behavior is exact** (PRD Section 6's tool table):
  - `get_quote`/`get_ohlc_history`/`get_positions_holdings` (Kite): 401 → session-expired error the caller should treat as fatal for this turn (report and stop, don't retry). 429 → rate-limit error, back off. Empty/stale data → state unavailable, don't invent numbers.
  - `get_ohlc_history`: expired options contracts are unsupported — state that explicitly rather than returning empty/wrong data silently.
  - `compute_indicator`: fails only if the underlying history pull failed, or fewer candles are available than the indicator requires — never compute on insufficient data.
  - `get_positions_holdings`: same retry/backoff as `get_quote`. Its data must never be passed into `get_news`'s query (a system-prompt/eval-set concern for Phase 5, not something this phase's code needs to structurally prevent, since the LLM decides what to call — but the tool registry documents this constraint for Phase 5's eval set to cover).
  - `get_news`: degrades gracefully — a search failure returns a result stating no results, never raises, never blocks the rest of the answer. Query text must never include account-specific data (a system-prompt concern, not enforced by this phase's code).
  - `capture_screen`: if the Kite window can't be located, return an explicit "window not found" result — never a blank or stale image.
- **Mock mode** (PRD Section 18.4, Phase 1's `mock.py`/`fixtures/`): `get_quote`, `get_ohlc_history`, `get_positions_holdings` must support `mode: "mock"`, reading from the exact fixture files Phase 1 already created (`fixtures/quote.json`, `fixtures/ohlc_history.json`, `fixtures/positions_holdings.json`) via `financial_voice_agent.mock.load_fixture()`. `get_news` and `capture_screen` have no PRD-specified mock mode — they always call their real backend (Tavily / real screen capture) regardless of `config.mode`.
- **Tool results must be concise, structured text, not raw API dumps** (PRD Section 10.4) — every tool function returns a small, normalized dict, never the vendor's raw JSON response.
- **Vendor API/SDK shapes in this plan are written from general knowledge, not guaranteed exact** — flagged explicitly per adapter, same as Phase 3's Groq/Cartesia adapters. Given your decision to build mock-first, the live Kite adapter's exact field-mapping is lower priority to verify right now; verify it against `kite.trade/docs/connect/v3` when you're ready to test live.
- Reuse Phase 1's `HTTPClients` (one reused client per vendor) and `Config` (for `mode`, API keys) — never construct a fresh HTTP client per tool call.

---

## File Structure

```
financial_voice_agent/
    tools/
        __init__.py
        kite_client.py    # kite_get() -- shared GET + error-mapping + retry for all Kite tools
        quotes.py         # get_quote(), get_positions_holdings()
        history.py        # get_ohlc_history()
        indicators.py      # compute_indicator() -- Bollinger/Fibonacci/MA/RSI
        news.py            # get_news() -- Tavily
        screen.py          # capture_screen() -- mss + pygetwindow
        registry.py         # TOOLS_SCHEMA + make_tool_executor() -- wires all six into Phase 3's run_llm_turn
tests/
    tools/
        __init__.py
        test_kite_client.py
        test_quotes.py
        test_history.py
        test_indicators.py
        test_news.py
        test_screen.py
        test_registry.py
```

- `kite_client.py`: owns the one shared GET+retry+error-mapping helper every Kite-backed tool uses. Nothing else maps Kite HTTP status codes to exceptions.
- `quotes.py`/`history.py`: own their one Kite endpoint each, normalizing to a small dict shape matching Phase 1's fixture shape exactly (so mock and live are interchangeable from the caller's perspective).
- `indicators.py`: owns indicator math. Depends on an injected `history_fn`, not on `history.py` directly — keeps indicator math testable with synthetic candle data, no Kite/mock wiring needed.
- `news.py`/`screen.py`: each owns one non-Kite vendor boundary.
- `registry.py`: owns tool-name → function dispatch and the `TOOLS_SCHEMA` Groq function-calling definitions. Nothing else builds a `ToolExecutor`.

---

### Task 1: Kite connection wrapper + get_quote + get_positions_holdings

**Files:**
- Create: `financial_voice_agent/tools/__init__.py` (empty)
- Create: `financial_voice_agent/tools/kite_client.py`
- Create: `financial_voice_agent/tools/quotes.py`
- Test: `tests/tools/__init__.py` (empty)
- Test: `tests/tools/test_kite_client.py`
- Test: `tests/tools/test_quotes.py`

**Interfaces:**
- Consumes: `financial_voice_agent.mock.load_fixture()` (Phase 1, exact signature already exists).
- Produces:
  - `class KiteSessionExpiredError(Exception)` — 401.
  - `class KiteRateLimitedError(Exception)` — 429.
  - `class KiteUnavailableError(Exception)` — 5xx after retries exhausted.
  - `async def kite_get(http_client, path: str, *, params: dict | None = None, max_attempts: int = 3, backoff_seconds: float = 1.0, sleep_fn=asyncio.sleep) -> dict` — GETs `path` on `http_client` (an `httpx.AsyncClient`-shaped object — tests inject a fake). 401/429 raise immediately (not transient, never retried). 5xx retries up to `max_attempts` total attempts with fixed backoff, then raises `KiteUnavailableError`. Any other non-2xx status raises via `response.raise_for_status()`.
  - `async def get_quote(symbol: str, *, http_client, mode: str = "live", fixtures_dir: str = "fixtures") -> dict` — `mode="mock"` reads `fixtures/quote.json` via `mock.load_fixture`; `mode="live"` calls `kite_get` and normalizes the response. Both paths return the same shape: `{"symbol": ..., "last_price": ..., "day_open": ..., "day_high": ..., "day_low": ..., "volume": ...}`.
  - `async def get_positions_holdings(*, http_client, mode: str = "live", fixtures_dir: str = "fixtures") -> dict` — same mock/live split, returns `{"positions": [...], "holdings": [...]}` matching `fixtures/positions_holdings.json`'s shape.

- [ ] **Step 1: Write the failing tests**

```python
# tests/tools/test_kite_client.py
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
```

```python
# tests/tools/test_quotes.py
import pytest

from financial_voice_agent.tools.kite_client import KiteSessionExpiredError
from financial_voice_agent.tools.quotes import get_positions_holdings, get_quote


@pytest.mark.asyncio
async def test_get_quote_mock_mode_reads_fixture():
    result = await get_quote("NIFTY 50", http_client=None, mode="mock")

    assert result == {
        "symbol": "NIFTY 50",
        "last_price": 24500.35,
        "day_open": 24380.1,
        "day_high": 24560.0,
        "day_low": 24350.0,
        "volume": 128340000,
    }


@pytest.mark.asyncio
async def test_get_quote_live_mode_calls_kite_get_and_normalizes():
    class _FakeHttpClient:
        async def get(self, path, params=None):
            class _Resp:
                status_code = 200
                def json(self):
                    return {
                        "data": {
                            "NSE:RELIANCE": {
                                "last_price": 2952.5,
                                "ohlc": {"open": 2950.0, "high": 2961.0, "low": 2945.0},
                                "volume": 4500000,
                            }
                        }
                    }
                def raise_for_status(self):
                    pass
            return _Resp()

    result = await get_quote("NSE:RELIANCE", http_client=_FakeHttpClient(), mode="live")

    assert result == {
        "symbol": "NSE:RELIANCE",
        "last_price": 2952.5,
        "day_open": 2950.0,
        "day_high": 2961.0,
        "day_low": 2945.0,
        "volume": 4500000,
    }


@pytest.mark.asyncio
async def test_get_quote_live_mode_propagates_session_expired():
    class _FakeHttpClient:
        async def get(self, path, params=None):
            class _Resp:
                status_code = 401
                def json(self):
                    return {}
            return _Resp()

    with pytest.raises(KiteSessionExpiredError):
        await get_quote("NIFTY 50", http_client=_FakeHttpClient(), mode="live")


@pytest.mark.asyncio
async def test_get_positions_holdings_mock_mode_reads_fixture():
    result = await get_positions_holdings(http_client=None, mode="mock")

    assert result == {
        "positions": [],
        "holdings": [
            {"symbol": "RELIANCE", "quantity": 10, "average_price": 2800.0, "last_price": 2952.5}
        ],
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tools/ -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.tools'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/tools/kite_client.py
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
```

```python
# financial_voice_agent/tools/quotes.py
from __future__ import annotations

from financial_voice_agent import mock
from financial_voice_agent.tools.kite_client import kite_get


async def get_quote(
    symbol: str, *, http_client, mode: str = "live", fixtures_dir: str = "fixtures"
) -> dict:
    if mode == "mock":
        return _summarize_quote(mock.load_fixture("quote", fixtures_dir=fixtures_dir), symbol)
    # NOTE: field mapping below is written from general knowledge of Kite
    # Connect's GET /quote response shape, not verified against a live
    # account -- verify against kite.trade/docs/connect/v3 when ready to go
    # live (per this project's mock-first decision).
    raw = await kite_get(http_client, "/quote", params={"i": symbol})
    instrument_data = raw["data"][symbol]
    ohlc = instrument_data["ohlc"]
    return {
        "symbol": symbol,
        "last_price": instrument_data["last_price"],
        "day_open": ohlc["open"],
        "day_high": ohlc["high"],
        "day_low": ohlc["low"],
        "volume": instrument_data["volume"],
    }


def _summarize_quote(fixture_data: dict, symbol: str) -> dict:
    return {
        "symbol": fixture_data.get("symbol", symbol),
        "last_price": fixture_data["ltp"],
        "day_open": fixture_data["day_open"],
        "day_high": fixture_data["day_high"],
        "day_low": fixture_data["day_low"],
        "volume": fixture_data["volume"],
    }


async def get_positions_holdings(
    *, http_client, mode: str = "live", fixtures_dir: str = "fixtures"
) -> dict:
    if mode == "mock":
        return mock.load_fixture("positions_holdings", fixtures_dir=fixtures_dir)
    # NOTE: same build-time-verify caveat as get_quote above.
    raw = await kite_get(http_client, "/portfolio/positions")
    holdings_raw = await kite_get(http_client, "/portfolio/holdings")
    return {
        "positions": raw.get("data", {}).get("net", []),
        "holdings": [
            {
                "symbol": h["tradingsymbol"],
                "quantity": h["quantity"],
                "average_price": h["average_price"],
                "last_price": h["last_price"],
            }
            for h in holdings_raw.get("data", [])
        ],
    }
```

Also create empty `financial_voice_agent/tools/__init__.py` and empty `tests/tools/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tools/test_kite_client.py tests/tools/test_quotes.py -v`
Expected: PASS (9 tests). The `test_get_quote_live_mode_*` tests exercise the flagged live-adapter field mapping against a hand-built fake response — they confirm the mapping is internally consistent, not that it matches the real Kite API (that's the noted build-time-verify step).

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/tools/__init__.py financial_voice_agent/tools/kite_client.py financial_voice_agent/tools/quotes.py tests/tools/__init__.py tests/tools/test_kite_client.py tests/tools/test_quotes.py
git commit -m "feat: add Kite client wrapper, get_quote, get_positions_holdings"
```

---

### Task 2: get_ohlc_history + compute_indicator

**Files:**
- Create: `financial_voice_agent/tools/history.py`
- Create: `financial_voice_agent/tools/indicators.py`
- Test: `tests/tools/test_history.py`
- Test: `tests/tools/test_indicators.py`
- Modify: `requirements.txt` — add `pandas`, `numpy` (if not already present from Phase 2 — check first; Phase 2 already added `numpy`, only `pandas` is new)

**Interfaces:**
- Consumes: `financial_voice_agent.tools.kite_client.kite_get` (Task 1), `financial_voice_agent.mock.load_fixture` (Phase 1).
- Produces:
  - `async def get_ohlc_history(symbol: str, interval: str, from_date: str, to_date: str, *, http_client, mode: str = "live", fixtures_dir: str = "fixtures") -> dict` — returns `{"symbol": ..., "interval": ..., "candles": [{"ts": ..., "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}, ...]}` matching `fixtures/ohlc_history.json`'s shape in both modes.
  - `class InsufficientDataError(Exception)`
  - `INDICATOR_MIN_CANDLES: dict[str, int]` — `{"bollinger": 20, "moving_average": 20, "rsi": 15, "fibonacci": 2}`.
  - `HistoryFn = Callable[..., Awaitable[dict]]` — the shape `get_ohlc_history` has, injected into `compute_indicator` so indicator math is testable without any Kite/mock wiring.
  - `async def compute_indicator(symbol: str, indicator: str, params: dict, *, history_fn: HistoryFn) -> dict` — calls `history_fn(symbol=symbol, interval=params.get("interval", "15minute"), from_date=params.get("from"), to_date=params.get("to"))`, raises `InsufficientDataError` if fewer candles than `INDICATOR_MIN_CANDLES[indicator]` are returned, otherwise computes and returns the indicator's values. Raises `ValueError` for an unknown indicator name.

- [ ] **Step 1: Write the failing tests**

```python
# tests/tools/test_history.py
import pytest

from financial_voice_agent.tools.history import get_ohlc_history


@pytest.mark.asyncio
async def test_get_ohlc_history_mock_mode_reads_fixture():
    result = await get_ohlc_history(
        "RELIANCE", "15minute", "2026-07-24", "2026-07-25", http_client=None, mode="mock"
    )

    assert result["symbol"] == "RELIANCE"
    assert result["interval"] == "15minute"
    assert len(result["candles"]) == 2
    assert result["candles"][0]["close"] == 2952.5


@pytest.mark.asyncio
async def test_get_ohlc_history_live_mode_calls_kite_get_and_normalizes():
    class _FakeHttpClient:
        async def get(self, path, params=None):
            class _Resp:
                status_code = 200
                def json(self):
                    return {
                        "data": {
                            "candles": [
                                ["2026-07-24T09:15:00+05:30", 2950.0, 2958.0, 2945.0, 2952.5, 45210],
                                ["2026-07-24T09:30:00+05:30", 2952.5, 2961.0, 2949.0, 2957.8, 38900],
                            ]
                        }
                    }
                def raise_for_status(self):
                    pass
            return _Resp()

    result = await get_ohlc_history(
        "RELIANCE", "15minute", "2026-07-24", "2026-07-25",
        http_client=_FakeHttpClient(), mode="live",
    )

    assert result["symbol"] == "RELIANCE"
    assert result["candles"][0] == {
        "ts": "2026-07-24T09:15:00+05:30",
        "open": 2950.0,
        "high": 2958.0,
        "low": 2945.0,
        "close": 2952.5,
        "volume": 45210,
    }
```

```python
# tests/tools/test_indicators.py
import pytest

from financial_voice_agent.tools.indicators import InsufficientDataError, compute_indicator


def _make_history_fn(closes: list[float]):
    async def history_fn(*, symbol, interval, from_date, to_date):
        return {
            "symbol": symbol,
            "interval": interval,
            "candles": [
                {"ts": f"t{i}", "open": c, "high": c, "low": c, "close": c, "volume": 100}
                for i, c in enumerate(closes)
            ],
        }
    return history_fn


@pytest.mark.asyncio
async def test_compute_indicator_moving_average():
    closes = [float(i) for i in range(1, 21)]  # 1..20
    history_fn = _make_history_fn(closes)

    result = await compute_indicator(
        "RELIANCE", "moving_average", {"window": 20}, history_fn=history_fn
    )

    assert result["moving_average"] == pytest.approx(10.5)  # mean of 1..20


@pytest.mark.asyncio
async def test_compute_indicator_rsi_all_gains_is_100():
    closes = [float(i) for i in range(1, 21)]  # strictly increasing -- no losses
    history_fn = _make_history_fn(closes)

    result = await compute_indicator("RELIANCE", "rsi", {"window": 14}, history_fn=history_fn)

    assert result["rsi"] == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_compute_indicator_bollinger_returns_three_bands():
    closes = [100.0] * 20  # constant series -- zero std, bands collapse to the mean
    history_fn = _make_history_fn(closes)

    result = await compute_indicator("RELIANCE", "bollinger", {"window": 20}, history_fn=history_fn)

    assert result["middle_band"] == pytest.approx(100.0)
    assert result["upper_band"] == pytest.approx(100.0)
    assert result["lower_band"] == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_compute_indicator_fibonacci_levels():
    closes = [100.0, 150.0, 120.0]  # high=150, low=100
    history_fn = _make_history_fn(closes)

    result = await compute_indicator("RELIANCE", "fibonacci", {}, history_fn=history_fn)

    levels = result["fibonacci_levels"]
    assert levels["0.0%"] == pytest.approx(150.0)
    assert levels["100.0%"] == pytest.approx(100.0)
    assert levels["50.0%"] == pytest.approx(125.0)


@pytest.mark.asyncio
async def test_compute_indicator_raises_on_insufficient_candles():
    closes = [100.0, 101.0]  # far fewer than bollinger's required 20
    history_fn = _make_history_fn(closes)

    with pytest.raises(InsufficientDataError, match="20"):
        await compute_indicator("RELIANCE", "bollinger", {"window": 20}, history_fn=history_fn)


@pytest.mark.asyncio
async def test_compute_indicator_raises_on_unknown_indicator():
    history_fn = _make_history_fn([100.0] * 20)

    with pytest.raises(ValueError, match="macd"):
        await compute_indicator("RELIANCE", "macd", {}, history_fn=history_fn)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tools/test_history.py tests/tools/test_indicators.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.tools.history'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/tools/history.py
from __future__ import annotations

from financial_voice_agent import mock
from financial_voice_agent.tools.kite_client import kite_get


async def get_ohlc_history(
    symbol: str,
    interval: str,
    from_date: str,
    to_date: str,
    *,
    http_client,
    mode: str = "live",
    fixtures_dir: str = "fixtures",
) -> dict:
    if mode == "mock":
        data = mock.load_fixture("ohlc_history", fixtures_dir=fixtures_dir)
        return {"symbol": data["symbol"], "interval": data["interval"], "candles": data["candles"]}
    # NOTE: field mapping below is written from general knowledge of Kite
    # Connect's GET /instruments/historical response shape (a list of
    # [timestamp, open, high, low, close, volume] arrays per candle) -- verify
    # against kite.trade/docs/connect/v3 when ready to go live.
    raw = await kite_get(
        http_client,
        f"/instruments/historical/{symbol}/{interval}",
        params={"from": from_date, "to": to_date},
    )
    candles = [
        {"ts": c[0], "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5]}
        for c in raw["data"]["candles"]
    ]
    return {"symbol": symbol, "interval": interval, "candles": candles}
```

```python
# financial_voice_agent/tools/indicators.py
from __future__ import annotations

from typing import Awaitable, Callable

import pandas as pd

HistoryFn = Callable[..., Awaitable[dict]]

INDICATOR_MIN_CANDLES = {"bollinger": 20, "moving_average": 20, "rsi": 15, "fibonacci": 2}


class InsufficientDataError(Exception):
    pass


async def compute_indicator(
    symbol: str, indicator: str, params: dict, *, history_fn: HistoryFn
) -> dict:
    if indicator not in INDICATOR_MIN_CANDLES:
        raise ValueError(f"Unknown indicator: {indicator}")

    history = await history_fn(
        symbol=symbol,
        interval=params.get("interval", "15minute"),
        from_date=params.get("from"),
        to_date=params.get("to"),
    )
    candles = history["candles"]
    min_required = INDICATOR_MIN_CANDLES[indicator]
    if len(candles) < min_required:
        raise InsufficientDataError(
            f"{indicator} requires at least {min_required} candles, got {len(candles)}"
        )

    closes = pd.Series([c["close"] for c in candles])
    if indicator == "bollinger":
        return _bollinger_bands(closes, window=params.get("window", 20))
    if indicator == "moving_average":
        return _moving_average(closes, window=params.get("window", 20))
    if indicator == "rsi":
        return _rsi(closes, window=params.get("window", 14))
    return _fibonacci_retracement(closes)


def _bollinger_bands(closes: pd.Series, *, window: int) -> dict:
    sma = closes.rolling(window).mean()
    std = closes.rolling(window).std(ddof=0)
    upper = sma + 2 * std
    lower = sma - 2 * std
    return {
        "middle_band": float(sma.iloc[-1]),
        "upper_band": float(upper.iloc[-1]),
        "lower_band": float(lower.iloc[-1]),
    }


def _moving_average(closes: pd.Series, *, window: int) -> dict:
    return {"moving_average": float(closes.rolling(window).mean().iloc[-1])}


def _rsi(closes: pd.Series, *, window: int) -> dict:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    last_avg_loss = avg_loss.iloc[-1]
    if last_avg_loss == 0:
        return {"rsi": 100.0}
    rs = avg_gain.iloc[-1] / last_avg_loss
    return {"rsi": float(100 - (100 / (1 + rs)))}


def _fibonacci_retracement(closes: pd.Series) -> dict:
    high = float(closes.max())
    low = float(closes.min())
    diff = high - low
    return {
        "fibonacci_levels": {
            "0.0%": high,
            "23.6%": high - 0.236 * diff,
            "38.2%": high - 0.382 * diff,
            "50.0%": high - 0.5 * diff,
            "61.8%": high - 0.618 * diff,
            "100.0%": low,
        }
    }
```

Add `pandas` to `requirements.txt` if not already present (check first — `numpy` was added in Phase 2). Run `pip install -r requirements.txt`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tools/test_history.py tests/tools/test_indicators.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/tools/history.py financial_voice_agent/tools/indicators.py tests/tools/test_history.py tests/tools/test_indicators.py requirements.txt
git commit -m "feat: add get_ohlc_history and compute_indicator (Bollinger/MA/RSI/Fibonacci)"
```

---

### Task 3: get_news (Tavily)

**Files:**
- Create: `financial_voice_agent/tools/news.py`
- Test: `tests/tools/test_news.py`

**Interfaces:**
- Consumes: nothing from earlier Phase 4 tasks (independent).
- Produces:
  - `async def get_news(query: str, *, http_client, api_key: str, max_results: int = 5) -> dict` — POSTs to Tavily's `/search` endpoint. **Never raises** — any failure (timeout, non-2xx, malformed response) is caught internally and returns `{"headlines": [], "error": "news search did not return results"}`, per PRD Section 6's "degrade gracefully" requirement. On success, returns `{"headlines": [{"title": ..., "summary": ..., "url": ...}, ...]}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/tools/test_news.py
import pytest

from financial_voice_agent.tools.news import get_news


class _FakeHttpClient:
    def __init__(self, response=None, raises: Exception | None = None):
        self._response = response
        self._raises = raises
        self.received_json: dict | None = None

    async def post(self, path, json=None):
        self.received_json = json
        if self._raises:
            raise self._raises
        return self._response


class _FakeResponse:
    def __init__(self, status_code: int, json_data: dict):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.mark.asyncio
async def test_get_news_returns_headlines_on_success():
    client = _FakeHttpClient(
        response=_FakeResponse(
            200,
            {
                "results": [
                    {"title": "Nifty hits new high", "content": "Markets rallied today.", "url": "https://example.com/1"},
                ]
            },
        )
    )

    result = await get_news("Nifty", http_client=client, api_key="test-key")

    assert result == {
        "headlines": [
            {"title": "Nifty hits new high", "summary": "Markets rallied today.", "url": "https://example.com/1"}
        ]
    }
    assert client.received_json["api_key"] == "test-key"
    assert client.received_json["query"] == "Nifty"


@pytest.mark.asyncio
async def test_get_news_degrades_gracefully_on_timeout():
    client = _FakeHttpClient(raises=TimeoutError("no response"))

    result = await get_news("Nifty", http_client=client, api_key="test-key")

    assert result["headlines"] == []
    assert "error" in result


@pytest.mark.asyncio
async def test_get_news_degrades_gracefully_on_http_error():
    client = _FakeHttpClient(response=_FakeResponse(500, {}))

    result = await get_news("Nifty", http_client=client, api_key="test-key")

    assert result["headlines"] == []
    assert "error" in result


@pytest.mark.asyncio
async def test_get_news_respects_max_results():
    client = _FakeHttpClient(
        response=_FakeResponse(
            200,
            {"results": [{"title": f"Headline {i}", "content": "x", "url": "u"} for i in range(10)]},
        )
    )

    result = await get_news("Nifty", http_client=client, api_key="test-key", max_results=3)

    assert len(result["headlines"]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_news.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.tools.news'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/tools/news.py
from __future__ import annotations


async def get_news(query: str, *, http_client, api_key: str, max_results: int = 5) -> dict:
    # Response parsing (not just the network call) must stay inside the try
    # block -- a malformed-but-200-status response (e.g. "results": null, or
    # non-dict result items) must also degrade gracefully, not raise. An
    # earlier draft of this plan parsed the response outside the try block,
    # which broke this exact guarantee for malformed responses; caught during
    # Task 3's review.
    try:
        response = await http_client.post(
            "/search", json={"api_key": api_key, "query": query, "max_results": max_results}
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results") or []
        headlines = [
            {"title": r.get("title"), "summary": r.get("content"), "url": r.get("url")}
            for r in results[:max_results]
        ]
    except Exception:  # noqa: BLE001 -- must degrade gracefully per PRD Section 6, never raise
        return {"headlines": [], "error": "news search did not return results"}

    return {"headlines": headlines}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_news.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/tools/news.py tests/tools/test_news.py
git commit -m "feat: add get_news via Tavily with graceful degradation"
```

---

### Task 4: capture_screen

**Files:**
- Create: `financial_voice_agent/tools/screen.py`
- Test: `tests/tools/test_screen.py`
- Modify: `requirements.txt` — add `mss`, `pygetwindow` (Windows; this machine is Windows per prior phases)

**Interfaces:**
- Consumes: nothing from earlier Phase 4 tasks.
- Produces:
  - `class WindowNotFoundError(Exception)`
  - `async def capture_screen(*, window_finder, screenshot_fn, jpeg_quality: int = 80) -> dict` — `window_finder: Callable[[], dict | None]` returns a region dict (`{"left": ..., "top": ..., "width": ..., "height": ...}`) or `None` if the Kite window can't be located; `screenshot_fn: Callable[[dict], bytes]` takes that region and returns a raw RGB/RGBA image buffer (both injected so tests never touch a real window manager or screen). Raises `WindowNotFoundError` if `window_finder()` returns `None` — per PRD Section 6, callers must speak "window not found" rather than guess from a stale frame. On success, returns `{"image_base64": <base64 JPEG string>}`.
  - `def find_kite_window() -> dict | None` — the real adapter, using `pygetwindow` to locate a window whose title contains "Kite". **Verify this title-matching heuristic against your actual browser/window setup at build time** — this is the one piece of this task with real-environment uncertainty (exact window title varies by browser and by whether Kite is a PWA, a browser tab, or a dedicated window).
  - `def capture_region(region: dict) -> bytes` — the real adapter, using `mss` to grab the given region as a raw BGRA buffer.

- [ ] **Step 1: Write the failing test**

```python
# tests/tools/test_screen.py
import base64
import io

import pytest
from PIL import Image

from financial_voice_agent.tools.screen import WindowNotFoundError, capture_screen


def _make_test_image_bytes(width: int = 10, height: int = 10) -> bytes:
    image = Image.new("RGB", (width, height), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_capture_screen_returns_base64_jpeg_when_window_found():
    def window_finder():
        return {"left": 0, "top": 0, "width": 10, "height": 10}

    def screenshot_fn(region):
        assert region == {"left": 0, "top": 0, "width": 10, "height": 10}
        return _make_test_image_bytes()

    result = await capture_screen(window_finder=window_finder, screenshot_fn=screenshot_fn)

    decoded = base64.b64decode(result["image_base64"])
    image = Image.open(io.BytesIO(decoded))
    assert image.format == "JPEG"


@pytest.mark.asyncio
async def test_capture_screen_raises_window_not_found_when_window_finder_returns_none():
    def window_finder():
        return None

    def screenshot_fn(region):
        raise AssertionError("must not be called if window not found")

    with pytest.raises(WindowNotFoundError):
        await capture_screen(window_finder=window_finder, screenshot_fn=screenshot_fn)
```

Note: this test uses `Pillow` (`PIL`) purely to construct/verify a valid JPEG in the test itself — the implementation's `screenshot_fn` is injected and returns raw JPEG bytes directly, so the implementation doesn't need to import PIL at all if `screenshot_fn` already returns JPEG-encoded bytes (see Step 3 — `capture_screen` treats `screenshot_fn`'s return value as already-encoded JPEG bytes, matching what the real `mss`+JPEG-encode adapter will produce). Add `pillow` to `requirements.txt` as a test-only convenience if it's not already installed (`pip show pillow` — Phase 2's `noisereduce`/`numpy` stack commonly pulls it in transitively; check before adding a redundant line).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_screen.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.tools.screen'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/tools/screen.py
from __future__ import annotations

import base64
from typing import Callable


class WindowNotFoundError(Exception):
    pass


async def capture_screen(
    *,
    window_finder: Callable[[], dict | None],
    screenshot_fn: Callable[[dict], bytes],
    jpeg_quality: int = 80,
) -> dict:
    region = window_finder()
    if region is None:
        raise WindowNotFoundError("Kite window not found")
    jpeg_bytes = screenshot_fn(region)
    return {"image_base64": base64.b64encode(jpeg_bytes).decode("ascii")}


def find_kite_window() -> dict | None:
    """Real adapter: locates a window whose title contains "Kite" using
    pygetwindow. Verify this title-matching heuristic against your actual
    browser/window setup at build time -- exact title varies by browser and
    by whether Kite is a PWA, browser tab, or dedicated window."""
    import pygetwindow as gw

    matches = [w for w in gw.getAllWindows() if "kite" in w.title.lower()]
    if not matches:
        return None
    window = matches[0]
    return {"left": window.left, "top": window.top, "width": window.width, "height": window.height}


def capture_region(region: dict, *, jpeg_quality: int = 80) -> bytes:
    """Real adapter: grabs `region` via mss and JPEG-encodes it. Verify
    against your actual display/DPI setup at build time -- mss's raw BGRA
    buffer needs PIL to re-encode as JPEG, matching PRD Section 10.3."""
    import io

    import mss
    from PIL import Image

    with mss.mss() as sct:
        raw = sct.grab(region)
    image = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()
```

Add `mss` and `pygetwindow` to `requirements.txt`. Run `pip install -r requirements.txt`. Also add `pillow` if `pip show pillow` shows it isn't already installed.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_screen.py -v`
Expected: PASS (2 tests). Neither test imports `pygetwindow`/`mss`/`find_kite_window`/`capture_region` — those real adapters are only exercised by a manual smoke test (Step 5), matching Phase 2's approach to PyAudio/Silero.

- [ ] **Step 5: Manual smoke test (not automated — record the result, do not skip)**

With the Kite web app open in a browser window:

```bash
python -c "
from financial_voice_agent.tools.screen import find_kite_window, capture_region
region = find_kite_window()
print('window found:', region)
if region:
    jpeg_bytes = capture_region(region)
    with open('kite_capture_test.jpg', 'wb') as f:
        f.write(jpeg_bytes)
    print('wrote kite_capture_test.jpg,', len(jpeg_bytes), 'bytes')
"
```

Expected: prints a non-`None` region and writes a JPEG file you can open and confirm shows the Kite window. If `find_kite_window()` returns `None`, inspect your actual window title (e.g. via `pygetwindow.getAllTitles()`) and adjust the `"kite" in w.title.lower()` heuristic in `find_kite_window` to match.

- [ ] **Step 6: Commit**

```bash
git add financial_voice_agent/tools/screen.py tests/tools/test_screen.py requirements.txt
git commit -m "feat: add capture_screen via mss + pygetwindow"
```

---

### Task 5: Wire all six tools into Phase 3's LLM loop

**Files:**
- Create: `financial_voice_agent/tools/registry.py`
- Test: `tests/tools/test_registry.py`

**Interfaces:**
- Consumes: `ToolCall` (Phase 3, `financial_voice_agent.orchestrator.llm`), all five tool modules from Tasks 1-4, `financial_voice_agent.Config`/`HTTPClients` (Phase 1).
- Produces:
  - `TOOLS_SCHEMA: list[dict]` — Groq function-calling schema definitions for all six tools, passed as `run_llm_turn`'s `tools_schema` argument.
  - `def make_tool_executor(config, http_clients) -> Callable[[ToolCall], Awaitable[dict]]` — returns a dispatcher matching `run_llm_turn`'s `tool_executor` parameter exactly, routing each `ToolCall` by `call.name` to the correct Task 1-4 function with `call.arguments` unpacked as keyword arguments, plus the config/client wiring each tool needs (`http_client`, `mode`, `api_key`). Raises `ValueError` for an unrecognized tool name.

- [ ] **Step 1: Write the failing test**

```python
# tests/tools/test_registry.py
import pytest

from financial_voice_agent.orchestrator.llm import ToolCall
from financial_voice_agent.tools.registry import TOOLS_SCHEMA, make_tool_executor


class _FakeConfig:
    mode = "mock"
    tavily_api_key = "test-tavily-key"


class _FakeHttpClients:
    kite = None  # unused in mock mode
    tavily = None  # overridden per-test where get_news is exercised


def test_tools_schema_names_match_all_six_tools():
    names = {tool["function"]["name"] for tool in TOOLS_SCHEMA}
    assert names == {
        "get_quote",
        "get_ohlc_history",
        "compute_indicator",
        "get_positions_holdings",
        "get_news",
        "capture_screen",
    }


@pytest.mark.asyncio
async def test_executor_dispatches_get_quote_in_mock_mode():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    result = await executor(ToolCall(id="1", name="get_quote", arguments={"symbol": "NIFTY 50"}))

    assert result["symbol"] == "NIFTY 50"
    assert result["last_price"] == 24500.35


@pytest.mark.asyncio
async def test_executor_dispatches_get_positions_holdings_in_mock_mode():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    result = await executor(ToolCall(id="1", name="get_positions_holdings", arguments={}))

    assert result["holdings"][0]["symbol"] == "RELIANCE"


@pytest.mark.asyncio
async def test_executor_dispatches_compute_indicator_using_mock_history():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    with pytest.raises(Exception):
        # fixtures/ohlc_history.json only has 2 candles -- far short of any
        # indicator's minimum, so this must raise InsufficientDataError,
        # proving compute_indicator is correctly wired to real
        # get_ohlc_history (mock mode) through the registry, not a stub.
        await executor(
            ToolCall(
                id="1",
                name="compute_indicator",
                arguments={"symbol": "RELIANCE", "indicator": "moving_average", "params": {}},
            )
        )


@pytest.mark.asyncio
async def test_executor_raises_value_error_for_unknown_tool():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    with pytest.raises(ValueError, match="unknown_tool"):
        await executor(ToolCall(id="1", name="unknown_tool", arguments={}))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.tools.registry'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/tools/registry.py
from __future__ import annotations

import functools
from typing import Awaitable, Callable

from financial_voice_agent.orchestrator.llm import ToolCall
from financial_voice_agent.tools.history import get_ohlc_history
from financial_voice_agent.tools.indicators import compute_indicator
from financial_voice_agent.tools.news import get_news
from financial_voice_agent.tools.quotes import get_positions_holdings, get_quote
from financial_voice_agent.tools.screen import capture_region, capture_screen, find_kite_window

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Get the latest price and day OHLC for a market instrument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "e.g. 'NIFTY 50' or 'RELIANCE'"}
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ohlc_history",
            "description": "Get historical OHLC candle data for an instrument over a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "interval": {"type": "string", "description": "e.g. '15minute', 'day'"},
                    "from_date": {"type": "string", "description": "ISO date, e.g. '2026-07-01'"},
                    "to_date": {"type": "string", "description": "ISO date, e.g. '2026-07-25'"},
                },
                "required": ["symbol", "interval", "from_date", "to_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_indicator",
            "description": "Compute a technical indicator (bollinger, moving_average, rsi, fibonacci) for an instrument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "indicator": {
                        "type": "string",
                        "enum": ["bollinger", "moving_average", "rsi", "fibonacci"],
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional: window, interval, from, to",
                    },
                },
                "required": ["symbol", "indicator"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_positions_holdings",
            "description": "Get the user's current holdings and positions.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Search for recent news headlines on a topic. Never include account-specific data in the query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_screen",
            "description": "Capture the current Kite window as an image, for questions about what's visible on screen.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def make_tool_executor(config, http_clients) -> Callable[[ToolCall], Awaitable[dict]]:
    async def executor(call: ToolCall) -> dict:
        if call.name == "get_quote":
            return await get_quote(**call.arguments, http_client=http_clients.kite, mode=config.mode)
        if call.name == "get_ohlc_history":
            return await get_ohlc_history(
                **call.arguments, http_client=http_clients.kite, mode=config.mode
            )
        if call.name == "compute_indicator":
            history_fn = functools.partial(
                get_ohlc_history, http_client=http_clients.kite, mode=config.mode
            )
            return await compute_indicator(**call.arguments, history_fn=history_fn)
        if call.name == "get_positions_holdings":
            return await get_positions_holdings(http_client=http_clients.kite, mode=config.mode)
        if call.name == "get_news":
            return await get_news(
                **call.arguments, http_client=http_clients.tavily, api_key=config.tavily_api_key
            )
        if call.name == "capture_screen":
            return await capture_screen(window_finder=find_kite_window, screenshot_fn=capture_region)
        raise ValueError(f"Unknown tool: {call.name}")

    return executor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_registry.py -v`
Expected: PASS (4 tests). The `compute_indicator` test intentionally exercises the real (mock-mode) `get_ohlc_history` through the registry and expects it to raise `InsufficientDataError` — proving genuine end-to-end wiring, not a stub.

- [ ] **Step 5: Run the full Phase 4 suite**

Run: `pytest tests/ -v`
Expected: PASS (85 baseline from Phases 1-3 + Phase 4's new tests, all green).

- [ ] **Step 6: Commit**

```bash
git add financial_voice_agent/tools/registry.py tests/tools/test_registry.py
git commit -m "feat: wire all six tools into the LLM tool-calling loop"
```

---

## Phase 4 Exit Criteria

- `pytest tests/ -v` passes with 0 failures.
- `TOOLS_SCHEMA` + `make_tool_executor(config, http_clients)` can be passed directly into Phase 3's `run_llm_turn(..., tools_schema=TOOLS_SCHEMA, tool_executor=make_tool_executor(config, http_clients))`, replacing the empty tool set Phase 3 used.
- Every Kite-backed tool works end-to-end in `mode: "mock"` against Phase 1's fixtures with zero Kite account required — matching this project's mock-first decision.
- No tool call in this phase's code reaches a Kite write/order endpoint (spot-check: grep for any HTTP method other than `GET` against `http_clients.kite`, and confirm none exists).
- The real (live) adapters (`get_quote`/`get_ohlc_history`/`get_positions_holdings`'s Kite field-mapping, `find_kite_window`/`capture_region`) are flagged for build-time verification and are not blocking — they become relevant once you set up a Kite Connect subscription.

## Post-Final-Review Fixes (applied, all independently verified by execution)

The final whole-branch review found one Critical and four Important cross-tool integration defects, all fixed in one round:

1. **`capture_screen` no longer returns a full base64 JPEG inline** (would have produced a ~500,000-character tool message and blown the LLM's context on every screen capture). It now writes the JPEG to disk and returns `{"screenshot_path": ..., "width": ..., "height": ...}`, matching `db.py`'s existing (still unwired — see below) `screenshot_path` column. The dead `jpeg_quality` parameter was removed from `capture_screen`'s signature.
2. **`compute_indicator` now sends a real, non-`None` default date range** to `get_ohlc_history` when the caller doesn't specify one (`_default_date_range()`), and uses `from_date`/`to_date` param keys — matching `get_ohlc_history`'s own naming — instead of the inconsistent `from`/`to`.
3. **`fixtures/ohlc_history.json` was expanded from 2 to 25 candles** (first candle's `open`/`close` unchanged at 2950.0/2952.5) so all four indicators, not just Fibonacci, actually compute successfully in `mode: "mock"`.
4. **`make_tool_executor` now validates/translates failures instead of propagating them raw**: Kite's three exceptions, `InsufficientDataError`, `WindowNotFoundError`, and any argument-mismatch `TypeError` are all caught and returned as `{"error": ...}` dicts (an unknown tool/indicator name's `ValueError` still propagates uncaught, by design). `_FIXTURES_DIR` is now resolved via `pathlib.Path(__file__)`, not the process's CWD.
5. Two registry tests that used `pytest.raises(Exception)` (too loose to guard their own fix) were rewritten to assert specific success/failure outcomes.

### Known Limitations (parked, not blocking, not load-bearing for Phase 5)

The fix-wave re-review found three new Minor issues in the fixes themselves — real, but no further fix round was spent on them per this project's one-fix-wave-per-final-review process:

- **`capture_screen`'s dispatch in `registry.py` doesn't pass a `screenshot_dir`, so it falls back to the CWD-relative `"screenshots"` default** — the same class of bug Fix 4 above just closed for fixture loading, reintroduced on the write side. Running the app from any directory other than the repo root will write screenshots to a stray relative folder and return a path that won't resolve later. **Fix for whoever picks this up:** resolve a `_SCREENSHOTS_DIR` the same way `_FIXTURES_DIR` is resolved (`pathlib.Path(__file__).resolve().parent.parent.parent / "screenshots"`) and pass it as `screenshot_dir=` in the `capture_screen` dispatch branch.
- **`make_tool_executor`'s `except TypeError` wraps the entire tool call, not just argument binding** — a genuine `TypeError` raised deep inside a tool (e.g. malformed data hitting a pandas operation) is currently mislabeled `"Invalid arguments for tool 'X'"` and swallowed as if it were a bad LLM argument, rather than surfacing as a real bug. Low priority for a personal prototype; revisit if this ever misleads debugging in practice.
- **No retention/cleanup for screenshots** — every `capture_screen` call writes a new ~1-1.5MB JPEG that is never deleted. Fine for occasional manual testing; would need a cleanup policy before any sustained real usage.

Also noted, not a code defect: **`orchestrator/turn.py` (Phase 3) still hardcodes `screenshot_path=None`** when logging a turn — Fix 1 above makes `capture_screen`'s return shape compatible with `db.py`'s `screenshot_path` column, but nothing wires the two together yet (that's a Phase 3/main-loop concern, not this phase's). And `_default_date_range` anchors "today" to UTC, which can lag the IST trading date by up to a day during IST evening hours in live mode — worth a fix when live Kite testing actually begins.

---

## Upcoming Phases (summaries — to be written in full detail after Phase 4 review)

**Phase 5 — Eval Harness:** JSON test cases (PRD Section 17) with input transcript, optional mocked screen result, expected tool calls, and tools that must NOT be called. A runner asserts on tool names/args only (not exact wording), run against `mode: "mock"` end-to-end through Phases 1-4 (using this phase's `TOOLS_SCHEMA`/`make_tool_executor` directly). Seeds the 8 starting cases from the PRD — including the constraint this phase's `registry.py` documented but didn't enforce in code: "Fetch me the latest news on Nifty" must never call `get_positions_holdings`, and "What are my current holdings?" must never call `get_news`. Provides the harness for adding a case every time real usage surfaces a wrong tool call or a hallucinated figure.
