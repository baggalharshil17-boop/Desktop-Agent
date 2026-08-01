# Momentum Stock Screener — Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let the voice agent answer "show me the top N momentum stocks in [sector]" by screening a basket of NSE/BSE symbols against Alpha Vantage's RSI data, without any UI automation against Kite.

**Context:** This originated from a broader ask ("mouse/UI automation so the agent can operate inside Kite"). Scoping that down: transacting is fully out of scope, and "search/discovery" turned out not to need UI automation at all — it needs a new data-driven screening tool. "View/monitor" (analyzing a chart the user already has open in Kite) is already covered by the existing `capture_screen` tool. So this spec covers only the one genuinely new capability: momentum screening via Alpha Vantage.

**Architecture:** A single new tool, `screen_stocks`, following the existing tool pattern in `financial_voice_agent/tools/` (same shape as `get_stock_fundamentals`). It maps a sector name to a hardcoded list of NSE/BSE ticker symbols, queries Alpha Vantage's `RSI` endpoint for each, filters by momentum threshold and price range, and returns the top N ranked by RSI descending.

**Tech Stack:** `httpx` (existing HTTP client pattern), Alpha Vantage REST API (`ALPHA_VANTAGE_API_KEY` env var, new).

## Global Constraints

- Read-only. No order placement, no UI automation, no write access to Kite. This assistant's read-only design is a deliberate project-level constraint (see README) and this feature does not change that.
- Alpha Vantage free tier: 5 requests/minute, 25 requests/day (current published limit — verify against live docs before implementing, since Alpha Vantage has changed this before). The tool must not silently exceed this; it must fail gracefully with a clear error instead.
- Sector-to-symbol mapping is hardcoded for the initial sectors used in testing (tech, finance, pharma, auto) — not a general-purpose sector database. Follows existing project pattern of small, focused, non-exhaustive lookups (see `_summarize_fundamentals` in `fundamentals.py` for the analogous fundamentals-shaping style).
- Indian Stock API integration for screening is explicitly deferred (Phase 2, not in this plan) — noted here so it isn't accidentally scope-crept into this work.
- New tool must be wired into `TOOLS_SCHEMA` and the dispatch table in `tools/registry.py`, following the exact pattern of the other tools already there (`get_stock_fundamentals` is the closest analog: external HTTP API, `mode: "live"`/`"mock"` split, graceful error dict on failure).
- Must support `mode: "mock"` via a fixture (consistent with every other live-data tool in this codebase — see `mock.load_fixture` usage in `fundamentals.py`).

---

## Components

### 1. `financial_voice_agent/tools/screener.py` (new file)

Responsible for: sector→symbol mapping, calling Alpha Vantage's RSI endpoint per symbol, filtering, ranking, and shaping the response.

```python
from __future__ import annotations

import asyncio

import httpx

from financial_voice_agent import mock

# Small, explicit sector -> NSE/BSE symbol list. Not exhaustive; extend as
# needed. Alpha Vantage requires an exchange suffix (.BSE here, since that's
# what's been verified live) -- see design doc for Alpha Vantage caveats.
SECTOR_SYMBOLS: dict[str, list[str]] = {
    "tech": ["TCS.BSE", "INFY.BSE", "WIPRO.BSE", "HCLTECH.BSE", "TECHM.BSE"],
    "finance": ["HDFCBANK.BSE", "ICICIBANK.BSE", "SBIN.BSE", "KOTAKBANK.BSE", "AXISBANK.BSE"],
    "pharma": ["SUNPHARMA.BSE", "DRREDDY.BSE", "CIPLA.BSE", "DIVISLAB.BSE", "LUPIN.BSE"],
    "auto": ["MARUTI.BSE", "TATAMOTORS.BSE", "M&M.BSE", "BAJAJ-AUTO.BSE", "EICHERMOT.BSE"],
}


async def _fetch_rsi(http_client, symbol: str) -> dict | None:
    """Fetches the latest RSI value for one symbol.

    Returns None (not an exception) on any per-symbol failure -- a single
    bad symbol must not fail the whole screen, it should just be skipped
    (see screen_stocks' filtering, which drops None entries).
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
        technical_analysis = data.get("Technical Analysis: RSI", {})
        if not technical_analysis:
            return None
        latest_date = max(technical_analysis.keys())
        rsi = float(technical_analysis[latest_date]["RSI"])
        return {"symbol": symbol, "rsi": rsi}
    except Exception:  # noqa: BLE001 -- any per-symbol failure just drops that symbol
        return None


async def _fetch_price(http_client, symbol: str) -> float | None:
    """Fetches the latest close price for one symbol, for price-range filtering."""
    try:
        response = await http_client.get(
            "/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol},
        )
        response.raise_for_status()
        data = response.json()
        quote = data.get("Global Quote", {})
        price = quote.get("05. price")
        return float(price) if price else None
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
) -> dict:
    if mode == "mock":
        return mock.load_fixture("stock_screener", fixtures_dir=fixtures_dir)

    sector_key = sector.strip().lower()
    symbols = SECTOR_SYMBOLS.get(sector_key)
    if symbols is None:
        return {
            "error": f"Unknown sector: '{sector}'. Try: {', '.join(sorted(SECTOR_SYMBOLS))}"
        }

    try:
        rsi_results = await asyncio.gather(*(_fetch_rsi(http_client, s) for s in symbols))
    except Exception:  # noqa: BLE001 -- network/timeout errors must degrade gracefully
        return {"error": f"Could not reach the screener data provider for sector '{sector}'"}

    candidates = [r for r in rsi_results if r is not None and r["rsi"] >= momentum_threshold]

    if not candidates:
        return {"results": [], "count": 0}

    # Price fetched only for symbols that already passed the RSI filter --
    # keeps API calls down against Alpha Vantage's tight free-tier limits.
    prices = await asyncio.gather(*(_fetch_price(http_client, c["symbol"]) for c in candidates))
    for candidate, price in zip(candidates, prices):
        candidate["price"] = price

    filtered = [
        c
        for c in candidates
        if c["price"] is not None
        and c["price"] >= price_min
        and (price_max is None or c["price"] <= price_max)
    ]

    filtered.sort(key=lambda c: c["rsi"], reverse=True)
    top = filtered[:limit]

    return {
        "results": [
            {"symbol": c["symbol"], "price": c["price"], "rsi": round(c["rsi"], 1)}
            for c in top
        ],
        "count": len(top),
    }
```

**Rate-limit note (must be validated during implementation, not assumed):** Alpha Vantage's free tier is 5 requests/minute. A single `screen_stocks` call against a 5-symbol sector list makes up to 10 requests (RSI + price, and price is only for candidates that already passed the RSI filter). That can exceed the per-minute cap in one screen call. The implementation task must add a rate limiter (simple sleep-based throttle between requests, e.g. via `asyncio.sleep`) or reduce default sector list size — this is a concrete task-level decision, not deferred to "later."

### 2. `financial_voice_agent/tools/registry.py` (modified)

Add to `TOOLS_SCHEMA`:

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
```

Add dispatch branch (follows the exact pattern of the `get_stock_fundamentals` branch already in the file):

```python
if call.name == "screen_stocks":
    return await screen_stocks(
        **args, http_client=http_clients.alpha_vantage, mode=config.mode,
    )
```

And the import at the top of the file:

```python
from financial_voice_agent.tools.screener import screen_stocks
```

### 3. `financial_voice_agent/config.py` (modified)

Add an `alpha_vantage` HTTP client alongside the existing `indian_stock` one, using the same construction pattern (base URL + API key from env, timeout). Add `ALPHA_VANTAGE_API_KEY` to the `.env.example` template and `scripts/setup.py`'s wizard (same flow as the other vendor keys — validates live during setup, per the existing wizard pattern).

### 4. `fixtures/stock_screener.json` (new file)

Mock-mode fixture, same shape as the `results`/`count` response above, with 2-3 example entries — mirrors the existing `fixtures/stock_fundamentals.json` pattern.

---

## Data Flow

1. User: "Show me momentum stocks in tech, top 10"
2. LLM parses intent → calls `screen_stocks(sector="tech", limit=10)` (momentum_threshold/price_min/price_max default if unspecified)
3. Tool maps `"tech"` → 5 hardcoded BSE symbols
4. Fetches RSI per symbol (throttled to respect Alpha Vantage's rate limit), filters to `rsi >= momentum_threshold`
5. Fetches price only for RSI-passing candidates, filters to price range
6. Sorts by RSI descending, returns top `limit`
7. Agent speaks results: "Top momentum tech stocks: TCS at ₹3850 (RSI 72.5), Infosys at ₹1920 (RSI 68.3)." and may offer to render a chart via the existing `show_chart` tool for any of them — no new integration needed there, the LLM can just chain tool calls in conversation.

## Error Handling

- Alpha Vantage rate limit exceeded (HTTP 429, or their documented "Note" field in a 200 response indicating throttling) → `{"error": "Screener temporarily rate-limited, try again in a minute"}`
- Unknown sector → `{"error": "Unknown sector: '<name>'. Try: auto, finance, pharma, tech"}`
- Individual symbol fetch failure → silently dropped from results (not a whole-call failure)
- No candidates pass the filters → `{"results": [], "count": 0}` (not an error — a valid empty result)
- Network/timeout at the gather level → `{"error": "Could not reach the screener data provider for sector '<sector>'"}`

## Testing

- Unit tests for `_fetch_rsi`/`_fetch_price` parsing against mocked Alpha Vantage JSON responses (both success and malformed/missing-key shapes)
- Unit tests for `screen_stocks` filtering/ranking logic with a fake `http_client` returning canned per-symbol responses (covers: threshold filtering, price range filtering, limit truncation, unknown sector, empty results)
- One live integration test (manual, not in CI) against 1-2 real symbols to confirm Alpha Vantage's actual response shape matches what parsing expects — Alpha Vantage's response format is a documented risk area (nested under a variable date-keyed dict) worth confirming against the real API before trusting the parsing code
- Mock-mode test: `mode="mock"` returns the fixture unchanged

## Out of Scope (explicitly)

- Any UI automation against Kite (browser or desktop) — search/discovery is fully served by this API-based tool instead
- Order placement / transacting
- Indian Stock API as a screening data source (deferred to Phase 2)
- General-purpose/dynamic sector-to-symbol mapping (e.g., fetched from an index/exchange listing) — hardcoded list is sufficient for now
