from __future__ import annotations

from financial_voice_agent import mock
from financial_voice_agent.tools.instruments import get_instrument_token
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
    exchange: str = "NSE",
    instrument_cache: dict | None = None,
) -> dict:
    if mode == "mock":
        data = mock.load_fixture("ohlc_history", fixtures_dir=fixtures_dir)
        return {"symbol": data["symbol"], "interval": data["interval"], "candles": data["candles"]}
    # Kite's historical-candles endpoint takes a numeric instrument_token in
    # the URL path, not the trading symbol -- there's no per-symbol lookup
    # endpoint, only a full instrument dump to search (confirmed against
    # kite.trade/docs/connect/v3/market-data-and-instruments). instrument_cache
    # should be one dict shared across calls for the process's lifetime (see
    # tools/registry.py's make_tool_executor) so the (large) dump is fetched
    # once, not per call.
    instrument_token = await get_instrument_token(
        symbol, http_client=http_client, cache=instrument_cache if instrument_cache is not None else {},
        exchange=exchange,
    )
    raw = await kite_get(
        http_client,
        f"/instruments/historical/{instrument_token}/{interval}",
        params={"from": from_date, "to": to_date},
    )
    candles = [
        {"ts": c[0], "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5]}
        for c in raw["data"]["candles"]
    ]
    return {"symbol": symbol, "interval": interval, "candles": candles}
