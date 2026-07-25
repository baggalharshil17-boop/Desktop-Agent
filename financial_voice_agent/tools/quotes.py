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
