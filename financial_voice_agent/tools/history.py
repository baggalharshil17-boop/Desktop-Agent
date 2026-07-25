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
