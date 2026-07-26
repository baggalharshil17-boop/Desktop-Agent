from __future__ import annotations

import csv
import io

from financial_voice_agent.tools.kite_client import kite_get_csv


class InstrumentNotFoundError(Exception):
    pass


async def get_instrument_token(
    symbol: str,
    *,
    http_client,
    cache: dict[tuple[str, str], int],
    exchange: str = "NSE",
) -> int:
    """Resolves a plain trading symbol (e.g. "CAPLIPOINT", "NIFTY 50") to the
    numeric instrument_token Kite's historical-candles endpoint requires --
    there is no way to query candles by symbol name directly (confirmed
    against kite.trade/docs/connect/v3/market-data-and-instruments: the
    dump's columns are instrument_token, exchange_token, tradingsymbol,
    name, last_price, expiry, strike, tick_size, lot_size, instrument_type,
    segment, exchange).

    Kite's own docs say the dump regenerates at most once a day and
    recommend fetching it sparingly (once a day), so the whole per-exchange
    dump is cached in the caller-supplied `cache` dict rather than
    re-fetched per lookup -- callers should hold onto one `cache` dict for
    the process's lifetime (see tools/registry.py's make_tool_executor).
    """
    cache_key = (exchange, symbol)
    if cache_key not in cache:
        csv_text = await kite_get_csv(http_client, f"/instruments/{exchange}")
        for row in csv.DictReader(io.StringIO(csv_text)):
            cache[(exchange, row["tradingsymbol"])] = int(row["instrument_token"])

    if cache_key not in cache:
        raise InstrumentNotFoundError(f"Instrument not found: {exchange}:{symbol}")
    return cache[cache_key]
