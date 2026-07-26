import pytest

from financial_voice_agent.tools.history import get_ohlc_history


@pytest.mark.asyncio
async def test_get_ohlc_history_mock_mode_reads_fixture():
    result = await get_ohlc_history(
        "RELIANCE", "15minute", "2026-07-24", "2026-07-25", http_client=None, mode="mock"
    )

    assert result["symbol"] == "RELIANCE"
    assert result["interval"] == "15minute"
    assert len(result["candles"]) == 25
    assert result["candles"][0]["close"] == 2952.5


@pytest.mark.asyncio
async def test_get_ohlc_history_live_mode_calls_kite_get_and_normalizes():
    class _InstrumentsResp:
        status_code = 200
        content = (
            b"instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,"
            b"tick_size,lot_size,instrument_type,segment,exchange\r\n"
            b"128031234,500325,RELIANCE,RELIANCE INDUSTRIES,0,,0,0.05,1,EQ,NSE,NSE\r\n"
        )
        def raise_for_status(self):
            pass

    class _CandlesResp:
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

    class _FakeHttpClient:
        def __init__(self):
            self.received_paths = []

        async def get(self, path, params=None):
            self.received_paths.append(path)
            if path == "/instruments/NSE":
                return _InstrumentsResp()
            return _CandlesResp()

    http_client = _FakeHttpClient()
    result = await get_ohlc_history(
        "RELIANCE", "15minute", "2026-07-24", "2026-07-25",
        http_client=http_client, mode="live",
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
    # Resolved the trading symbol to its numeric instrument_token via the
    # instruments dump, then used that token (not the bare symbol) in the
    # historical-candles path.
    assert http_client.received_paths == ["/instruments/NSE", "/instruments/historical/128031234/15minute"]
