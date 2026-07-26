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
async def test_get_quote_live_mode_prepends_nse_exchange_for_bare_symbol():
    # The tool schema tells the LLM to pass bare symbols like "RELIANCE"
    # (no exchange prefix) -- Kite's /quote rejects that with a 403 unless
    # it's sent as "exchange:tradingsymbol".
    class _FakeHttpClient:
        def __init__(self):
            self.received_params = None

        async def get(self, path, params=None):
            self.received_params = params

            class _Resp:
                status_code = 200
                def json(self):
                    return {
                        "data": {
                            "NSE:CAPLIPOINT": {
                                "last_price": 2500.0,
                                "ohlc": {"open": 2480.0, "high": 2510.0, "low": 2470.0},
                                "volume": 100000,
                            }
                        }
                    }
                def raise_for_status(self):
                    pass
            return _Resp()

    client = _FakeHttpClient()
    result = await get_quote("CAPLIPOINT", http_client=client, mode="live")

    assert client.received_params == {"i": "NSE:CAPLIPOINT"}
    assert result["last_price"] == 2500.0


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
