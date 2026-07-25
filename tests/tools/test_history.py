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
