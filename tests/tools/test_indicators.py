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


@pytest.mark.asyncio
async def test_compute_indicator_raises_when_caller_requests_larger_window_than_available_candles():
    # 15 candles is enough for the default rsi window (14, needs 15 candles),
    # but NOT enough for a caller-requested window of 20 (needs 21 candles).
    # Must still raise InsufficientDataError rather than silently computing
    # NaN via a rolling(20) window that doesn't fit.
    closes = [float(i) for i in range(1, 16)]  # 15 candles
    history_fn = _make_history_fn(closes)

    with pytest.raises(InsufficientDataError, match="21"):
        await compute_indicator(
            "RELIANCE", "rsi", {"window": 20}, history_fn=history_fn
        )


@pytest.mark.asyncio
async def test_compute_indicator_defaults_params_to_empty_dict_when_omitted():
    closes = [100.0] * 20
    history_fn = _make_history_fn(closes)

    # Simulates the real registry dispatcher's call shape when the LLM omits
    # "params" entirely from a tool call -- must not raise TypeError.
    result = await compute_indicator("RELIANCE", "moving_average", history_fn=history_fn)

    assert result["moving_average"] == pytest.approx(100.0)
