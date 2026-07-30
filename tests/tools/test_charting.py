import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

import pandas as pd
import pytest

from financial_voice_agent.tools.charting import UnknownChartIndicatorError, _prepare_chart_data, render_chart
from financial_voice_agent.tools.indicators import InsufficientDataError


def _make_history_fn(closes: list[float]):
    async def history_fn(*, symbol, interval, from_date, to_date):
        return {
            "symbol": symbol,
            "interval": interval,
            "candles": [
                {
                    "ts": f"2026-01-{i + 1:02d}T00:00:00+05:30",
                    "open": c,
                    "high": c + 1,
                    "low": c - 1,
                    "close": c,
                    "volume": 1000,
                }
                for i, c in enumerate(closes)
            ],
        }
    return history_fn


def test_prepare_chart_data_builds_ohlc_dataframe_with_datetime_index():
    candles = [
        {"ts": "2026-01-01T00:00:00+05:30", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 500},
        {"ts": "2026-01-02T00:00:00+05:30", "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 600},
    ]

    df, indicator_data = _prepare_chart_data(candles, [])

    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df["Close"].iloc[-1] == pytest.approx(101.5)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert indicator_data == {}


def test_prepare_chart_data_computes_requested_moving_average():
    candles = [
        {"ts": f"2026-01-{i + 1:02d}T00:00:00+05:30", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 500}
        for i in range(20)
    ]

    df, indicator_data = _prepare_chart_data(candles, ["moving_average"])

    assert "moving_average" in indicator_data
    assert indicator_data["moving_average"].iloc[-1] == pytest.approx(100.0)


def test_prepare_chart_data_computes_bollinger_and_rsi_and_fibonacci():
    candles = [
        {"ts": f"2026-01-{(i % 28) + 1:02d}T00:00:00+05:30", "open": float(100 + i), "high": float(101 + i), "low": float(99 + i), "close": float(100 + i), "volume": 500}
        for i in range(20)
    ]

    df, indicator_data = _prepare_chart_data(candles, ["bollinger", "rsi", "fibonacci"])

    assert set(indicator_data.keys()) == {"bollinger_upper", "bollinger_lower", "rsi", "fibonacci_levels"}
    assert isinstance(indicator_data["fibonacci_levels"], dict)
    assert "0.0%" in indicator_data["fibonacci_levels"]


def test_prepare_chart_data_raises_on_unknown_indicator():
    candles = [{"ts": "2026-01-01T00:00:00+05:30", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}]

    with pytest.raises(UnknownChartIndicatorError, match="macd"):
        _prepare_chart_data(candles, ["macd"])


@pytest.mark.asyncio
async def test_render_chart_raises_on_unknown_indicator_before_fetching_history():
    history_fn = _make_history_fn([100.0] * 20)

    with pytest.raises(UnknownChartIndicatorError):
        await render_chart("RELIANCE", ["macd"], history_fn=history_fn)


@pytest.mark.asyncio
async def test_render_chart_raises_insufficient_data_when_not_enough_candles_for_requested_indicator():
    history_fn = _make_history_fn([100.0, 101.0])  # far fewer than bollinger's required 20

    with pytest.raises(InsufficientDataError, match="20"):
        await render_chart("RELIANCE", ["bollinger"], history_fn=history_fn)


@pytest.mark.asyncio
async def test_render_chart_saves_a_png_and_returns_its_path(tmp_path):
    history_fn = _make_history_fn([float(100 + i) for i in range(30)])

    path = await render_chart(
        "RELIANCE", ["moving_average", "rsi"], history_fn=history_fn,
        interval="day", output_dir=str(tmp_path),
    )

    import os
    assert os.path.exists(path)
    assert path.endswith(".png")


@pytest.mark.asyncio
async def test_render_chart_saves_plain_candlestick_with_no_indicators(tmp_path):
    """Regression test: render_chart with no indicators (addplot=None path) should produce PNG."""
    history_fn = _make_history_fn([float(100 + i) for i in range(30)])

    path = await render_chart(
        "RELIANCE", indicators=None, history_fn=history_fn,
        interval="day", output_dir=str(tmp_path),
    )

    import os
    assert os.path.exists(path)
    assert path.endswith(".png")


@pytest.mark.asyncio
async def test_render_chart_produces_png_matching_overlay_panel_dimensions(tmp_path):
    """Regression test: the rendered PNG must be exactly 520x420px so it isn't
    cropped by the overlay's tk.PhotoImage, which does no scaling (see
    financial_voice_agent/overlay/screen_overlay.py's _CHART_PANEL_WIDTH/_HEIGHT)."""
    from PIL import Image

    history_fn = _make_history_fn([float(100 + i) for i in range(30)])

    path = await render_chart(
        "RELIANCE", ["moving_average", "rsi"], history_fn=history_fn,
        interval="day", output_dir=str(tmp_path),
    )

    with Image.open(path) as img:
        assert img.size == (520, 420)
