from __future__ import annotations

from typing import Awaitable, Callable

import pandas as pd

HistoryFn = Callable[..., Awaitable[dict]]

INDICATOR_MIN_CANDLES = {"bollinger": 20, "moving_average": 20, "rsi": 15, "fibonacci": 2}


class InsufficientDataError(Exception):
    pass


async def compute_indicator(
    symbol: str, indicator: str, params: dict, *, history_fn: HistoryFn
) -> dict:
    if indicator not in INDICATOR_MIN_CANDLES:
        raise ValueError(f"Unknown indicator: {indicator}")

    history = await history_fn(
        symbol=symbol,
        interval=params.get("interval", "15minute"),
        from_date=params.get("from"),
        to_date=params.get("to"),
    )
    candles = history["candles"]
    min_required = INDICATOR_MIN_CANDLES[indicator]
    if len(candles) < min_required:
        raise InsufficientDataError(
            f"{indicator} requires at least {min_required} candles, got {len(candles)}"
        )

    closes = pd.Series([c["close"] for c in candles])
    if indicator == "bollinger":
        return _bollinger_bands(closes, window=params.get("window", 20))
    if indicator == "moving_average":
        return _moving_average(closes, window=params.get("window", 20))
    if indicator == "rsi":
        return _rsi(closes, window=params.get("window", 14))
    return _fibonacci_retracement(closes)


def _bollinger_bands(closes: pd.Series, *, window: int) -> dict:
    sma = closes.rolling(window).mean()
    std = closes.rolling(window).std(ddof=0)
    upper = sma + 2 * std
    lower = sma - 2 * std
    return {
        "middle_band": float(sma.iloc[-1]),
        "upper_band": float(upper.iloc[-1]),
        "lower_band": float(lower.iloc[-1]),
    }


def _moving_average(closes: pd.Series, *, window: int) -> dict:
    return {"moving_average": float(closes.rolling(window).mean().iloc[-1])}


def _rsi(closes: pd.Series, *, window: int) -> dict:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    last_avg_loss = avg_loss.iloc[-1]
    if last_avg_loss == 0:
        return {"rsi": 100.0}
    rs = avg_gain.iloc[-1] / last_avg_loss
    return {"rsi": float(100 - (100 / (1 + rs)))}


def _fibonacci_retracement(closes: pd.Series) -> dict:
    high = float(closes.max())
    low = float(closes.min())
    diff = high - low
    return {
        "fibonacci_levels": {
            "0.0%": high,
            "23.6%": high - 0.236 * diff,
            "38.2%": high - 0.382 * diff,
            "50.0%": high - 0.5 * diff,
            "61.8%": high - 0.618 * diff,
            "100.0%": low,
        }
    }
