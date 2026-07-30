from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

import matplotlib

# This module only ever saves figures to disk (mpf.plot(..., savefig=path)) and
# never displays them, so it must not depend on a GUI toolkit (e.g. Tk) being
# available/working on the host machine. The backend has to be selected before
# mplfinance (which imports matplotlib.pyplot) picks one on its own.
matplotlib.use("Agg")

import mplfinance as mpf
import pandas as pd

from financial_voice_agent.tools.indicators import (
    INDICATOR_MIN_CANDLES,
    InsufficientDataError,
    bollinger_bands_series,
    fibonacci_levels,
    moving_average_series,
    rsi_series,
)

HistoryFn = Callable[..., Awaitable[dict]]

_VALID_CHART_INDICATORS = {"moving_average", "bollinger", "rsi", "fibonacci"}
_MOVING_AVERAGE_WINDOW = 20
_BOLLINGER_WINDOW = 20
_RSI_WINDOW = 14
# Matches the overlay panel's _CHART_PANEL_WIDTH/_CHART_PANEL_HEIGHT (520x420px)
# at mplfinance's default DPI of 100. Must stay in sync with
# financial_voice_agent/overlay/screen_overlay.py's panel dimensions -- tk.PhotoImage
# does no scaling, so a mismatch here crops the chart in the overlay panel.
_CHART_FIGSIZE_INCHES = (5.2, 4.2)


class UnknownChartIndicatorError(ValueError):
    pass


class ChartPathError(ValueError):
    pass


# Whitelist, not blacklist. `symbol` reaches here from the LLM, which in turn
# reads untrusted third-party content (news results, screenshots), so it must
# be treated as attacker-influenced. A blacklist that only replaced "/" left
# Windows' "\" separator live: a symbol like "\..\..\evil" escaped output_dir
# entirely once os.path.join normalized it.
_UNSAFE_SYMBOL_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]")


def _chart_output_path(symbol: str, output_dir: str) -> str:
    safe_symbol = _UNSAFE_SYMBOL_CHARS_RE.sub("_", symbol)[:32]
    path = os.path.join(output_dir, f"chart_{safe_symbol}_{int(time.time() * 1000)}.png")
    # Belt-and-braces: even with the whitelist above, assert the resolved file
    # really lands inside output_dir before anything writes to it.
    resolved_dir = os.path.realpath(output_dir)
    resolved_path = os.path.realpath(path)
    if os.path.commonpath([resolved_path, resolved_dir]) != resolved_dir:
        raise ChartPathError(f"refusing to write chart outside {output_dir!r}")
    return path


def _default_chart_range(interval: str) -> tuple[str, str]:
    to_date = datetime.now(timezone.utc)
    days_back = 90 if interval == "day" else 5
    from_date = to_date - timedelta(days=days_back)
    return from_date.date().isoformat(), to_date.date().isoformat()


def _candles_to_dataframe(candles: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    df = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _prepare_chart_data(candles: list[dict], indicators: list[str]) -> tuple[pd.DataFrame, dict]:
    for name in indicators:
        if name not in _VALID_CHART_INDICATORS:
            raise UnknownChartIndicatorError(f"Unknown chart indicator: {name}")

    min_required = max((INDICATOR_MIN_CANDLES[i] for i in indicators), default=1)
    if len(candles) < min_required:
        raise InsufficientDataError(
            f"chart requires at least {min_required} candles for {indicators}, got {len(candles)}"
        )

    df = _candles_to_dataframe(candles)
    closes = df["Close"]
    indicator_data: dict = {}

    if "moving_average" in indicators:
        indicator_data["moving_average"] = moving_average_series(closes, window=_MOVING_AVERAGE_WINDOW)
    if "bollinger" in indicators:
        _middle, upper, lower = bollinger_bands_series(closes, window=_BOLLINGER_WINDOW)
        indicator_data["bollinger_upper"] = upper
        indicator_data["bollinger_lower"] = lower
    if "rsi" in indicators:
        indicator_data["rsi"] = rsi_series(closes, window=_RSI_WINDOW)
    if "fibonacci" in indicators:
        indicator_data["fibonacci_levels"] = fibonacci_levels(closes)

    return df, indicator_data


async def render_chart(
    symbol: str,
    indicators: list[str] | None = None,
    *,
    history_fn: HistoryFn,
    interval: str = "day",
    output_dir: str = "charts",
) -> str:
    indicators = indicators or []
    # Validated here too (not just inside _prepare_chart_data) so an unknown
    # indicator is rejected before spending a network call on history_fn.
    for name in indicators:
        if name not in _VALID_CHART_INDICATORS:
            raise UnknownChartIndicatorError(f"Unknown chart indicator: {name}")

    from_date, to_date = _default_chart_range(interval)
    history = await history_fn(symbol=symbol, interval=interval, from_date=from_date, to_date=to_date)
    df, indicator_data = _prepare_chart_data(history["candles"], indicators)

    addplots = []
    hlines = None
    panel_ratios = None

    if "moving_average" in indicator_data:
        addplots.append(mpf.make_addplot(indicator_data["moving_average"], color="blue"))
    if "bollinger_upper" in indicator_data:
        addplots.append(mpf.make_addplot(indicator_data["bollinger_upper"], color="gray"))
        addplots.append(mpf.make_addplot(indicator_data["bollinger_lower"], color="gray"))
    if "rsi" in indicator_data:
        addplots.append(mpf.make_addplot(indicator_data["rsi"], panel=1, color="purple", ylabel="RSI"))
        panel_ratios = (3, 1)
    if "fibonacci_levels" in indicator_data:
        levels = list(indicator_data["fibonacci_levels"].values())
        hlines = dict(hlines=levels, colors=["#999999"] * len(levels), linestyle="--")

    os.makedirs(output_dir, exist_ok=True)
    path = _chart_output_path(symbol, output_dir)

    plot_kwargs = dict(
        type="candle",
        volume=False,
        style="charles",
        savefig=path,
        figsize=_CHART_FIGSIZE_INCHES,
    )
    if addplots:
        plot_kwargs["addplot"] = addplots
    if hlines is not None:
        plot_kwargs["hlines"] = hlines
    if panel_ratios is not None:
        plot_kwargs["panel_ratios"] = panel_ratios

    mpf.plot(df, **plot_kwargs)
    return path
