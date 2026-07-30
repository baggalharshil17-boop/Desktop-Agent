# v4.0: Screen Overlay, Technical Charts, and Investment-Analyst Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a processing-feedback screen overlay (glow while the agent is silently working), a `show_chart` tool that renders candlestick charts with technical indicator overlays into a sliding on-screen panel, and a `get_stock_fundamentals` tool backed by a new external API, plus investment-analyst-style system prompt guidance.

**Architecture:** A standalone `screen_overlay.py` process (tkinter, always-on-top) is launched alongside the voice agent and driven by single-packet UDP messages sent from the agent process over localhost — `processing_on`/`processing_off` toggle a click-through edge glow, `show_chart:<path>` slides in a chart panel loaded from a PNG. Chart rendering reuses the existing `get_ohlc_history` tool and extends `indicators.py`'s existing math to expose full series (not just latest value) via `mplfinance`. Stock fundamentals is a new DI-pattern tool (`fundamentals.py`) calling the Indian Stock API, matching the existing `get_quote`/`get_news` shape.

**Tech Stack:** `tkinter` + `ctypes` (stdlib, for the overlay window), `mplfinance` (new dependency, candlestick rendering), `httpx` (already a dependency, for the fundamentals API), `pandas` (already a dependency).

## Global Constraints

- Windows only.
- New dependency: `mplfinance` (added to `requirements.txt`) — everything else needed (`httpx`, `pandas`, `tkinter`) is already available.
- The overlay process must never block or crash the voice agent — all agent-side sends to it are best-effort (swallow all exceptions).
- `show_chart` and `get_stock_fundamentals` are LLM-callable tools the model decides when to use, same as `capture_screen` today — no new trigger mechanism.
- Tool results that fail must return `{"error": ...}` rather than raise, matching every existing tool in `tools/registry.py`'s executor.
- The overlay glow is click-through; the chart panel is not (it needs a clickable close button).
- The `screen_overlay.py` GUI process itself is not unit tested, matching this project's existing convention for interactive/GUI scripts (`scripts/kite_login.py`, `scripts/setup.py`) — verified by one real manual run instead.
- Follow the existing DI testing pattern throughout: injectable `history_fn`/`http_client`/factory parameters with real defaults, matching `tools/quotes.py`, `tools/news.py`, `tools/indicators.py`.

---

### Task 1: Overlay signal client

**Files:**
- Create: `financial_voice_agent/overlay/__init__.py` (empty package marker)
- Create: `financial_voice_agent/overlay/signal_client.py`
- Test: `tests/overlay/__init__.py` (empty package marker)
- Test: `tests/overlay/test_signal_client.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DEFAULT_OVERLAY_PORT: int = 47765` and `make_overlay_sender(port: int = DEFAULT_OVERLAY_PORT, *, socket_factory: Callable[[], socket.socket] | None = None) -> Callable[[str], None]`. Later tasks (5, 6) call this by this exact name/signature to get a `send(message: str) -> None` callable.

- [ ] **Step 1: Write the failing tests**

Create `tests/overlay/__init__.py` (empty file).

Create `tests/overlay/test_signal_client.py`:
```python
from financial_voice_agent.overlay.signal_client import DEFAULT_OVERLAY_PORT, make_overlay_sender


class _FakeSocket:
    def __init__(self, *, raise_on_send: bool = False):
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self._raise_on_send = raise_on_send

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        if self._raise_on_send:
            raise OSError("no listener")
        self.sent.append((data, addr))


def test_make_overlay_sender_sends_utf8_encoded_message_to_loopback():
    fake_socket = _FakeSocket()
    send = make_overlay_sender(47765, socket_factory=lambda: fake_socket)

    send("processing_on")

    assert fake_socket.sent == [(b"processing_on", ("127.0.0.1", 47765))]


def test_make_overlay_sender_uses_default_port_when_not_specified():
    fake_socket = _FakeSocket()
    send = make_overlay_sender(socket_factory=lambda: fake_socket)

    send("processing_off")

    assert fake_socket.sent[0][1] == ("127.0.0.1", DEFAULT_OVERLAY_PORT)


def test_make_overlay_sender_swallows_os_error_when_overlay_not_running():
    fake_socket = _FakeSocket(raise_on_send=True)
    send = make_overlay_sender(47765, socket_factory=lambda: fake_socket)

    send("processing_on")  # must not raise


def test_make_overlay_sender_reuses_same_socket_across_calls():
    fake_socket = _FakeSocket()
    factory_calls = []

    def factory():
        factory_calls.append(1)
        return fake_socket

    send = make_overlay_sender(47765, socket_factory=factory)
    send("processing_on")
    send("processing_off")

    assert len(factory_calls) == 1
    assert len(fake_socket.sent) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/overlay/test_signal_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.overlay'`

- [ ] **Step 3: Write the implementation**

Create `financial_voice_agent/overlay/__init__.py` (empty file).

Create `financial_voice_agent/overlay/signal_client.py`:
```python
from __future__ import annotations

import socket
from typing import Callable

# Arbitrary fixed high port for localhost-only agent<->overlay signaling.
# Chosen to be unlikely to collide with common dev-server ports.
DEFAULT_OVERLAY_PORT = 47765


def make_overlay_sender(
    port: int = DEFAULT_OVERLAY_PORT,
    *,
    socket_factory: Callable[[], socket.socket] | None = None,
) -> Callable[[str], None]:
    sock = (socket_factory or (lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)))()

    def send(message: str) -> None:
        # Fire-and-forget: the overlay process may not be running (disabled
        # via config, crashed, or not yet started), and a missing overlay
        # must never block or crash the voice agent.
        try:
            sock.sendto(message.encode("utf-8"), ("127.0.0.1", port))
        except OSError:
            pass

    return send
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/overlay/test_signal_client.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/overlay/__init__.py financial_voice_agent/overlay/signal_client.py tests/overlay/__init__.py tests/overlay/test_signal_client.py
git commit -m "Add overlay signal client for agent-to-overlay UDP messaging"
```

---

### Task 2: Indicator series functions

**Files:**
- Modify: `financial_voice_agent/tools/indicators.py`
- Test: `tests/tools/test_indicators.py` (add new tests; existing tests must keep passing unmodified)

**Interfaces:**
- Consumes: nothing new.
- Produces: `moving_average_series(closes: pd.Series, *, window: int) -> pd.Series`, `bollinger_bands_series(closes: pd.Series, *, window: int) -> tuple[pd.Series, pd.Series, pd.Series]` (returns `(middle, upper, lower)`), `rsi_series(closes: pd.Series, *, window: int) -> pd.Series`, `fibonacci_levels(closes: pd.Series) -> dict[str, float]` (the existing level-name-to-price dict, extracted as a standalone public function). Task 3 (`charting.py`) imports and calls these by these exact names.

- [ ] **Step 1: Write the failing tests**

Add to `tests/tools/test_indicators.py` (append; keep all existing tests and imports, add `pandas as pd` and these names to the existing import line):
```python
from financial_voice_agent.tools.indicators import (
    InsufficientDataError,
    bollinger_bands_series,
    compute_indicator,
    fibonacci_levels,
    moving_average_series,
    rsi_series,
)


def test_moving_average_series_last_value_matches_scalar_result():
    closes = pd.Series([float(i) for i in range(1, 21)])

    series = moving_average_series(closes, window=20)

    assert series.iloc[-1] == pytest.approx(10.5)
    assert len(series) == 20


def test_bollinger_bands_series_returns_three_series_of_matching_length():
    closes = pd.Series([100.0] * 20)

    middle, upper, lower = bollinger_bands_series(closes, window=20)

    assert middle.iloc[-1] == pytest.approx(100.0)
    assert upper.iloc[-1] == pytest.approx(100.0)
    assert lower.iloc[-1] == pytest.approx(100.0)
    assert len(middle) == len(upper) == len(lower) == 20


def test_rsi_series_all_gains_last_value_is_100():
    closes = pd.Series([float(i) for i in range(1, 21)])

    series = rsi_series(closes, window=14)

    assert series.iloc[-1] == pytest.approx(100.0)


def test_rsi_series_matches_scalar_compute_indicator_result():
    import numpy as np

    np.random.seed(0)
    closes = pd.Series(100 + np.cumsum(np.random.randn(40)))

    async def history_fn(*, symbol, interval, from_date, to_date):
        return {
            "symbol": symbol,
            "interval": interval,
            "candles": [
                {"ts": f"t{i}", "open": c, "high": c, "low": c, "close": c, "volume": 100}
                for i, c in enumerate(closes)
            ],
        }

    scalar_result = await_helper(
        compute_indicator("RELIANCE", "rsi", {"window": 14}, history_fn=history_fn)
    )
    series = rsi_series(closes, window=14)

    assert series.iloc[-1] == pytest.approx(scalar_result["rsi"])


def test_fibonacci_levels_returns_same_shape_as_compute_indicator():
    closes = pd.Series([100.0, 150.0, 120.0])

    levels = fibonacci_levels(closes)

    assert levels["0.0%"] == pytest.approx(150.0)
    assert levels["100.0%"] == pytest.approx(100.0)
    assert levels["50.0%"] == pytest.approx(125.0)
```

This test file needs a small async-call helper since `test_rsi_series_matches_scalar_compute_indicator_result` is not itself an `async def` test (it mixes sync series computation with one async call). Add this helper near the top of `tests/tools/test_indicators.py`, after the imports:
```python
import asyncio


def await_helper(coro):
    return asyncio.get_event_loop().run_until_complete(coro)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tools/test_indicators.py -v`
Expected: FAIL with `ImportError: cannot import name 'moving_average_series'`

- [ ] **Step 3: Write the implementation**

Replace the bodies of `_moving_average`, `_bollinger_bands`, `_rsi`, `_fibonacci_retracement` in `financial_voice_agent/tools/indicators.py` (keep every other line in the file — `INDICATOR_MIN_CANDLES`, `_required_candles`, `_default_date_range`, `compute_indicator`, `InsufficientDataError` — exactly as-is):

```python
def _moving_average(closes: pd.Series, *, window: int) -> dict:
    return {"moving_average": float(moving_average_series(closes, window=window).iloc[-1])}


def moving_average_series(closes: pd.Series, *, window: int) -> pd.Series:
    return closes.rolling(window).mean()


def _bollinger_bands(closes: pd.Series, *, window: int) -> dict:
    middle, upper, lower = bollinger_bands_series(closes, window=window)
    return {
        "middle_band": float(middle.iloc[-1]),
        "upper_band": float(upper.iloc[-1]),
        "lower_band": float(lower.iloc[-1]),
    }


def bollinger_bands_series(closes: pd.Series, *, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    sma = closes.rolling(window).mean()
    std = closes.rolling(window).std(ddof=0)
    return sma, sma + 2 * std, sma - 2 * std


def _rsi(closes: pd.Series, *, window: int) -> dict:
    return {"rsi": float(rsi_series(closes, window=window).iloc[-1])}


def rsi_series(closes: pd.Series, *, window: int) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    # avg_loss.replace(0, nan) avoids a literal division-by-zero warning;
    # .where(...) below then fills those positions with the correct RSI=100
    # (no losses at all in the window means maximally overbought).
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi.where(avg_loss != 0, 100.0)


def _fibonacci_retracement(closes: pd.Series) -> dict:
    return {"fibonacci_levels": fibonacci_levels(closes)}


def fibonacci_levels(closes: pd.Series) -> dict:
    high = float(closes.max())
    low = float(closes.min())
    diff = high - low
    return {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "100.0%": low,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tools/test_indicators.py -v`
Expected: PASS (all existing tests plus the 5 new ones, 13 total)

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/tools/indicators.py tests/tools/test_indicators.py
git commit -m "Expose full indicator series alongside existing scalar results"
```

---

### Task 3: Charting module

**Files:**
- Create: `financial_voice_agent/tools/charting.py`
- Test: `tests/tools/test_charting.py`

**Interfaces:**
- Consumes: `financial_voice_agent.tools.indicators.{InsufficientDataError, INDICATOR_MIN_CANDLES, moving_average_series, bollinger_bands_series, rsi_series, fibonacci_levels}` (Task 2). `history_fn` has the same shape as `get_ohlc_history` (Task-2-independent, already exists): `async def history_fn(*, symbol, interval, from_date, to_date) -> dict` returning `{"symbol": ..., "interval": ..., "candles": [{"ts": ..., "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}, ...]}`.
- Produces: `render_chart(symbol: str, indicators: list[str] | None = None, *, history_fn, interval: str = "day", output_dir: str = "charts") -> str` (returns the saved PNG's file path). `UnknownChartIndicatorError(ValueError)`. Task 6 calls `render_chart` by this exact name/signature.

- [ ] **Step 1: Write the failing tests**

Create `tests/tools/test_charting.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tools/test_charting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.tools.charting'`

- [ ] **Step 3: Write the implementation**

Create `financial_voice_agent/tools/charting.py`:
```python
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

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


class UnknownChartIndicatorError(ValueError):
    pass


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
    safe_symbol = symbol.replace(" ", "_").replace("/", "_")
    path = os.path.join(output_dir, f"chart_{safe_symbol}_{int(time.time() * 1000)}.png")

    mpf.plot(
        df,
        type="candle",
        addplot=addplots or None,
        hlines=hlines,
        volume=False,
        panel_ratios=panel_ratios,
        style="charles",
        savefig=path,
    )
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tools/test_charting.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/tools/charting.py tests/tools/test_charting.py
git commit -m "Add chart rendering module (candlestick + indicator overlays)"
```

---

### Task 4: Screen overlay GUI process

**Files:**
- Create: `financial_voice_agent/overlay/screen_overlay.py`

**Interfaces:**
- Consumes: nothing from other tasks' Python code directly — it's a standalone process. Protocol contract (must match Task 1's sender and Task 5/6's message strings exactly): listens on UDP port `financial_voice_agent.overlay.signal_client.DEFAULT_OVERLAY_PORT` for `"processing_on"`, `"processing_off"`, and `"show_chart:<absolute file path>"` text messages.
- Produces: nothing consumed by other Python modules — it's launched as a subprocess (Task 5) via `python -m financial_voice_agent.overlay.screen_overlay`.

This task has no automated tests (matches this project's convention for interactive/GUI scripts — see Global Constraints). Verify by running it directly and confirming it doesn't crash on startup; full visual/behavioral verification happens in Task 9's manual verification step.

- [ ] **Step 1: Write the implementation**

Create `financial_voice_agent/overlay/screen_overlay.py`:
```python
"""Standalone always-on-top screen overlay process for the Financial Voice
Agent. Shows a click-through edge glow while the agent is processing, and a
sliding chart panel when a chart is rendered. Launched by __main__.py as a
subprocess (see financial_voice_agent/overlay/signal_client.py for the
message protocol this listens for). Run directly for manual testing:

    python -m financial_voice_agent.overlay.screen_overlay
"""

from __future__ import annotations

import ctypes
import queue
import socket
import threading
import tkinter as tk

from financial_voice_agent.overlay.signal_client import DEFAULT_OVERLAY_PORT

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020

_GLOW_COLOR = "#3399ff"
_GLOW_BORDER_PX = 8
_TRANSPARENT_COLOR = "black"  # chosen as the -transparentcolor key; never drawn otherwise
_CHART_PANEL_WIDTH = 520
_CHART_PANEL_HEIGHT = 420
_SLIDE_STEP_PX = 40
_SLIDE_INTERVAL_MS = 15


def _make_click_through(root: tk.Tk) -> None:
    root.update_idletasks()
    hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
    if hwnd == 0:
        hwnd = root.winfo_id()
    ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style | WS_EX_LAYERED | WS_EX_TRANSPARENT)


def _listen(inbox: "queue.Queue[str]", port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    while True:
        data, _addr = sock.recvfrom(2048)
        inbox.put(data.decode("utf-8"))


class ChartPanel:
    """A non-click-through Toplevel that slides in from the right edge of
    the screen when shown, and slides back out when closed."""

    def __init__(self, root: tk.Tk):
        self._root = root
        self._window = tk.Toplevel(root)
        self._window.overrideredirect(True)
        self._window.attributes("-topmost", True)
        self._window.withdraw()
        self._screen_w = root.winfo_screenwidth()
        self._screen_h = root.winfo_screenheight()
        self._target_x = self._screen_w - _CHART_PANEL_WIDTH - 20
        self._hidden_x = self._screen_w
        self._y = (self._screen_h - _CHART_PANEL_HEIGHT) // 2

        close_bar = tk.Frame(self._window, bg="#222222", height=24)
        close_bar.pack(fill="x", side="top")
        close_button = tk.Button(
            close_bar, text="×", command=self.hide, bg="#222222", fg="white",
            bd=0, activebackground="#444444", activeforeground="white",
        )
        close_button.pack(side="right")

        self._image_label = tk.Label(self._window)
        self._image_label.pack(fill="both", expand=True)
        self._current_image = None  # keep a reference so tkinter doesn't garbage-collect it

    def show(self, image_path: str) -> None:
        self._current_image = tk.PhotoImage(file=image_path)
        self._image_label.configure(image=self._current_image)
        self._window.geometry(
            f"{_CHART_PANEL_WIDTH}x{_CHART_PANEL_HEIGHT}+{self._hidden_x}+{self._y}"
        )
        self._window.deiconify()
        self._animate_to(self._target_x)

    def hide(self) -> None:
        self._animate_to(self._hidden_x, on_done=self._window.withdraw)

    def _animate_to(self, target_x: int, *, on_done=None) -> None:
        current_x = self._window.winfo_x()
        step = _SLIDE_STEP_PX if target_x > current_x else -_SLIDE_STEP_PX

        def tick() -> None:
            nonlocal current_x
            current_x += step
            reached = (step > 0 and current_x >= target_x) or (step < 0 and current_x <= target_x)
            if reached:
                current_x = target_x
            self._window.geometry(f"{_CHART_PANEL_WIDTH}x{_CHART_PANEL_HEIGHT}+{current_x}+{self._y}")
            if not reached:
                self._window.after(_SLIDE_INTERVAL_MS, tick)
            elif on_done is not None:
                on_done()

        tick()


def main() -> None:
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg=_TRANSPARENT_COLOR)
    root.attributes("-transparentcolor", _TRANSPARENT_COLOR)

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.geometry(f"{screen_w}x{screen_h}+0+0")

    canvas = tk.Canvas(root, width=screen_w, height=screen_h, bg=_TRANSPARENT_COLOR, highlightthickness=0)
    canvas.pack()
    glow_id = canvas.create_rectangle(
        _GLOW_BORDER_PX // 2, _GLOW_BORDER_PX // 2,
        screen_w - _GLOW_BORDER_PX // 2, screen_h - _GLOW_BORDER_PX // 2,
        outline=_GLOW_COLOR, width=_GLOW_BORDER_PX, state="hidden",
    )

    _make_click_through(root)

    chart_panel = ChartPanel(root)

    inbox: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_listen, args=(inbox, DEFAULT_OVERLAY_PORT), daemon=True).start()

    def poll() -> None:
        try:
            while True:
                message = inbox.get_nowait()
                if message == "processing_on":
                    canvas.itemconfigure(glow_id, state="normal")
                elif message == "processing_off":
                    canvas.itemconfigure(glow_id, state="hidden")
                elif message.startswith("show_chart:"):
                    chart_panel.show(message[len("show_chart:") :])
        except queue.Empty:
            pass
        root.after(30, poll)

    root.after(30, poll)
    root.mainloop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it starts without error**

Run: `python -m financial_voice_agent.overlay.screen_overlay`
Expected: a fullscreen transparent window opens with no visible glow (state="hidden" by default) and no traceback. Close it with Ctrl+C in the terminal (or Alt+F4 won't work since it's borderless — Ctrl+C is the way to stop it for this manual check).

- [ ] **Step 3: Commit**

```bash
git add financial_voice_agent/overlay/screen_overlay.py
git commit -m "Add screen overlay GUI process (processing glow + chart panel)"
```

---

### Task 5: Wire processing-glow signals into the voice loop

**Files:**
- Modify: `financial_voice_agent/orchestrator/main_loop.py`
- Modify: `financial_voice_agent/__main__.py`
- Modify: `financial_voice_agent/config.py`
- Modify: `config.yaml`
- Test: `tests/orchestrator/test_main_loop.py` (add new tests; existing tests must keep passing unmodified)
- Test: `tests/test_config.py` (add new test for the new field)

**Interfaces:**
- Consumes: `financial_voice_agent.overlay.signal_client.make_overlay_sender` (Task 1).
- Produces: `run_voice_loop(..., on_processing_start: Callable[[], None] | None = None, on_processing_end: Callable[[], None] | None = None)` — two new optional keyword parameters, defaulting to `None` (no-op), so all existing call sites and tests keep working unmodified. `Config.processing_overlay_enabled: bool` field.

- [ ] **Step 1: Write the failing tests**

Add to `tests/orchestrator/test_main_loop.py` (append; uses the existing `_FakePipeline`/`_FakePlayback` fixtures already defined in this file):
```python
@pytest.mark.asyncio
async def test_run_voice_loop_calls_on_processing_start_and_end_around_each_turn(tmp_path):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    pipeline = _FakePipeline([b"utterance-1"])
    playback = _FakePlayback()
    events: list[str] = []

    async def stt_fn(wav):
        return "hello"

    async def llm_fn(transcript, history):
        events.append("during_turn")
        return LlmTurnResult(response_text="hi there", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return b"audio-bytes"

    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path,
        on_processing_start=lambda: events.append("start"),
        on_processing_end=lambda: events.append("end"),
    )

    assert events == ["start", "during_turn", "end"]


@pytest.mark.asyncio
async def test_run_voice_loop_works_without_processing_callbacks(tmp_path):
    # Both callbacks default to None -- must not raise when omitted.
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    pipeline = _FakePipeline([b"utterance-1"])
    playback = _FakePlayback()

    async def stt_fn(wav):
        return "hello"

    async def llm_fn(transcript, history):
        return LlmTurnResult(response_text="hi there", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return b"audio-bytes"

    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
    )

    assert playback.play_calls == [b"audio-bytes"]
```

Add to `tests/test_config.py` (append, using this file's existing `_write_yaml`/`_write_env`/`VALID_YAML` helpers — `VALID_YAML` has no `overlay:` section, which is exactly what the first test below needs):
```python
def test_processing_overlay_enabled_defaults_to_true_when_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    yaml_path = _write_yaml(tmp_path, VALID_YAML)  # no overlay: section
    env_path = _write_env(
        tmp_path, "GROQ_API_KEY=groq-secret\nCARTESIA_API_KEY=cartesia-secret\n"
    )

    config = load_config(config_path=yaml_path, env_path=env_path)

    assert config.processing_overlay_enabled is True


def test_processing_overlay_enabled_reads_explicit_false(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    yaml_path = _write_yaml(tmp_path, VALID_YAML + "\noverlay:\n  enabled: false\n")
    env_path = _write_env(
        tmp_path, "GROQ_API_KEY=groq-secret\nCARTESIA_API_KEY=cartesia-secret\n"
    )

    config = load_config(config_path=yaml_path, env_path=env_path)

    assert config.processing_overlay_enabled is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/orchestrator/test_main_loop.py tests/test_config.py -v`
Expected: FAIL — `test_run_voice_loop_calls_on_processing_start_and_end_around_each_turn` fails with `TypeError: run_voice_loop() got an unexpected keyword argument 'on_processing_start'`.

- [ ] **Step 3: Write the implementation**

In `financial_voice_agent/orchestrator/main_loop.py`, modify `run_voice_loop`'s signature and body (around line 199-230):
```python
async def run_voice_loop(
    pipeline,
    playback,
    *,
    stt_fn,
    llm_fn,
    tts_fn,
    db_path: str,
    max_turns_history: int = 8,
    choice_fn: Callable[[list[str]], str] = random.choice,
    recent_fallback_window: int = 5,
    barge_in_enabled: bool = True,
    on_processing_start: Callable[[], None] | None = None,
    on_processing_end: Callable[[], None] | None = None,
) -> None:
    output_queue: asyncio.Queue = asyncio.Queue()
    drive_task = asyncio.create_task(drive_pipeline(pipeline, output_queue))
    history: list[dict] = []
    recent_fallback_messages: deque = deque(maxlen=recent_fallback_window)

    try:
        while not drive_task.done() or not output_queue.empty():
            try:
                utterance_wav = await asyncio.wait_for(output_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                if drive_task.done():
                    exc = drive_task.exception()
                    if exc is not None:
                        raise exc
                continue

            if on_processing_start is not None:
                on_processing_start()
            try:
                result = await run_turn(
                    utterance_wav, history, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
                )
            finally:
                if on_processing_end is not None:
                    on_processing_end()
```

(Everything below that `result = await run_turn(...)` call — the `if result.transcript is not None...` through the end of the `while` loop and the `finally: drive_task.cancel()` — stays exactly as it already is; only the signature and the two new callback invocations around the `run_turn` call change.)

In `financial_voice_agent/config.py`:
1. Add `processing_overlay_enabled: bool` to the `Config` dataclass (after `mode: str`, before `groq_api_key`).
2. In `load_config`, add `processing_overlay_enabled=raw.get("overlay", {}).get("enabled", True),` to the `Config(...)` constructor call (place it near `mode=raw["mode"],`).

In `config.yaml`, add a new top-level section (after the `mode: "live"  # or "mock"` line):
```yaml
overlay:
  # Shows a glow around the screen edges while the agent is silently
  # working (speech-to-text, LLM, and TTS generation) so there's visual
  # feedback during that gap. Set to false to disable the overlay process
  # entirely.
  enabled: true
```

In `financial_voice_agent/__main__.py`:
1. Add imports near the top (with the other `financial_voice_agent` imports):
```python
import subprocess
import sys

from financial_voice_agent.overlay.signal_client import make_overlay_sender
```
2. Immediately before `tool_executor = make_tool_executor(config, http_clients)` (around line 84), insert:
```python
    overlay_process = None
    overlay_sender = None
    if config.processing_overlay_enabled:
        overlay_process = subprocess.Popen(
            [sys.executable, "-m", "financial_voice_agent.overlay.screen_overlay"]
        )
        overlay_sender = make_overlay_sender()
```
(This ordering matters: Task 6 will change the `tool_executor = make_tool_executor(config, http_clients)` line itself to pass `overlay_sender=overlay_sender`, so `overlay_sender` must already be assigned above it — this task only inserts the block above; it does not yet change the `make_tool_executor(...)` call's arguments.)
3. In the `run_voice_loop(...)` call (around line 167-175), add the two new keyword arguments:
```python
        await run_voice_loop(
            pipeline,
            playback,
            stt_fn=stt_fn,
            llm_fn=llm_fn,
            tts_fn=tts_fn,
            db_path=config.storage_db_path,
            barge_in_enabled=config.barge_in_enabled,
            on_processing_start=(lambda: overlay_sender("processing_on")) if overlay_sender else None,
            on_processing_end=(lambda: overlay_sender("processing_off")) if overlay_sender else None,
        )
```
4. In the `finally:` block (around line 178-181), terminate the overlay process alongside the existing cleanup:
```python
    finally:
        capture.stop()
        playback.close()
        await close_http_clients(http_clients)
        if overlay_process is not None:
            overlay_process.terminate()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/orchestrator/test_main_loop.py tests/test_config.py -v`
Expected: PASS (all existing tests plus the new ones)

Then run the full suite to confirm no regressions elsewhere:
Run: `python -m pytest -q`
Expected: PASS, no failures

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/orchestrator/main_loop.py financial_voice_agent/__main__.py financial_voice_agent/config.py config.yaml tests/orchestrator/test_main_loop.py tests/test_config.py
git commit -m "Wire processing-glow signals into the voice loop and launch the overlay process"
```

---

### Task 6: `show_chart` tool

**Files:**
- Modify: `financial_voice_agent/tools/registry.py`
- Modify: `financial_voice_agent/__main__.py`
- Test: `tests/tools/test_registry.py` (add new tests; existing tests must keep passing unmodified)

**Interfaces:**
- Consumes: `financial_voice_agent.tools.charting.render_chart` (Task 3), `financial_voice_agent.overlay.signal_client.make_overlay_sender`-produced callable shape `Callable[[str], None]` (Task 1) — passed in as `overlay_sender`, not constructed here.
- Produces: `make_tool_executor(config, http_clients, overlay_sender: Callable[[str], None] | None = None)` — one new optional keyword parameter, defaulting to `None`, so the existing `__main__.py` call (before Task 5's changes are considered — Task 5 already updated `__main__.py`'s other wiring, but the `make_tool_executor` call site needs one more edit here) and all existing tests keep working. A new `"show_chart"` entry in `TOOLS_SCHEMA`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/tools/test_registry.py` (append; extend the existing `test_tools_schema_names_match_all_six_tools` test's expected set — rename it since it'll no longer be six — and add new dispatch tests):

First, modify the existing test (it currently asserts exactly 6 tool names) to add just `show_chart` — Task 7 will extend this same test's set again when it adds `get_stock_fundamentals`, so this task's version must only assert what this task itself adds:
```python
def test_tools_schema_names_include_all_expected_tools():
    names = {tool["function"]["name"] for tool in TOOLS_SCHEMA}
    assert names == {
        "get_quote",
        "get_ohlc_history",
        "compute_indicator",
        "get_positions_holdings",
        "get_news",
        "capture_screen",
        "show_chart",
    }
```
(This replaces `test_tools_schema_names_match_all_six_tools` — same test, renamed and updated set. Task 7 modifies this exact test again, adding `"get_stock_fundamentals"` to the set — see Task 7's Step 1.)

Then add:
```python
@pytest.mark.asyncio
async def test_executor_dispatches_show_chart_in_mock_mode_and_returns_chart_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # so the "charts/" dir render_chart creates lands in tmp_path
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    result = await executor(
        ToolCall(id="1", name="show_chart", arguments={"symbol": "RELIANCE", "indicators": ["moving_average"]})
    )

    assert "error" not in result
    assert result["chart_path"].endswith(".png")
    import os
    assert os.path.exists(result["chart_path"])


@pytest.mark.asyncio
async def test_executor_sends_show_chart_message_to_overlay_sender(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sent_messages = []
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients(), overlay_sender=sent_messages.append)

    result = await executor(ToolCall(id="1", name="show_chart", arguments={"symbol": "RELIANCE"}))

    assert len(sent_messages) == 1
    assert sent_messages[0] == f"show_chart:{result['chart_path']}"


@pytest.mark.asyncio
async def test_executor_show_chart_works_without_overlay_sender():
    # overlay_sender defaults to None -- must not raise when the overlay is disabled.
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    result = await executor(ToolCall(id="1", name="show_chart", arguments={"symbol": "RELIANCE"}))

    assert "error" not in result


@pytest.mark.asyncio
async def test_executor_show_chart_returns_error_on_unknown_indicator():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    result = await executor(
        ToolCall(id="1", name="show_chart", arguments={"symbol": "RELIANCE", "indicators": ["macd"]})
    )

    assert "error" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tools/test_registry.py -v`
Expected: FAIL — schema test fails (missing `show_chart`/`get_stock_fundamentals`), dispatch tests fail with `ValueError: unknown_tool` style errors / `UnknownToolError`.

- [ ] **Step 3: Write the implementation**

In `financial_voice_agent/tools/registry.py`:

1. Add imports (near the top, with the other `financial_voice_agent.tools` imports):
```python
from financial_voice_agent.tools.charting import UnknownChartIndicatorError, render_chart
```
2. Add `Callable` to the existing `from typing import Awaitable, Callable` import if not already there (it already is, per the current file).
3. Add a new entry to `TOOLS_SCHEMA` (append to the list, after the `capture_screen` entry):
```python
    {
        "type": "function",
        "function": {
            "name": "show_chart",
            "description": (
                "Render and display a candlestick chart for an instrument, optionally with "
                "technical indicator overlays."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "indicators": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["moving_average", "bollinger", "rsi", "fibonacci"],
                        },
                        "description": "Optional list of indicators to overlay on the chart.",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
```
4. In `_dispatch`, add a `show_chart` branch (after the `capture_screen` branch, before `raise UnknownToolError`) — note `_dispatch`'s signature needs one new parameter, `overlay_sender`:
```python
async def _dispatch(
    call: ToolCall, config, http_clients, instrument_cache: dict, overlay_sender: Callable[[str], None] | None
) -> dict:
    # All existing branches (get_quote, get_ohlc_history, compute_indicator,
    # get_positions_holdings, get_news, capture_screen) stay exactly as they
    # already are -- only the signature above and the new branch below change.
    if call.name == "show_chart":
        history_fn = functools.partial(
            get_ohlc_history,
            http_client=http_clients.kite,
            mode=config.mode,
            fixtures_dir=_FIXTURES_DIR,
            instrument_cache=instrument_cache,
        )
        path = await render_chart(**call.arguments, history_fn=history_fn)
        if overlay_sender is not None:
            overlay_sender(f"show_chart:{path}")
        return {"chart_path": path}
    raise UnknownToolError(f"Unknown tool: {call.name}")
```
5. Update `make_tool_executor`'s signature and its call to `_dispatch`, and add `UnknownChartIndicatorError` to the executor's caught exceptions:
```python
def make_tool_executor(
    config, http_clients, overlay_sender: Callable[[str], None] | None = None
) -> Callable[[ToolCall], Awaitable[dict]]:
    instrument_cache: dict[tuple[str, str], int] = {}

    async def executor(call: ToolCall) -> dict:
        try:
            return await _dispatch(call, config, http_clients, instrument_cache, overlay_sender)
        except KiteSessionExpiredError:
            return {"error": "Kite session expired, please log in again"}
        except KiteRateLimitedError:
            return {"error": "Kite Connect rate limit hit, please try again shortly"}
        except KiteUnavailableError:
            return {"error": "Kite Connect is temporarily unavailable"}
        except InsufficientDataError as exc:
            return {"error": str(exc)}
        except InstrumentNotFoundError as exc:
            return {"error": str(exc)}
        except WindowNotFoundError:
            return {"error": "Could not find the Kite window on screen"}
        except UnknownChartIndicatorError as exc:
            return {"error": str(exc)}
        except TypeError as exc:
            return {"error": f"Invalid arguments for tool '{call.name}': {exc}"}
        except UnknownToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Tool '{call.name}' failed: {exc}"}

    return executor
```

Finally, update `financial_voice_agent/__main__.py`'s existing `tool_executor = make_tool_executor(config, http_clients)` line (Task 5 placed the `overlay_sender`-assigning block immediately above this line, so `overlay_sender` is already in scope here) to pass the new parameter:
```python
    tool_executor = make_tool_executor(config, http_clients, overlay_sender=overlay_sender)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tools/test_registry.py -v`
Expected: PASS (schema test + 4 new dispatch tests, plus all pre-existing tests)

Then run the full suite:
Run: `python -m pytest -q`
Expected: PASS, no failures

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/tools/registry.py financial_voice_agent/__main__.py
git commit -m "Add show_chart tool, wired to the overlay signal sender"
```

---

### Task 7: Stock fundamentals tool

**Files:**
- Create: `financial_voice_agent/tools/fundamentals.py`
- Create: `fixtures/stock_fundamentals.json`
- Modify: `financial_voice_agent/http_clients.py`
- Modify: `financial_voice_agent/config.py`
- Modify: `financial_voice_agent/tools/registry.py`
- Modify: `.env.example`
- Test: `tests/tools/test_fundamentals.py`
- Test: `tests/test_http_clients.py` (add new test; existing tests must keep passing unmodified)
- Test: `tests/tools/test_registry.py` (add new dispatch tests)

**Interfaces:**
- Consumes: nothing from Tasks 1-6.
- Produces: `get_stock_fundamentals(name: str, *, http_client, mode: str = "live", fixtures_dir: str = "fixtures") -> dict`. `HTTPClients.indian_stock: httpx.AsyncClient` field. `Config.indian_stock_api_key: str | None` field.

- [ ] **Step 1: Write the failing tests**

Create `tests/tools/test_fundamentals.py`. This uses `httpx.MockTransport` (already used elsewhere in this codebase's test suite for mocking `httpx` calls — see `tests/tools/test_news.py` or `tests/test_http_clients.py` for the exact pattern already in use, and follow it) to simulate the real Indian Stock API's verified response shape:
```python
import httpx
import pytest

from financial_voice_agent.tools.fundamentals import get_stock_fundamentals


def _client_with_response(json_body: dict, status_code: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=json_body)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://stock.indianapi.in")


@pytest.mark.asyncio
async def test_get_stock_fundamentals_mock_mode_returns_fixture():
    result = await get_stock_fundamentals("Reliance", http_client=None, mode="mock", fixtures_dir="fixtures")

    assert result["company_name"] == "Reliance Industries"
    assert "error" not in result


@pytest.mark.asyncio
async def test_get_stock_fundamentals_live_mode_summarizes_real_response_shape():
    # Shape verified against a real call to https://stock.indianapi.in/stock?name=Reliance
    client = _client_with_response({
        "companyName": "Reliance Industries",
        "industry": "Oil & Gas Operations",
        "stockDetailsReusableData": {
            "price": "1276.00",
            "percentChange": "0.64",
            "marketCap": "1726751.94",
            "yhigh": "1611.20",
            "ylow": "1250.55",
            "pPerEBasicExcludingExtraordinaryItemsTTM": "23.43",
        },
    })

    result = await get_stock_fundamentals("Reliance", http_client=client, mode="live")

    assert result["company_name"] == "Reliance Industries"
    assert result["industry"] == "Oil & Gas Operations"
    assert result["price"] == "1276.00"
    assert result["percent_change"] == "0.64"
    assert result["market_cap"] == "1726751.94"
    assert result["year_high"] == "1611.20"
    assert result["year_low"] == "1250.55"
    assert result["pe_ratio"] == "23.43"


@pytest.mark.asyncio
async def test_get_stock_fundamentals_returns_error_for_unknown_company():
    # Verified: the real API returns HTTP 200 with {"error": "Stock not found"}
    # for an unrecognized name, not a 4xx.
    client = _client_with_response({"error": "Stock not found"})

    result = await get_stock_fundamentals("NotARealCompanyXYZ", http_client=client, mode="live")

    assert result == {"error": "Stock not found"}


@pytest.mark.asyncio
async def test_get_stock_fundamentals_returns_error_on_invalid_api_key():
    # Verified: the real API returns HTTP 401 with a plain-text body for a bad key.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Invalid API key")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://stock.indianapi.in")

    result = await get_stock_fundamentals("Reliance", http_client=client, mode="live")

    assert "error" in result


@pytest.mark.asyncio
async def test_get_stock_fundamentals_returns_error_on_network_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://stock.indianapi.in")

    result = await get_stock_fundamentals("Reliance", http_client=client, mode="live")

    assert "error" in result
```

Create `fixtures/stock_fundamentals.json` (in the app's own summarized shape, matching this project's existing convention — see `fixtures/quote.json`, which is stored pre-summarized rather than mirroring the live vendor's raw wire format):
```json
{
  "company_name": "Reliance Industries",
  "industry": "Oil & Gas Operations",
  "price": "1276.00",
  "percent_change": "0.64",
  "market_cap": "1726751.94",
  "year_high": "1611.20",
  "year_low": "1250.55",
  "pe_ratio": "23.43"
}
```

Add to `tests/test_http_clients.py` (append, extending the existing `_make_config` helper call sites is not needed since it already accepts no new required args — just add `indian_stock_api_key="indian-stock-secret"` as a new keyword default inside the existing `_make_config` helper function's `Config(...)` call, then add this test):
```python
@pytest.mark.asyncio
async def test_indian_stock_client_carries_api_key_header():
    config = _make_config()

    clients = await create_http_clients(config)
    try:
        assert clients.indian_stock.headers["X-Api-Key"] == "indian-stock-secret"
    finally:
        await close_http_clients(clients)
```
(Also add `assert isinstance(clients.indian_stock, httpx.AsyncClient)` to the existing `test_create_http_clients_returns_one_client_per_vendor` test, and include `id(clients.indian_stock)` in that test's uniqueness-count set, updating `== 4` to `== 5`.)

In `tests/tools/test_registry.py`:

1. Extend `test_tools_schema_names_include_all_expected_tools` (added in Task 6) to add `"get_stock_fundamentals"` to its expected set:
```python
def test_tools_schema_names_include_all_expected_tools():
    names = {tool["function"]["name"] for tool in TOOLS_SCHEMA}
    assert names == {
        "get_quote",
        "get_ohlc_history",
        "compute_indicator",
        "get_positions_holdings",
        "get_news",
        "capture_screen",
        "show_chart",
        "get_stock_fundamentals",
    }
```
2. Add `indian_stock = None  # unused in mock mode` as a class attribute on `_FakeHttpClients` (matching its existing `kite`/`tavily` attributes).
3. Append a new dispatch test:
```python
@pytest.mark.asyncio
async def test_executor_dispatches_get_stock_fundamentals_in_mock_mode():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    result = await executor(ToolCall(id="1", name="get_stock_fundamentals", arguments={"name": "Reliance"}))

    assert result["company_name"] == "Reliance Industries"
    assert "error" not in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tools/test_fundamentals.py tests/test_http_clients.py tests/tools/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.tools.fundamentals'`, then (after that's created) `AttributeError: 'HTTPClients' object has no attribute 'indian_stock'`, then `UnknownToolError` for the registry test.

- [ ] **Step 3: Write the implementation**

Create `financial_voice_agent/tools/fundamentals.py`:
```python
from __future__ import annotations

import httpx

from financial_voice_agent import mock


async def get_stock_fundamentals(
    name: str, *, http_client, mode: str = "live", fixtures_dir: str = "fixtures"
) -> dict:
    if mode == "mock":
        return mock.load_fixture("stock_fundamentals", fixtures_dir=fixtures_dir)

    try:
        response = await http_client.get("/stock", params={"name": name})
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return {"error": f"Could not look up '{name}': {exc.response.status_code}"}
    except Exception:  # noqa: BLE001 -- network/timeout errors must degrade gracefully, never crash the turn
        return {"error": f"Could not reach the stock data provider for '{name}'"}

    data = response.json()
    if "error" in data:
        # The real API returns HTTP 200 with {"error": "Stock not found"} for
        # an unrecognized company name -- not a 4xx (verified against a live call).
        return {"error": data["error"]}

    return _summarize_fundamentals(data)


def _summarize_fundamentals(data: dict) -> dict:
    details = data.get("stockDetailsReusableData", {})
    return {
        "company_name": data.get("companyName"),
        "industry": data.get("industry"),
        "price": details.get("price"),
        "percent_change": details.get("percentChange"),
        "market_cap": details.get("marketCap"),
        "year_high": details.get("yhigh"),
        "year_low": details.get("ylow"),
        "pe_ratio": details.get("pPerEBasicExcludingExtraordinaryItemsTTM"),
    }
```

In `financial_voice_agent/http_clients.py`:
1. Add `INDIAN_STOCK_BASE_URL = "https://stock.indianapi.in"` near the other `*_BASE_URL` constants.
2. Add `indian_stock: httpx.AsyncClient` to the `HTTPClients` dataclass.
3. In `create_http_clients`, before the `return HTTPClients(...)` line, add:
```python
    indian_stock = httpx.AsyncClient(
        base_url=INDIAN_STOCK_BASE_URL,
        headers={"X-Api-Key": config.indian_stock_api_key or ""},
        timeout=REST_TIMEOUT,
    )
```
   and add `indian_stock=indian_stock` to the `HTTPClients(...)` constructor call.
4. In `close_http_clients`, add `await clients.indian_stock.aclose()`.

In `financial_voice_agent/config.py`:
1. Add `indian_stock_api_key: str | None` to the `Config` dataclass (with the other `*_api_key` fields).
2. In `_load_env`, add `"INDIAN_STOCK_API_KEY"` to the tuple of keys read from `os.environ`.
3. In `load_config`, add `indian_stock_api_key=env.get("INDIAN_STOCK_API_KEY"),` to the `Config(...)` constructor call (with the other `*_api_key` fields).

In `.env.example`, add a new line: `INDIAN_STOCK_API_KEY=`

In `financial_voice_agent/tools/registry.py`:
1. Add import: `from financial_voice_agent.tools.fundamentals import get_stock_fundamentals`
2. Add a new `TOOLS_SCHEMA` entry (append, after the `show_chart` entry added in Task 6):
```python
    {
        "type": "function",
        "function": {
            "name": "get_stock_fundamentals",
            "description": (
                "Get fundamental data (P/E ratio, market cap, 52-week range) for a company "
                "by name -- also resolves fuzzy/partial company names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Company name, e.g. 'Reliance' or 'HDFC Bank'"}
                },
                "required": ["name"],
            },
        },
    },
```
3. In `_dispatch`, add a branch (after the `show_chart` branch, before `raise UnknownToolError`):
```python
    if call.name == "get_stock_fundamentals":
        return await get_stock_fundamentals(
            **call.arguments, http_client=http_clients.indian_stock, mode=config.mode,
            fixtures_dir=_FIXTURES_DIR,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tools/test_fundamentals.py tests/test_http_clients.py tests/tools/test_registry.py -v`
Expected: PASS (all new tests plus every pre-existing test in these files)

Then run the full suite:
Run: `python -m pytest -q`
Expected: PASS, no failures

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/tools/fundamentals.py fixtures/stock_fundamentals.json financial_voice_agent/http_clients.py financial_voice_agent/config.py financial_voice_agent/tools/registry.py .env.example tests/tools/test_fundamentals.py tests/test_http_clients.py tests/tools/test_registry.py
git commit -m "Add get_stock_fundamentals tool backed by the Indian Stock API"
```

---

### Task 8: Investment-analyst system prompt, requirements.txt, and .gitignore

**Files:**
- Modify: `financial_voice_agent/orchestrator/system_prompt.py`
- Modify: `requirements.txt`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing (the prompt just needs to know the tool names `show_chart` and `get_stock_fundamentals` exist, which it does by this point in the plan).
- Produces: nothing new consumed by other tasks — this is the last code task before manual verification.

This task has no dedicated unit test — `SYSTEM_PROMPT` is an unstructured string constant with no existing test coverage in this codebase (there is no `test_system_prompt.py`), consistent with how prompt wording isn't unit tested elsewhere in this project.

- [ ] **Step 1: Add `mplfinance` to requirements.txt and gitignore generated charts**

Add a new line to `requirements.txt`: `mplfinance`

Add a new line to `.gitignore`, alongside the existing `screenshots/` entry: `charts/` (this is where `render_chart`, Task 3, saves its output by default — generated images shouldn't be committed, matching the existing convention for `capture_screen`'s `screenshots/`).

- [ ] **Step 2: Update the system prompt**

Replace `financial_voice_agent/orchestrator/system_prompt.py`'s `SYSTEM_PROMPT` value with:
```python
SYSTEM_PROMPT = (
    "You are a read-only voice assistant for a personal trading desk running Zerodha Kite. "
    "You observe market data, positions, and news — you never place, modify, or cancel orders, "
    "and you have no tool capable of doing so. Speak in short, natural sentences meant to be "
    "heard, not read. When you need to call a tool that will take a moment, say a brief natural "
    "acknowledgment first — e.g. \"let me check that\" or \"one sec, pulling the chart\" — and "
    "vary the phrasing so it doesn't sound scripted. If you cannot see the relevant instrument "
    "on screen and the query depends on what's currently visible, call capture_screen rather "
    "than guessing. If a query is ambiguous about which instrument or timeframe is meant, ask a "
    "short clarifying question instead of assuming. Never speculate about figures you have not "
    "retrieved from a tool in this turn. If a tool call returns an error, say so plainly and in "
    "plain language — e.g. \"I couldn't pull that up, looks like a permissions issue on the Kite "
    "side\" — rather than staying silent, giving up without explanation, or pretending it worked. "
    "Briefly suggest what the user could check or do next if it's obvious from the error.\n\n"
    "For analysis-style questions about a stock, act like an investment analyst: pull fundamentals "
    "(get_stock_fundamentals), technicals (compute_indicator, and show_chart when a visual would "
    "help), and recent news (get_news) as relevant, then synthesize them into a clear picture of "
    "what the numbers mean — e.g. \"the P/E is above the sector average, RSI suggests overbought "
    "conditions, and recent news is mixed.\" Explain and interpret; do not issue buy/sell/hold "
    "recommendations, price targets, or tell the user what they should do — describe what the data "
    "shows and let them draw their own conclusion."
)
```

- [ ] **Step 3: Install the new dependency and run the full suite**

Run: `pip install -r requirements.txt`
Run: `python -m pytest -q`
Expected: PASS, no failures (this task doesn't add tests, but must not break anything)

- [ ] **Step 4: Commit**

```bash
git add financial_voice_agent/orchestrator/system_prompt.py requirements.txt .gitignore
git commit -m "Add investment-analyst guidance to the system prompt"
```

---

### Task 9: Manual verification

**Files:**
- None (verification only, no code changes expected — see the exception below).

**Interfaces:**
- Consumes: everything from Tasks 1-8, running end-to-end.

- [ ] **Step 1: Run the full test suite one more time**

Run: `python -m pytest -q`
Expected: PASS, all tests (existing suite plus every new test from Tasks 1-8)

- [ ] **Step 2: Manually verify the processing glow**

Run: `python -m financial_voice_agent`

Confirm: a screen_overlay process starts (visible in Task Manager as a `python.exe` process, or infer from no crash/traceback in the main terminal). Speak a query. Confirm a thin blue glow appears around the screen edges while the agent is processing (STT → LLM → TTS generation) and disappears once the response starts playing. Confirm you can still click into other windows underneath the glow while it's visible (click-through).

If the glow doesn't appear, doesn't disappear, or blocks clicks: fix the relevant code in `financial_voice_agent/overlay/screen_overlay.py` or the wiring in `financial_voice_agent/orchestrator/main_loop.py` / `financial_voice_agent/__main__.py`, then re-run this step.

- [ ] **Step 3: Manually verify show_chart**

While the agent is running, ask something like "show me a chart for RELIANCE with a moving average." Confirm: the agent calls `show_chart`, a panel slides in from the right edge of the screen showing a real candlestick chart with the moving average line overlaid, and clicking the panel's close button (×) slides it back out.

If the panel doesn't appear, doesn't render correctly, or the close button doesn't work: fix `financial_voice_agent/overlay/screen_overlay.py` or `financial_voice_agent/tools/charting.py`, then re-run this step.

- [ ] **Step 4: Manually verify get_stock_fundamentals**

Ask something like "what's Reliance's P/E ratio and market cap?" Confirm: the agent calls `get_stock_fundamentals`, gets real data back, and speaks a sensible answer using it.

If this fails: check `INDIAN_STOCK_API_KEY` is set in `.env`, and check the real API's response shape hasn't changed since verification (re-run a manual `httpx.get` call against `https://stock.indianapi.in/stock?name=Reliance` with the `X-Api-Key` header to compare against `financial_voice_agent/tools/fundamentals.py`'s `_summarize_fundamentals` field paths). Fix `fundamentals.py` if the shape has drifted, then re-run this step.

- [ ] **Step 5: Manually verify investment-analyst behavior**

Ask an analysis-style question, e.g. "how does Reliance look right now?" Confirm the agent pulls fundamentals + technicals/news as relevant and gives a synthesized, descriptive answer — and confirm it does NOT say something like "you should buy" or give a price target. If it does, that's a prompt-tuning issue, not a code bug — adjust the guardrail wording in `financial_voice_agent/orchestrator/system_prompt.py` (from Task 8) if needed, and re-run this step.

- [ ] **Step 6: Verify processing_overlay_enabled: false actually disables the overlay**

Temporarily set `overlay: enabled: false` in `config.yaml`, run `python -m financial_voice_agent` again, and confirm no overlay process starts and no glow/chart ever appears, with no errors. Then set `enabled: true` back (or remove the line, since the default is `true`) before finishing.

- [ ] **Step 7: Commit any manual-verification fixes**

If Steps 2-6 required code changes, commit them now with a message describing what was fixed (e.g. "Fix chart panel close button not registering clicks"). If no changes were needed, there is nothing to commit for this task.
