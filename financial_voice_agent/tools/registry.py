from __future__ import annotations

import functools
import os
import pathlib
from typing import Awaitable, Callable

from financial_voice_agent.orchestrator.llm import ToolCall
from financial_voice_agent.tools.charting import (
    ChartPathError,
    UnknownChartIndicatorError,
    render_chart,
)
from financial_voice_agent.tools.fundamentals import get_stock_fundamentals
from financial_voice_agent.tools.history import get_ohlc_history
from financial_voice_agent.tools.indicators import InsufficientDataError, compute_indicator
from financial_voice_agent.tools.instruments import InstrumentNotFoundError
from financial_voice_agent.tools.kite_client import (
    KiteRateLimitedError,
    KiteSessionExpiredError,
    KiteUnavailableError,
)
from financial_voice_agent.tools.news import get_news
from financial_voice_agent.tools.quotes import get_positions_holdings, get_quote
from financial_voice_agent.tools.screen import (
    WindowNotFoundError,
    capture_region,
    capture_screen,
    find_active_window,
)

_FIXTURES_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "fixtures")


class UnknownToolError(ValueError):
    """A tool_call name outside TOOLS_SCHEMA -- a dev-time contract bug (the
    LLM should never be able to name a tool that isn't in TOOLS_SCHEMA), not
    a runtime failure -- so this must keep crashing loudly rather than being
    swallowed into a spoken {"error": ...} by executor()'s catch-all below."""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Get the latest price and day OHLC for a market instrument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "e.g. 'NIFTY 50' or 'RELIANCE'"}
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ohlc_history",
            "description": "Get historical OHLC candle data for an instrument over a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "interval": {"type": "string", "description": "e.g. '15minute', 'day'"},
                    "from_date": {"type": "string", "description": "ISO date, e.g. '2026-07-01'"},
                    "to_date": {"type": "string", "description": "ISO date, e.g. '2026-07-25'"},
                },
                "required": ["symbol", "interval", "from_date", "to_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_indicator",
            "description": "Compute a technical indicator (bollinger, moving_average, rsi, fibonacci) for an instrument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "indicator": {
                        "type": "string",
                        "enum": ["bollinger", "moving_average", "rsi", "fibonacci"],
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional: window, interval, from_date, to_date",
                    },
                },
                "required": ["symbol", "indicator"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_positions_holdings",
            "description": "Get the user's current holdings and positions.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Search for recent news headlines on a topic. Never include account-specific data in the query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_screen",
            "description": "Capture the currently active window as an image, for questions about what's visible on screen.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
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
    {
        "type": "function",
        "function": {
            "name": "get_stock_fundamentals",
            "description": (
                "Get fundamental data (P/E ratio, market cap, 52-week range) for a company "
                "by name -- also resolves fuzzy/partial company names. Also returns the "
                "resolved NSE ticker symbol."
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
]


# Every parameter each tool actually declares to the model, derived from
# TOOLS_SCHEMA itself so the two can't drift. TOOLS_SCHEMA is only advisory
# metadata *sent to* the model -- nothing makes the model's reply conform to
# it, so the argument dict coming back is untrusted input and must be
# filtered before it reaches a **splat. Without this, any keyword-only
# parameter a tool function happens to expose (e.g. render_chart's
# output_dir) is silently model-settable, which turns a chart request into
# an arbitrary-location file write.
_SCHEMA_PROPERTIES = {
    tool["function"]["name"]: set(tool["function"]["parameters"].get("properties", {}))
    for tool in TOOLS_SCHEMA
}


def filter_tool_arguments(tool_name: str, arguments: dict) -> dict:
    """Drops any argument the tool's schema doesn't declare."""
    allowed = _SCHEMA_PROPERTIES.get(tool_name, set())
    return {key: value for key, value in arguments.items() if key in allowed}


async def _dispatch(
    call: ToolCall, config, http_clients, instrument_cache: dict, overlay_sender: Callable[[str], None] | None
) -> dict:
    args = filter_tool_arguments(call.name, call.arguments)
    if call.name == "get_quote":
        return await get_quote(
            **args, http_client=http_clients.kite, mode=config.mode, fixtures_dir=_FIXTURES_DIR
        )
    if call.name == "get_ohlc_history":
        return await get_ohlc_history(
            **args,
            http_client=http_clients.kite,
            mode=config.mode,
            fixtures_dir=_FIXTURES_DIR,
            instrument_cache=instrument_cache,
        )
    if call.name == "compute_indicator":
        history_fn = functools.partial(
            get_ohlc_history,
            http_client=http_clients.kite,
            mode=config.mode,
            fixtures_dir=_FIXTURES_DIR,
            instrument_cache=instrument_cache,
        )
        return await compute_indicator(**args, history_fn=history_fn)
    if call.name == "get_positions_holdings":
        return await get_positions_holdings(
            http_client=http_clients.kite, mode=config.mode, fixtures_dir=_FIXTURES_DIR
        )
    if call.name == "get_news":
        return await get_news(
            **args, http_client=http_clients.tavily, api_key=config.tavily_api_key,
            mode=config.mode, fixtures_dir=_FIXTURES_DIR,
        )
    if call.name == "capture_screen":
        return await capture_screen(window_finder=find_active_window, screenshot_fn=capture_region)
    if call.name == "show_chart":
        history_fn = functools.partial(
            get_ohlc_history,
            http_client=http_clients.kite,
            mode=config.mode,
            fixtures_dir=_FIXTURES_DIR,
            instrument_cache=instrument_cache,
        )
        path = os.path.abspath(await render_chart(**args, history_fn=history_fn))
        if overlay_sender is not None:
            overlay_sender(f"show_chart:{path}")
        return {"chart_path": path}
    if call.name == "get_stock_fundamentals":
        return await get_stock_fundamentals(
            **args, http_client=http_clients.indian_stock, mode=config.mode,
            fixtures_dir=_FIXTURES_DIR,
        )
    raise UnknownToolError(f"Unknown tool: {call.name}")


def make_tool_executor(
    config, http_clients, overlay_sender: Callable[[str], None] | None = None
) -> Callable[[ToolCall], Awaitable[dict]]:
    # Kite's instrument dump is meant to be fetched at most once a day (per
    # Kite's own docs), so this cache is created once here and shared across
    # every get_ohlc_history/compute_indicator call for this executor's
    # lifetime, rather than re-fetched per tool call.
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
            return {"error": "Could not find an active window to capture"}
        except UnknownChartIndicatorError as exc:
            return {"error": str(exc)}
        except ChartPathError as exc:
            return {"error": str(exc)}
        except TypeError as exc:
            return {"error": f"Invalid arguments for tool '{call.name}': {exc}"}
        except UnknownToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Any tool failure not covered by a specific case above (e.g. a
            # raw httpx.HTTPStatusError from a Kite error this code doesn't
            # specifically recognize, like a 403 PermissionException for an
            # account without historical-data access) must still become a
            # tool result the LLM can see and explain out loud -- not an
            # uncaught exception. Letting one escape crashes the whole turn,
            # and the caller's fallback ("I didn't catch that, one moment")
            # is actively misleading for a tool/API failure that has nothing
            # to do with hearing the user.
            return {"error": f"Tool '{call.name}' failed: {exc}"}

    return executor
