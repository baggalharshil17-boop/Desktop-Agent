from __future__ import annotations

import functools
import pathlib
from typing import Awaitable, Callable

from financial_voice_agent.orchestrator.llm import ToolCall
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
    find_kite_window,
)

_FIXTURES_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "fixtures")

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
            "description": "Capture the current Kite window as an image, for questions about what's visible on screen.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


async def _dispatch(call: ToolCall, config, http_clients, instrument_cache: dict) -> dict:
    if call.name == "get_quote":
        return await get_quote(
            **call.arguments, http_client=http_clients.kite, mode=config.mode, fixtures_dir=_FIXTURES_DIR
        )
    if call.name == "get_ohlc_history":
        return await get_ohlc_history(
            **call.arguments,
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
        return await compute_indicator(**call.arguments, history_fn=history_fn)
    if call.name == "get_positions_holdings":
        return await get_positions_holdings(
            http_client=http_clients.kite, mode=config.mode, fixtures_dir=_FIXTURES_DIR
        )
    if call.name == "get_news":
        return await get_news(
            **call.arguments, http_client=http_clients.tavily, api_key=config.tavily_api_key,
            mode=config.mode, fixtures_dir=_FIXTURES_DIR,
        )
    if call.name == "capture_screen":
        return await capture_screen(window_finder=find_kite_window, screenshot_fn=capture_region)
    raise ValueError(f"Unknown tool: {call.name}")


def make_tool_executor(config, http_clients) -> Callable[[ToolCall], Awaitable[dict]]:
    # Kite's instrument dump is meant to be fetched at most once a day (per
    # Kite's own docs), so this cache is created once here and shared across
    # every get_ohlc_history/compute_indicator call for this executor's
    # lifetime, rather than re-fetched per tool call.
    instrument_cache: dict[tuple[str, str], int] = {}

    async def executor(call: ToolCall) -> dict:
        try:
            return await _dispatch(call, config, http_clients, instrument_cache)
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
        except TypeError as exc:
            return {"error": f"Invalid arguments for tool '{call.name}': {exc}"}

    return executor
