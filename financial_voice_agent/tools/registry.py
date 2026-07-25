from __future__ import annotations

import functools
from typing import Awaitable, Callable

from financial_voice_agent.orchestrator.llm import ToolCall
from financial_voice_agent.tools.history import get_ohlc_history
from financial_voice_agent.tools.indicators import compute_indicator
from financial_voice_agent.tools.news import get_news
from financial_voice_agent.tools.quotes import get_positions_holdings, get_quote
from financial_voice_agent.tools.screen import capture_region, capture_screen, find_kite_window

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
                        "description": "Optional: window, interval, from, to",
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


def make_tool_executor(config, http_clients) -> Callable[[ToolCall], Awaitable[dict]]:
    async def executor(call: ToolCall) -> dict:
        if call.name == "get_quote":
            return await get_quote(**call.arguments, http_client=http_clients.kite, mode=config.mode)
        if call.name == "get_ohlc_history":
            return await get_ohlc_history(
                **call.arguments, http_client=http_clients.kite, mode=config.mode
            )
        if call.name == "compute_indicator":
            history_fn = functools.partial(
                get_ohlc_history, http_client=http_clients.kite, mode=config.mode
            )
            return await compute_indicator(**call.arguments, history_fn=history_fn)
        if call.name == "get_positions_holdings":
            return await get_positions_holdings(http_client=http_clients.kite, mode=config.mode)
        if call.name == "get_news":
            return await get_news(
                **call.arguments, http_client=http_clients.tavily, api_key=config.tavily_api_key
            )
        if call.name == "capture_screen":
            return await capture_screen(window_finder=find_kite_window, screenshot_fn=capture_region)
        raise ValueError(f"Unknown tool: {call.name}")

    return executor
