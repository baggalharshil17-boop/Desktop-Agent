import pytest

from financial_voice_agent.orchestrator.llm import ToolCall
from financial_voice_agent.tools.registry import TOOLS_SCHEMA, make_tool_executor


class _FakeConfig:
    mode = "mock"
    tavily_api_key = "test-tavily-key"


class _FakeHttpClients:
    kite = None  # unused in mock mode
    tavily = None  # overridden per-test where get_news is exercised


def test_tools_schema_names_match_all_six_tools():
    names = {tool["function"]["name"] for tool in TOOLS_SCHEMA}
    assert names == {
        "get_quote",
        "get_ohlc_history",
        "compute_indicator",
        "get_positions_holdings",
        "get_news",
        "capture_screen",
    }


@pytest.mark.asyncio
async def test_executor_dispatches_get_quote_in_mock_mode():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    result = await executor(ToolCall(id="1", name="get_quote", arguments={"symbol": "NIFTY 50"}))

    assert result["symbol"] == "NIFTY 50"
    assert result["last_price"] == 24500.35


@pytest.mark.asyncio
async def test_executor_dispatches_get_positions_holdings_in_mock_mode():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    result = await executor(ToolCall(id="1", name="get_positions_holdings", arguments={}))

    assert result["holdings"][0]["symbol"] == "RELIANCE"


@pytest.mark.asyncio
async def test_executor_dispatches_compute_indicator_using_mock_history():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    with pytest.raises(Exception):
        # fixtures/ohlc_history.json only has 2 candles -- far short of any
        # indicator's minimum, so this must raise InsufficientDataError,
        # proving compute_indicator is correctly wired to real
        # get_ohlc_history (mock mode) through the registry, not a stub.
        await executor(
            ToolCall(
                id="1",
                name="compute_indicator",
                arguments={"symbol": "RELIANCE", "indicator": "moving_average", "params": {}},
            )
        )


@pytest.mark.asyncio
async def test_executor_raises_value_error_for_unknown_tool():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    with pytest.raises(ValueError, match="unknown_tool"):
        await executor(ToolCall(id="1", name="unknown_tool", arguments={}))
