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

    result = await executor(
        ToolCall(
            id="1",
            name="compute_indicator",
            arguments={"symbol": "RELIANCE", "indicator": "moving_average", "params": {}},
        )
    )

    assert "moving_average" in result
    assert "error" not in result


@pytest.mark.asyncio
async def test_executor_raises_value_error_for_unknown_tool():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    with pytest.raises(ValueError, match="unknown_tool"):
        await executor(ToolCall(id="1", name="unknown_tool", arguments={}))


@pytest.mark.asyncio
async def test_executor_dispatches_compute_indicator_without_params_key():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    result = await executor(
        ToolCall(id="1", name="compute_indicator", arguments={"symbol": "RELIANCE", "indicator": "moving_average"})
    )

    assert "moving_average" in result
    assert "error" not in result


@pytest.mark.asyncio
async def test_executor_translates_insufficient_data_error_to_error_dict():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    # The 25-candle fixture is not enough for a caller-requested window of 30.
    result = await executor(
        ToolCall(
            id="1",
            name="compute_indicator",
            arguments={"symbol": "RELIANCE", "indicator": "moving_average", "params": {"window": 30}},
        )
    )

    assert "error" in result


@pytest.mark.asyncio
async def test_executor_translates_bad_arguments_to_error_dict():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    # An extra/unexpected keyword argument from the LLM must not crash the tool loop.
    result = await executor(
        ToolCall(id="1", name="get_quote", arguments={"symbol": "NIFTY 50", "unexpected_arg": "oops"})
    )

    assert "error" in result


@pytest.mark.asyncio
async def test_executor_translates_unexpected_exceptions_to_error_dict_instead_of_crashing():
    # A raw exception this executor doesn't specifically recognize (e.g. an
    # httpx.HTTPStatusError from a Kite error code this code doesn't handle
    # by name, like a 403 PermissionException for missing historical-data
    # access) must still become a tool result the LLM can see and explain
    # out loud -- not an uncaught exception that crashes the whole turn and
    # forces a generic, misleading "I didn't catch that" fallback.
    class _FakeConfigLiveMode:
        mode = "live"
        tavily_api_key = "test-tavily-key"

    class _FailingKiteHttpClient:
        async def get(self, path, params=None):
            raise RuntimeError("Client error '403 Forbidden' for url '...': Insufficient permission")

    class _HttpClientsWithFailingKite:
        kite = _FailingKiteHttpClient()
        tavily = None

    executor = make_tool_executor(_FakeConfigLiveMode(), _HttpClientsWithFailingKite())

    result = await executor(ToolCall(id="1", name="get_quote", arguments={"symbol": "NIFTY 50"}))

    assert "error" in result
    assert "get_quote" in result["error"]


@pytest.mark.asyncio
async def test_executor_dispatches_get_news_in_mock_mode_without_network():
    class _HttpClientsWithFailingTavily:
        kite = None
        class tavily:
            @staticmethod
            async def post(path, json=None):
                raise AssertionError("real Tavily call must not happen when config.mode == 'mock'")

    executor = make_tool_executor(_FakeConfig(), _HttpClientsWithFailingTavily())

    result = await executor(ToolCall(id="1", name="get_news", arguments={"query": "Nifty"}))

    assert "error" not in result
    assert len(result["headlines"]) == 2
