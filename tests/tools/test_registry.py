import os

import pytest

from financial_voice_agent.orchestrator.llm import ToolCall
from financial_voice_agent.tools.registry import (
    TOOLS_SCHEMA,
    filter_tool_arguments,
    make_tool_executor,
)


class _FakeConfig:
    mode = "mock"
    tavily_api_key = "test-tavily-key"


class _FakeHttpClients:
    kite = None  # unused in mock mode
    tavily = None  # overridden per-test where get_news is exercised
    indian_stock = None  # unused in mock mode


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
async def test_executor_drops_undeclared_arguments_instead_of_crashing():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    # An extra/unexpected keyword argument from the LLM must not crash the tool
    # loop. It is now dropped by the schema filter before dispatch (it used to
    # reach the tool function and raise TypeError), so the call still succeeds.
    result = await executor(
        ToolCall(id="1", name="get_quote", arguments={"symbol": "NIFTY 50", "unexpected_arg": "oops"})
    )

    assert "error" not in result
    assert result["symbol"] == "NIFTY 50"


@pytest.mark.asyncio
async def test_executor_still_reports_missing_required_arguments_as_an_error():
    # The schema filter drops *undeclared* keys, but a declared-yet-missing
    # required one must still surface as a spoken error rather than crash the
    # turn -- this keeps the executor's `except TypeError` branch covered.
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    result = await executor(ToolCall(id="1", name="get_quote", arguments={}))

    assert "error" in result
    assert "get_quote" in result["error"]


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
async def test_executor_show_chart_works_without_overlay_sender(tmp_path, monkeypatch):
    # overlay_sender defaults to None -- must not raise when the overlay is disabled.
    monkeypatch.chdir(tmp_path)  # so the "charts/" dir render_chart creates lands in tmp_path
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


@pytest.mark.asyncio
async def test_executor_dispatches_get_stock_fundamentals_in_mock_mode():
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    result = await executor(ToolCall(id="1", name="get_stock_fundamentals", arguments={"name": "Reliance"}))

    assert result["company_name"] == "Reliance Industries"
    assert "error" not in result


def test_filter_tool_arguments_keeps_only_schema_declared_keys():
    assert filter_tool_arguments("show_chart", {"symbol": "RELIANCE"}) == {"symbol": "RELIANCE"}
    # output_dir/interval are real render_chart parameters but are NOT declared
    # in show_chart's schema, so the model must not be able to set them.
    assert filter_tool_arguments(
        "show_chart", {"symbol": "RELIANCE", "output_dir": "C:/evil", "interval": "day"}
    ) == {"symbol": "RELIANCE"}
    assert filter_tool_arguments("get_positions_holdings", {"anything": 1}) == {}
    assert filter_tool_arguments("unknown_tool", {"symbol": "X"}) == {}


@pytest.mark.asyncio
async def test_executor_ignores_model_supplied_output_dir(tmp_path, monkeypatch):
    # Regression test for the confirmed arbitrary-file-write: a model-supplied
    # output_dir used to be splatted straight into render_chart, letting
    # prompt-injected content choose where on disk the PNG landed.
    monkeypatch.chdir(tmp_path)
    escape_target = tmp_path / "escaped"
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    result = await executor(
        ToolCall(
            id="1",
            name="show_chart",
            arguments={"symbol": "RELIANCE", "output_dir": str(escape_target)},
        )
    )

    assert "error" not in result
    assert not escape_target.exists()
    assert (tmp_path / "charts").is_dir()
    assert os.path.realpath(result["chart_path"]).startswith(
        os.path.realpath(str(tmp_path / "charts"))
    )


@pytest.mark.asyncio
async def test_executor_show_chart_symbol_cannot_escape_the_charts_directory(tmp_path, monkeypatch):
    # Regression test for the confirmed Windows path traversal: "\" was not
    # sanitized, so a leading-backslash symbol climbed out of charts/.
    monkeypatch.chdir(tmp_path)
    executor = make_tool_executor(_FakeConfig(), _FakeHttpClients())

    result = await executor(
        ToolCall(id="1", name="show_chart", arguments={"symbol": "\\..\\..\\..\\evil"})
    )

    assert "error" not in result
    charts_dir = os.path.realpath(str(tmp_path / "charts"))
    assert os.path.realpath(result["chart_path"]).startswith(charts_dir)
    assert not list(tmp_path.glob("*.png"))
    assert not list(tmp_path.parent.glob("evil*.png"))
