import httpx
import pytest

from financial_voice_agent.tools.screener import SECTOR_SYMBOLS, screen_stocks


def _rsi_response(rsi_value: float, date: str = "2026-08-01") -> dict:
    return {"Technical Analysis: RSI": {date: {"RSI": str(rsi_value)}}}


def _quote_response(price: float) -> dict:
    return {"Global Quote": {"05. price": str(price)}}


def _client_with_responses(responses_by_symbol_and_function: dict) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        key = (params.get("symbol"), params.get("function"))
        body = responses_by_symbol_and_function.get(key, {})
        return httpx.Response(200, json=body)
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://www.alphavantage.co"
    )


@pytest.mark.asyncio
async def test_screen_stocks_mock_mode_returns_fixture():
    result = await screen_stocks("tech", http_client=None, mode="mock", fixtures_dir="fixtures")

    assert "results" in result
    assert "error" not in result


@pytest.mark.asyncio
async def test_screen_stocks_filters_by_momentum_threshold():
    symbols = SECTOR_SYMBOLS["tech"]
    responses = {}
    for i, symbol in enumerate(symbols):
        rsi = 75.0 if i == 0 else 40.0
        responses[(symbol, "RSI")] = _rsi_response(rsi)
    responses[(symbols[0], "GLOBAL_QUOTE")] = _quote_response(1000.0)
    client = _client_with_responses(responses)

    result = await screen_stocks(
        "tech", http_client=client, mode="live", momentum_threshold=70.0,
        request_interval_seconds=0.0,
    )

    assert result["count"] == 1
    assert result["results"][0]["symbol"] == symbols[0]
    assert result["results"][0]["rsi"] == 75.0


@pytest.mark.asyncio
async def test_screen_stocks_filters_by_price_range():
    symbols = SECTOR_SYMBOLS["tech"]
    responses = {symbol: None for symbol in symbols}
    responses = {(symbol, "RSI"): _rsi_response(80.0) for symbol in symbols}
    responses[(symbols[0], "GLOBAL_QUOTE")] = _quote_response(500.0)
    for symbol in symbols[1:]:
        responses[(symbol, "GLOBAL_QUOTE")] = _quote_response(5000.0)
    client = _client_with_responses(responses)

    result = await screen_stocks(
        "tech", http_client=client, mode="live", momentum_threshold=70.0,
        price_max=1000.0, request_interval_seconds=0.0,
    )

    assert result["count"] == 1
    assert result["results"][0]["symbol"] == symbols[0]


@pytest.mark.asyncio
async def test_screen_stocks_respects_limit_and_ranks_by_rsi_descending():
    symbols = SECTOR_SYMBOLS["tech"]
    responses = {}
    for i, symbol in enumerate(symbols):
        responses[(symbol, "RSI")] = _rsi_response(70.0 + i)
    for symbol in symbols:
        responses[(symbol, "GLOBAL_QUOTE")] = _quote_response(100.0)
    client = _client_with_responses(responses)

    result = await screen_stocks(
        "tech", http_client=client, mode="live", momentum_threshold=70.0,
        limit=2, request_interval_seconds=0.0,
    )

    assert result["count"] == 2
    returned_symbols = {r["symbol"] for r in result["results"]}
    assert returned_symbols == {symbols[-1], symbols[-2]}


@pytest.mark.asyncio
async def test_screen_stocks_returns_error_for_unknown_sector():
    result = await screen_stocks("crypto", http_client=None, mode="live")

    assert "error" in result
    assert "crypto" in result["error"]


@pytest.mark.asyncio
async def test_screen_stocks_returns_empty_results_when_no_candidates_pass():
    symbols = SECTOR_SYMBOLS["tech"]
    responses = {(symbol, "RSI"): _rsi_response(10.0) for symbol in symbols}
    client = _client_with_responses(responses)

    result = await screen_stocks(
        "tech", http_client=client, mode="live", momentum_threshold=70.0,
        request_interval_seconds=0.0,
    )

    assert result == {"results": [], "count": 0}


@pytest.mark.asyncio
async def test_screen_stocks_drops_symbol_on_malformed_rsi_response():
    symbols = SECTOR_SYMBOLS["tech"]
    responses = {(symbols[0], "RSI"): {"unexpected": "shape"}}
    for symbol in symbols[1:]:
        responses[(symbol, "RSI")] = _rsi_response(75.0)
        responses[(symbol, "GLOBAL_QUOTE")] = _quote_response(100.0)
    client = _client_with_responses(responses)

    result = await screen_stocks(
        "tech", http_client=client, mode="live", momentum_threshold=70.0,
        request_interval_seconds=0.0,
    )

    returned_symbols = {r["symbol"] for r in result["results"]}
    assert symbols[0] not in returned_symbols


@pytest.mark.asyncio
async def test_screen_stocks_returns_error_on_network_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://www.alphavantage.co")

    result = await screen_stocks("tech", http_client=client, mode="live", request_interval_seconds=0.0)

    assert "error" in result


@pytest.mark.asyncio
async def test_screen_stocks_returns_rate_limit_error_on_alpha_vantage_throttle_note():
    # Verified against Alpha Vantage's documented behavior: a throttled call
    # returns HTTP 200 with a "Note" key instead of the expected data key --
    # not an HTTP 429 -- so this must be detected from the response body.
    symbols = SECTOR_SYMBOLS["tech"]
    responses = {
        (symbol, "RSI"): {"Note": "Thank you for using Alpha Vantage! Our standard API rate limit is..."}
        for symbol in symbols
    }
    client = _client_with_responses(responses)

    result = await screen_stocks("tech", http_client=client, mode="live", request_interval_seconds=0.0)

    assert result == {"error": "Screener temporarily rate-limited, try again in a minute"}


@pytest.mark.asyncio
async def test_screen_stocks_paces_requests_by_configured_interval():
    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    symbols = SECTOR_SYMBOLS["tech"]
    # All fail the momentum threshold, so no GLOBAL_QUOTE calls happen --
    # this isolates the test to just the 5 RSI calls (4 sleeps between them).
    responses = {(symbol, "RSI"): _rsi_response(10.0) for symbol in symbols}
    client = _client_with_responses(responses)

    await screen_stocks(
        "tech", http_client=client, mode="live", momentum_threshold=70.0,
        request_interval_seconds=5.0, sleep_fn=fake_sleep,
    )

    assert sleep_calls == [5.0, 5.0, 5.0, 5.0]
