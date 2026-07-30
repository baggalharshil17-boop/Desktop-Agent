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
    assert result["nse_symbol"] == "RELIANCE"
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
        "companyProfile": {"exchangeCodeNse": "RELIANCE"},
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
    assert result["nse_symbol"] == "RELIANCE"


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
