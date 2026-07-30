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
