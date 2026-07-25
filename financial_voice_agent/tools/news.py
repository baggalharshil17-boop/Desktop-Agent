from __future__ import annotations


async def get_news(query: str, *, http_client, api_key: str, max_results: int = 5) -> dict:
    try:
        response = await http_client.post(
            "/search", json={"api_key": api_key, "query": query, "max_results": max_results}
        )
        response.raise_for_status()
        data = response.json()
    except Exception:  # noqa: BLE001 -- must degrade gracefully per PRD Section 6, never raise
        return {"headlines": [], "error": "news search did not return results"}

    results = data.get("results", [])
    return {
        "headlines": [
            {"title": r.get("title"), "summary": r.get("content"), "url": r.get("url")}
            for r in results[:max_results]
        ]
    }
