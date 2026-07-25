import pytest

from financial_voice_agent.tools.news import get_news


class _FakeHttpClient:
    def __init__(self, response=None, raises: Exception | None = None):
        self._response = response
        self._raises = raises
        self.received_json: dict | None = None

    async def post(self, path, json=None):
        self.received_json = json
        if self._raises:
            raise self._raises
        return self._response


class _FakeResponse:
    def __init__(self, status_code: int, json_data: dict):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.mark.asyncio
async def test_get_news_returns_headlines_on_success():
    client = _FakeHttpClient(
        response=_FakeResponse(
            200,
            {
                "results": [
                    {"title": "Nifty hits new high", "content": "Markets rallied today.", "url": "https://example.com/1"},
                ]
            },
        )
    )

    result = await get_news("Nifty", http_client=client, api_key="test-key")

    assert result == {
        "headlines": [
            {"title": "Nifty hits new high", "summary": "Markets rallied today.", "url": "https://example.com/1"}
        ]
    }
    assert client.received_json["api_key"] == "test-key"
    assert client.received_json["query"] == "Nifty"


@pytest.mark.asyncio
async def test_get_news_degrades_gracefully_on_timeout():
    client = _FakeHttpClient(raises=TimeoutError("no response"))

    result = await get_news("Nifty", http_client=client, api_key="test-key")

    assert result["headlines"] == []
    assert "error" in result


@pytest.mark.asyncio
async def test_get_news_degrades_gracefully_on_http_error():
    client = _FakeHttpClient(response=_FakeResponse(500, {}))

    result = await get_news("Nifty", http_client=client, api_key="test-key")

    assert result["headlines"] == []
    assert "error" in result


@pytest.mark.asyncio
async def test_get_news_respects_max_results():
    client = _FakeHttpClient(
        response=_FakeResponse(
            200,
            {"results": [{"title": f"Headline {i}", "content": "x", "url": "u"} for i in range(10)]},
        )
    )

    result = await get_news("Nifty", http_client=client, api_key="test-key", max_results=3)

    assert len(result["headlines"]) == 3


@pytest.mark.asyncio
async def test_get_news_degrades_gracefully_on_null_results():
    client = _FakeHttpClient(response=_FakeResponse(200, {"results": None}))

    result = await get_news("Nifty", http_client=client, api_key="test-key")

    assert result["headlines"] == []
    assert "error" in result


@pytest.mark.asyncio
async def test_get_news_degrades_gracefully_on_malformed_result_items():
    client = _FakeHttpClient(response=_FakeResponse(200, {"results": ["not-a-dict"]}))

    result = await get_news("Nifty", http_client=client, api_key="test-key")

    assert result["headlines"] == []
    assert "error" in result
