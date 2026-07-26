import pytest

from financial_voice_agent.tools.instruments import InstrumentNotFoundError, get_instrument_token

_SAMPLE_CSV = (
    "instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,"
    "tick_size,lot_size,instrument_type,segment,exchange\r\n"
    "128031234,500325,RELIANCE,RELIANCE INDUSTRIES,0,,0,0.05,1,EQ,NSE,NSE\r\n"
    "256265,1001,NIFTY 50,NIFTY 50,0,,0,0.05,1,EQ,INDICES,NSE\r\n"
)


class _FakeResponse:
    def __init__(self, content: bytes):
        self.status_code = 200
        self.content = content

    def raise_for_status(self):
        pass


class _FakeHttpClient:
    def __init__(self, csv_text: str):
        self._content = csv_text.encode("utf-8")
        self.received_paths = []

    async def get(self, path, params=None):
        self.received_paths.append(path)
        return _FakeResponse(self._content)


@pytest.mark.asyncio
async def test_get_instrument_token_looks_up_symbol_from_dump():
    client = _FakeHttpClient(_SAMPLE_CSV)
    cache: dict = {}

    token = await get_instrument_token("RELIANCE", http_client=client, cache=cache)

    assert token == 128031234
    assert client.received_paths == ["/instruments/NSE"]


@pytest.mark.asyncio
async def test_get_instrument_token_handles_symbols_with_spaces():
    client = _FakeHttpClient(_SAMPLE_CSV)
    cache: dict = {}

    token = await get_instrument_token("NIFTY 50", http_client=client, cache=cache)

    assert token == 256265


@pytest.mark.asyncio
async def test_get_instrument_token_reuses_cache_without_refetching():
    client = _FakeHttpClient(_SAMPLE_CSV)
    cache: dict = {}

    await get_instrument_token("RELIANCE", http_client=client, cache=cache)
    await get_instrument_token("NIFTY 50", http_client=client, cache=cache)

    # Second lookup must be served from cache, not a second dump fetch.
    assert client.received_paths == ["/instruments/NSE"]


@pytest.mark.asyncio
async def test_get_instrument_token_raises_for_unknown_symbol():
    client = _FakeHttpClient(_SAMPLE_CSV)
    cache: dict = {}

    with pytest.raises(InstrumentNotFoundError, match="DOES_NOT_EXIST"):
        await get_instrument_token("DOES_NOT_EXIST", http_client=client, cache=cache)


@pytest.mark.asyncio
async def test_get_instrument_token_uses_separate_cache_key_per_exchange():
    client = _FakeHttpClient(_SAMPLE_CSV)
    cache: dict = {("BSE", "RELIANCE"): 999}

    token = await get_instrument_token("RELIANCE", http_client=client, cache=cache, exchange="NSE")

    assert token == 128031234
