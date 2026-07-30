# Add Fish Audio as a TTS Provider Option Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Fish Audio as a real, selectable `tts.provider` option (`"cartesia"` or `"fish_audio"`), replacing the Deepgram slot that was configured but never implemented, and let the setup wizard collect/validate a Fish Audio key.

**Architecture:** A new `RealFishAudioTtsClient` (plain `httpx` REST calls, no vendor SDK) implements the existing `TtsClient` Protocol. A new `make_tts_client(config, http_clients)` factory (matching the existing `make_stt_client`/`make_llm_client` pattern) picks the right adapter at startup. Config, `http_clients.py`, and the setup wizard get matching plumbing.

**Tech Stack:** `httpx` (already a dependency) — no new dependency needed.

## Global Constraints

- Windows only.
- No new dependencies — Fish Audio's REST API needs only `httpx`, already used throughout this codebase.
- Fish Audio's TTS request format, verified against real live API calls (not just documentation): `POST https://api.fish.audio/v1/tts`, `Authorization: Bearer <key>` header, model selection via a `model` **request header** (not a JSON body field — confirmed from a real code sample, not the OpenAPI schema, which omits this), JSON body `{"text": ..., "format": "pcm", "sample_rate": 16000}`. `format: "pcm"` returns raw PCM bytes with no container/header (verified: response starts with raw sample bytes, not a `RIFF` signature) — this is the format to use, matching `CARTESIA_OUTPUT_FORMAT`'s shape (16kHz, `pcm_s16le`) so nothing downstream needs resampling. `format: "wav"` was also tested and does NOT match (adds a RIFF header) — do not use it.
- `s2.1-pro-free` is a free-tier model (verified working with zero API credit balance); other models (`s2.1-pro`, `s2-pro`, `s1`) require funded API credit (confirmed via a real `402 Insufficient API credit` response) — use `s2.1-pro-free` as the default.
- Deepgram's placeholder (config validation + an unused `http_clients.py` branch, no real adapter ever existed) is fully removed as part of this plan.
- `tts.provider` remains a single mutually-exclusive choice (like `stt.provider`/`llm.provider`) — no live automatic fallback wiring; `synthesize_with_fallback()`'s `fallback` parameter stays unwired, same as today.
- Match the existing DI testing pattern: `RealFishAudioTtsClient` and `make_tts_client` are unit-testable via injected fakes/`httpx.MockTransport`, matching how `financial_voice_agent/tools/fundamentals.py` is tested.
- Fish Audio's ASR endpoint is explicitly out of scope for this plan (no free tier found).

---

### Task 1: Config and HTTP client plumbing

**Files:**
- Modify: `financial_voice_agent/config.py`
- Modify: `financial_voice_agent/http_clients.py`
- Modify: `config.yaml`
- Modify: `.env.example`
- Test: `tests/test_config.py` (add new tests, modify one existing test; other existing tests must keep passing unmodified)
- Test: `tests/test_http_clients.py` (modify `_make_config` helper and one existing test; other existing tests must keep passing unmodified)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Config.fish_audio_api_key: str | None`, `Config.fish_audio_model: str | None` fields. `_VALID_TTS_PROVIDERS = {"cartesia", "fish_audio"}`. `HTTPClients.tts` (unchanged field name) now also supports `config.tts_provider == "fish_audio"`, carrying an `Authorization: Bearer <fish_audio_api_key>` header when so configured. Task 2 consumes `http_clients.tts` and `config.fish_audio_model` by these exact names.

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.py`, modify the existing `test_load_config_invalid_tts_provider_raises` test — change `monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)` to `monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)` (the rest of that test is unchanged — it still asserts on an unrelated invalid provider name, `"elevenlabs"`, so no other line needs to change).

Then append these new tests to `tests/test_config.py`:
```python
def test_load_config_missing_fish_audio_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
    yaml_path = _write_yaml(
        tmp_path,
        """\
        vad:
          speech_threshold: 0.5
          silence_duration_ms: 600
          min_speech_duration_ms: 200
        audio:
          output_device_index: null
        input_mode: "always_on"
        tts:
          provider: "fish_audio"
        stt:
          provider: "groq"
          model: "test-stt-model"
        llm:
          provider: "groq"
          model: "test-model"
        storage:
          db_path: "./agent_turns.db"
        mode: "live"
        """,
    )
    env_path = _write_env(tmp_path, "GROQ_API_KEY=groq-secret\n")

    with pytest.raises(ConfigError, match="FISH_AUDIO_API_KEY"):
        load_config(config_path=yaml_path, env_path=env_path)


def test_load_config_reads_fish_audio_provider_and_model(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
    yaml_path = _write_yaml(
        tmp_path,
        """\
        vad:
          speech_threshold: 0.5
          silence_duration_ms: 600
          min_speech_duration_ms: 200
        audio:
          output_device_index: null
        input_mode: "always_on"
        tts:
          provider: "fish_audio"
          fish_audio_model: "s2.1-pro"
        stt:
          provider: "groq"
          model: "test-stt-model"
        llm:
          provider: "groq"
          model: "test-model"
        storage:
          db_path: "./agent_turns.db"
        mode: "live"
        """,
    )
    env_path = _write_env(tmp_path, "GROQ_API_KEY=groq-secret\nFISH_AUDIO_API_KEY=fish-secret\n")

    config = load_config(config_path=yaml_path, env_path=env_path)

    assert config.tts_provider == "fish_audio"
    assert config.fish_audio_model == "s2.1-pro"
    assert config.fish_audio_api_key == "fish-secret"


def test_load_config_fish_audio_model_defaults_to_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
    yaml_path = _write_yaml(
        tmp_path,
        """\
        vad:
          speech_threshold: 0.5
          silence_duration_ms: 600
          min_speech_duration_ms: 200
        audio:
          output_device_index: null
        input_mode: "always_on"
        tts:
          provider: "fish_audio"
        stt:
          provider: "groq"
          model: "test-stt-model"
        llm:
          provider: "groq"
          model: "test-model"
        storage:
          db_path: "./agent_turns.db"
        mode: "live"
        """,
    )
    env_path = _write_env(tmp_path, "GROQ_API_KEY=groq-secret\nFISH_AUDIO_API_KEY=fish-secret\n")

    config = load_config(config_path=yaml_path, env_path=env_path)

    assert config.fish_audio_model is None
```

In `tests/test_http_clients.py`, modify `_make_config`'s body: replace the `deepgram_api_key="deepgram-secret",` line with `fish_audio_api_key="fish-audio-secret",`.

Then replace the existing `test_tts_client_uses_deepgram_key_when_provider_is_deepgram` test with:
```python
@pytest.mark.asyncio
async def test_tts_client_uses_fish_audio_key_when_provider_is_fish_audio():
    config = _make_config(tts_provider="fish_audio")

    clients = await create_http_clients(config)
    try:
        assert clients.tts.headers["Authorization"] == "Bearer fish-audio-secret"
    finally:
        await close_http_clients(clients)
```
(All other existing tests in this file — `test_create_http_clients_returns_one_client_per_vendor`, `test_groq_client_carries_auth_header`, `test_tts_client_uses_cartesia_key_when_provider_is_cartesia`, `test_close_http_clients_closes_all`, `test_create_http_clients_rejects_unsupported_provider`, `test_groq_client_has_extended_timeout`, `test_kite_client_has_auth_header_when_credentials_present`, `test_kite_client_has_no_auth_header_when_credentials_absent`, `test_kite_client_sends_x_kite_version_header`, `test_indian_stock_client_carries_api_key_header` — are untouched; they don't reference Deepgram or Fish Audio directly and must keep passing unmodified.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py tests/test_http_clients.py -v`
Expected: FAIL — the new/modified tests fail because `Config` has no `fish_audio_api_key`/`fish_audio_model` fields yet, and `_VALID_TTS_PROVIDERS` doesn't include `"fish_audio"` yet.

- [ ] **Step 3: Write the implementation**

In `financial_voice_agent/config.py`:

1. Change line 16 from:
```python
_VALID_TTS_PROVIDERS = {"cartesia", "deepgram"}
```
to:
```python
_VALID_TTS_PROVIDERS = {"cartesia", "fish_audio"}
```

2. In the `Config` dataclass, replace the `deepgram_api_key: str | None` line with `fish_audio_api_key: str | None`, and add a new field `fish_audio_model: str | None` right after `cartesia_voice_id: str | None` (near the other `tts`-related fields):
```python
@dataclass(frozen=True)
class Config:
    vad_speech_threshold: float
    vad_silence_duration_ms: int
    vad_min_speech_duration_ms: int
    barge_in_enabled: bool
    barge_in_min_speech_ms: float
    audio_output_device_index: int | None
    audio_input_device_index: int | None
    echo_suppression_enabled: bool
    echo_margin: float
    echo_gain: float | None
    input_mode: str
    tts_provider: str
    cartesia_voice_id: str | None
    fish_audio_model: str | None
    stt_provider: str
    stt_model: str
    llm_provider: str
    llm_model: str
    storage_db_path: str
    mode: str
    processing_overlay_enabled: bool
    groq_api_key: str | None
    cartesia_api_key: str | None
    fish_audio_api_key: str | None
    huggingface_api_key: str | None
    kite_api_key: str | None
    kite_access_token: str | None
    tavily_api_key: str | None
    indian_stock_api_key: str | None
```

3. In `_load_env`'s tuple of env-var keys, replace `"DEEPGRAM_API_KEY",` with `"FISH_AUDIO_API_KEY",`.

4. In `load_config`, replace this block:
```python
        cartesia_api_key = env.get("CARTESIA_API_KEY")
        deepgram_api_key = env.get("DEEPGRAM_API_KEY")
        if tts_provider == "cartesia" and not cartesia_api_key:
            raise ConfigError("tts.provider is 'cartesia' but CARTESIA_API_KEY is not set")
        if tts_provider == "deepgram" and not deepgram_api_key:
            raise ConfigError("tts.provider is 'deepgram' but DEEPGRAM_API_KEY is not set")
```
with:
```python
        cartesia_api_key = env.get("CARTESIA_API_KEY")
        fish_audio_api_key = env.get("FISH_AUDIO_API_KEY")
        if tts_provider == "cartesia" and not cartesia_api_key:
            raise ConfigError("tts.provider is 'cartesia' but CARTESIA_API_KEY is not set")
        if tts_provider == "fish_audio" and not fish_audio_api_key:
            raise ConfigError("tts.provider is 'fish_audio' but FISH_AUDIO_API_KEY is not set")
```

5. Also update the error message text when `tts_provider not in _VALID_TTS_PROVIDERS` (still raises `ConfigError`, just fix the wording to match the new valid set):
```python
        if tts_provider not in _VALID_TTS_PROVIDERS:
            raise ConfigError(
                f"tts.provider must be 'cartesia' or 'fish_audio', got {tts_provider!r}"
            )
```

6. Right after the existing `cartesia_voice_id = raw.get("tts", {}).get("voice_id")` line (and its validation block, which stays exactly as-is), add:
```python
        fish_audio_model = raw.get("tts", {}).get("fish_audio_model")
```

7. In the `Config(...)` constructor call inside `load_config`, replace `deepgram_api_key=deepgram_api_key,` with `fish_audio_api_key=fish_audio_api_key,`, and add `fish_audio_model=fish_audio_model,` right after the existing `cartesia_voice_id=cartesia_voice_id,` line.

In `financial_voice_agent/http_clients.py`:

1. Replace line 11 (`DEEPGRAM_BASE_URL = "https://api.deepgram.com"`) with:
```python
FISH_AUDIO_BASE_URL = "https://api.fish.audio"
```

2. Replace this block in `create_http_clients`:
```python
    elif config.tts_provider == "deepgram":
        tts = httpx.AsyncClient(
            base_url=DEEPGRAM_BASE_URL,
            headers={"Authorization": f"Token {config.deepgram_api_key or ''}"},
            timeout=TTS_TIMEOUT,
        )
```
with:
```python
    elif config.tts_provider == "fish_audio":
        tts = httpx.AsyncClient(
            base_url=FISH_AUDIO_BASE_URL,
            headers={"Authorization": f"Bearer {config.fish_audio_api_key or ''}"},
            timeout=TTS_TIMEOUT,
        )
```
(The `else: raise ValueError(...)` branch right after stays exactly as-is.)

In `config.yaml` (the real checked-in file), the `tts:` section stays on `provider: "cartesia"` (no change to the active choice) but gets an updated comment and the new optional field documented — replace:
```yaml
tts:
  provider: "cartesia"  # or "deepgram"
  voice_id: "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"
```
with:
```yaml
tts:
  provider: "cartesia"  # or "fish_audio"
  voice_id: "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"
  # Only used when provider is "fish_audio". "s2.1-pro-free" works with
  # zero Fish Audio API credit balance; other models ("s2.1-pro", "s2-pro",
  # "s1") require a funded account.
  # fish_audio_model: "s2.1-pro-free"
```

In `.env.example`, replace the `DEEPGRAM_API_KEY=` line with `FISH_AUDIO_API_KEY=`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py tests/test_http_clients.py -v`
Expected: PASS (all existing tests plus the new ones)

Then run the full suite to confirm no regressions elsewhere:
Run: `python -m pytest -q`
Expected: PASS, no failures

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/config.py financial_voice_agent/http_clients.py config.yaml .env.example tests/test_config.py tests/test_http_clients.py
git commit -m "Add Fish Audio config/HTTP client support, remove unused Deepgram placeholder"
```

---

### Task 2: TTS adapter and live wiring

**Files:**
- Modify: `financial_voice_agent/orchestrator/tts.py`
- Modify: `financial_voice_agent/__main__.py`
- Test: `tests/orchestrator/test_tts.py` (add new tests; existing tests must keep passing unmodified)

**Interfaces:**
- Consumes: `financial_voice_agent.http_clients.HTTPClients.tts` (Task 1), `Config.fish_audio_model`/`Config.fish_audio_api_key`/`Config.tts_provider`/`Config.cartesia_api_key`/`Config.cartesia_voice_id` (Task 1 and pre-existing).
- Produces: `RealFishAudioTtsClient(http_client, *, model: str = FISH_AUDIO_DEFAULT_MODEL)` implementing `TtsClient`. `make_tts_client(config, http_clients) -> TtsClient`. `__main__.py` calls `make_tts_client` by this exact name/signature.

- [ ] **Step 1: Write the failing tests**

Add to `tests/orchestrator/test_tts.py` (append; keep the existing imports and all existing tests exactly as-is — add `httpx` to the imports at the top, since the new tests need it):
```python
import httpx
```
(add this as a new top-level import line, alongside the existing `import pytest`)

Then update the existing import line from:
```python
from financial_voice_agent.orchestrator.tts import CARTESIA_OUTPUT_FORMAT, TtsError, synthesize_with_fallback
```
to:
```python
from financial_voice_agent.orchestrator.tts import (
    CARTESIA_OUTPUT_FORMAT,
    FISH_AUDIO_DEFAULT_MODEL,
    RealFishAudioTtsClient,
    TtsError,
    make_tts_client,
    synthesize_with_fallback,
)
```

Then append these new tests:
```python
@pytest.mark.asyncio
async def test_fish_audio_client_sends_model_header_and_pcm_format():
    captured_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, content=b"raw-pcm-bytes")

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.fish.audio"
    )
    client = RealFishAudioTtsClient(http_client, model="s2.1-pro-free")

    result = await client.synthesize("Hello there")

    assert result == b"raw-pcm-bytes"
    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.headers["model"] == "s2.1-pro-free"
    import json
    body = json.loads(request.content)
    assert body == {"text": "Hello there", "format": "pcm", "sample_rate": 16000}


@pytest.mark.asyncio
async def test_fish_audio_client_defaults_to_free_model():
    captured_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, content=b"audio")

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.fish.audio"
    )
    client = RealFishAudioTtsClient(http_client)

    await client.synthesize("test")

    assert captured_requests[0].headers["model"] == FISH_AUDIO_DEFAULT_MODEL


@pytest.mark.asyncio
async def test_fish_audio_client_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"message": "Insufficient API credit", "status": 402})

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.fish.audio"
    )
    client = RealFishAudioTtsClient(http_client)

    with pytest.raises(httpx.HTTPStatusError):
        await client.synthesize("test")


def test_make_tts_client_returns_fish_audio_client_for_fish_audio_provider():
    class _FakeConfig:
        tts_provider = "fish_audio"
        fish_audio_model = "s2.1-pro"

    class _FakeHttpClients:
        tts = httpx.AsyncClient(base_url="https://api.fish.audio")

    client = make_tts_client(_FakeConfig(), _FakeHttpClients())

    assert isinstance(client, RealFishAudioTtsClient)
    assert client._model == "s2.1-pro"


def test_make_tts_client_uses_default_model_when_config_model_is_none():
    class _FakeConfig:
        tts_provider = "fish_audio"
        fish_audio_model = None

    class _FakeHttpClients:
        tts = httpx.AsyncClient(base_url="https://api.fish.audio")

    client = make_tts_client(_FakeConfig(), _FakeHttpClients())

    assert client._model == FISH_AUDIO_DEFAULT_MODEL


def test_make_tts_client_raises_for_unsupported_provider():
    class _FakeConfig:
        tts_provider = "elevenlabs"
        fish_audio_model = None

    class _FakeHttpClients:
        tts = None

    with pytest.raises(ValueError, match="elevenlabs"):
        make_tts_client(_FakeConfig(), _FakeHttpClients())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/orchestrator/test_tts.py -v`
Expected: FAIL with `ImportError: cannot import name 'FISH_AUDIO_DEFAULT_MODEL'`

- [ ] **Step 3: Write the implementation**

In `financial_voice_agent/orchestrator/tts.py`:

1. Add a new constant right after `CARTESIA_OUTPUT_FORMAT` (line 8):
```python
FISH_AUDIO_DEFAULT_MODEL = "s2.1-pro-free"
```

2. Update `synthesize_with_fallback`'s two hardcoded error messages, since "Cartesia" is no longer the only possible provider — replace:
```python
    except RetryExhaustedError as exc:
        if fallback is not None:
            try:
                return await fallback.synthesize(text)
            except Exception as fallback_exc:  # noqa: BLE001
                raise TtsError("Both Cartesia and Deepgram TTS failed") from fallback_exc
        raise TtsError("Cartesia TTS failed and no fallback configured") from exc
```
with:
```python
    except RetryExhaustedError as exc:
        if fallback is not None:
            try:
                return await fallback.synthesize(text)
            except Exception as fallback_exc:  # noqa: BLE001
                raise TtsError("Both primary and fallback TTS failed") from fallback_exc
        raise TtsError("TTS failed and no fallback configured") from exc
```

3. Add the new adapter class and factory function at the end of the file, after `RealCartesiaTtsClient`:
```python
class RealFishAudioTtsClient:
    """Thin adapter around Fish Audio's REST TTS API.

    Verified against real live API calls: model selection is a request
    HEADER (not a JSON body field -- the OpenAPI schema doesn't document
    this; found from a real code sample instead). format="pcm" returns raw
    PCM bytes with no container/header, matching CARTESIA_OUTPUT_FORMAT's
    shape (16kHz, pcm_s16le) so nothing downstream needs resampling --
    format="wav" was also tested and does NOT match (adds a RIFF header).
    """

    def __init__(self, http_client, *, model: str = FISH_AUDIO_DEFAULT_MODEL) -> None:
        self._client = http_client
        self._model = model

    async def synthesize(self, text: str) -> bytes:
        response = await self._client.post(
            "/v1/tts",
            headers={"model": self._model},
            json={"text": text, "format": "pcm", "sample_rate": 16000},
        )
        response.raise_for_status()
        return response.content


def make_tts_client(config, http_clients) -> TtsClient:
    """Picks the real TTS adapter based on config.tts_provider -- the seam
    that lets tts.provider in config.yaml switch between Cartesia and Fish
    Audio without touching any calling code."""
    if config.tts_provider == "fish_audio":
        return RealFishAudioTtsClient(
            http_clients.tts, model=config.fish_audio_model or FISH_AUDIO_DEFAULT_MODEL
        )
    if config.tts_provider == "cartesia":
        from cartesia import AsyncCartesia

        return RealCartesiaTtsClient(
            AsyncCartesia(api_key=config.cartesia_api_key), voice_id=config.cartesia_voice_id
        )
    raise ValueError(f"Unsupported tts_provider: {config.tts_provider!r}")
```

In `financial_voice_agent/__main__.py`:

1. Remove the now-unused import (line 29): `from cartesia import AsyncCartesia` — delete this line entirely (the import moves into `tts.py`'s `make_tts_client`, matching how `stt.py`/`llm.py`'s factories locally import their vendor SDKs).

2. Change the import on line 48 from:
```python
from financial_voice_agent.orchestrator.tts import RealCartesiaTtsClient, synthesize_with_fallback
```
to:
```python
from financial_voice_agent.orchestrator.tts import make_tts_client, synthesize_with_fallback
```

3. Remove the hard guard (lines 55-59):
```python
    if config.tts_provider != "cartesia":
        raise NotImplementedError(
            f"Only tts.provider: 'cartesia' is wired up in the live loop right now, "
            f"got {config.tts_provider!r} -- no Deepgram TTS adapter exists yet"
        )
```
(Delete this block entirely — `make_tts_client` now handles every supported provider, and raises its own clear `ValueError` for anything unsupported.)

4. Replace lines 97-99:
```python
    tts_client = RealCartesiaTtsClient(
        AsyncCartesia(api_key=config.cartesia_api_key), voice_id=config.cartesia_voice_id
    )
```
with:
```python
    tts_client = make_tts_client(config, http_clients)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/orchestrator/test_tts.py -v`
Expected: PASS (all existing tests plus the new ones)

Verify `__main__.py` still imports cleanly (it's the live entry point, not directly unit tested):
Run: `python -c "import financial_voice_agent.__main__"`
Expected: no traceback (all runtime deps — pyaudio, janus, cartesia — resolve; `main()` itself is not invoked since it needs real audio hardware)

Then run the full suite:
Run: `python -m pytest -q`
Expected: PASS, no failures

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/orchestrator/tts.py financial_voice_agent/__main__.py tests/orchestrator/test_tts.py
git commit -m "Add Fish Audio TTS adapter and wire it into the live entry point"
```

---

### Task 3: Setup wizard support

**Files:**
- Modify: `financial_voice_agent/setup/validators.py`
- Modify: `financial_voice_agent/setup/config_template.py`
- Modify: `scripts/setup.py`
- Modify: `README.md`
- Test: `tests/setup/test_validators.py` (add new tests; existing tests must keep passing unmodified)
- Test: `tests/setup/test_config_template.py` (modify all three existing tests to pass the new required `tts_provider` argument; add new tests)

**Interfaces:**
- Consumes: nothing from Tasks 1-2 directly (the wizard writes `config.yaml`/`.env` files that `load_config` later reads independently — it does not import `Config`/`make_tts_client`).
- Produces: `validate_fish_audio_key(api_key: str, *, http_client: httpx.Client | None = None) -> ValidationResult`. `render_config_yaml(..., tts_provider: str, fish_audio_model: str = FISH_AUDIO_DEFAULT_MODEL_STR, ...)` — note `render_config_yaml` gains a new **required** keyword-only parameter `tts_provider` (breaking its existing call shape, hence all three existing tests need updating in Step 1 below).

- [ ] **Step 1: Write the failing tests**

Add to `tests/setup/test_validators.py` (append; this file already imports `httpx` and `pytest` for the Tavily tests — check the existing imports at the top and add `validate_fish_audio_key` to the existing `from financial_voice_agent.setup.validators import (...)` import line):
```python
def test_validate_fish_audio_key_ok_on_200_response():
    def handler(request):
        return httpx.Response(200, content=b"audio-bytes")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://api.fish.audio", transport=transport)

    result = validate_fish_audio_key("test-key", http_client=client)

    assert result.ok is True


def test_validate_fish_audio_key_reports_failure_on_402():
    def handler(request):
        return httpx.Response(402, json={"message": "Insufficient API credit", "status": 402})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://api.fish.audio", transport=transport)

    result = validate_fish_audio_key("test-key", http_client=client)

    assert result.ok is False
```

In `tests/setup/test_config_template.py`, update all three existing tests to add `tts_provider="cartesia"` as a new argument to every `render_config_yaml(...)` call (each of the three existing calls currently passes `cartesia_voice_id=...` but not `tts_provider` — add `tts_provider="cartesia",` as a new line in each call, right before the `cartesia_voice_id=...` line). For example, the first test's call becomes:
```python
    text = render_config_yaml(
        stt_provider="groq",
        stt_model="whisper-large-v3-turbo",
        llm_provider="groq",
        llm_model="qwen/qwen3.6-27b",
        tts_provider="cartesia",
        cartesia_voice_id="db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
        mode="mock",
    )
```
(Apply the same one-line addition, `tts_provider="cartesia",`, to the other two existing `render_config_yaml(...)` calls in this file — their other arguments and assertions are unchanged.)

Then append a new test:
```python
def test_render_config_yaml_renders_fish_audio_model_when_provider_is_fish_audio():
    text = render_config_yaml(
        stt_provider="groq",
        stt_model="whisper-large-v3-turbo",
        llm_provider="groq",
        llm_model="qwen/qwen3.6-27b",
        tts_provider="fish_audio",
        fish_audio_model="s2.1-pro-free",
        mode="mock",
    )

    parsed = yaml.safe_load(text)

    assert parsed["tts"]["provider"] == "fish_audio"
    assert parsed["tts"]["fish_audio_model"] == "s2.1-pro-free"
    assert "voice_id" not in parsed["tts"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/setup/test_validators.py tests/setup/test_config_template.py -v`
Expected: FAIL — `validate_fish_audio_key` doesn't exist yet (`ImportError`); the three modified `render_config_yaml` calls fail with `TypeError: render_config_yaml() got an unexpected keyword argument 'tts_provider'`.

- [ ] **Step 3: Write the implementation**

In `financial_voice_agent/setup/validators.py`, add a new function at the end of the file (after `validate_tavily_key`), matching `validate_tavily_key`'s exact shape (sync `httpx.Client`, injectable):
```python
def validate_fish_audio_key(api_key: str, *, http_client: httpx.Client | None = None) -> ValidationResult:
    owns_client = http_client is None
    client = http_client or httpx.Client(base_url="https://api.fish.audio", timeout=15.0)
    try:
        response = client.post(
            "/v1/tts",
            headers={"Authorization": f"Bearer {api_key}", "model": "s2.1-pro-free"},
            json={"text": "Test.", "format": "pcm", "sample_rate": 16000},
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return ValidationResult(ok=False, message=f"Fish Audio key rejected: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(ok=False, message=f"Could not reach Fish Audio: {exc}")
    finally:
        if owns_client:
            client.close()
    return ValidationResult(ok=True, message="Fish Audio key OK")
```

In `financial_voice_agent/setup/config_template.py`, replace the whole file's content with:
```python
from __future__ import annotations

_TEMPLATE = """\
vad:
  speech_threshold: 0.5
  silence_duration_ms: 600
  min_speech_duration_ms: 200
  barge_in_enabled: true
  barge_in_min_speech_ms: 200

audio:
  output_device_index: null
  input_device_index: null
  echo_suppression: true
  echo_margin: 1.75

input_mode: "always_on"  # or "ptt"
tts:
  provider: "{tts_provider}"  # or "cartesia"/"fish_audio"
{tts_provider_field}

stt:
  provider: "{stt_provider}"  # or "groq"/"huggingface"
  model: "{stt_model}"

llm:
  provider: "{llm_provider}"  # or "groq"/"huggingface"
  # Must support both tool calling and vision (capture_screen needs the
  # latter) -- check console.groq.com/docs/models and
  # console.groq.com/docs/vision before changing.
  model: "{llm_model}"

storage:
  db_path: "./agent_turns.db"

mode: "{mode}"  # or "live"/"mock"
"""

FISH_AUDIO_DEFAULT_MODEL = "s2.1-pro-free"


def render_config_yaml(
    *,
    stt_provider: str,
    stt_model: str,
    llm_provider: str,
    llm_model: str,
    tts_provider: str,
    cartesia_voice_id: str = "",
    fish_audio_model: str = FISH_AUDIO_DEFAULT_MODEL,
    mode: str,
) -> str:
    if tts_provider == "cartesia":
        tts_provider_field = f'  voice_id: "{cartesia_voice_id}"'
    else:
        tts_provider_field = f'  fish_audio_model: "{fish_audio_model}"'
    return _TEMPLATE.format(
        stt_provider=stt_provider,
        stt_model=stt_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
        tts_provider=tts_provider,
        tts_provider_field=tts_provider_field,
        mode=mode,
    )
```
(This is a full-file replacement: the template's `vad`/`audio`/`stt`/`llm`/`storage`/`mode` sections are unchanged from the current file — only the `tts:` section and the function signature/body change, to make the provider and its provider-specific field dynamic instead of hardcoded to Cartesia.)

In `scripts/setup.py`:

1. Update the import block (currently lines 28-34) to add `validate_fish_audio_key`:
```python
from financial_voice_agent.setup.validators import (
    ValidationResult,
    validate_cartesia_key,
    validate_fish_audio_key,
    validate_groq_key,
    validate_huggingface_key,
    validate_tavily_key,
)
```

2. Add a `tts_provider` choice prompt. Insert it right after the existing `llm_provider = _ask_choice(...)` line and its follow-up `print(...)` note (i.e., right before the `mode = _ask_choice(...)` block):
```python
    tts_provider = _ask_choice("TTS provider", ["cartesia", "fish_audio"], default="cartesia")
```

3. Replace the current unconditional Cartesia key collection block:
```python
    cartesia_key, cartesia_result = _collect_and_validate(
        "CARTESIA_API_KEY", "Cartesia API key (play.cartesia.ai)", validate_cartesia_key
    )
    if cartesia_key:
        new_env["CARTESIA_API_KEY"] = cartesia_key
        voices = (cartesia_result.data or {}).get("voices", [])
        if voices:
            print("\n  Available voices:")
            for i, (voice_id, name) in enumerate(voices[:15], start=1):
                print(f"    {i}. {name} ({voice_id})")
            choice = input("  Pick a voice number (or press Enter to type an id manually): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(voices[:15]):
                cartesia_voice_id = voices[int(choice) - 1][0]
            else:
                cartesia_voice_id = input("  Paste a voice_id: ").strip()
```
with a version conditional on `tts_provider`, and add a new `fish_audio_model` variable initialized alongside the existing `cartesia_voice_id = ""` line near the top of `main()`:
```python
    cartesia_voice_id = ""
    fish_audio_model = "s2.1-pro-free"
```
(This replaces the current standalone `cartesia_voice_id = ""` line — add `fish_audio_model = "s2.1-pro-free"` right after it.)

Then the conditional key collection:
```python
    if tts_provider == "cartesia":
        cartesia_key, cartesia_result = _collect_and_validate(
            "CARTESIA_API_KEY", "Cartesia API key (play.cartesia.ai)", validate_cartesia_key
        )
        if cartesia_key:
            new_env["CARTESIA_API_KEY"] = cartesia_key
            voices = (cartesia_result.data or {}).get("voices", [])
            if voices:
                print("\n  Available voices:")
                for i, (voice_id, name) in enumerate(voices[:15], start=1):
                    print(f"    {i}. {name} ({voice_id})")
                choice = input("  Pick a voice number (or press Enter to type an id manually): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(voices[:15]):
                    cartesia_voice_id = voices[int(choice) - 1][0]
                else:
                    cartesia_voice_id = input("  Paste a voice_id: ").strip()
    else:
        fish_audio_key, _ = _collect_and_validate(
            "FISH_AUDIO_API_KEY", "Fish Audio API key (fish.audio/app/api-keys)", validate_fish_audio_key
        )
        if fish_audio_key:
            new_env["FISH_AUDIO_API_KEY"] = fish_audio_key
```

4. Update the `render_config_yaml(...)` call (inside the `if write_config:` block) to pass the new `tts_provider`/`fish_audio_model` arguments — replace:
```python
        config_text = render_config_yaml(
            stt_provider=stt_provider,
            stt_model=stt_model,
            llm_provider=llm_provider,
            llm_model=llm_model,
            cartesia_voice_id=cartesia_voice_id or "<pick a voice id from https://play.cartesia.ai/voices>",
            mode=mode,
        )
```
with:
```python
        config_text = render_config_yaml(
            stt_provider=stt_provider,
            stt_model=stt_model,
            llm_provider=llm_provider,
            llm_model=llm_model,
            tts_provider=tts_provider,
            cartesia_voice_id=cartesia_voice_id or "<pick a voice id from https://play.cartesia.ai/voices>",
            fish_audio_model=fish_audio_model,
            mode=mode,
        )
```

In `README.md`, add a new row to the "What each provider needs" table (currently 5 rows) — replace:
```markdown
| Cartesia (TTS) | `CARTESIA_API_KEY` | play.cartesia.ai | Credit-based free tier |
```
with:
```markdown
| Cartesia (TTS) | `CARTESIA_API_KEY` | play.cartesia.ai | Credit-based free tier |
| Fish Audio (TTS alternative) | `FISH_AUDIO_API_KEY` | fish.audio/app/api-keys | `s2.1-pro-free` model works free; other models need funded API credit |
```
(This inserts a new row right after the existing Cartesia row, before the Tavily row.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/setup/test_validators.py tests/setup/test_config_template.py -v`
Expected: PASS (all existing tests plus the new ones)

Sanity-check `scripts/setup.py` still has no syntax errors (this script is not unit tested, per this project's established convention for interactive scripts — matches `scripts/kite_login.py`):
Run: `python -m py_compile scripts/setup.py`
Expected: no output, no error

Then run the full suite:
Run: `python -m pytest -q`
Expected: PASS, no failures

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/setup/validators.py financial_voice_agent/setup/config_template.py scripts/setup.py README.md tests/setup/test_validators.py tests/setup/test_config_template.py
git commit -m "Let the setup wizard collect and validate a Fish Audio key"
```

---

### Task 4: Manual verification

**Files:**
- None (verification only, no code changes expected — see the exception below).

**Interfaces:**
- Consumes: everything from Tasks 1-3, running end-to-end.

- [ ] **Step 1: Run the full test suite one more time**

Run: `python -m pytest -q`
Expected: PASS, all tests (existing suite plus every new test from Tasks 1-3)

- [ ] **Step 2: Manually verify Fish Audio TTS with the real agent**

In a scratch copy of `config.yaml` (or by temporarily editing the real one), set:
```yaml
tts:
  provider: "fish_audio"
  fish_audio_model: "s2.1-pro-free"
```
Ensure `FISH_AUDIO_API_KEY` is set in `.env`. Run:
```
python -m financial_voice_agent
```
Speak a query and confirm real, audible, correctly-formed audio plays back (not static/noise, which would indicate a format mismatch between what Fish Audio returned and what the playback pipeline expects). If it doesn't sound right, double check the request actually used `format: "pcm"` (not `"wav"`) — re-run this plan's Global Constraints section's verified request shape against a real call if something seems off.

Set `config.yaml`'s `tts.provider` back to `"cartesia"` afterward (or leave as `"fish_audio"` if you intend to keep using it — your call).

- [ ] **Step 3: Manually verify the setup wizard's Fish Audio path**

Run `python scripts/setup.py` in a scratch environment (or answer "No" to the overwrite prompts to avoid touching your real `.env`/`config.yaml` if running it for real) and choose `fish_audio` when prompted for TTS provider. Confirm the wizard prompts for `FISH_AUDIO_API_KEY`, validates it live (shows "OK: Fish Audio key OK" or a clear failure message), and — if `write_config` is chosen — the generated `config.yaml` has `tts.provider: "fish_audio"` and a `fish_audio_model` field, not a `voice_id` field.

- [ ] **Step 4: Commit any manual-verification fixes**

If Steps 2-3 required code changes, commit them now with a message describing what was fixed. If no changes were needed, there is nothing to commit for this task.
