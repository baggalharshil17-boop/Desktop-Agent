# Financial Voice Agent — Phase 1: Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the non-negotiable substrate every later phase depends on: config loading (yaml + env), SQLite turn logging, mock-mode fixture loading, and one reused HTTP client per vendor. No audio, no LLM calls, no tools yet — this phase produces zero user-facing behavior but everything else imports from it.

**Architecture:** Four small, independent modules under `financial_voice_agent/`, each with its own test file. No module depends on another except `http_clients.py`, which reads `Config` produced by `config.py`. All modules are synchronous except `http_clients.py` (async, since httpx.AsyncClient and later Groq/Cartesia calls are async).

**Tech Stack:** Python 3.11+, pyyaml, python-dotenv, httpx, pytest, pytest-asyncio, sqlite3 (stdlib).

## Global Constraints

- Python 3.11+ only (per PRD Section 9 "Tech Stack & Dependencies").
- `storage.db_path` defaults to `./agent_turns.db` (PRD Section 12 config.yaml).
- Screenshots stored as files, not BLOBs (PRD Section 12 "Why SQLite").
- One `httpx.AsyncClient` per vendor, created once at startup, reused — never a fresh client per call (PRD Section 14 "Network & Transport").
- `mode: "mock"` must let `get_quote`, `get_ohlc_history`, `get_positions_holdings` load from `fixtures/` without a live Kite session (PRD Section 18.4).
- Treat the Groq model string as config, not a hardcoded constant (PRD Section 18.2 / Section 20).
- No app-level encryption; rely on OS disk encryption (PRD "Privacy & Compliance").

---

## File Structure

```
financial_voice_agent/
    __init__.py
    config.py          # load_config() -> Config dataclass (yaml + env vars)
    db.py               # init_db(), log_turn()
    mock.py             # load_fixture() for mode="mock"
    http_clients.py     # HTTPClients — one httpx.AsyncClient per vendor, reused
fixtures/
    quote.json
    ohlc_history.json
    positions_holdings.json
config.yaml
.env.example
tests/
    __init__.py
    test_config.py
    test_db.py
    test_mock.py
    test_http_clients.py
```

- `config.py`: owns the `Config` dataclass and all env/yaml parsing. Nothing else parses yaml or reads `os.environ` for these keys.
- `db.py`: owns the `turns` table schema and all writes to it. Nothing else touches `agent_turns.db` directly.
- `mock.py`: owns fixture file I/O. Nothing else reads from `fixtures/`.
- `http_clients.py`: owns vendor `httpx.AsyncClient` lifecycle (create once, close once). Nothing else instantiates `httpx.AsyncClient` for these vendors.

---

### Task 1: Config loading (`config.py`)

**Files:**
- Create: `financial_voice_agent/__init__.py` (empty)
- Create: `financial_voice_agent/config.py`
- Create: `config.yaml`
- Create: `.env.example`
- Test: `tests/__init__.py` (empty)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `class Config` (frozen dataclass) with fields:
    - `vad_speech_threshold: float`
    - `vad_silence_duration_ms: int`
    - `vad_min_speech_duration_ms: int`
    - `audio_output_device_index: int | None`
    - `input_mode: str` (`"always_on"` or `"ptt"`)
    - `tts_provider: str` (`"cartesia"` or `"deepgram"`)
    - `llm_model: str`
    - `storage_db_path: str`
    - `mode: str` (`"live"` or `"mock"`)
    - `groq_api_key: str`
    - `cartesia_api_key: str | None`
    - `deepgram_api_key: str | None`
    - `kite_api_key: str | None`
    - `kite_access_token: str | None`
    - `tavily_api_key: str | None`
  - `load_config(config_path: str = "config.yaml", env_path: str = ".env") -> Config`
  - `class ConfigError(Exception)` — raised when a required env var is missing for the selected providers (`GROQ_API_KEY` always required; `CARTESIA_API_KEY` required if `tts_provider == "cartesia"`, `DEEPGRAM_API_KEY` required if `tts_provider == "deepgram"`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import os
import textwrap

import pytest

from financial_voice_agent.config import Config, ConfigError, load_config


def _write_yaml(tmp_path, contents):
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(contents))
    return str(path)


def _write_env(tmp_path, contents):
    path = tmp_path / ".env"
    path.write_text(textwrap.dedent(contents))
    return str(path)


VALID_YAML = """\
    vad:
      speech_threshold: 0.5
      silence_duration_ms: 600
      min_speech_duration_ms: 200
    audio:
      output_device_index: null
    input_mode: "always_on"
    tts:
      provider: "cartesia"
    llm:
      model: "test-model"
    storage:
      db_path: "./agent_turns.db"
    mode: "live"
    """


def test_load_config_reads_yaml_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    yaml_path = _write_yaml(tmp_path, VALID_YAML)
    env_path = _write_env(
        tmp_path,
        """\
        GROQ_API_KEY=groq-secret
        CARTESIA_API_KEY=cartesia-secret
        """,
    )

    config = load_config(config_path=yaml_path, env_path=env_path)

    assert isinstance(config, Config)
    assert config.vad_speech_threshold == 0.5
    assert config.vad_silence_duration_ms == 600
    assert config.vad_min_speech_duration_ms == 200
    assert config.audio_output_device_index is None
    assert config.input_mode == "always_on"
    assert config.tts_provider == "cartesia"
    assert config.llm_model == "test-model"
    assert config.storage_db_path == "./agent_turns.db"
    assert config.mode == "live"
    assert config.groq_api_key == "groq-secret"
    assert config.cartesia_api_key == "cartesia-secret"


def test_load_config_missing_groq_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    yaml_path = _write_yaml(tmp_path, VALID_YAML)
    env_path = _write_env(tmp_path, "CARTESIA_API_KEY=cartesia-secret\n")

    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        load_config(config_path=yaml_path, env_path=env_path)


def test_load_config_missing_provider_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    yaml_path = _write_yaml(tmp_path, VALID_YAML)  # tts.provider: cartesia
    env_path = _write_env(tmp_path, "GROQ_API_KEY=groq-secret\n")

    with pytest.raises(ConfigError, match="CARTESIA_API_KEY"):
        load_config(config_path=yaml_path, env_path=env_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent'` (or `ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/config.py
from __future__ import annotations

import os
from dataclasses import dataclass

import yaml
from dotenv import dotenv_values


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    vad_speech_threshold: float
    vad_silence_duration_ms: int
    vad_min_speech_duration_ms: int
    audio_output_device_index: int | None
    input_mode: str
    tts_provider: str
    llm_model: str
    storage_db_path: str
    mode: str
    groq_api_key: str
    cartesia_api_key: str | None
    deepgram_api_key: str | None
    kite_api_key: str | None
    kite_access_token: str | None
    tavily_api_key: str | None


def _load_env(env_path: str) -> dict[str, str]:
    file_values = dotenv_values(env_path) if os.path.exists(env_path) else {}
    merged = dict(file_values)
    for key in (
        "GROQ_API_KEY",
        "CARTESIA_API_KEY",
        "DEEPGRAM_API_KEY",
        "KITE_API_KEY",
        "KITE_ACCESS_TOKEN",
        "TAVILY_API_KEY",
    ):
        if key in os.environ:
            merged[key] = os.environ[key]
    return merged


def load_config(config_path: str = "config.yaml", env_path: str = ".env") -> Config:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    env = _load_env(env_path)

    groq_api_key = env.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ConfigError("Missing required environment variable: GROQ_API_KEY")

    tts_provider = raw["tts"]["provider"]
    cartesia_api_key = env.get("CARTESIA_API_KEY")
    deepgram_api_key = env.get("DEEPGRAM_API_KEY")
    if tts_provider == "cartesia" and not cartesia_api_key:
        raise ConfigError("tts.provider is 'cartesia' but CARTESIA_API_KEY is not set")
    if tts_provider == "deepgram" and not deepgram_api_key:
        raise ConfigError("tts.provider is 'deepgram' but DEEPGRAM_API_KEY is not set")

    return Config(
        vad_speech_threshold=raw["vad"]["speech_threshold"],
        vad_silence_duration_ms=raw["vad"]["silence_duration_ms"],
        vad_min_speech_duration_ms=raw["vad"]["min_speech_duration_ms"],
        audio_output_device_index=raw["audio"]["output_device_index"],
        input_mode=raw["input_mode"],
        tts_provider=tts_provider,
        llm_model=raw["llm"]["model"],
        storage_db_path=raw["storage"]["db_path"],
        mode=raw["mode"],
        groq_api_key=groq_api_key,
        cartesia_api_key=cartesia_api_key,
        deepgram_api_key=deepgram_api_key,
        kite_api_key=env.get("KITE_API_KEY"),
        kite_access_token=env.get("KITE_ACCESS_TOKEN"),
        tavily_api_key=env.get("TAVILY_API_KEY"),
    )
```

```yaml
# config.yaml
vad:
  speech_threshold: 0.5
  silence_duration_ms: 600
  min_speech_duration_ms: 200

audio:
  output_device_index: null  # null = OS default

input_mode: "always_on"  # or "ptt"
tts:
  provider: "cartesia"  # or "deepgram"

llm:
  model: "<check console.groq.com/docs/models>"  # treat as config, not constant

storage:
  db_path: "./agent_turns.db"

mode: "live"  # or "mock"
```

```
# .env.example
GROQ_API_KEY=
CARTESIA_API_KEY=
DEEPGRAM_API_KEY=
KITE_API_KEY=
KITE_ACCESS_TOKEN=
TAVILY_API_KEY=
```

Also create empty `tests/__init__.py` and empty `financial_voice_agent/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 tests). If `ModuleNotFoundError: No module named 'yaml'` or `'dotenv'`, run `pip install pyyaml python-dotenv` first.

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/__init__.py financial_voice_agent/config.py config.yaml .env.example tests/__init__.py tests/test_config.py
git commit -m "feat: add config loading from yaml + env"
```

---

### Task 2: SQLite turn logging (`db.py`)

**Files:**
- Create: `financial_voice_agent/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: nothing (takes a plain `db_path: str`, not `Config`, to keep the module testable in isolation).
- Produces:
  - `init_db(db_path: str) -> None` — creates the `turns` table if it doesn't exist.
  - `log_turn(db_path: str, *, transcript: str | None, tool_calls_json: str | None, tool_results_json: str | None, response_text: str | None, screenshot_path: str | None, latency_stt_ms: int | None, latency_llm_ms: int | None, latency_tool_ms: int | None, latency_tts_ms: int | None, latency_total_ms: int | None, error: str | None) -> int` — inserts a row with `ts_utc` set to the current UTC time in ISO 8601, returns the new `turn_id`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
import sqlite3

from financial_voice_agent.db import init_db, log_turn


def test_init_db_creates_turns_table(tmp_path):
    db_path = str(tmp_path / "agent_turns.db")

    init_db(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='turns'")
        assert cursor.fetchone() is not None
    finally:
        conn.close()


def test_init_db_is_idempotent(tmp_path):
    db_path = str(tmp_path / "agent_turns.db")

    init_db(db_path)
    init_db(db_path)  # must not raise

    conn = sqlite3.connect(db_path)
    conn.close()


def test_log_turn_inserts_row_and_returns_id(tmp_path):
    db_path = str(tmp_path / "agent_turns.db")
    init_db(db_path)

    turn_id = log_turn(
        db_path,
        transcript="what's the nifty level",
        tool_calls_json='[{"tool": "get_quote", "args": {"symbol": "NIFTY 50"}}]',
        tool_results_json='{"ltp": 24500}',
        response_text="Nifty is at 24500.",
        screenshot_path=None,
        latency_stt_ms=120,
        latency_llm_ms=430,
        latency_tool_ms=90,
        latency_tts_ms=60,
        latency_total_ms=700,
        error=None,
    )

    assert isinstance(turn_id, int)
    assert turn_id >= 1

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT transcript, response_text, latency_total_ms, ts_utc FROM turns WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    transcript, response_text, latency_total_ms, ts_utc = row
    assert transcript == "what's the nifty level"
    assert response_text == "Nifty is at 24500."
    assert latency_total_ms == 700
    assert ts_utc  # non-empty ISO timestamp string


def test_log_turn_second_call_increments_turn_id(tmp_path):
    db_path = str(tmp_path / "agent_turns.db")
    init_db(db_path)

    first_id = log_turn(
        db_path,
        transcript="a",
        tool_calls_json=None,
        tool_results_json=None,
        response_text="b",
        screenshot_path=None,
        latency_stt_ms=None,
        latency_llm_ms=None,
        latency_tool_ms=None,
        latency_tts_ms=None,
        latency_total_ms=None,
        error=None,
    )
    second_id = log_turn(
        db_path,
        transcript="c",
        tool_calls_json=None,
        tool_results_json=None,
        response_text="d",
        screenshot_path=None,
        latency_stt_ms=None,
        latency_llm_ms=None,
        latency_tool_ms=None,
        latency_tts_ms=None,
        latency_total_ms=None,
        error=None,
    )

    assert second_id == first_id + 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.db'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/db.py
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
  turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc TEXT NOT NULL,
  transcript TEXT,
  tool_calls_json TEXT,
  tool_results_json TEXT,
  response_text TEXT,
  screenshot_path TEXT,
  latency_stt_ms INTEGER,
  latency_llm_ms INTEGER,
  latency_tool_ms INTEGER,
  latency_tts_ms INTEGER,
  latency_total_ms INTEGER,
  error TEXT
);
"""


def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def log_turn(
    db_path: str,
    *,
    transcript: str | None,
    tool_calls_json: str | None,
    tool_results_json: str | None,
    response_text: str | None,
    screenshot_path: str | None,
    latency_stt_ms: int | None,
    latency_llm_ms: int | None,
    latency_tool_ms: int | None,
    latency_tts_ms: int | None,
    latency_total_ms: int | None,
    error: str | None,
) -> int:
    ts_utc = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO turns (
                ts_utc, transcript, tool_calls_json, tool_results_json,
                response_text, screenshot_path, latency_stt_ms, latency_llm_ms,
                latency_tool_ms, latency_tts_ms, latency_total_ms, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts_utc,
                transcript,
                tool_calls_json,
                tool_results_json,
                response_text,
                screenshot_path,
                latency_stt_ms,
                latency_llm_ms,
                latency_tool_ms,
                latency_tts_ms,
                latency_total_ms,
                error,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_db.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/db.py tests/test_db.py
git commit -m "feat: add SQLite turn logging"
```

---

### Task 3: Mock-mode fixture loader (`mock.py`)

**Files:**
- Create: `financial_voice_agent/mock.py`
- Create: `fixtures/quote.json`
- Create: `fixtures/ohlc_history.json`
- Create: `fixtures/positions_holdings.json`
- Test: `tests/test_mock.py`

**Interfaces:**
- Consumes: nothing (takes a plain `fixtures_dir: str`, defaulting to `"fixtures"`).
- Produces:
  - `class FixtureNotFoundError(Exception)`
  - `load_fixture(name: str, fixtures_dir: str = "fixtures") -> dict` — reads `{fixtures_dir}/{name}.json`, returns parsed JSON as a dict. Raises `FixtureNotFoundError` (not a bare `FileNotFoundError`) if the file doesn't exist, so callers in later phases can catch one exception type regardless of OS.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mock.py
import json

import pytest

from financial_voice_agent.mock import FixtureNotFoundError, load_fixture


def test_load_fixture_reads_json(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "quote.json").write_text(json.dumps({"symbol": "NIFTY 50", "ltp": 24500}))

    result = load_fixture("quote", fixtures_dir=str(fixtures_dir))

    assert result == {"symbol": "NIFTY 50", "ltp": 24500}


def test_load_fixture_missing_file_raises(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()

    with pytest.raises(FixtureNotFoundError, match="does_not_exist"):
        load_fixture("does_not_exist", fixtures_dir=str(fixtures_dir))


def test_repo_fixtures_are_valid_json():
    # Guards the checked-in fixtures used by mode="mock" at runtime.
    for name in ("quote", "ohlc_history", "positions_holdings"):
        result = load_fixture(name, fixtures_dir="fixtures")
        assert isinstance(result, dict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mock.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.mock'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/mock.py
from __future__ import annotations

import json
import os


class FixtureNotFoundError(Exception):
    pass


def load_fixture(name: str, fixtures_dir: str = "fixtures") -> dict:
    path = os.path.join(fixtures_dir, f"{name}.json")
    if not os.path.exists(path):
        raise FixtureNotFoundError(f"Fixture '{name}' not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
```

```json
// fixtures/quote.json
{
  "symbol": "NIFTY 50",
  "ltp": 24500.35,
  "day_open": 24380.1,
  "day_high": 24560.0,
  "day_low": 24350.0,
  "volume": 128340000
}
```

```json
// fixtures/ohlc_history.json
{
  "symbol": "RELIANCE",
  "interval": "15minute",
  "candles": [
    { "ts": "2026-07-24T09:15:00+05:30", "open": 2950.0, "high": 2958.0, "low": 2945.0, "close": 2952.5, "volume": 45210 },
    { "ts": "2026-07-24T09:30:00+05:30", "open": 2952.5, "high": 2961.0, "low": 2949.0, "close": 2957.8, "volume": 38900 }
  ]
}
```

```json
// fixtures/positions_holdings.json
{
  "positions": [],
  "holdings": [
    { "symbol": "RELIANCE", "quantity": 10, "average_price": 2800.0, "last_price": 2952.5 }
  ]
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mock.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/mock.py fixtures/quote.json fixtures/ohlc_history.json fixtures/positions_holdings.json tests/test_mock.py
git commit -m "feat: add mock-mode fixture loader and starter fixtures"
```

---

### Task 4: Per-vendor HTTP client manager (`http_clients.py`)

**Files:**
- Create: `financial_voice_agent/http_clients.py`
- Test: `tests/test_http_clients.py`
- Modify: `requirements.txt` (create if absent) — add `pyyaml`, `python-dotenv`, `httpx`, `pytest`, `pytest-asyncio`

**Interfaces:**
- Consumes: `Config` from Task 1 (`financial_voice_agent.config.Config`) — specifically `config.tts_provider`.
- Produces:
  - `class HTTPClients` — holds one `httpx.AsyncClient` per vendor:
    - `.groq: httpx.AsyncClient`
    - `.tts: httpx.AsyncClient` (points at Cartesia or Deepgram base URL depending on `config.tts_provider` — Cartesia TTS itself is used over WebSocket per the PRD, but this REST client is for any non-streaming Cartesia/Deepgram REST calls later phases need)
    - `.kite: httpx.AsyncClient`
    - `.tavily: httpx.AsyncClient`
  - `async def create_http_clients(config: Config) -> HTTPClients` — constructs all four clients once.
  - `async def close_http_clients(clients: HTTPClients) -> None` — calls `.aclose()` on all four, used at shutdown.
  - Usable as an async context manager: `async with create_http_clients(config) as clients:` is NOT required — later phases call `create_http_clients` at startup and `close_http_clients` at shutdown explicitly, matching the PRD's "created once at startup" requirement.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_http_clients.py
import httpx
import pytest

from financial_voice_agent.config import Config
from financial_voice_agent.http_clients import HTTPClients, close_http_clients, create_http_clients


def _make_config(tts_provider="cartesia") -> Config:
    return Config(
        vad_speech_threshold=0.5,
        vad_silence_duration_ms=600,
        vad_min_speech_duration_ms=200,
        audio_output_device_index=None,
        input_mode="always_on",
        tts_provider=tts_provider,
        llm_model="test-model",
        storage_db_path="./agent_turns.db",
        mode="live",
        groq_api_key="groq-secret",
        cartesia_api_key="cartesia-secret",
        deepgram_api_key="deepgram-secret",
        kite_api_key="kite-key",
        kite_access_token="kite-token",
        tavily_api_key="tavily-key",
    )


@pytest.mark.asyncio
async def test_create_http_clients_returns_one_client_per_vendor():
    config = _make_config()

    clients = await create_http_clients(config)
    try:
        assert isinstance(clients, HTTPClients)
        assert isinstance(clients.groq, httpx.AsyncClient)
        assert isinstance(clients.tts, httpx.AsyncClient)
        assert isinstance(clients.kite, httpx.AsyncClient)
        assert isinstance(clients.tavily, httpx.AsyncClient)
        # Each vendor gets a distinct client instance (no accidental sharing).
        assert len({id(clients.groq), id(clients.tts), id(clients.kite), id(clients.tavily)}) == 4
    finally:
        await close_http_clients(clients)


@pytest.mark.asyncio
async def test_groq_client_carries_auth_header():
    config = _make_config()

    clients = await create_http_clients(config)
    try:
        assert clients.groq.headers["Authorization"] == "Bearer groq-secret"
    finally:
        await close_http_clients(clients)


@pytest.mark.asyncio
async def test_tts_client_uses_cartesia_key_when_provider_is_cartesia():
    config = _make_config(tts_provider="cartesia")

    clients = await create_http_clients(config)
    try:
        assert clients.tts.headers["X-API-Key"] == "cartesia-secret"
    finally:
        await close_http_clients(clients)


@pytest.mark.asyncio
async def test_tts_client_uses_deepgram_key_when_provider_is_deepgram():
    config = _make_config(tts_provider="deepgram")

    clients = await create_http_clients(config)
    try:
        assert clients.tts.headers["Authorization"] == "Token deepgram-secret"
    finally:
        await close_http_clients(clients)


@pytest.mark.asyncio
async def test_close_http_clients_closes_all():
    config = _make_config()
    clients = await create_http_clients(config)

    await close_http_clients(clients)

    assert clients.groq.is_closed
    assert clients.tts.is_closed
    assert clients.kite.is_closed
    assert clients.tavily.is_closed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_http_clients.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.http_clients'` (or a `pytest-asyncio` "async def functions are not natively supported" error if the plugin/config is missing — see Step 3b).

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/http_clients.py
from __future__ import annotations

from dataclasses import dataclass

import httpx

from financial_voice_agent.config import Config

GROQ_BASE_URL = "https://api.groq.com"
CARTESIA_BASE_URL = "https://api.cartesia.ai"
DEEPGRAM_BASE_URL = "https://api.deepgram.com"
KITE_BASE_URL = "https://api.kite.trade"
TAVILY_BASE_URL = "https://api.tavily.com"


@dataclass
class HTTPClients:
    groq: httpx.AsyncClient
    tts: httpx.AsyncClient
    kite: httpx.AsyncClient
    tavily: httpx.AsyncClient


async def create_http_clients(config: Config) -> HTTPClients:
    groq = httpx.AsyncClient(
        base_url=GROQ_BASE_URL,
        headers={"Authorization": f"Bearer {config.groq_api_key}"},
    )

    if config.tts_provider == "cartesia":
        tts = httpx.AsyncClient(
            base_url=CARTESIA_BASE_URL,
            headers={"X-API-Key": config.cartesia_api_key or ""},
        )
    else:
        tts = httpx.AsyncClient(
            base_url=DEEPGRAM_BASE_URL,
            headers={"Authorization": f"Token {config.deepgram_api_key or ''}"},
        )

    kite = httpx.AsyncClient(
        base_url=KITE_BASE_URL,
        headers={"Authorization": f"token {config.kite_api_key}:{config.kite_access_token}"},
    )

    tavily = httpx.AsyncClient(base_url=TAVILY_BASE_URL)

    return HTTPClients(groq=groq, tts=tts, kite=kite, tavily=tavily)


async def close_http_clients(clients: HTTPClients) -> None:
    await clients.groq.aclose()
    await clients.tts.aclose()
    await clients.kite.aclose()
    await clients.tavily.aclose()
```

- [ ] **Step 3b: Enable pytest-asyncio**

Create or update `pytest.ini` at repo root:

```ini
[pytest]
asyncio_mode = auto
```

Create or update `requirements.txt`:

```
pyyaml
python-dotenv
httpx
pytest
pytest-asyncio
```

Run: `pip install -r requirements.txt`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_http_clients.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the full Phase 1 suite**

Run: `pytest tests/ -v`
Expected: PASS (all tests from Tasks 1-4, 15 tests total).

- [ ] **Step 6: Commit**

```bash
git add financial_voice_agent/http_clients.py tests/test_http_clients.py pytest.ini requirements.txt
git commit -m "feat: add reused per-vendor HTTP client manager"
```

---

## Phase 1 Exit Criteria

- `pytest tests/ -v` passes with 0 failures.
- `python -c "from financial_voice_agent.config import load_config; print(load_config())"` fails cleanly with a `ConfigError` naming the missing key when `.env` is absent (manual smoke check — requires a real `config.yaml`, already created in Task 1).
- `fixtures/quote.json`, `fixtures/ohlc_history.json`, `fixtures/positions_holdings.json` exist and are loadable via `load_fixture`.
- No module in `financial_voice_agent/` imports `pyaudio`, `groq`, `cartesia`, or any audio/LLM SDK yet — Phase 1 is pure plumbing.

---

## Upcoming Phases (summaries — to be written in full detail after Phase 1 review)

**Phase 2 — Audio Pipeline:** PyAudio capture callback on an OS thread → `janus` thread-safe queue → asyncio consumer → `noisereduce` noise suppression → Silero VAD gating → buffer-until-end-of-speech → WAV encoding via stdlib `wave`. Covers the PRD's Section 11.2 threading gotcha (plain `asyncio.Queue` is unsafe across the callback thread boundary) and the exact 16-bit/mono/16000Hz/512-or-1024-chunk data contract from Section 10.

**Phase 3 — Turn Orchestrator Core:** Wires Phase 1 (config, db, http_clients) and Phase 2 (VAD-gated WAV) into the full turn loop: Groq Whisper STT → Groq LLM tool-calling loop (`asyncio.gather` for concurrent tool calls) → Cartesia WebSocket TTS → PyAudio playback, with barge-in (`asyncio.Event` set by VAD during playback, checked between PCM chunks) and `latency_*_ms` timing captured at each stage for `log_turn`. Tool calls themselves are stubbed/mocked in this phase; Phase 4 implements them for real.

**Phase 4 — Tools:** Implements `get_quote`, `get_ohlc_history`, `compute_indicator` (Bollinger/Fibonacci/MA/RSI via `ta`/`pandas`/`numpy`), `get_positions_holdings`, `get_news` (Tavily), `capture_screen` (mss + pygetwindow/Quartz), each following the PRD Section 6 tool table's exact error/retry behavior (401 → "session expired", 429 → backoff, mock mode reads Phase 1 fixtures). Registers all six with the Phase 3 LLM tool loop.

**Phase 5 — Eval Harness:** JSON test cases (PRD Section 17) with input transcript, optional mocked screen result, expected tool calls, and tools that must NOT be called. A runner asserts on tool names/args only (not exact wording), run against `mode: "mock"` end-to-end through Phases 1-4. Seeds the 8 starting cases from the PRD and provides the harness for adding a case every time real usage surfaces a wrong tool call.
