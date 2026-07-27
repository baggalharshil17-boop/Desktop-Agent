# v3.0 Onboarding Setup Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/setup.py`, an interactive wizard that lets a technical friend go from a fresh `git clone` to a running agent with their own API keys, validating each key live as they enter it.

**Architecture:** The wizard's actual logic (env file merging, config.yaml generation, per-vendor key validation) lives in small, pure/testable modules under `financial_voice_agent/setup/`. `scripts/setup.py` itself is a thin interactive shell (prompts, `getpass`, print statements) that calls into those modules — it is not unit tested directly, matching this project's existing pattern for `scripts/kite_login.py`. A new `README.md` documents the whole fresh-clone flow.

**Tech Stack:** Python 3.14 stdlib (`getpass`, `dataclasses`), `python-dotenv` (already a dependency, via `dotenv_values`), `groq`, `huggingface_hub`, `cartesia`, `httpx` (all already dependencies) for live key validation.

## Global Constraints

- Windows only — do not add any OS-detection branching or cross-platform code paths.
- No new third-party dependencies. Every library used here (`groq`, `huggingface_hub`, `cartesia`, `httpx`, `python-dotenv`) is already in `requirements.txt`.
- `scripts/setup.py` must never overwrite an existing `.env`/`config.yaml` without asking first.
- A failed live key validation must show the real error and let the user retry-this-key or skip-and-fix-later — never crash the whole wizard.
- Match this codebase's existing dependency-injection testing pattern (see `financial_voice_agent/orchestrator/stt.py`'s `make_stt_client`, `financial_voice_agent/orchestrator/llm.py`'s `make_llm_client` for the style — an injectable factory/client parameter with a real default).
- No Kite key collection in this wizard — `scripts/kite_login.py` already owns that flow; this wizard only points to it.

---

### Task 1: `.env` file read/write/merge helpers

**Files:**
- Create: `financial_voice_agent/setup/__init__.py` (empty)
- Create: `financial_voice_agent/setup/env_file.py`
- Test: `tests/setup/__init__.py` (empty)
- Test: `tests/setup/test_env_file.py`

**Interfaces:**
- Produces: `read_env_file(path: str) -> dict[str, str]`, `merge_env_values(existing: dict[str, str], new_values: dict[str, str], *, overwrite: bool) -> dict[str, str]`, `write_env_file(path: str, values: dict[str, str]) -> None`. Later tasks (Task 4, `scripts/setup.py`) call all three by these exact names.

- [ ] **Step 1: Write the failing tests**

Create `tests/setup/__init__.py` (empty file).

Create `tests/setup/test_env_file.py`:

```python
from financial_voice_agent.setup.env_file import (
    merge_env_values,
    read_env_file,
    write_env_file,
)


def test_read_env_file_returns_empty_dict_when_file_missing(tmp_path):
    result = read_env_file(str(tmp_path / "does_not_exist.env"))

    assert result == {}


def test_read_env_file_parses_key_value_pairs(tmp_path):
    path = tmp_path / ".env"
    path.write_text("GROQ_API_KEY=abc123\nCARTESIA_API_KEY=xyz789\n")

    result = read_env_file(str(path))

    assert result == {"GROQ_API_KEY": "abc123", "CARTESIA_API_KEY": "xyz789"}


def test_merge_env_values_with_overwrite_true_prefers_new_values():
    existing = {"GROQ_API_KEY": "old", "TAVILY_API_KEY": "keep-me"}
    new_values = {"GROQ_API_KEY": "new"}

    result = merge_env_values(existing, new_values, overwrite=True)

    assert result == {"GROQ_API_KEY": "new", "TAVILY_API_KEY": "keep-me"}


def test_merge_env_values_with_overwrite_false_keeps_existing_values():
    # Re-running setup.py to add a previously-skipped key must not clobber
    # a value the user already has -- only fill in what's missing.
    existing = {"GROQ_API_KEY": "old"}
    new_values = {"GROQ_API_KEY": "new", "CARTESIA_API_KEY": "added"}

    result = merge_env_values(existing, new_values, overwrite=False)

    assert result == {"GROQ_API_KEY": "old", "CARTESIA_API_KEY": "added"}


def test_write_env_file_writes_sorted_key_value_lines(tmp_path):
    path = tmp_path / ".env"

    write_env_file(str(path), {"TAVILY_API_KEY": "b", "GROQ_API_KEY": "a"})

    # Sorted so re-running setup.py produces a stable diff, not a
    # randomly-reordered file every time.
    assert path.read_text() == "GROQ_API_KEY=a\nTAVILY_API_KEY=b\n"


def test_write_env_file_round_trips_through_read_env_file(tmp_path):
    path = tmp_path / ".env"
    values = {"GROQ_API_KEY": "abc", "HF_TOKEN": "def"}

    write_env_file(str(path), values)
    result = read_env_file(str(path))

    assert result == values
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/setup/test_env_file.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.setup'`

- [ ] **Step 3: Write the implementation**

Create `financial_voice_agent/setup/__init__.py` (empty file).

Create `financial_voice_agent/setup/env_file.py`:

```python
from __future__ import annotations

import os

from dotenv import dotenv_values


def read_env_file(path: str) -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    return dict(dotenv_values(path))


def merge_env_values(
    existing: dict[str, str], new_values: dict[str, str], *, overwrite: bool
) -> dict[str, str]:
    if overwrite:
        return {**existing, **new_values}
    return {**new_values, **existing}


def write_env_file(path: str, values: dict[str, str]) -> None:
    lines = [f"{key}={values[key]}" for key in sorted(values)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" if lines else "")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/setup/test_env_file.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/setup/__init__.py financial_voice_agent/setup/env_file.py tests/setup/__init__.py tests/setup/test_env_file.py
git commit -m "Add .env read/write/merge helpers for the setup wizard"
```

---

### Task 2: `config.yaml` template generator

**Files:**
- Create: `financial_voice_agent/setup/config_template.py`
- Test: `tests/setup/test_config_template.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `render_config_yaml(*, stt_provider: str, stt_model: str, llm_provider: str, llm_model: str, cartesia_voice_id: str, mode: str) -> str`. Task 4 calls this by this exact name/signature and writes the returned string to `config.yaml`.

**Context:** This intentionally does NOT read or patch the existing `config.yaml` — it always renders a complete fresh file from the template below, matching the real structure/comments in the current `config.yaml` at the repo root (read that file if you need to double check current field names before implementing — `vad.*`, `audio.*`, `input_mode`, `tts.*`, `stt.*`, `llm.*`, `storage.*`, `mode`). This plan hardcodes sane defaults for `vad`/`audio`/`storage` fields (matching the current checked-in `config.yaml`) since the wizard does not prompt for those — only for provider/model/voice/mode choices.

- [ ] **Step 1: Write the failing test**

Create `tests/setup/test_config_template.py`:

```python
import yaml

from financial_voice_agent.setup.config_template import render_config_yaml


def test_render_config_yaml_produces_valid_yaml_with_chosen_values():
    text = render_config_yaml(
        stt_provider="groq",
        stt_model="whisper-large-v3-turbo",
        llm_provider="groq",
        llm_model="qwen/qwen3.6-27b",
        cartesia_voice_id="db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
        mode="mock",
    )

    parsed = yaml.safe_load(text)

    assert parsed["stt"]["provider"] == "groq"
    assert parsed["stt"]["model"] == "whisper-large-v3-turbo"
    assert parsed["llm"]["provider"] == "groq"
    assert parsed["llm"]["model"] == "qwen/qwen3.6-27b"
    assert parsed["tts"]["provider"] == "cartesia"
    assert parsed["tts"]["voice_id"] == "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"
    assert parsed["mode"] == "mock"


def test_render_config_yaml_includes_required_top_level_sections():
    # These must be present with sane defaults since the wizard doesn't
    # prompt for them -- financial_voice_agent.config.load_config requires
    # all of these keys to exist or it raises ConfigError.
    text = render_config_yaml(
        stt_provider="huggingface",
        stt_model="openai/whisper-large-v3-turbo",
        llm_provider="huggingface",
        llm_model="qwen/qwen3.6-27b",
        cartesia_voice_id="some-voice-id",
        mode="live",
    )

    parsed = yaml.safe_load(text)

    assert "vad" in parsed
    assert set(parsed["vad"].keys()) >= {"speech_threshold", "silence_duration_ms", "min_speech_duration_ms"}
    assert "audio" in parsed
    assert parsed["audio"]["output_device_index"] is None
    assert "storage" in parsed
    assert parsed["storage"]["db_path"] == "./agent_turns.db"


def test_render_config_yaml_is_loadable_by_the_real_config_loader(tmp_path):
    # The strongest possible check: the app's own loader must accept this
    # output without raising, using real provider/env combinations.
    from financial_voice_agent.config import load_config

    text = render_config_yaml(
        stt_provider="groq",
        stt_model="whisper-large-v3-turbo",
        llm_provider="groq",
        llm_model="qwen/qwen3.6-27b",
        cartesia_voice_id="db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
        mode="mock",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(text)
    env_path = tmp_path / ".env"
    env_path.write_text("GROQ_API_KEY=x\nCARTESIA_API_KEY=y\n")

    config = load_config(config_path=str(config_path), env_path=str(env_path))

    assert config.stt_provider == "groq"
    assert config.llm_model == "qwen/qwen3.6-27b"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/setup/test_config_template.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.setup.config_template'`

- [ ] **Step 3: Write the implementation**

Create `financial_voice_agent/setup/config_template.py`:

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
  provider: "cartesia"  # or "deepgram" -- deepgram has no adapter yet
  voice_id: "{cartesia_voice_id}"

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


def render_config_yaml(
    *,
    stt_provider: str,
    stt_model: str,
    llm_provider: str,
    llm_model: str,
    cartesia_voice_id: str,
    mode: str,
) -> str:
    return _TEMPLATE.format(
        stt_provider=stt_provider,
        stt_model=stt_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
        cartesia_voice_id=cartesia_voice_id,
        mode=mode,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/setup/test_config_template.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/setup/config_template.py tests/setup/test_config_template.py
git commit -m "Add config.yaml template renderer for the setup wizard"
```

---

### Task 3: Vendor key validators

**Files:**
- Create: `financial_voice_agent/setup/validators.py`
- Test: `tests/setup/test_validators.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `@dataclass(frozen=True) class ValidationResult: ok: bool; message: str; data: dict | None = None`
  - `validate_groq_key(api_key: str, *, client_factory=groq.Groq) -> ValidationResult`
  - `validate_huggingface_key(token: str, *, whoami_fn=huggingface_hub.whoami) -> ValidationResult`
  - `validate_cartesia_key(api_key: str, *, client_factory=cartesia.Cartesia) -> ValidationResult` — on success, `data["voices"]` is a `list[tuple[str, str]]` of `(voice_id, voice_name)`.
  - `validate_tavily_key(api_key: str, *, http_client=None) -> ValidationResult` — `http_client` defaults to a fresh `httpx.Client(base_url="https://api.tavily.com")` if not given.

  Task 4 (`scripts/setup.py`) calls all four by these exact names and reads `.ok`/`.message`/`.data` off the returned `ValidationResult`.

**Context:** Verified against the installed SDKs at plan-writing time (do not re-guess these):
- `groq.Groq(api_key=...).models.list()` is a real, cheap (no inference cost) call; raises `groq.AuthenticationError` (subclass of `groq.GroqError`) on a bad key.
- `huggingface_hub.whoami(token=...)` raises `huggingface_hub.errors.HfHubHTTPError` on a bad token.
- `cartesia.Cartesia(api_key=...).voices.list(limit=...)` returns an iterable of `Voice` objects with `.id` and `.name` fields; raises `cartesia.AuthenticationError` on a bad key.
- Tavily has no Python SDK in this project (`financial_voice_agent/tools/news.py` calls it via raw `httpx` POST to `/search` with `api_key` in the JSON body) and no free "check my key" endpoint, so validation here is a real minimal search (`query="test", max_results=1`) — this costs one Tavily search credit (free tier is 1k/month, per the PRD's cost table), which is an accepted, documented cost, not a bug.

- [ ] **Step 1: Write the failing tests**

Create `tests/setup/test_validators.py`:

```python
import httpx
import pytest

from financial_voice_agent.setup.validators import (
    validate_cartesia_key,
    validate_groq_key,
    validate_huggingface_key,
    validate_tavily_key,
)


class _FakeGroqModels:
    def __init__(self, *, should_raise=False):
        self._should_raise = should_raise

    def list(self):
        if self._should_raise:
            import groq

            raise groq.AuthenticationError("bad key", response=None, body=None)
        return ["model-a", "model-b"]


class _FakeGroqClient:
    def __init__(self, *, api_key, should_raise=False):
        self.models = _FakeGroqModels(should_raise=should_raise)


def test_validate_groq_key_ok_on_successful_list_call():
    result = validate_groq_key(
        "test-key", client_factory=lambda api_key: _FakeGroqClient(api_key=api_key)
    )

    assert result.ok is True


def test_validate_groq_key_reports_failure_on_bad_key():
    result = validate_groq_key(
        "bad-key",
        client_factory=lambda api_key: _FakeGroqClient(api_key=api_key, should_raise=True),
    )

    assert result.ok is False
    assert "bad key" in result.message.lower() or "authentication" in result.message.lower()


def test_validate_huggingface_key_ok_on_successful_whoami():
    result = validate_huggingface_key("test-token", whoami_fn=lambda token: {"name": "someone"})

    assert result.ok is True


def test_validate_huggingface_key_reports_failure_on_bad_token():
    import huggingface_hub.errors

    def failing_whoami(token):
        raise huggingface_hub.errors.HfHubHTTPError("bad token")

    result = validate_huggingface_key("bad-token", whoami_fn=failing_whoami)

    assert result.ok is False


class _FakeVoice:
    def __init__(self, id, name):
        self.id = id
        self.name = name


class _FakeCartesiaVoices:
    def __init__(self, *, should_raise=False):
        self._should_raise = should_raise

    def list(self, *, limit):
        if self._should_raise:
            import cartesia

            raise cartesia.AuthenticationError("bad key", response=None, body=None)
        return [_FakeVoice("voice-1", "Friendly Guide"), _FakeVoice("voice-2", "Calm Narrator")]


class _FakeCartesiaClient:
    def __init__(self, *, api_key, should_raise=False):
        self.voices = _FakeCartesiaVoices(should_raise=should_raise)


def test_validate_cartesia_key_ok_and_returns_voices_on_success():
    result = validate_cartesia_key(
        "test-key", client_factory=lambda api_key: _FakeCartesiaClient(api_key=api_key)
    )

    assert result.ok is True
    assert result.data["voices"] == [("voice-1", "Friendly Guide"), ("voice-2", "Calm Narrator")]


def test_validate_cartesia_key_reports_failure_on_bad_key():
    result = validate_cartesia_key(
        "bad-key",
        client_factory=lambda api_key: _FakeCartesiaClient(api_key=api_key, should_raise=True),
    )

    assert result.ok is False


@pytest.mark.asyncio
async def test_validate_tavily_key_ok_on_200_response():
    def handler(request):
        return httpx.Response(200, json={"results": []})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://api.tavily.com", transport=transport)

    result = validate_tavily_key("test-key", http_client=client)

    assert result.ok is True


@pytest.mark.asyncio
async def test_validate_tavily_key_reports_failure_on_401():
    def handler(request):
        return httpx.Response(401, json={"error": "invalid api key"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://api.tavily.com", transport=transport)

    result = validate_tavily_key("bad-key", http_client=client)

    assert result.ok is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/setup/test_validators.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.setup.validators'`

- [ ] **Step 3: Write the implementation**

Create `financial_voice_agent/setup/validators.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import cartesia
import groq
import httpx
import huggingface_hub
import huggingface_hub.errors


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    message: str
    data: dict | None = None


def validate_groq_key(api_key: str, *, client_factory=groq.Groq) -> ValidationResult:
    try:
        client = client_factory(api_key=api_key)
        client.models.list()
    except groq.GroqError as exc:
        return ValidationResult(ok=False, message=f"Groq key rejected: {exc}")
    except Exception as exc:  # noqa: BLE001 -- any other failure (network, etc.) is still "not validated"
        return ValidationResult(ok=False, message=f"Could not reach Groq: {exc}")
    return ValidationResult(ok=True, message="Groq key OK")


def validate_huggingface_key(token: str, *, whoami_fn=huggingface_hub.whoami) -> ValidationResult:
    try:
        whoami_fn(token=token)
    except huggingface_hub.errors.HfHubHTTPError as exc:
        return ValidationResult(ok=False, message=f"Hugging Face token rejected: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(ok=False, message=f"Could not reach Hugging Face: {exc}")
    return ValidationResult(ok=True, message="Hugging Face token OK")


def validate_cartesia_key(api_key: str, *, client_factory=cartesia.Cartesia) -> ValidationResult:
    try:
        client = client_factory(api_key=api_key)
        voices = [(v.id, v.name) for v in client.voices.list(limit=50)]
    except cartesia.CartesiaError as exc:
        return ValidationResult(ok=False, message=f"Cartesia key rejected: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(ok=False, message=f"Could not reach Cartesia: {exc}")
    return ValidationResult(ok=True, message="Cartesia key OK", data={"voices": voices})


def validate_tavily_key(api_key: str, *, http_client: httpx.Client | None = None) -> ValidationResult:
    owns_client = http_client is None
    client = http_client or httpx.Client(base_url="https://api.tavily.com", timeout=15.0)
    try:
        response = client.post(
            "/search", json={"api_key": api_key, "query": "test", "max_results": 1}
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return ValidationResult(ok=False, message=f"Tavily key rejected: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(ok=False, message=f"Could not reach Tavily: {exc}")
    finally:
        if owns_client:
            client.close()
    return ValidationResult(ok=True, message="Tavily key OK")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/setup/test_validators.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/setup/validators.py tests/setup/test_validators.py
git commit -m "Add live vendor key validators for the setup wizard"
```

---

### Task 4: `scripts/setup.py` interactive wizard

**Files:**
- Create: `scripts/setup.py`

**Interfaces:**
- Consumes: `financial_voice_agent.setup.env_file.{read_env_file, merge_env_values, write_env_file}` (Task 1), `financial_voice_agent.setup.config_template.render_config_yaml` (Task 2), `financial_voice_agent.setup.validators.{ValidationResult, validate_groq_key, validate_huggingface_key, validate_cartesia_key, validate_tavily_key}` (Task 3).
- Produces: nothing consumed by other tasks — this is the final interactive entry point, run directly as `python scripts/setup.py`.

**Context:** This file is intentionally NOT unit tested (per the approved spec: "not literally scripting interactive prompts in CI") — its only job is orchestrating already-tested logic with `input()`/`getpass()` calls. Keep it as a thin, readable sequence; all actual decision logic (merging, rendering, validating) lives in the tested modules from Tasks 1-3. It is manually verified in Task 5 by actually running it.

Mirror the existing `scripts/kite_login.py` for style (docstring explaining what it's for and how to run it, `ENV_PATH`/`CONFIG_PATH` computed relative to the repo root via `pathlib.Path(__file__).resolve().parent.parent`, a `main()` function, `if __name__ == "__main__": main()`).

- [ ] **Step 1: Write the script**

Create `scripts/setup.py`:

```python
"""Interactive first-run setup wizard.

Walks through choosing providers (STT/LLM/TTS), collecting the API keys
those choices need, validating each one live, and writing .env and
config.yaml. Run once after cloning:

    python scripts/setup.py

Does not collect Kite credentials -- that's a separate OAuth-style browser
flow, see scripts/kite_login.py. If you don't want to deal with a live
Zerodha account yet, choose mode "mock" when prompted below; the agent will
use canned fixture data for Kite-backed tools instead.
"""

from __future__ import annotations

import getpass
from pathlib import Path

from financial_voice_agent.setup.config_template import render_config_yaml
from financial_voice_agent.setup.env_file import merge_env_values, read_env_file, write_env_file
from financial_voice_agent.setup.validators import (
    ValidationResult,
    validate_cartesia_key,
    validate_groq_key,
    validate_huggingface_key,
    validate_tavily_key,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
CONFIG_PATH = REPO_ROOT / "config.yaml"


def _ask_choice(prompt: str, options: list[str], default: str) -> str:
    options_str = "/".join(o if o != default else o.upper() for o in options)
    while True:
        raw = input(f"{prompt} [{options_str}]: ").strip().lower()
        if not raw:
            return default
        if raw in options:
            return raw
        print(f"  Please enter one of: {', '.join(options)}")


def _ask_yes_no(prompt: str, *, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def _collect_and_validate(
    env_var: str, prompt: str, validate_fn
) -> tuple[str, ValidationResult]:
    while True:
        value = getpass.getpass(f"{prompt} ({env_var}): ").strip()
        if not value:
            print("  Skipped -- you can add this later by re-running this script or editing .env directly.")
            return "", ValidationResult(ok=False, message="skipped by user")
        print("  Checking...")
        result = validate_fn(value)
        if result.ok:
            print(f"  OK: {result.message}")
            return value, result
        print(f"  Failed: {result.message}")
        if not _ask_yes_no("  Try this key again?", default=True):
            return value, result


def main() -> None:
    print("Financial Voice Agent -- setup wizard\n")

    existing_env = read_env_file(str(ENV_PATH))
    overwrite_env = True
    if existing_env:
        overwrite_env = _ask_yes_no(
            "Found an existing .env. Overwrite everything (choose No to only fill in what's missing)?",
            default=False,
        )

    config_exists = CONFIG_PATH.exists()
    write_config = True
    if config_exists:
        write_config = _ask_yes_no(
            "Found an existing config.yaml. Regenerate it from your choices below?", default=False
        )

    new_env: dict[str, str] = {}
    cartesia_voice_id = ""

    stt_provider = _ask_choice("STT provider", ["groq", "huggingface"], default="groq")
    llm_provider = _ask_choice("LLM provider", ["groq", "huggingface"], default="groq")
    print(
        "  Note: whichever LLM model you use must support BOTH tool calling and vision "
        "for capture_screen results to actually be described, not just confirmed."
    )
    mode = _ask_choice(
        "Mode -- 'mock' uses canned data and needs no Zerodha account; 'live' needs Kite Connect set up separately",
        ["mock", "live"],
        default="mock",
    )

    if stt_provider == "groq" or llm_provider == "groq":
        value, _ = _collect_and_validate(
            "GROQ_API_KEY", "Groq API key (console.groq.com)", validate_groq_key
        )
        if value:
            new_env["GROQ_API_KEY"] = value

    if stt_provider == "huggingface" or llm_provider == "huggingface":
        value, _ = _collect_and_validate(
            "HF_TOKEN", "Hugging Face token (huggingface.co/settings/tokens)", validate_huggingface_key
        )
        if value:
            new_env["HF_TOKEN"] = value

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

    tavily_value, _ = _collect_and_validate(
        "TAVILY_API_KEY", "Tavily API key (tavily.com) -- for news search", validate_tavily_key
    )
    if tavily_value:
        new_env["TAVILY_API_KEY"] = tavily_value

    merged_env = merge_env_values(existing_env, new_env, overwrite=overwrite_env)
    write_env_file(str(ENV_PATH), merged_env)
    print(f"\nWrote {ENV_PATH}")

    if write_config:
        stt_model = "whisper-large-v3-turbo" if stt_provider == "groq" else "openai/whisper-large-v3-turbo"
        llm_model = "qwen/qwen3.6-27b"
        config_text = render_config_yaml(
            stt_provider=stt_provider,
            stt_model=stt_model,
            llm_provider=llm_provider,
            llm_model=llm_model,
            cartesia_voice_id=cartesia_voice_id or "<pick a voice id from https://play.cartesia.ai/voices>",
            mode=mode,
        )
        CONFIG_PATH.write_text(config_text)
        print(f"Wrote {CONFIG_PATH}")

    print("\nNext steps:")
    if mode == "live":
        print("  1. python scripts/kite_login.py   (get today's Kite access token)")
        print("  2. python -m financial_voice_agent")
    else:
        print("  1. python -m financial_voice_agent")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script imports cleanly**

Run: `python -c "import ast; ast.parse(open('scripts/setup.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/setup.py
git commit -m "Add interactive setup wizard (scripts/setup.py)"
```

---

### Task 5: `README.md` and manual verification

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: nothing (documentation only).

- [ ] **Step 1: Write `README.md`**

Create `README.md` at the repo root:

```markdown
# Financial Voice Agent

A read-only AI voice assistant for a Zerodha Kite trading dashboard. See
`Financial_Voice_Agent.claude.md` for the full design/developer reference.

## Getting Started (Fresh Clone)

### 1. Prerequisites

- Python 3.11+ (developed against 3.14).
- `pip install -r requirements.txt` installs everything, but two packages
  need a callout on Windows:
  - **torch** (used for voice-activity detection): if you don't have or
    want a GPU, install the smaller CPU-only build first, or plain
    `pip install torch` may pull a multi-GB CUDA build:
    ```
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```
  - **pyaudio** (microphone/speaker access): sometimes fails to build from
    source on Windows. If `pip install -r requirements.txt` fails on it
    specifically, install a prebuilt wheel instead:
    ```
    pip install pipwin
    pipwin install pyaudio
    ```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the setup wizard

```
python scripts/setup.py
```

Walks you through picking providers and pasting in your own API keys,
validating each one live as you go. Writes `.env` and `config.yaml` for
you -- see the table below for what each provider needs.

### 4. Run it

```
python -m financial_voice_agent
```

Talk after it prints "Listening...". Ctrl+C to stop.

If you chose `mode: "live"` during setup, get today's Kite access token
first (it expires daily):
```
python scripts/kite_login.py
```

## What each provider needs

| Provider | Env var | Get it from | Notes |
|---|---|---|---|
| Groq (STT and/or LLM) | `GROQ_API_KEY` | console.groq.com | Free tier available |
| Hugging Face (STT and/or LLM alternative) | `HF_TOKEN` | huggingface.co/settings/tokens | Free tier available, separate quota from Groq |
| Cartesia (TTS) | `CARTESIA_API_KEY` | play.cartesia.ai | Credit-based free tier |
| Tavily (news search) | `TAVILY_API_KEY` | tavily.com | Free to 1k searches/month |
| Zerodha Kite Connect (live trading data) | `KITE_API_KEY` / `KITE_API_SECRET` / `KITE_ACCESS_TOKEN` | developers.kite.trade | Paid (~₹500/month), only needed for `mode: "live"` -- skip entirely with `mode: "mock"` |

## Development

Run tests: `python -m pytest -q`
```

- [ ] **Step 2: Manually run the setup wizard end-to-end**

Run: `python scripts/setup.py`

Walk through it once for real: choose providers, paste in a real (or intentionally wrong, to confirm the retry path) key for at least one vendor, confirm the printed validation result matches what you'd expect, and confirm `.env` and `config.yaml` get written. This is the manual verification called for in the spec's Testing section -- every other feature in this project was confirmed against real vendor APIs before being considered done, and this wizard's whole job is calling those same real APIs.

If anything about the actual live prompts/output doesn't match what's described in `README.md`, fix the README to match reality before moving on.

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS, all tests (including the 17 new ones from Tasks 1-3) plus everything pre-existing.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Add README with fresh-clone setup instructions"
```
