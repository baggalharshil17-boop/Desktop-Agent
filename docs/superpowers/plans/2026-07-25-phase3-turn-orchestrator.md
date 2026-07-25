# Financial Voice Agent — Phase 3: Turn Orchestrator Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the actual conversation loop: microphone utterance (Phase 2) → Groq Whisper STT → Groq LLM tool-calling loop → Cartesia TTS → PyAudio playback, with barge-in and full turn logging (Phase 1), wired together end to end. Tools are stubbed in this phase (real tools are Phase 4) — the loop mechanics, retry policy, and data contracts are what this phase proves out.

**Architecture:** Five modules under `financial_voice_agent/orchestrator/`. Every module that talks to a real vendor (Groq, Cartesia) is split into (a) pure, fully-tested business logic — retry policy, tool-loop mechanics, turn timing — that depends only on a small `Protocol` interface, and (b) a thin adapter class implementing that Protocol against the real SDK. This isolates the one thing genuinely uncertain here (exact current vendor SDK call shapes, which both the PRD and this plan explicitly flag as "verify at build time") from everything else, which is fully deterministic and unit-testable with fakes — the same pattern Phase 2 used for PyAudio and Silero. Task 1 builds and tests the whole skeleton against fake stand-ins for STT/LLM/TTS, so it runs and proves itself out at zero cost before any real API key is needed.

**Tech Stack:** Python 3.11+, `groq` (Groq SDK), `cartesia` (Cartesia SDK), `pyaudio` (already installed from Phase 2), `pytest`, `pytest-asyncio` (already configured).

## Global Constraints

- Python 3.11+ only (PRD Section 9).
- **TTS/playback data contract is exact and non-negotiable** (PRD Section 10.2): Cartesia `output_format` must be `{"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000}`, and the PyAudio *output* stream must be opened at the same 16000 Hz / 16-bit / mono. A mismatch doesn't error — it produces "chipmunk voice" or "slow-motion voice" silently.
- Send the LLM's complete response text to Cartesia in one call after generation finishes — no token-level streaming into TTS in this phase (PRD Section 10.2, explicitly deferred).
- **Concurrency model** (PRD Section 11.1): the turn orchestrator runs as an asyncio task; independent tool calls within one turn run concurrently via `asyncio.gather` (mechanics built now, even though Phase 3's only tool is a stub — Phase 4 plugs in real tools without touching this loop). Playback must be interruptible mid-stream on a barge-in signal (Section 11.3), reusing Phase 2's `AudioPipeline.speech_active` event — per that event's documented liveness contract, `AudioPipeline.run()` must be driven from a dedicated background task, never blocked on inline.
- **Session/context window** (PRD Section 12): keep a rolling window of the last 6–8 turns as plain `{role, content}` messages. Never resend raw `tool_calls_json`/`tool_results_json` from prior turns into the LLM context — only the natural-language outcome (`response_text`).
- **Turn logging is not optional or deferred** (PRD Section 13, Section 2.2's whole justification for cascade-over-S2S): every turn — success or failure — calls `db.log_turn()` with the full `latency_stt_ms`/`latency_llm_ms`/`latency_tool_ms`/`latency_tts_ms`/`latency_total_ms` breakdown. This is built into Task 1's skeleton, not added later.
- **Per-vendor retry policy is exact** (PRD Section 15.1):
  - Groq STT / LLM, timeout or 5xx: retry twice with 1s backoff (3 total attempts); on final failure, speak "I didn't catch that, one moment" and re-listen — never crash.
  - Groq LLM, 429 rate limit: exponential backoff; speak "I've hit a usage limit — give me a second".
  - Cartesia TTS, timeout or 5xx: retry once (2 total attempts); on repeated failure, fall back to Deepgram Aura-2 if configured, else fail with a logged (not spoken) error.
- Treat the Groq model string as config, not a constant (already true — `Config.llm_model`, Phase 1).
- Reuse the one `httpx.AsyncClient` per vendor from Phase 1's `http_clients.py` wherever the vendor SDK accepts an injected HTTP client; otherwise reuse the vendor SDK's own client instance across the process lifetime, never construct one per call.
- **Vendor SDK call shapes in this plan are written from current documentation research, not guaranteed exact** — every adapter class below is explicitly flagged for verification against `console.groq.com/docs/api-reference` and Cartesia's current Python SDK docs at build time (PRD Section 2.3's model-churn warning applies to SDK surfaces generally, not just model names). This uncertainty is isolated to the adapter classes only — the retry policy, tool-loop mechanics, and turn timing they sit inside of are fully specified and fully tested independent of that uncertainty.
- **Every function that retries (`transcribe_with_retry`, `run_llm_turn`, `synthesize_with_fallback`) must accept and forward a `sleep_fn` parameter defaulting to `asyncio.sleep`, rather than relying on tests to monkeypatch `asyncio.sleep` after the fact.** Python default-argument values are evaluated once, at function-definition time — `def f(sleep_fn=asyncio.sleep)` captures a direct reference to the function object that existed at import time, so `monkeypatch.setattr("module.asyncio.sleep", fake)` executed later in a test has no effect on that already-bound default. Threading `sleep_fn` explicitly is what makes retry-timing tests both correct and fast (no real multi-second sleeps in the suite).

---

## File Structure

```
financial_voice_agent/
    orchestrator/
        __init__.py
        retry.py         # retry_with_fixed_backoff(), retry_with_exponential_backoff()
        turn.py           # TurnResult, run_turn(), update_history()
        stubs.py          # fake_stt(), fake_llm(), fake_tts() -- the pretend implementations
        stt.py            # transcribe_with_retry(), SttError, RealGroqSttClient adapter
        llm.py            # run_llm_turn(), LlmError, ToolCall, RealGroqLlmClient adapter
        tts.py            # synthesize_with_fallback(), TtsError, RealCartesiaTtsClient adapter
        playback.py       # AudioPlayback -- PyAudio output stream, barge-in interruptible
        main_loop.py      # run_voice_loop() -- ties Phase 2 capture/pipeline + this phase together
tests/
    orchestrator/
        __init__.py
        test_retry.py
        test_turn.py
        test_stt.py
        test_llm.py
        test_tts.py
        test_playback.py
        test_main_loop.py
```

- `retry.py`: owns all backoff/retry mechanics. Nothing else implements a sleep-and-retry loop.
- `turn.py`: owns one turn's shape and its logging. Nothing else calls `db.log_turn()`.
- `stt.py` / `llm.py` / `tts.py`: each owns one vendor boundary — a `Protocol`, the business logic against that Protocol, and a `Real*Client` adapter. Nothing else imports `groq` or `cartesia`.
- `playback.py`: owns the PyAudio *output* stream. Nothing else opens one.
- `main_loop.py`: owns wiring Phase 2's capture/pipeline to this phase's turn/playback. Nothing upstream of it knows about PyAudio at all.

---

### Task 1: Turn orchestrator skeleton (`turn.py`, `stubs.py`)

**Files:**
- Create: `financial_voice_agent/orchestrator/__init__.py` (empty)
- Create: `financial_voice_agent/orchestrator/turn.py`
- Create: `financial_voice_agent/orchestrator/stubs.py`
- Create: `tests/orchestrator/__init__.py` (empty)
- Test: `tests/orchestrator/test_turn.py`

**Interfaces:**
- Consumes: `financial_voice_agent.db.log_turn()` (Phase 1, exact signature already exists).
- Produces:
  - `@dataclass(frozen=True) class LlmTurnResult: response_text: str; tool_calls_json: str | None; tool_results_json: str | None` — the shape every later task's LLM call must return.
  - `@dataclass(frozen=True) class TurnResult: transcript: str | None; response_text: str | None; tts_audio: bytes | None; latency_stt_ms: int; latency_llm_ms: int; latency_tts_ms: int; latency_total_ms: int; error: str | None`
  - `async def run_turn(utterance_wav: bytes, history: list[dict], *, stt_fn, llm_fn, tts_fn, db_path: str) -> TurnResult` — `stt_fn: Callable[[bytes], Awaitable[str]]`, `llm_fn: Callable[[str, list[dict]], Awaitable[LlmTurnResult]]`, `tts_fn: Callable[[str], Awaitable[bytes]]`. Never raises — catches any stage failure, logs it, returns a `TurnResult` with `error` set.
  - `def update_history(history: list[dict], transcript: str, response_text: str, *, max_turns: int = 8) -> list[dict]` — appends the new user/assistant turn as plain `{role, content}` messages and trims to the last `max_turns` turns (i.e. `2 * max_turns` messages), per PRD Section 12.
  - `async def fake_stt(wav_bytes: bytes) -> str`, `async def fake_llm(transcript: str, history: list[dict]) -> LlmTurnResult`, `async def fake_tts(text: str) -> bytes` (in `stubs.py`) — the pretend implementations for running the whole loop at zero cost.

- [ ] **Step 1: Write the failing tests**

```python
# tests/orchestrator/test_turn.py
import sqlite3

import pytest

from financial_voice_agent.db import init_db
from financial_voice_agent.orchestrator.turn import LlmTurnResult, TurnResult, run_turn, update_history


def _fake_monotonic(values):
    it = iter(values)
    return lambda: next(it)


@pytest.mark.asyncio
async def test_run_turn_computes_latencies_and_logs_success(tmp_path, monkeypatch):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    # 8 time.monotonic() calls: turn_start, stt_start, stt_end, llm_start,
    # llm_end, tts_start, tts_end, total_end.
    monkeypatch.setattr(
        "financial_voice_agent.orchestrator.turn.time.monotonic",
        _fake_monotonic([0.0, 0.0, 0.1, 0.1, 0.35, 0.35, 0.40, 0.60]),
    )

    async def stt_fn(wav_bytes):
        return "hello"

    async def llm_fn(transcript, history):
        return LlmTurnResult(response_text="hi there", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return b"audio-bytes"

    result = await run_turn(
        b"wav-bytes", [], stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
    )

    assert result == TurnResult(
        transcript="hello",
        response_text="hi there",
        tts_audio=b"audio-bytes",
        latency_stt_ms=100,
        latency_llm_ms=250,
        latency_tts_ms=50,
        latency_total_ms=600,
        error=None,
    )

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT transcript, response_text, latency_stt_ms, latency_llm_ms, "
            "latency_tts_ms, latency_total_ms, error FROM turns"
        ).fetchone()
    finally:
        conn.close()
    assert row == ("hello", "hi there", 100, 250, 50, 600, None)


@pytest.mark.asyncio
async def test_run_turn_catches_stage_failure_and_logs_error_without_raising(tmp_path):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)

    async def stt_fn(wav_bytes):
        raise RuntimeError("groq is down")

    async def llm_fn(transcript, history):
        raise AssertionError("must not be called after stt_fn fails")

    async def tts_fn(text):
        raise AssertionError("must not be called after stt_fn fails")

    result = await run_turn(
        b"wav-bytes", [], stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
    )

    assert result.transcript is None
    assert result.response_text is None
    assert result.tts_audio is None
    assert result.error == "groq is down"

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT error FROM turns").fetchone()
    finally:
        conn.close()
    assert row == ("groq is down",)


@pytest.mark.asyncio
async def test_run_turn_with_real_stub_functions_end_to_end(tmp_path):
    from financial_voice_agent.orchestrator.stubs import fake_llm, fake_stt, fake_tts

    db_path = str(tmp_path / "turns.db")
    init_db(db_path)

    result = await run_turn(
        b"wav-bytes", [], stt_fn=fake_stt, llm_fn=fake_llm, tts_fn=fake_tts, db_path=db_path
    )

    assert isinstance(result.transcript, str) and result.transcript
    assert isinstance(result.response_text, str) and result.response_text
    assert isinstance(result.tts_audio, bytes)
    assert result.error is None


def test_update_history_appends_user_and_assistant_turns():
    history = update_history([], "what's the nifty level", "Nifty is at 24500")

    assert history == [
        {"role": "user", "content": "what's the nifty level"},
        {"role": "assistant", "content": "Nifty is at 24500"},
    ]


def test_update_history_trims_to_max_turns():
    history: list[dict] = []
    for i in range(10):
        history = update_history(history, f"question {i}", f"answer {i}", max_turns=8)

    assert len(history) == 16  # 8 turns * 2 messages
    assert history[0] == {"role": "user", "content": "question 2"}
    assert history[-1] == {"role": "assistant", "content": "answer 9"}


def test_update_history_does_not_mutate_input():
    original = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]

    result = update_history(original, "c", "d")

    assert len(original) == 2
    assert len(result) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/orchestrator/test_turn.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.orchestrator'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/orchestrator/turn.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from financial_voice_agent import db


@dataclass(frozen=True)
class LlmTurnResult:
    response_text: str
    tool_calls_json: str | None
    tool_results_json: str | None


@dataclass(frozen=True)
class TurnResult:
    transcript: str | None
    response_text: str | None
    tts_audio: bytes | None
    latency_stt_ms: int
    latency_llm_ms: int
    latency_tts_ms: int
    latency_total_ms: int
    error: str | None


async def run_turn(
    utterance_wav: bytes,
    history: list[dict],
    *,
    stt_fn: Callable[[bytes], Awaitable[str]],
    llm_fn: Callable[[str, list[dict]], Awaitable[LlmTurnResult]],
    tts_fn: Callable[[str], Awaitable[bytes]],
    db_path: str,
) -> TurnResult:
    turn_start = time.monotonic()
    transcript: str | None = None
    llm_result: LlmTurnResult | None = None
    tts_audio: bytes | None = None
    latency_stt_ms = 0
    latency_llm_ms = 0
    latency_tts_ms = 0
    error: str | None = None

    try:
        stt_start = time.monotonic()
        transcript = await stt_fn(utterance_wav)
        latency_stt_ms = int((time.monotonic() - stt_start) * 1000)

        llm_start = time.monotonic()
        llm_result = await llm_fn(transcript, history)
        latency_llm_ms = int((time.monotonic() - llm_start) * 1000)

        tts_start = time.monotonic()
        tts_audio = await tts_fn(llm_result.response_text)
        latency_tts_ms = int((time.monotonic() - tts_start) * 1000)
    except Exception as exc:  # noqa: BLE001 -- a turn must never crash the caller
        error = str(exc)

    latency_total_ms = int((time.monotonic() - turn_start) * 1000)

    db.log_turn(
        db_path,
        transcript=transcript,
        tool_calls_json=llm_result.tool_calls_json if llm_result else None,
        tool_results_json=llm_result.tool_results_json if llm_result else None,
        response_text=llm_result.response_text if llm_result else None,
        screenshot_path=None,
        latency_stt_ms=latency_stt_ms or None,
        latency_llm_ms=latency_llm_ms or None,
        latency_tool_ms=None,
        latency_tts_ms=latency_tts_ms or None,
        latency_total_ms=latency_total_ms,
        error=error,
    )

    return TurnResult(
        transcript=transcript,
        response_text=llm_result.response_text if llm_result else None,
        tts_audio=tts_audio,
        latency_stt_ms=latency_stt_ms,
        latency_llm_ms=latency_llm_ms,
        latency_tts_ms=latency_tts_ms,
        latency_total_ms=latency_total_ms,
        error=error,
    )


def update_history(
    history: list[dict], transcript: str, response_text: str, *, max_turns: int = 8
) -> list[dict]:
    updated = [
        *history,
        {"role": "user", "content": transcript},
        {"role": "assistant", "content": response_text},
    ]
    return updated[-(max_turns * 2):]
```

```python
# financial_voice_agent/orchestrator/stubs.py
from __future__ import annotations

import asyncio

from financial_voice_agent.orchestrator.turn import LlmTurnResult


async def fake_stt(wav_bytes: bytes) -> str:
    await asyncio.sleep(0)
    return "this is a fake transcript"


async def fake_llm(transcript: str, history: list[dict]) -> LlmTurnResult:
    await asyncio.sleep(0)
    return LlmTurnResult(
        response_text=f"You said: {transcript}",
        tool_calls_json=None,
        tool_results_json=None,
    )


async def fake_tts(text: str) -> bytes:
    await asyncio.sleep(0)
    return b""
```

Also create empty `financial_voice_agent/orchestrator/__init__.py` and empty `tests/orchestrator/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/orchestrator/test_turn.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/orchestrator/__init__.py financial_voice_agent/orchestrator/turn.py financial_voice_agent/orchestrator/stubs.py tests/orchestrator/__init__.py tests/orchestrator/test_turn.py
git commit -m "feat: add turn orchestrator skeleton with logging and fake stages"
```

---

### Task 2: Real Groq STT with retry policy (`retry.py`, `stt.py`)

**Files:**
- Create: `financial_voice_agent/orchestrator/retry.py`
- Create: `financial_voice_agent/orchestrator/stt.py`
- Test: `tests/orchestrator/test_retry.py`
- Test: `tests/orchestrator/test_stt.py`
- Modify: `requirements.txt` — add `groq`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `class RetryExhaustedError(Exception)` — `.last_exception: Exception | None` attribute.
  - `async def retry_with_fixed_backoff(fn, *, max_attempts=3, backoff_seconds=1.0, sleep_fn=asyncio.sleep) -> T` — `max_attempts` counts total attempts (1 initial + `max_attempts - 1` retries), sleeping `backoff_seconds` between each.
  - `async def retry_with_exponential_backoff(fn, *, max_attempts=4, base_delay_seconds=1.0, sleep_fn=asyncio.sleep) -> T` — sleeps `base_delay_seconds * 2**attempt` between attempts.
  - `class SttError(Exception)` — raised after retries are exhausted; the orchestrator (Task 5) catches this to speak "I didn't catch that, one moment" per PRD 15.1.
  - `class GroqSttClient(Protocol): async def transcribe(self, wav_bytes: bytes, *, model: str) -> str: ...`
  - `async def transcribe_with_retry(client: GroqSttClient, wav_bytes: bytes, *, model: str, sleep_fn=asyncio.sleep) -> str` — 3 total attempts, 1s fixed backoff, raises `SttError` on exhaustion. `sleep_fn` is exposed (not buried behind a default bound at import time) so tests can inject an instant fake — see the Global Constraints note on why `monkeypatch`ing `asyncio.sleep` after the fact does not work here: default argument values bind once, at function-definition time, not per call.
  - `class RealGroqSttClient` — adapter implementing `GroqSttClient` against the real `groq.AsyncGroq` client. **Verify against `console.groq.com/docs/api-reference` at build time** — this is the one piece of this task with real external-API uncertainty.

- [ ] **Step 1: Write the failing tests**

```python
# tests/orchestrator/test_retry.py
import pytest

from financial_voice_agent.orchestrator.retry import (
    RetryExhaustedError,
    retry_with_exponential_backoff,
    retry_with_fixed_backoff,
)


class _FakeSleeper:
    def __init__(self):
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.mark.asyncio
async def test_retry_with_fixed_backoff_returns_on_eventual_success():
    attempts = {"count": 0}

    async def fn():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("not yet")
        return "ok"

    sleeper = _FakeSleeper()
    result = await retry_with_fixed_backoff(fn, max_attempts=3, backoff_seconds=1.0, sleep_fn=sleeper)

    assert result == "ok"
    assert attempts["count"] == 3
    assert sleeper.calls == [1.0, 1.0]


@pytest.mark.asyncio
async def test_retry_with_fixed_backoff_raises_after_exhausting_attempts():
    async def fn():
        raise RuntimeError("always fails")

    sleeper = _FakeSleeper()
    with pytest.raises(RetryExhaustedError) as exc_info:
        await retry_with_fixed_backoff(fn, max_attempts=3, backoff_seconds=1.0, sleep_fn=sleeper)

    assert isinstance(exc_info.value.last_exception, RuntimeError)
    assert sleeper.calls == [1.0, 1.0]  # sleeps between attempts, not after the last


@pytest.mark.asyncio
async def test_retry_with_exponential_backoff_doubles_delay_each_attempt():
    async def fn():
        raise RuntimeError("always fails")

    sleeper = _FakeSleeper()
    with pytest.raises(RetryExhaustedError):
        await retry_with_exponential_backoff(fn, max_attempts=4, base_delay_seconds=1.0, sleep_fn=sleeper)

    assert sleeper.calls == [1.0, 2.0, 4.0]
```

```python
# tests/orchestrator/test_stt.py
import pytest

from financial_voice_agent.orchestrator.stt import RealGroqSttClient, SttError, transcribe_with_retry


class _FailNTimesClient:
    def __init__(self, fail_count: int, result: str = "transcribed text"):
        self._fail_count = fail_count
        self._result = result
        self.calls = 0

    async def transcribe(self, wav_bytes: bytes, *, model: str) -> str:
        self.calls += 1
        if self.calls <= self._fail_count:
            raise RuntimeError("groq unavailable")
        return self._result


class _AlwaysFailsClient:
    async def transcribe(self, wav_bytes: bytes, *, model: str) -> str:
        raise RuntimeError("groq unavailable")


@pytest.mark.asyncio
async def test_transcribe_with_retry_succeeds_after_transient_failures():
    async def instant_sleep(seconds):
        pass

    client = _FailNTimesClient(fail_count=2)

    result = await transcribe_with_retry(
        client, b"wav-bytes", model="whisper-large-v3-turbo", sleep_fn=instant_sleep
    )

    assert result == "transcribed text"
    assert client.calls == 3


@pytest.mark.asyncio
async def test_transcribe_with_retry_raises_sttexception_after_exhaustion():
    client = _AlwaysFailsClient()

    with pytest.raises(SttError):
        await transcribe_with_retry(client, b"wav-bytes", model="whisper-large-v3-turbo")


@pytest.mark.asyncio
async def test_real_groq_stt_client_calls_transcriptions_create_with_wav_bytes():
    class _FakeTranscriptionsResource:
        def __init__(self):
            self.received_kwargs = None

        async def create(self, **kwargs):
            self.received_kwargs = kwargs
            return "adapter transcribed text"

    class _FakeAudioNamespace:
        def __init__(self, transcriptions):
            self.transcriptions = transcriptions

    class _FakeGroqAsyncClient:
        def __init__(self, transcriptions):
            self.audio = _FakeAudioNamespace(transcriptions)

    transcriptions = _FakeTranscriptionsResource()
    fake_groq_client = _FakeGroqAsyncClient(transcriptions)
    adapter = RealGroqSttClient(fake_groq_client)

    result = await adapter.transcribe(b"wav-bytes", model="whisper-large-v3-turbo")

    assert result == "adapter transcribed text"
    assert transcriptions.received_kwargs["model"] == "whisper-large-v3-turbo"
    assert transcriptions.received_kwargs["file"][1] == b"wav-bytes"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/orchestrator/test_retry.py tests/orchestrator/test_stt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.orchestrator.retry'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/orchestrator/retry.py
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class RetryExhaustedError(Exception):
    def __init__(self, message: str, *, last_exception: Exception | None = None):
        super().__init__(message)
        self.last_exception = last_exception


async def retry_with_fixed_backoff(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts - 1:
                await sleep_fn(backoff_seconds)
    raise RetryExhaustedError(f"failed after {max_attempts} attempts", last_exception=last_exc)


async def retry_with_exponential_backoff(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 4,
    base_delay_seconds: float = 1.0,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts - 1:
                await sleep_fn(base_delay_seconds * (2**attempt))
    raise RetryExhaustedError(f"failed after {max_attempts} attempts", last_exception=last_exc)
```

```python
# financial_voice_agent/orchestrator/stt.py
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Protocol

from financial_voice_agent.orchestrator.retry import RetryExhaustedError, retry_with_fixed_backoff


class SttError(Exception):
    """Raised when STT fails after all retries. The turn orchestrator (Task 5)
    catches this and speaks "I didn't catch that, one moment" per PRD
    Section 15.1, then re-listens rather than crashing."""


class GroqSttClient(Protocol):
    async def transcribe(self, wav_bytes: bytes, *, model: str) -> str: ...


async def transcribe_with_retry(
    client: GroqSttClient,
    wav_bytes: bytes,
    *,
    model: str,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> str:
    async def _attempt() -> str:
        return await client.transcribe(wav_bytes, model=model)

    try:
        return await retry_with_fixed_backoff(
            _attempt, max_attempts=3, backoff_seconds=1.0, sleep_fn=sleep_fn
        )
    except RetryExhaustedError as exc:
        raise SttError("Speech-to-text failed after 3 attempts") from exc


class RealGroqSttClient:
    """Thin adapter around groq.AsyncGroq's real transcription API.

    Verify against console.groq.com/docs/api-reference at build time --
    Groq's SDK surface can change (PRD Section 2.3's model-churn warning
    applies to the SDK, not just the model name).
    """

    def __init__(self, groq_async_client) -> None:
        self._client = groq_async_client

    async def transcribe(self, wav_bytes: bytes, *, model: str) -> str:
        response = await self._client.audio.transcriptions.create(
            file=("utterance.wav", wav_bytes),
            model=model,
            response_format="text",
        )
        return response if isinstance(response, str) else response.text
```

Add `groq` to `requirements.txt`. Run `pip install -r requirements.txt`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/orchestrator/test_retry.py tests/orchestrator/test_stt.py -v`
Expected: PASS (6 tests, all fast — every retry-timing test passes an instant `sleep_fn` explicitly, so no test actually waits out a real backoff delay).

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/orchestrator/retry.py financial_voice_agent/orchestrator/stt.py tests/orchestrator/test_retry.py tests/orchestrator/test_stt.py requirements.txt
git commit -m "feat: add retry policy and Groq STT with retry"
```

---

### Task 3: Real Groq LLM tool-calling loop with retry (`llm.py`)

**Files:**
- Create: `financial_voice_agent/orchestrator/llm.py`
- Test: `tests/orchestrator/test_llm.py`

**Interfaces:**
- Consumes: `retry_with_fixed_backoff`, `retry_with_exponential_backoff`, `RetryExhaustedError` (Task 2). `LlmTurnResult` (Task 1).
- Produces:
  - `@dataclass(frozen=True) class ToolCall: id: str; name: str; arguments: dict`
  - `@dataclass(frozen=True) class LlmCompletion: text: str | None; tool_calls: list[ToolCall]`
  - `class LlmClient(Protocol): async def complete(self, messages: list[dict], *, model: str, tools_schema: list[dict]) -> LlmCompletion: ...`
  - `ToolExecutor = Callable[[ToolCall], Awaitable[dict]]`
  - `class LlmError(Exception)` — `.rate_limited: bool` attribute.
  - `async def run_llm_turn(client: LlmClient, transcript: str, history: list[dict], *, model: str, tools_schema: list[dict], tool_executor: ToolExecutor, max_tool_rounds: int = 3, sleep_fn=asyncio.sleep) -> LlmTurnResult` — runs the tool loop, executing concurrent tool calls per round via `asyncio.gather`, raising `LlmError` if it doesn't converge within `max_tool_rounds` or if the underlying call fails after retries. `sleep_fn` is forwarded to the retry helpers for the same reason as `stt.py` (see Global Constraints).
  - `class RealGroqLlmClient` — adapter implementing `LlmClient` against `groq.AsyncGroq.chat.completions.create`. **Verify against current Groq docs at build time.**

- [ ] **Step 1: Write the failing tests**

```python
# tests/orchestrator/test_llm.py
import pytest

from financial_voice_agent.orchestrator.llm import (
    LlmCompletion,
    LlmError,
    RealGroqLlmClient,
    ToolCall,
    run_llm_turn,
)


class _NoToolCallsClient:
    async def complete(self, messages, *, model, tools_schema):
        return LlmCompletion(text="Nifty is at 24500.", tool_calls=[])


class _OneRoundToolCallClient:
    def __init__(self):
        self.round = 0

    async def complete(self, messages, *, model, tools_schema):
        self.round += 1
        if self.round == 1:
            return LlmCompletion(
                text=None,
                tool_calls=[ToolCall(id="call_1", name="get_quote", arguments={"symbol": "NIFTY 50"})],
            )
        return LlmCompletion(text="Nifty is at 24500.", tool_calls=[])


class _NeverConvergesClient:
    async def complete(self, messages, *, model, tools_schema):
        return LlmCompletion(
            text=None,
            tool_calls=[ToolCall(id="call_x", name="get_quote", arguments={})],
        )


class _AlwaysFailsClient:
    async def complete(self, messages, *, model, tools_schema):
        raise RuntimeError("groq unavailable")


class _RateLimitedClient:
    async def complete(self, messages, *, model, tools_schema):
        exc = RuntimeError("rate limited")
        exc.status_code = 429
        raise exc


async def _tool_executor(call: ToolCall) -> dict:
    return {"ltp": 24500}


@pytest.mark.asyncio
async def test_run_llm_turn_with_no_tool_calls_returns_immediately():
    result = await run_llm_turn(
        _NoToolCallsClient(), "what's the nifty level", [],
        model="test-model", tools_schema=[], tool_executor=_tool_executor,
    )

    assert result.response_text == "Nifty is at 24500."
    assert result.tool_calls_json is None
    assert result.tool_results_json is None


@pytest.mark.asyncio
async def test_run_llm_turn_executes_tool_call_then_returns_final_text():
    result = await run_llm_turn(
        _OneRoundToolCallClient(), "what's the nifty level", [],
        model="test-model", tools_schema=[], tool_executor=_tool_executor,
    )

    assert result.response_text == "Nifty is at 24500."
    assert result.tool_calls_json == '[{"tool": "get_quote", "args": {"symbol": "NIFTY 50"}}]'
    assert result.tool_results_json == '[{"ltp": 24500}]'


@pytest.mark.asyncio
async def test_run_llm_turn_executes_concurrent_tool_calls_via_gather():
    call_order: list[str] = []

    async def slow_tool_executor(call: ToolCall) -> dict:
        call_order.append(f"start:{call.name}")
        import asyncio
        await asyncio.sleep(0.01)
        call_order.append(f"end:{call.name}")
        return {"result": call.name}

    class _TwoToolCallsClient:
        def __init__(self):
            self.round = 0

        async def complete(self, messages, *, model, tools_schema):
            self.round += 1
            if self.round == 1:
                return LlmCompletion(
                    text=None,
                    tool_calls=[
                        ToolCall(id="c1", name="get_quote", arguments={}),
                        ToolCall(id="c2", name="get_news", arguments={}),
                    ],
                )
            return LlmCompletion(text="done", tool_calls=[])

    await run_llm_turn(
        _TwoToolCallsClient(), "transcript", [],
        model="test-model", tools_schema=[], tool_executor=slow_tool_executor,
    )

    # Both tools must start before either finishes -- proves asyncio.gather
    # concurrency, not sequential await.
    assert call_order[0].startswith("start:")
    assert call_order[1].startswith("start:")


@pytest.mark.asyncio
async def test_run_llm_turn_raises_llmerror_if_tool_loop_never_converges():
    with pytest.raises(LlmError):
        await run_llm_turn(
            _NeverConvergesClient(), "transcript", [],
            model="test-model", tools_schema=[], tool_executor=_tool_executor, max_tool_rounds=2,
        )


@pytest.mark.asyncio
async def test_run_llm_turn_raises_llmerror_after_generic_failure_retries():
    async def instant_sleep(seconds):
        pass

    with pytest.raises(LlmError) as exc_info:
        await run_llm_turn(
            _AlwaysFailsClient(), "transcript", [],
            model="test-model", tools_schema=[], tool_executor=_tool_executor, sleep_fn=instant_sleep,
        )
    assert exc_info.value.rate_limited is False


@pytest.mark.asyncio
async def test_run_llm_turn_raises_rate_limited_llmerror_on_429():
    async def instant_sleep(seconds):
        pass

    with pytest.raises(LlmError) as exc_info:
        await run_llm_turn(
            _RateLimitedClient(), "transcript", [],
            model="test-model", tools_schema=[], tool_executor=_tool_executor, sleep_fn=instant_sleep,
        )
    assert exc_info.value.rate_limited is True


@pytest.mark.asyncio
async def test_real_groq_llm_client_calls_chat_completions_create():
    class _FakeMessage:
        def __init__(self, content, tool_calls):
            self.content = content
            self.tool_calls = tool_calls

    class _FakeToolCallFunction:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class _FakeToolCall:
        def __init__(self, id, name, arguments):
            self.id = id
            self.function = _FakeToolCallFunction(name, arguments)

    class _FakeChoice:
        def __init__(self, message):
            self.message = message

    class _FakeCompletion:
        def __init__(self, choices):
            self.choices = choices

    class _FakeCompletionsResource:
        def __init__(self, response):
            self._response = response
            self.received_kwargs = None

        async def create(self, **kwargs):
            self.received_kwargs = kwargs
            return self._response

    class _FakeChatNamespace:
        def __init__(self, completions):
            self.completions = completions

    class _FakeGroqAsyncClient:
        def __init__(self, completions):
            self.chat = _FakeChatNamespace(completions)

    fake_response = _FakeCompletion(
        [_FakeChoice(_FakeMessage(None, [_FakeToolCall("call_1", "get_quote", '{"symbol": "NIFTY 50"}')]))]
    )
    completions = _FakeCompletionsResource(fake_response)
    fake_groq_client = _FakeGroqAsyncClient(completions)
    adapter = RealGroqLlmClient(fake_groq_client)

    result = await adapter.complete([{"role": "user", "content": "hi"}], model="test-model", tools_schema=[])

    assert result.text is None
    assert result.tool_calls == [ToolCall(id="call_1", name="get_quote", arguments={"symbol": "NIFTY 50"})]
    assert completions.received_kwargs["model"] == "test-model"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/orchestrator/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.orchestrator.llm'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/orchestrator/llm.py
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from financial_voice_agent.orchestrator.retry import (
    RetryExhaustedError,
    retry_with_exponential_backoff,
    retry_with_fixed_backoff,
)
from financial_voice_agent.orchestrator.turn import LlmTurnResult


class LlmError(Exception):
    def __init__(self, message: str, *, rate_limited: bool = False):
        super().__init__(message)
        self.rate_limited = rate_limited


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class LlmCompletion:
    text: str | None
    tool_calls: list[ToolCall]


class LlmClient(Protocol):
    async def complete(
        self, messages: list[dict], *, model: str, tools_schema: list[dict]
    ) -> LlmCompletion: ...


ToolExecutor = Callable[[ToolCall], Awaitable[dict]]


async def run_llm_turn(
    client: LlmClient,
    transcript: str,
    history: list[dict],
    *,
    model: str,
    tools_schema: list[dict],
    tool_executor: ToolExecutor,
    max_tool_rounds: int = 3,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> LlmTurnResult:
    messages = [*history, {"role": "user", "content": transcript}]
    all_tool_calls: list[dict] = []
    all_tool_results: list[dict] = []

    for _round in range(max_tool_rounds):
        completion = await _complete_with_retry(
            client, messages, model=model, tools_schema=tools_schema, sleep_fn=sleep_fn
        )

        if not completion.tool_calls:
            return LlmTurnResult(
                response_text=completion.text or "",
                tool_calls_json=json.dumps(all_tool_calls) if all_tool_calls else None,
                tool_results_json=json.dumps(all_tool_results) if all_tool_results else None,
            )

        results = await asyncio.gather(*(tool_executor(call) for call in completion.tool_calls))
        for call, result in zip(completion.tool_calls, results):
            all_tool_calls.append({"tool": call.name, "args": call.arguments})
            all_tool_results.append(result)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

    raise LlmError("LLM tool loop did not converge within max_tool_rounds")


async def _complete_with_retry(
    client: LlmClient, messages, *, model, tools_schema, sleep_fn
) -> LlmCompletion:
    async def _attempt() -> LlmCompletion:
        return await client.complete(messages, model=model, tools_schema=tools_schema)

    try:
        return await _attempt()
    except Exception as first_exc:  # noqa: BLE001
        if getattr(first_exc, "status_code", None) == 429:
            try:
                return await retry_with_exponential_backoff(
                    _attempt, max_attempts=4, base_delay_seconds=1.0, sleep_fn=sleep_fn
                )
            except RetryExhaustedError as exc:
                raise LlmError("Groq LLM rate limited repeatedly", rate_limited=True) from exc
        try:
            return await retry_with_fixed_backoff(
                _attempt, max_attempts=2, backoff_seconds=1.0, sleep_fn=sleep_fn
            )
        except RetryExhaustedError as exc:
            raise LlmError("Groq LLM failed after retries") from exc


class RealGroqLlmClient:
    """Thin adapter around groq.AsyncGroq's chat.completions.create.

    Verify against console.groq.com/docs/api-reference at build time.
    """

    def __init__(self, groq_async_client) -> None:
        self._client = groq_async_client

    async def complete(
        self, messages: list[dict], *, model: str, tools_schema: list[dict]
    ) -> LlmCompletion:
        response = await self._client.chat.completions.create(
            messages=messages,
            model=model,
            tools=tools_schema or None,
            tool_choice="auto" if tools_schema else None,
        )
        message = response.choices[0].message
        raw_tool_calls = message.tool_calls or []
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
            for tc in raw_tool_calls
        ]
        return LlmCompletion(text=message.content, tool_calls=tool_calls)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/orchestrator/test_llm.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/orchestrator/llm.py tests/orchestrator/test_llm.py
git commit -m "feat: add Groq LLM tool-calling loop with retry policy"
```

---

### Task 4: Real Cartesia TTS + PyAudio playback (`tts.py`, `playback.py`)

**Files:**
- Create: `financial_voice_agent/orchestrator/tts.py`
- Create: `financial_voice_agent/orchestrator/playback.py`
- Test: `tests/orchestrator/test_tts.py`
- Test: `tests/orchestrator/test_playback.py`
- Modify: `requirements.txt` — add `cartesia`

**Interfaces:**
- Consumes: `retry_with_fixed_backoff`, `RetryExhaustedError` (Task 2).
- Produces:
  - `CARTESIA_OUTPUT_FORMAT: dict` — `{"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000}`, the exact PRD Section 10.2 contract.
  - `class TtsClient(Protocol): async def synthesize(self, text: str) -> bytes: ...`
  - `class TtsError(Exception)`
  - `async def synthesize_with_fallback(primary: TtsClient, text: str, *, fallback: TtsClient | None = None, sleep_fn=asyncio.sleep) -> bytes` — retry once on primary (2 total attempts), then fall back if configured, else raise `TtsError`. `sleep_fn` forwarded for the same reason as `stt.py`/`llm.py` (see Global Constraints).
  - `class RealCartesiaTtsClient` — adapter implementing `TtsClient` against Cartesia's websocket API with `CARTESIA_OUTPUT_FORMAT`. **Verify against Cartesia's current Python SDK docs at build time.**
  - `class AudioPlayback:` — `__init__(self, *, sample_rate=16000, channels=1, output_device_index=None, pyaudio_factory=pyaudio.PyAudio)`, `def open(self) -> None`, `def close(self) -> None`, `def play(self, pcm_bytes: bytes, *, chunk_size: int = 1024, interrupt_event=None) -> bool` — writes `pcm_bytes` to the output stream in chunks, checking `interrupt_event.is_set()` between chunks; returns `True` if playback completed, `False` if interrupted early (matching PRD Section 11.3: "stops immediately, discarding any buffered-but-unplayed audio").

- [ ] **Step 1: Write the failing tests**

```python
# tests/orchestrator/test_tts.py
import pytest

from financial_voice_agent.orchestrator.tts import CARTESIA_OUTPUT_FORMAT, TtsError, synthesize_with_fallback


class _FailNTimesClient:
    def __init__(self, fail_count: int, result: bytes = b"audio"):
        self._fail_count = fail_count
        self._result = result
        self.calls = 0

    async def synthesize(self, text: str) -> bytes:
        self.calls += 1
        if self.calls <= self._fail_count:
            raise RuntimeError("cartesia unavailable")
        return self._result


class _AlwaysFailsClient:
    def __init__(self):
        self.calls = 0

    async def synthesize(self, text: str) -> bytes:
        self.calls += 1
        raise RuntimeError("cartesia unavailable")


async def _instant_sleep(seconds: float) -> None:
    pass


@pytest.mark.asyncio
async def test_synthesize_with_fallback_succeeds_on_retry():
    primary = _FailNTimesClient(fail_count=1)

    result = await synthesize_with_fallback(primary, "hello", sleep_fn=_instant_sleep)

    assert result == b"audio"
    assert primary.calls == 2


@pytest.mark.asyncio
async def test_synthesize_with_fallback_uses_fallback_after_primary_exhausted():
    primary = _AlwaysFailsClient()
    fallback = _FailNTimesClient(fail_count=0, result=b"deepgram-audio")

    result = await synthesize_with_fallback(primary, "hello", fallback=fallback, sleep_fn=_instant_sleep)

    assert result == b"deepgram-audio"
    assert primary.calls == 2  # retry once = 2 total attempts


@pytest.mark.asyncio
async def test_synthesize_with_fallback_raises_ttserror_when_no_fallback_configured():
    primary = _AlwaysFailsClient()

    with pytest.raises(TtsError):
        await synthesize_with_fallback(primary, "hello", fallback=None, sleep_fn=_instant_sleep)


@pytest.mark.asyncio
async def test_synthesize_with_fallback_raises_ttserror_when_both_fail():
    primary = _AlwaysFailsClient()
    fallback = _AlwaysFailsClient()

    with pytest.raises(TtsError):
        await synthesize_with_fallback(primary, "hello", fallback=fallback, sleep_fn=_instant_sleep)


def test_cartesia_output_format_matches_prd_data_contract():
    assert CARTESIA_OUTPUT_FORMAT == {
        "container": "raw",
        "encoding": "pcm_s16le",
        "sample_rate": 16000,
    }
```

```python
# tests/orchestrator/test_playback.py
from financial_voice_agent.orchestrator.playback import AudioPlayback


class _FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.written_chunks: list[bytes] = []
        self.stopped = False
        self.closed = False

    def write(self, chunk: bytes) -> None:
        self.written_chunks.append(chunk)

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _FakePyAudio:
    def __init__(self):
        self.terminated = False
        self.opened_stream = None

    def open(self, **kwargs):
        self.opened_stream = _FakeStream(**kwargs)
        return self.opened_stream

    def terminate(self):
        self.terminated = True


def test_open_uses_documented_output_data_contract():
    import pyaudio

    fake_pa = _FakePyAudio()
    playback = AudioPlayback(pyaudio_factory=lambda: fake_pa)

    playback.open()

    stream = fake_pa.opened_stream
    assert stream.kwargs["format"] == pyaudio.paInt16
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["rate"] == 16000
    assert stream.kwargs["output"] is True


def test_play_writes_all_chunks_when_not_interrupted():
    fake_pa = _FakePyAudio()
    playback = AudioPlayback(pyaudio_factory=lambda: fake_pa)
    playback.open()
    pcm_bytes = b"\x00\x01" * 3000  # spans multiple 1024-byte chunks

    completed = playback.play(pcm_bytes, chunk_size=1024)

    stream = fake_pa.opened_stream
    assert completed is True
    assert b"".join(stream.written_chunks) == pcm_bytes


def test_play_stops_early_when_interrupt_event_is_set():
    import threading

    fake_pa = _FakePyAudio()
    playback = AudioPlayback(pyaudio_factory=lambda: fake_pa)
    playback.open()
    pcm_bytes = b"\x00\x01" * 3000
    interrupt_event = threading.Event()

    stream = fake_pa.opened_stream
    original_write = stream.write

    def _write_and_interrupt_after_first_chunk(chunk):
        original_write(chunk)
        if len(stream.written_chunks) == 1:
            interrupt_event.set()

    stream.write = _write_and_interrupt_after_first_chunk

    completed = playback.play(pcm_bytes, chunk_size=1024, interrupt_event=interrupt_event)

    assert completed is False
    assert len(stream.written_chunks) == 1  # stopped after the first chunk, not all 3


def test_close_stops_and_closes_stream_and_terminates_pyaudio():
    fake_pa = _FakePyAudio()
    playback = AudioPlayback(pyaudio_factory=lambda: fake_pa)
    playback.open()
    stream = fake_pa.opened_stream

    playback.close()

    assert stream.stopped is True
    assert stream.closed is True
    assert fake_pa.terminated is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/orchestrator/test_tts.py tests/orchestrator/test_playback.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.orchestrator.tts'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/orchestrator/tts.py
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Protocol

from financial_voice_agent.orchestrator.retry import RetryExhaustedError, retry_with_fixed_backoff

CARTESIA_OUTPUT_FORMAT = {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000}


class TtsError(Exception):
    pass


class TtsClient(Protocol):
    async def synthesize(self, text: str) -> bytes: ...


async def synthesize_with_fallback(
    primary: TtsClient,
    text: str,
    *,
    fallback: TtsClient | None = None,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> bytes:
    async def _primary_attempt() -> bytes:
        return await primary.synthesize(text)

    try:
        return await retry_with_fixed_backoff(
            _primary_attempt, max_attempts=2, backoff_seconds=1.0, sleep_fn=sleep_fn
        )
    except RetryExhaustedError as exc:
        if fallback is not None:
            try:
                return await fallback.synthesize(text)
            except Exception as fallback_exc:  # noqa: BLE001
                raise TtsError("Both Cartesia and Deepgram TTS failed") from fallback_exc
        raise TtsError("Cartesia TTS failed and no fallback configured") from exc


class RealCartesiaTtsClient:
    """Thin adapter around Cartesia's websocket TTS API, using the exact
    PRD Section 10.2 output format (16kHz, pcm_s16le, matching mic capture
    rate so nothing needs resampling).

    Verify against Cartesia's current Python SDK docs at build time -- this
    adapter's websocket call shape is the highest-uncertainty piece of this
    task.
    """

    def __init__(self, cartesia_async_client, *, voice_id: str, model_id: str = "sonic-2") -> None:
        self._client = cartesia_async_client
        self._voice_id = voice_id
        self._model_id = model_id

    async def synthesize(self, text: str) -> bytes:
        chunks: list[bytes] = []
        async with self._client.tts.websocket() as ws:
            async for response in ws.send(
                model_id=self._model_id,
                transcript=text,
                voice={"mode": "id", "id": self._voice_id},
                output_format=CARTESIA_OUTPUT_FORMAT,
            ):
                if response.get("audio"):
                    chunks.append(response["audio"])
        return b"".join(chunks)
```

```python
# financial_voice_agent/orchestrator/playback.py
from __future__ import annotations

import pyaudio


class AudioPlayback:
    """PyAudio output stream matching Cartesia's PCM format exactly (16kHz,
    16-bit, mono) -- a mismatch here is PRD Section 10.2's "chipmunk voice"
    bug. Interruptible mid-stream via a threading/asyncio Event
    (barge-in, Section 11.3).
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        output_device_index: int | None = None,
        pyaudio_factory=pyaudio.PyAudio,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._output_device_index = output_device_index
        self._pyaudio_factory = pyaudio_factory
        self._pa = None
        self._stream = None

    def open(self) -> None:
        self._pa = self._pyaudio_factory()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._sample_rate,
            output=True,
            output_device_index=self._output_device_index,
        )

    def play(self, pcm_bytes: bytes, *, chunk_size: int = 1024, interrupt_event=None) -> bool:
        for offset in range(0, len(pcm_bytes), chunk_size):
            if interrupt_event is not None and interrupt_event.is_set():
                return False
            self._stream.write(pcm_bytes[offset : offset + chunk_size])
        return True

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
```

Add `cartesia` to `requirements.txt`. Run `pip install -r requirements.txt`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/orchestrator/test_tts.py tests/orchestrator/test_playback.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/orchestrator/tts.py financial_voice_agent/orchestrator/playback.py tests/orchestrator/test_tts.py tests/orchestrator/test_playback.py requirements.txt
git commit -m "feat: add Cartesia TTS with Deepgram fallback and interruptible playback"
```

---

### Task 5: Barge-in wiring and the real voice loop (`main_loop.py`)

**Files:**
- Create: `financial_voice_agent/orchestrator/main_loop.py`
- Test: `tests/orchestrator/test_main_loop.py`

**Interfaces:**
- Consumes: `financial_voice_agent.audio.pipeline.AudioPipeline` (Phase 2, `speech_active` event + `run()` async generator), `run_turn`, `update_history` (Task 1), `AudioPlayback` (Task 4).
- Produces:
  - `async def drive_pipeline(pipeline, output_queue) -> None` — the dedicated background task `pipeline.speech_active`'s docstring requires: continuously iterates `pipeline.run()` and pushes each utterance onto `output_queue`, never blocking on turn processing (satisfies the liveness contract documented in Phase 2's `pipeline.py`).
  - `async def run_voice_loop(pipeline, playback: AudioPlayback, *, stt_fn, llm_fn, tts_fn, db_path: str, max_turns_history: int = 8) -> None` — the real loop: starts `drive_pipeline` as a background task, then for each utterance popped from the output queue: calls `run_turn`, updates history, and calls `playback.play(..., interrupt_event=<a threading.Event mirrored from pipeline.speech_active>)`. If playback is interrupted, the loop immediately goes back to waiting for the next utterance instead of finishing playback. Runs until cancelled.

- [ ] **Step 1: Write the failing tests**

```python
# tests/orchestrator/test_main_loop.py
import asyncio

import pytest

from financial_voice_agent.orchestrator.main_loop import drive_pipeline, run_voice_loop
from financial_voice_agent.orchestrator.turn import LlmTurnResult
from financial_voice_agent.db import init_db


class _FakePipeline:
    """Mimics AudioPipeline: an async generator yielding scripted utterances,
    plus a speech_active event a test can set to simulate barge-in."""

    def __init__(self, utterances: list[bytes]):
        self._utterances = utterances
        self.speech_active = asyncio.Event()

    async def run(self):
        for u in self._utterances:
            yield u


class _FakePlayback:
    def __init__(self, *, interrupted_after: int | None = None):
        self.play_calls: list[bytes] = []
        self._interrupted_after = interrupted_after

    def play(self, pcm_bytes: bytes, *, chunk_size: int = 1024, interrupt_event=None) -> bool:
        self.play_calls.append(pcm_bytes)
        if self._interrupted_after is not None and len(self.play_calls) > self._interrupted_after:
            return False
        return True


@pytest.mark.asyncio
async def test_drive_pipeline_forwards_utterances_to_output_queue():
    pipeline = _FakePipeline([b"utterance-1", b"utterance-2"])
    output_queue: asyncio.Queue = asyncio.Queue()

    await drive_pipeline(pipeline, output_queue)

    assert output_queue.get_nowait() == b"utterance-1"
    assert output_queue.get_nowait() == b"utterance-2"


@pytest.mark.asyncio
async def test_run_voice_loop_processes_each_utterance_and_plays_response(tmp_path):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    pipeline = _FakePipeline([b"utterance-1"])
    playback = _FakePlayback()

    async def stt_fn(wav):
        return "hello"

    async def llm_fn(transcript, history):
        return LlmTurnResult(response_text="hi there", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return b"audio-bytes"

    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
    )

    assert playback.play_calls == [b"audio-bytes"]


@pytest.mark.asyncio
async def test_run_voice_loop_processes_multiple_utterances_in_order(tmp_path):
    db_path = str(tmp_path / "turns.db")
    init_db(db_path)
    pipeline = _FakePipeline([b"utterance-1", b"utterance-2"])
    playback = _FakePlayback()
    seen_transcripts: list[str] = []

    async def stt_fn(wav):
        transcript = "first" if wav == b"utterance-1" else "second"
        seen_transcripts.append(transcript)
        return transcript

    async def llm_fn(transcript, history):
        return LlmTurnResult(response_text=f"response to {transcript}", tool_calls_json=None, tool_results_json=None)

    async def tts_fn(text):
        return text.encode()

    await run_voice_loop(
        pipeline, playback, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
    )

    assert seen_transcripts == ["first", "second"]
    assert playback.play_calls == [b"response to first", b"response to second"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/orchestrator/test_main_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.orchestrator.main_loop'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/orchestrator/main_loop.py
from __future__ import annotations

import asyncio
import threading

from financial_voice_agent.orchestrator.turn import run_turn, update_history


async def drive_pipeline(pipeline, output_queue: asyncio.Queue) -> None:
    """Continuously pulls utterances from pipeline.run() and forwards them to
    output_queue. Must run as its own task, per AudioPipeline.speech_active's
    documented liveness contract (financial_voice_agent/audio/pipeline.py) --
    it must never be blocked waiting on turn processing, or speech_active
    (and therefore barge-in) freezes."""
    async for utterance_wav in pipeline.run():
        await output_queue.put(utterance_wav)


async def run_voice_loop(
    pipeline,
    playback,
    *,
    stt_fn,
    llm_fn,
    tts_fn,
    db_path: str,
    max_turns_history: int = 8,
) -> None:
    output_queue: asyncio.Queue = asyncio.Queue()
    drive_task = asyncio.create_task(drive_pipeline(pipeline, output_queue))
    history: list[dict] = []

    try:
        while not drive_task.done() or not output_queue.empty():
            try:
                utterance_wav = await asyncio.wait_for(output_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            result = await run_turn(
                utterance_wav, history, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
            )

            if result.transcript is not None and result.response_text is not None:
                history = update_history(
                    history, result.transcript, result.response_text, max_turns=max_turns_history
                )

            if result.tts_audio:
                interrupt_event = threading.Event()
                if pipeline.speech_active.is_set():
                    interrupt_event.set()
                playback.play(result.tts_audio, interrupt_event=interrupt_event)
    finally:
        drive_task.cancel()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/orchestrator/test_main_loop.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full Phase 3 suite**

Run: `pytest tests/ -v`
Expected: PASS (48 baseline + Phase 3's new tests, all green).

- [ ] **Step 6: Manual smoke test (not automated — record the result, do not skip)**

Once `GROQ_API_KEY` and `CARTESIA_API_KEY` are set in `.env` (per your earlier choice to build against fakes first, do this only once you're ready to go live):

```bash
python -c "
import asyncio
from financial_voice_agent.config import load_config
from financial_voice_agent.audio.capture import AudioCapture
from financial_voice_agent.audio.pipeline import AudioPipeline
from financial_voice_agent.audio.vad import SileroVadScorer
from financial_voice_agent.orchestrator.playback import AudioPlayback
from financial_voice_agent.orchestrator.main_loop import run_voice_loop
from financial_voice_agent.orchestrator.stubs import fake_stt, fake_llm, fake_tts
import janus

config = load_config()
queue = janus.Queue()
capture = AudioCapture(queue)
pipeline = AudioPipeline(queue.async_q, SileroVadScorer())
playback = AudioPlayback()
playback.open()
capture.start()
print('Say something -- Ctrl+C to stop.')
try:
    asyncio.run(run_voice_loop(pipeline, playback, stt_fn=fake_stt, llm_fn=fake_llm, tts_fn=fake_tts, db_path=config.storage_db_path))
except KeyboardInterrupt:
    pass
finally:
    capture.stop()
    playback.close()
"
```

Expected: speaking into the microphone produces a logged turn in the SQLite DB (query `SELECT * FROM turns ORDER BY turn_id DESC LIMIT 1`) even with fake STT/LLM/TTS — this proves the real audio hardware path (capture → VAD → utterance) is wired correctly end to end. Swap `fake_stt`/`fake_llm`/`fake_tts` for `RealGroqSttClient`/`RealGroqLlmClient`/`RealCartesiaTtsClient`-backed functions once ready to go live, per your earlier decision to build fake-first.

- [ ] **Step 7: Commit**

```bash
git add financial_voice_agent/orchestrator/main_loop.py tests/orchestrator/test_main_loop.py
git commit -m "feat: add barge-in-aware voice loop wiring capture, pipeline, turns, and playback"
```

---

## Phase 3 Exit Criteria

- `pytest tests/ -v` passes with 0 failures.
- Running the Task 5 manual smoke test with fake STT/LLM/TTS produces a real logged turn from real microphone input — the full hardware→software path works even before any paid API key is used.
- No test in `tests/orchestrator/` makes a real network call to Groq or Cartesia — all vendor-boundary logic is tested via the `Protocol` fakes; only the thin adapter classes touch the real SDKs, and those are exercised only by the manual smoke test once real keys are configured.
- Every turn — success or failure — produces a row in the `turns` table with a populated `latency_total_ms`.

---

## Upcoming Phases (summaries — to be written in full detail after Phase 3 review)

**Phase 4 — Tools:** Implements `get_quote`, `get_ohlc_history`, `compute_indicator` (Bollinger/Fibonacci/MA/RSI via `ta`/`pandas`/`numpy`), `get_positions_holdings`, `get_news` (Tavily), `capture_screen` (mss + pygetwindow/Quartz), each following the PRD Section 6 tool table's exact error/retry behavior (401 → "session expired", 429 → backoff, mock mode reads Phase 1 fixtures). Registers all six as `tools_schema` + a real `tool_executor` with Phase 3's `run_llm_turn`, replacing the currently-empty tool set.

**Phase 5 — Eval Harness:** JSON test cases (PRD Section 17) with input transcript, optional mocked screen result, expected tool calls, and tools that must NOT be called. A runner asserts on tool names/args only (not exact wording), run against `mode: "mock"` end-to-end through Phases 1-4. Seeds the 8 starting cases from the PRD and provides the harness for adding a case every time real usage surfaces a wrong tool call.
