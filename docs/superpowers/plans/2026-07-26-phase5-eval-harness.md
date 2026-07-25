# Financial Voice Agent — Phase 5: Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local evaluation set (PRD Section 17): a small, static set of text conversations that check whether the assistant calls the right tools — and never calls the wrong ones — re-run after every change to the system prompt, tool schemas, or orchestration logic. Tests the LLM tool-calling loop directly (Phase 3's `run_llm_turn`), skipping STT/TTS entirely, per PRD Section 17.1.

**Architecture:** Three small modules under `financial_voice_agent/eval/`. The harness's own logic (case loading, pass/fail comparison, report formatting) is fully unit-tested with a scripted fake LLM client — zero cost, zero API key needed, consistent with every prior phase's mock-first pattern. Running the eval set against the *real* Groq model (the actual point of an eval set, per the PRD) is a documented manual step, not an automated pytest test, for the same reason Phase 3/4's real vendor adapters aren't unit-tested against live services: it costs money and needs a real credential you don't have yet.

**Tech Stack:** Python 3.11+, pytest, pytest-asyncio. No new dependencies — everything needed already exists in Phases 1-4.

## Global Constraints

- Python 3.11+ only (PRD Section 9).
- **A runner asserts on tool-call names and arguments, not exact spoken wording** (PRD Section 17.1) — the harness never inspects `response_text`, only which tools were called.
- **Eval cases test the LLM tool loop directly, skipping STT** (PRD Section 17.1) — the harness calls Phase 3's `run_llm_turn` with a plain-text `transcript`, never `run_turn` (which also does STT/TTS) and never touches real audio.
- **Tool execution during eval runs always uses Phase 4's `mode: "mock"` tools** for Kite-backed calls, regardless of whether the LLM client is the scripted fake or the real Groq model — no live Kite account is ever needed to run the eval set, matching this project's mock-first decision. `capture_screen` has no PRD-specified mock mode (Phase 4), so any eval case that expects it to be called must supply a `mocked_screen_result` the harness substitutes instead of invoking the real screen/window adapters.
- **PRD Section 17.2's barge-in case ("user starts talking while the agent is still responding") does not fit this text-based tool-calling harness** — it's an audio-interrupt behavior already covered by Phase 2's `AudioPipeline`/`speech_active` tests and Phase 3's `main_loop`/barge-in tests. It is documented here as not applicable, not force-fit into `eval/cases.json`.
- **Add a case every time real usage (visible in Phase 1's turn log) surfaces a wrong tool call or a hallucinated figure** (PRD Section 17.2's closing instruction) — this is an ongoing process note for you, not something this phase builds.

---

## File Structure

```
financial_voice_agent/
    eval/
        __init__.py
        cases.py       # EvalCase, load_eval_cases()
        runner.py       # EvalResult, run_eval_case(), run_eval_set()
        report.py        # format_report()
eval/
    cases.json          # the 7 starting test cases (PRD Section 17.2, minus the barge-in case)
tests/
    eval/
        __init__.py
        test_cases.py
        test_runner.py
        test_report.py
```

- `cases.py`: owns the eval case data shape and JSON loading. Nothing else parses `eval/cases.json`.
- `runner.py`: owns driving one case (or a whole set) through `run_llm_turn` and comparing actual vs. expected/forbidden tool calls. Nothing else calls `run_llm_turn` for eval purposes.
- `report.py`: owns turning a list of `EvalResult`s into a human-readable summary. Nothing else formats eval output.

---

### Task 1: Eval case format + loader + starting cases

**Files:**
- Create: `financial_voice_agent/eval/__init__.py` (empty)
- Create: `financial_voice_agent/eval/cases.py`
- Create: `eval/cases.json`
- Test: `tests/eval/__init__.py` (empty)
- Test: `tests/eval/test_cases.py`

**Interfaces:**
- Consumes: nothing from earlier phases.
- Produces:
  - `@dataclass(frozen=True) class EvalCase: name: str; transcript: str; expected_tools: list[str]; forbidden_tools: list[str]; mocked_screen_result: dict | None = None`
  - `def load_eval_cases(path: str) -> list[EvalCase]` — reads a JSON file (a list of case objects), defaults `expected_tools`/`forbidden_tools` to `[]` and `mocked_screen_result` to `None` if omitted from a case's JSON.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_cases.py
from financial_voice_agent.eval.cases import EvalCase, load_eval_cases


def test_load_eval_cases_reads_json_file(tmp_path):
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(
        '[{"name": "t1", "transcript": "hi", "expected_tools": ["get_quote"], "forbidden_tools": ["capture_screen"]}]'
    )

    cases = load_eval_cases(str(cases_file))

    assert cases == [
        EvalCase(name="t1", transcript="hi", expected_tools=["get_quote"], forbidden_tools=["capture_screen"])
    ]


def test_load_eval_cases_supports_mocked_screen_result(tmp_path):
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(
        '[{"name": "t2", "transcript": "whats on screen", "expected_tools": ["capture_screen"], '
        '"forbidden_tools": [], "mocked_screen_result": {"screenshot_path": "x.jpg", "width": 100, "height": 100}}]'
    )

    cases = load_eval_cases(str(cases_file))

    assert cases[0].mocked_screen_result == {"screenshot_path": "x.jpg", "width": 100, "height": 100}


def test_load_eval_cases_defaults_missing_optional_fields(tmp_path):
    cases_file = tmp_path / "cases.json"
    cases_file.write_text('[{"name": "t3", "transcript": "buy shares"}]')

    cases = load_eval_cases(str(cases_file))

    assert cases[0].expected_tools == []
    assert cases[0].forbidden_tools == []
    assert cases[0].mocked_screen_result is None


def test_repo_eval_cases_file_is_valid_and_loadable():
    # Guards the checked-in starting cases (PRD Section 17.2) used by the
    # manual eval run.
    cases = load_eval_cases("eval/cases.json")

    assert len(cases) == 7
    names = {c.name for c in cases}
    assert names == {
        "nifty_level_quote",
        "screen_instrument_rsi",
        "latest_nifty_news",
        "current_holdings",
        "decline_buy_order",
        "bollinger_bands_reliance",
        "whats_on_screen",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/eval/test_cases.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.eval'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/eval/cases.py
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    name: str
    transcript: str
    expected_tools: list[str]
    forbidden_tools: list[str]
    mocked_screen_result: dict | None = None


def load_eval_cases(path: str) -> list[EvalCase]:
    with open(path, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)
    return [
        EvalCase(
            name=c["name"],
            transcript=c["transcript"],
            expected_tools=c.get("expected_tools", []),
            forbidden_tools=c.get("forbidden_tools", []),
            mocked_screen_result=c.get("mocked_screen_result"),
        )
        for c in raw_cases
    ]
```

```json
// eval/cases.json
[
  {
    "name": "nifty_level_quote",
    "transcript": "What's the Nifty level right now?",
    "expected_tools": ["get_quote"],
    "forbidden_tools": ["capture_screen"]
  },
  {
    "name": "screen_instrument_rsi",
    "transcript": "What is this instrument and what's its RSI?",
    "expected_tools": ["capture_screen", "compute_indicator"],
    "forbidden_tools": [],
    "mocked_screen_result": {"screenshot_path": "mock_screenshot.jpg", "width": 1920, "height": 1080}
  },
  {
    "name": "latest_nifty_news",
    "transcript": "Fetch me the latest news on Nifty.",
    "expected_tools": ["get_news"],
    "forbidden_tools": ["get_positions_holdings"]
  },
  {
    "name": "current_holdings",
    "transcript": "What are my current holdings?",
    "expected_tools": ["get_positions_holdings"],
    "forbidden_tools": ["get_news"]
  },
  {
    "name": "decline_buy_order",
    "transcript": "Buy 10 shares of Reliance.",
    "expected_tools": [],
    "forbidden_tools": []
  },
  {
    "name": "bollinger_bands_reliance",
    "transcript": "Show me Bollinger Bands for Reliance, 15-minute chart.",
    "expected_tools": ["get_ohlc_history", "compute_indicator"],
    "forbidden_tools": []
  },
  {
    "name": "whats_on_screen",
    "transcript": "What's on my screen?",
    "expected_tools": ["capture_screen"],
    "forbidden_tools": ["get_quote", "get_ohlc_history", "get_positions_holdings", "get_news"],
    "mocked_screen_result": {"screenshot_path": "mock_screenshot.jpg", "width": 1920, "height": 1080}
  }
]
```

Also create empty `financial_voice_agent/eval/__init__.py` and empty `tests/eval/__init__.py`.

Note: `decline_buy_order` has `expected_tools: []` — this is PRD case 5 ("agent should verbally decline, no tool call"). Task 2's runner treats an empty `expected_tools` list as "no tool call at all is allowed," not merely "nothing specific is required" — see Task 2's interface note.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/eval/test_cases.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/eval/__init__.py financial_voice_agent/eval/cases.py eval/cases.json tests/eval/__init__.py tests/eval/test_cases.py
git commit -m "feat: add eval case format, loader, and PRD Section 17.2 starting cases"
```

---

### Task 2: Eval runner

**Files:**
- Create: `financial_voice_agent/eval/runner.py`
- Test: `tests/eval/test_runner.py`

**Interfaces:**
- Consumes: `EvalCase` (Task 1); `run_llm_turn`, `LlmClient`, `LlmCompletion`, `ToolCall` (Phase 3, `financial_voice_agent.orchestrator.llm` — exact signatures already exist, do not recreate).
- Produces:
  - `@dataclass(frozen=True) class EvalResult: case_name: str; passed: bool; actual_tools: list[str]; missing_tools: list[str]; unexpected_forbidden_tools: list[str]; error: str | None = None`
  - `async def run_eval_case(case: EvalCase, *, llm_client, base_tool_executor, model: str, tools_schema: list[dict]) -> EvalResult` — drives `run_llm_turn(llm_client, case.transcript, [], model=model, tools_schema=tools_schema, tool_executor=<wrapped>)`, where `<wrapped>` records every tool name called (in order) and substitutes `case.mocked_screen_result` for a real `capture_screen` call when the case specifies one, instead of delegating to `base_tool_executor` for that one tool. **Never raises** — a failure in the LLM call itself is caught and recorded as `EvalResult.error`, with `passed=False`.
    - Pass/fail rule: if `case.expected_tools` is non-empty, `passed` requires every name in it to appear among the tools actually called, AND no name in `case.forbidden_tools` was called. If `case.expected_tools` is empty, `passed` requires **no tool call at all** (matching PRD case 5's "the agent should decline, not call anything").
  - `async def run_eval_set(cases: list[EvalCase], *, llm_client, base_tool_executor, model: str, tools_schema: list[dict]) -> list[EvalResult]` — runs each case in sequence via `run_eval_case`, returning one `EvalResult` per case.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_runner.py
import pytest

from financial_voice_agent.eval.cases import EvalCase
from financial_voice_agent.eval.runner import run_eval_case, run_eval_set
from financial_voice_agent.orchestrator.llm import LlmCompletion, ToolCall


class _ScriptedLlmClient:
    """Returns each round's scripted tool_calls in order, then finalizes with
    no tool_calls once the script is exhausted -- exactly one round list per
    logical "batch" of tool calls a single eval case needs."""

    def __init__(self, rounds: list[list[ToolCall]]):
        self._rounds = rounds
        self._index = 0

    async def complete(self, messages, *, model, tools_schema):
        if self._index < len(self._rounds):
            calls = self._rounds[self._index]
            self._index += 1
            return LlmCompletion(text=None if calls else "done", tool_calls=calls)
        return LlmCompletion(text="done", tool_calls=[])


class _TranscriptKeyedLlmClient:
    """A smarter fake for exercising run_eval_set across MULTIPLE cases with
    one shared client instance (matching how a real Groq client is reused
    across cases). Keys its scripted tool_calls off the case's transcript and
    off whether a tool reply has already been seen in this case's own
    message history, rather than a single global call counter -- this stays
    correct no matter how many `.complete()` calls a previous case in the set
    already consumed, since each case starts a fresh `history=[]`."""

    def __init__(self, tool_calls_by_transcript: dict[str, list[ToolCall]]):
        self._tool_calls_by_transcript = tool_calls_by_transcript

    async def complete(self, messages, *, model, tools_schema):
        has_tool_reply = any(m.get("role") == "tool" for m in messages)
        if not has_tool_reply:
            last_user_message = next(m["content"] for m in reversed(messages) if m["role"] == "user")
            calls = self._tool_calls_by_transcript.get(last_user_message, [])
            if calls:
                return LlmCompletion(text=None, tool_calls=calls)
        return LlmCompletion(text="done", tool_calls=[])


async def _fake_tool_executor(call: ToolCall) -> dict:
    return {"result": f"executed {call.name}"}


@pytest.mark.asyncio
async def test_run_eval_case_passes_when_expected_tool_called():
    case = EvalCase(
        name="quote_case", transcript="what's the nifty level",
        expected_tools=["get_quote"], forbidden_tools=["capture_screen"],
    )
    client = _ScriptedLlmClient([[ToolCall(id="1", name="get_quote", arguments={"symbol": "NIFTY 50"})]])

    result = await run_eval_case(
        case, llm_client=client, base_tool_executor=_fake_tool_executor, model="test-model", tools_schema=[]
    )

    assert result.passed is True
    assert result.actual_tools == ["get_quote"]
    assert result.missing_tools == []
    assert result.unexpected_forbidden_tools == []


@pytest.mark.asyncio
async def test_run_eval_case_fails_when_expected_tool_missing():
    case = EvalCase(name="quote_case", transcript="what's the nifty level", expected_tools=["get_quote"], forbidden_tools=[])
    client = _ScriptedLlmClient([[]])  # no tool calls at all

    result = await run_eval_case(
        case, llm_client=client, base_tool_executor=_fake_tool_executor, model="test-model", tools_schema=[]
    )

    assert result.passed is False
    assert result.missing_tools == ["get_quote"]


@pytest.mark.asyncio
async def test_run_eval_case_fails_when_forbidden_tool_called():
    case = EvalCase(
        name="news_case", transcript="latest news on nifty",
        expected_tools=["get_news"], forbidden_tools=["get_positions_holdings"],
    )
    client = _ScriptedLlmClient([[
        ToolCall(id="1", name="get_news", arguments={"query": "Nifty"}),
        ToolCall(id="2", name="get_positions_holdings", arguments={}),
    ]])

    result = await run_eval_case(
        case, llm_client=client, base_tool_executor=_fake_tool_executor, model="test-model", tools_schema=[]
    )

    assert result.passed is False
    assert "get_positions_holdings" in result.unexpected_forbidden_tools


@pytest.mark.asyncio
async def test_run_eval_case_fails_when_any_tool_called_but_none_expected():
    case = EvalCase(name="decline_case", transcript="buy 10 shares of reliance", expected_tools=[], forbidden_tools=[])
    client = _ScriptedLlmClient([[ToolCall(id="1", name="get_quote", arguments={"symbol": "RELIANCE"})]])

    result = await run_eval_case(
        case, llm_client=client, base_tool_executor=_fake_tool_executor, model="test-model", tools_schema=[]
    )

    assert result.passed is False
    assert "get_quote" in result.unexpected_forbidden_tools


@pytest.mark.asyncio
async def test_run_eval_case_passes_when_none_expected_and_none_called():
    case = EvalCase(name="decline_case", transcript="buy 10 shares of reliance", expected_tools=[], forbidden_tools=[])
    client = _ScriptedLlmClient([[]])

    result = await run_eval_case(
        case, llm_client=client, base_tool_executor=_fake_tool_executor, model="test-model", tools_schema=[]
    )

    assert result.passed is True
    assert result.actual_tools == []


@pytest.mark.asyncio
async def test_run_eval_case_uses_mocked_screen_result_instead_of_real_capture():
    case = EvalCase(
        name="screen_case", transcript="what's on my screen?",
        expected_tools=["capture_screen"], forbidden_tools=[],
        mocked_screen_result={"screenshot_path": "mock.jpg", "width": 1920, "height": 1080},
    )
    reached_base_executor = {"called": False}

    async def tool_executor_that_would_fail_on_real_capture(call: ToolCall) -> dict:
        reached_base_executor["called"] = True
        raise AssertionError("real capture_screen must not be invoked when mocked_screen_result is provided")

    client = _ScriptedLlmClient([[ToolCall(id="1", name="capture_screen", arguments={})]])

    result = await run_eval_case(
        case, llm_client=client, base_tool_executor=tool_executor_that_would_fail_on_real_capture,
        model="test-model", tools_schema=[],
    )

    assert result.passed is True
    assert reached_base_executor["called"] is False


@pytest.mark.asyncio
async def test_run_eval_case_records_error_without_crashing():
    case = EvalCase(name="broken_case", transcript="x", expected_tools=["get_quote"], forbidden_tools=[])

    class _AlwaysFailsClient:
        async def complete(self, messages, *, model, tools_schema):
            raise RuntimeError("llm unavailable")

    result = await run_eval_case(
        case, llm_client=_AlwaysFailsClient(), base_tool_executor=_fake_tool_executor,
        model="test-model", tools_schema=[],
    )

    assert result.passed is False
    assert result.error is not None


@pytest.mark.asyncio
async def test_run_eval_set_runs_all_cases_independently():
    cases = [
        EvalCase(name="a", transcript="t1", expected_tools=["get_quote"], forbidden_tools=[]),
        EvalCase(name="b", transcript="t2", expected_tools=["get_news"], forbidden_tools=[]),
    ]
    client = _TranscriptKeyedLlmClient({
        "t1": [ToolCall(id="1", name="get_quote", arguments={})],
        "t2": [ToolCall(id="2", name="get_news", arguments={"query": "x"})],
    })

    results = await run_eval_set(
        cases, llm_client=client, base_tool_executor=_fake_tool_executor, model="test-model", tools_schema=[]
    )

    assert len(results) == 2
    assert results[0].case_name == "a"
    assert results[0].passed is True
    assert results[1].case_name == "b"
    assert results[1].passed is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/eval/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.eval.runner'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/eval/runner.py
from __future__ import annotations

from dataclasses import dataclass

from financial_voice_agent.eval.cases import EvalCase
from financial_voice_agent.orchestrator.llm import ToolCall, run_llm_turn


@dataclass(frozen=True)
class EvalResult:
    case_name: str
    passed: bool
    actual_tools: list[str]
    missing_tools: list[str]
    unexpected_forbidden_tools: list[str]
    error: str | None = None


def _wrap_tool_executor(base_tool_executor, mocked_screen_result: dict | None, recorded_calls: list[str]):
    async def executor(call: ToolCall) -> dict:
        recorded_calls.append(call.name)
        if call.name == "capture_screen" and mocked_screen_result is not None:
            return mocked_screen_result
        return await base_tool_executor(call)

    return executor


async def run_eval_case(
    case: EvalCase, *, llm_client, base_tool_executor, model: str, tools_schema: list[dict]
) -> EvalResult:
    recorded_calls: list[str] = []
    wrapped_executor = _wrap_tool_executor(base_tool_executor, case.mocked_screen_result, recorded_calls)

    error: str | None = None
    try:
        await run_llm_turn(
            llm_client,
            case.transcript,
            [],
            model=model,
            tools_schema=tools_schema,
            tool_executor=wrapped_executor,
        )
    except Exception as exc:  # noqa: BLE001 -- one broken eval case must not crash the whole run
        error = str(exc)

    missing = [t for t in case.expected_tools if t not in recorded_calls]
    forbidden_hit = [t for t in case.forbidden_tools if t in recorded_calls]
    if not case.expected_tools:
        # PRD case 5: "agent should decline, no tool call" -- ANY tool call is a failure.
        unexpected_when_none_expected = list(recorded_calls)
    else:
        unexpected_when_none_expected = []

    passed = (
        error is None
        and not missing
        and not forbidden_hit
        and not unexpected_when_none_expected
    )

    return EvalResult(
        case_name=case.name,
        passed=passed,
        actual_tools=recorded_calls,
        missing_tools=missing,
        unexpected_forbidden_tools=forbidden_hit + unexpected_when_none_expected,
        error=error,
    )


async def run_eval_set(
    cases: list[EvalCase], *, llm_client, base_tool_executor, model: str, tools_schema: list[dict]
) -> list[EvalResult]:
    results = []
    for case in cases:
        result = await run_eval_case(
            case,
            llm_client=llm_client,
            base_tool_executor=base_tool_executor,
            model=model,
            tools_schema=tools_schema,
        )
        results.append(result)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/eval/test_runner.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/eval/runner.py tests/eval/test_runner.py
git commit -m "feat: add eval runner driving run_llm_turn with tool-call assertions"
```

---

### Task 3: Report formatting + manual entry point for running against real Groq

**Files:**
- Create: `financial_voice_agent/eval/report.py`
- Test: `tests/eval/test_report.py`

**Interfaces:**
- Consumes: `EvalResult` (Task 2).
- Produces: `def format_report(results: list[EvalResult]) -> str` — a human-readable summary: a `passed/total` header line, then one line per case (`[PASS]`/`[FAIL]` + case name), with missing/forbidden tools and any error indented underneath each failing case.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_report.py
from financial_voice_agent.eval.report import format_report
from financial_voice_agent.eval.runner import EvalResult


def test_format_report_summarizes_pass_fail_counts():
    results = [
        EvalResult(case_name="a", passed=True, actual_tools=["get_quote"], missing_tools=[], unexpected_forbidden_tools=[]),
        EvalResult(case_name="b", passed=False, actual_tools=[], missing_tools=["get_news"], unexpected_forbidden_tools=[]),
    ]

    report = format_report(results)

    assert "1/2 passed" in report
    assert "[PASS] a" in report
    assert "[FAIL] b" in report
    assert "missing expected tools: ['get_news']" in report


def test_format_report_includes_forbidden_tools_and_error_for_failures():
    results = [
        EvalResult(
            case_name="c", passed=False, actual_tools=["get_news", "get_positions_holdings"],
            missing_tools=[], unexpected_forbidden_tools=["get_positions_holdings"],
        ),
        EvalResult(
            case_name="d", passed=False, actual_tools=[], missing_tools=["get_quote"],
            unexpected_forbidden_tools=[], error="llm unavailable",
        ),
    ]

    report = format_report(results)

    assert "0/2 passed" in report
    assert "called forbidden/unexpected tools: ['get_positions_holdings']" in report
    assert "error: llm unavailable" in report


def test_format_report_omits_detail_lines_for_passing_cases():
    results = [
        EvalResult(case_name="a", passed=True, actual_tools=["get_quote"], missing_tools=[], unexpected_forbidden_tools=[]),
    ]

    report = format_report(results)

    lines = report.splitlines()
    pass_line_index = next(i for i, line in enumerate(lines) if line == "[PASS] a")
    # No indented detail line follows a passing case.
    assert pass_line_index == len(lines) - 1 or not lines[pass_line_index + 1].startswith("    ")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.eval.report'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/eval/report.py
from __future__ import annotations

from financial_voice_agent.eval.runner import EvalResult


def format_report(results: list[EvalResult]) -> str:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    lines = [f"Eval results: {passed}/{total} passed", ""]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"[{status}] {r.case_name}")
        if not r.passed:
            if r.missing_tools:
                lines.append(f"    missing expected tools: {r.missing_tools}")
            if r.unexpected_forbidden_tools:
                lines.append(f"    called forbidden/unexpected tools: {r.unexpected_forbidden_tools}")
            if r.error:
                lines.append(f"    error: {r.error}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_report.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full Phase 5 suite**

Run: `pytest tests/ -v`
Expected: PASS (120 baseline from Phases 1-4 + Phase 5's new tests, all green).

- [ ] **Step 6: Manual run against the real Groq model (not automated — record the result, do not skip)**

Once `GROQ_API_KEY` is set (per your earlier mock-first decision, do this only once you're ready — this step costs a small amount of real API usage):

```bash
python -c "
import asyncio
import groq
from financial_voice_agent.config import load_config
from financial_voice_agent.http_clients import create_http_clients, close_http_clients
from financial_voice_agent.orchestrator.llm import RealGroqLlmClient
from financial_voice_agent.tools.registry import TOOLS_SCHEMA, make_tool_executor
from financial_voice_agent.eval.cases import load_eval_cases
from financial_voice_agent.eval.runner import run_eval_set
from financial_voice_agent.eval.report import format_report

async def main():
    config = load_config()
    http_clients = await create_http_clients(config)
    try:
        llm_client = RealGroqLlmClient(groq.AsyncGroq(api_key=config.groq_api_key))
        tool_executor = make_tool_executor(config, http_clients)
        cases = load_eval_cases('eval/cases.json')
        results = await run_eval_set(
            cases, llm_client=llm_client, base_tool_executor=tool_executor,
            model=config.llm_model, tools_schema=TOOLS_SCHEMA,
        )
        print(format_report(results))
    finally:
        await close_http_clients(http_clients)

asyncio.run(main())
"
```

Expected: a report showing 7/7 (or fewer, if the real model surfaces a genuine prompt/tool-schema issue — that's the actual point of the eval set per PRD Section 17). Set `mode: "mock"` in `config.yaml` before running this so Kite-backed tools use Phase 1's fixtures rather than requiring a live Kite session — the real Groq model is what's under test here, not live Kite connectivity. Record the pass/fail result; if any case fails against the real model, that's a real signal about the system prompt or tool schemas, not a bug in this harness.

- [ ] **Step 7: Commit**

```bash
git add financial_voice_agent/eval/report.py tests/eval/test_report.py
git commit -m "feat: add eval report formatting and manual real-Groq run entry point"
```

---

## Phase 5 Exit Criteria

- `pytest tests/ -v` passes with 0 failures.
- `eval/cases.json` contains all 7 applicable PRD Section 17.2 starting cases (the 8th, barge-in, is documented as not applicable to this harness).
- Running the Step 6 manual script against the real Groq model produces a report; the result (pass/fail per case) is recorded, whatever it is — a failing case here is valuable signal, not a defect in the harness.
- No automated test in `tests/eval/` makes a real network call to Groq — all runner/report logic is tested via scripted fake `LlmClient`s.
- Adding a new eval case (from a real usage session surfacing a bad tool call) requires only appending one object to `eval/cases.json` — no code change.

---

## Project Status After Phase 5

All five phases from the original build-order plan are complete: Foundation (config/db/mock/http_clients), Audio Pipeline (capture/VAD/WAV), Turn Orchestrator (STT/LLM/TTS/barge-in), Tools (all six PRD Section 6 tools), and this Eval Harness. What remains before treating this as more than a personal prototype is entirely the PRD's own documented next steps, not a new phase:

- Set up a Kite Connect subscription and flip `mode: "live"` for the Kite-backed tools (PRD Section 18.1/18.4) — the point at which every adapter flagged "verify at build time" across Phases 3-4 actually needs verifying.
- Run the Section 17 eval set (this phase) after every prompt/tool/orchestration change from here on, per the PRD's own instruction.
- Validate the Section 5 latency targets against 20+ real logged turns (Phase 1's turn log makes this possible; it hasn't been done yet because there haven't been 20 real turns).
- Wire `capture_screen`'s `screenshot_path` result into `orchestrator/turn.py`'s actual vision-message assembly — flagged as a known gap in Phase 4's final review, not yet built in any phase.
- Work through the parked minor items across Phases 2-4's plan documents' "Known Limitations" sections whenever they're next touched.
