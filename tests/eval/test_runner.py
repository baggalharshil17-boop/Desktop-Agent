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
