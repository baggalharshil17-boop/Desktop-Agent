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
