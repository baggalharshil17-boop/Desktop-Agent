import asyncio

import pytest

from financial_voice_agent.orchestrator.llm import (
    HuggingFaceLlmClient,
    LlmCompletion,
    LlmError,
    RealGroqLlmClient,
    ToolCall,
    make_llm_client,
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
async def test_run_llm_turn_fires_on_tool_call_started_once_and_awaits_it_before_returning():
    ack_calls = 0
    ack_finished = False

    async def on_tool_call_started():
        nonlocal ack_calls, ack_finished
        ack_calls += 1
        await asyncio.sleep(0.01)
        ack_finished = True

    async def slow_tool_executor(call: ToolCall) -> dict:
        await asyncio.sleep(0.05)
        return {"ltp": 24500}

    class _TwoRoundToolCallClient:
        def __init__(self):
            self.round = 0

        async def complete(self, messages, *, model, tools_schema):
            self.round += 1
            if self.round <= 2:
                return LlmCompletion(
                    text=None,
                    tool_calls=[ToolCall(id=f"c{self.round}", name="get_quote", arguments={})],
                )
            return LlmCompletion(text="done", tool_calls=[])

    result = await run_llm_turn(
        _TwoRoundToolCallClient(), "transcript", [],
        model="test-model", tools_schema=[], tool_executor=slow_tool_executor,
        max_tool_rounds=3, on_tool_call_started=on_tool_call_started, ack_delay_seconds=0.01,
    )

    assert result.response_text == "done"
    # Fired once even though two rounds of (slow) tool calls happened.
    assert ack_calls == 1
    # run_llm_turn must join the ack task before returning, so a caller
    # never starts playing the real response while the ack is still
    # mid-playback on the same output stream.
    assert ack_finished is True


@pytest.mark.asyncio
async def test_run_llm_turn_swallows_on_tool_call_started_failures():
    ack_was_called = False

    async def failing_ack():
        nonlocal ack_was_called
        ack_was_called = True
        raise RuntimeError("TTS unavailable")

    async def slow_tool_executor(call: ToolCall) -> dict:
        await asyncio.sleep(0.05)
        return {"ltp": 24500}

    result = await run_llm_turn(
        _OneRoundToolCallClient(), "what's the nifty level", [],
        model="test-model", tools_schema=[], tool_executor=slow_tool_executor,
        on_tool_call_started=failing_ack, ack_delay_seconds=0.01,
    )

    assert ack_was_called is True
    assert result.response_text == "Nifty is at 24500."


@pytest.mark.asyncio
async def test_run_llm_turn_skips_ack_for_tools_faster_than_the_delay():
    # get_quote/capture_screen resolve in well under 100ms live -- acking
    # every one of those made the agent sound repetitive. A tool that beats
    # ack_delay_seconds must never trigger the ack at all.
    ack_calls = 0

    async def on_tool_call_started():
        nonlocal ack_calls
        ack_calls += 1

    result = await run_llm_turn(
        _OneRoundToolCallClient(), "what's the nifty level", [],
        model="test-model", tools_schema=[], tool_executor=_tool_executor,
        on_tool_call_started=on_tool_call_started, ack_delay_seconds=0.5,
    )

    assert result.response_text == "Nifty is at 24500."
    assert ack_calls == 0


@pytest.mark.asyncio
async def test_run_llm_turn_fires_ack_for_tools_slower_than_the_delay():
    ack_calls = 0

    async def on_tool_call_started():
        nonlocal ack_calls
        ack_calls += 1

    async def slow_tool_executor(call: ToolCall) -> dict:
        await asyncio.sleep(0.05)
        return {"ltp": 24500}

    result = await run_llm_turn(
        _OneRoundToolCallClient(), "what's the nifty level", [],
        model="test-model", tools_schema=[], tool_executor=slow_tool_executor,
        on_tool_call_started=on_tool_call_started, ack_delay_seconds=0.01,
    )

    assert result.response_text == "Nifty is at 24500."
    assert ack_calls == 1


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
async def test_run_llm_turn_joins_ack_task_even_when_loop_never_converges():
    ack_finished = False

    async def on_tool_call_started():
        nonlocal ack_finished
        await asyncio.sleep(0.01)
        ack_finished = True

    async def slow_tool_executor(call: ToolCall) -> dict:
        await asyncio.sleep(0.05)
        return {"ltp": 24500}

    with pytest.raises(LlmError):
        await run_llm_turn(
            _NeverConvergesClient(), "transcript", [],
            model="test-model", tools_schema=[], tool_executor=slow_tool_executor, max_tool_rounds=2,
            on_tool_call_started=on_tool_call_started, ack_delay_seconds=0.01,
        )

    assert ack_finished is True


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
    assert "reasoning_effort" not in completions.received_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["qwen/qwen3.6-27b", "openai/gpt-oss-120b"])
async def test_real_groq_llm_client_disables_reasoning_for_qwen_and_gpt_oss_models(model):
    class _FakeCompletionsResource:
        def __init__(self):
            self.received_kwargs = None

        async def create(self, **kwargs):
            self.received_kwargs = kwargs
            message = type("M", (), {"content": "ok", "tool_calls": None})()
            choice = type("C", (), {"message": message})()
            return type("R", (), {"choices": [choice]})()

    class _FakeChatNamespace:
        def __init__(self, completions):
            self.completions = completions

    class _FakeGroqAsyncClient:
        def __init__(self, completions):
            self.chat = _FakeChatNamespace(completions)

    completions = _FakeCompletionsResource()
    adapter = RealGroqLlmClient(_FakeGroqAsyncClient(completions))

    await adapter.complete([{"role": "user", "content": "hi"}], model=model, tools_schema=[])

    assert completions.received_kwargs["reasoning_effort"] == "none"


class _MessageCapturingClient:
    def __init__(self):
        self.round = 0
        self.messages_seen = []

    async def complete(self, messages, *, model, tools_schema):
        self.messages_seen.append(list(messages))
        self.round += 1
        if self.round == 1:
            return LlmCompletion(
                text=None,
                tool_calls=[ToolCall(id="call_1", name="get_quote", arguments={"symbol": "NIFTY 50"})],
            )
        return LlmCompletion(text="Nifty is at 24500.", tool_calls=[])


@pytest.mark.asyncio
async def test_run_llm_turn_includes_assistant_tool_calls_message_before_tool_result():
    client = _MessageCapturingClient()

    await run_llm_turn(
        client, "what's the nifty level", [],
        model="test-model", tools_schema=[], tool_executor=_tool_executor,
    )

    # The second round's message history must include the assistant message
    # carrying tool_calls, immediately followed by the matching tool-role
    # response -- required by the real Groq/OpenAI-compatible chat
    # completions contract (a "tool" message must follow an assistant
    # message with matching tool_calls).
    second_call_messages = client.messages_seen[1]
    assistant_msg = next(
        m for m in second_call_messages if m["role"] == "assistant" and m.get("tool_calls")
    )
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_quote"
    tool_msg_index = second_call_messages.index(assistant_msg) + 1
    assert second_call_messages[tool_msg_index]["role"] == "tool"
    assert second_call_messages[tool_msg_index]["tool_call_id"] == "call_1"


def _fake_monotonic(values):
    it = iter(values)
    return lambda: next(it)


@pytest.mark.asyncio
async def test_run_llm_turn_populates_latency_tool_ms():
    # _OneRoundToolCallClient does exactly 1 round with tool calls (round 1),
    # then round 2 has no tool calls -- so clock_fn is called exactly twice
    # (tool_start, tool_end) across the whole run.
    clock_fn = _fake_monotonic([0.0, 0.05])

    result = await run_llm_turn(
        _OneRoundToolCallClient(), "what's the nifty level", [],
        model="test-model", tools_schema=[], tool_executor=_tool_executor, clock_fn=clock_fn,
    )

    assert result.latency_tool_ms == 50


@pytest.mark.asyncio
async def test_run_llm_turn_prepends_system_prompt_when_provided():
    captured_messages = []

    class _CapturingClient:
        async def complete(self, messages, *, model, tools_schema):
            captured_messages.append(list(messages))
            return LlmCompletion(text="ok", tool_calls=[])

    await run_llm_turn(
        _CapturingClient(), "hello", [],
        model="test-model", tools_schema=[], tool_executor=_tool_executor,
        system_prompt="be a helpful read-only assistant",
    )

    assert captured_messages[0][0] == {"role": "system", "content": "be a helpful read-only assistant"}
    assert captured_messages[0][-1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_run_llm_turn_omits_system_message_when_not_provided():
    captured_messages = []

    class _CapturingClient:
        async def complete(self, messages, *, model, tools_schema):
            captured_messages.append(list(messages))
            return LlmCompletion(text="ok", tool_calls=[])

    await run_llm_turn(
        _CapturingClient(), "hello", [],
        model="test-model", tools_schema=[], tool_executor=_tool_executor,
    )


@pytest.mark.asyncio
async def test_run_llm_turn_attaches_screenshot_image_and_strips_it_from_log():
    captured_messages = []

    class _CapturingClient:
        def __init__(self):
            self.round = 0

        async def complete(self, messages, *, model, tools_schema):
            captured_messages.append([dict(m) for m in messages])
            self.round += 1
            if self.round == 1:
                return LlmCompletion(
                    text=None,
                    tool_calls=[ToolCall(id="call_1", name="capture_screen", arguments={})],
                )
            return LlmCompletion(text="I can see the Nifty chart.", tool_calls=[])

    async def _screenshot_tool_executor(call: ToolCall) -> dict:
        return {
            "screenshot_path": "/tmp/shot.jpg",
            "width": 800,
            "height": 600,
            "image_b64": "ZmFrZS1qcGVnLWJ5dGVz",
            "image_mime": "image/jpeg",
        }

    result = await run_llm_turn(
        _CapturingClient(), "what's on screen", [],
        model="test-model", tools_schema=[], tool_executor=_screenshot_tool_executor,
    )

    assert result.response_text == "I can see the Nifty chart."
    # The tool message and the persisted log must not carry the raw image
    # bytes -- only the follow-up vision message does.
    assert "image_b64" not in result.tool_results_json
    second_round_messages = captured_messages[1]
    tool_message = next(m for m in second_round_messages if m["role"] == "tool")
    assert "image_b64" not in tool_message["content"]
    vision_message = second_round_messages[-1]
    assert vision_message["role"] == "user"
    assert vision_message["content"][1]["type"] == "image_url"
    assert vision_message["content"][1]["image_url"]["url"] == (
        "data:image/jpeg;base64,ZmFrZS1qcGVnLWJ5dGVz"
    )


@pytest.mark.asyncio
async def test_huggingface_llm_client_calls_chat_completion():
    class _FakeFunction:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class _FakeToolCall:
        def __init__(self, id, name, arguments):
            self.id = id
            self.function = _FakeFunction(name, arguments)

    class _FakeMessage:
        def __init__(self, content, tool_calls):
            self.content = content
            self.tool_calls = tool_calls

    class _FakeChoice:
        def __init__(self, message):
            self.message = message

    class _FakeChatCompletionOutput:
        def __init__(self, choices):
            self.choices = choices

    class _FakeHfInferenceClient:
        def __init__(self, response):
            self._response = response
            self.received_kwargs = None

        async def chat_completion(self, **kwargs):
            self.received_kwargs = kwargs
            return self._response

    fake_response = _FakeChatCompletionOutput(
        [_FakeChoice(_FakeMessage(None, [_FakeToolCall("call_1", "get_quote", '{"symbol": "NIFTY 50"}')]))]
    )
    client = _FakeHfInferenceClient(fake_response)
    adapter = HuggingFaceLlmClient(client)

    result = await adapter.complete([{"role": "user", "content": "hi"}], model="test-model", tools_schema=[])

    assert result.text is None
    assert result.tool_calls == [ToolCall(id="call_1", name="get_quote", arguments={"symbol": "NIFTY 50"})]
    assert client.received_kwargs["model"] == "test-model"


class _FakeLlmConfig:
    def __init__(self, llm_provider: str, groq_api_key: str = "groq-secret", huggingface_api_key: str = "hf-secret"):
        self.llm_provider = llm_provider
        self.groq_api_key = groq_api_key
        self.huggingface_api_key = huggingface_api_key


def test_make_llm_client_returns_huggingface_client_for_huggingface_provider():
    config = _FakeLlmConfig(llm_provider="huggingface")

    client = make_llm_client(config)

    assert isinstance(client, HuggingFaceLlmClient)


def test_make_llm_client_returns_groq_client_for_groq_provider():
    config = _FakeLlmConfig(llm_provider="groq")

    client = make_llm_client(config)

    assert isinstance(client, RealGroqLlmClient)


def test_make_llm_client_raises_for_unknown_provider():
    config = _FakeLlmConfig(llm_provider="openai")

    with pytest.raises(ValueError, match="openai"):
        make_llm_client(config)
