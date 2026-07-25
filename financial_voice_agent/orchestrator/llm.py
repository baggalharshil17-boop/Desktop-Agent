from __future__ import annotations

import asyncio
import json
import time
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
    clock_fn: Callable[[], float] = time.monotonic,
) -> LlmTurnResult:
    messages = [*history, {"role": "user", "content": transcript}]
    all_tool_calls: list[dict] = []
    all_tool_results: list[dict] = []
    total_tool_ms = 0

    for _round in range(max_tool_rounds):
        completion = await _complete_with_retry(
            client, messages, model=model, tools_schema=tools_schema, sleep_fn=sleep_fn
        )

        if not completion.tool_calls:
            return LlmTurnResult(
                response_text=completion.text or "",
                tool_calls_json=json.dumps(all_tool_calls) if all_tool_calls else None,
                tool_results_json=json.dumps(all_tool_results) if all_tool_results else None,
                latency_tool_ms=total_tool_ms or None,
            )

        messages.append(
            {
                "role": "assistant",
                "content": completion.text,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in completion.tool_calls
                ],
            }
        )

        tool_start = clock_fn()
        results = await asyncio.gather(*(tool_executor(call) for call in completion.tool_calls))
        total_tool_ms += round((clock_fn() - tool_start) * 1000)
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
