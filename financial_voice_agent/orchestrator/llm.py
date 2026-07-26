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
    system_prompt: str | None = None,
    max_tool_rounds: int = 3,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
) -> LlmTurnResult:
    messages = list(history)
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}, *messages]
    messages.append({"role": "user", "content": transcript})
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
            # capture_screen's result carries the actual image (image_b64/
            # image_mime) so a vision-capable model can describe what's on
            # screen -- see financial_voice_agent/tools/screen.py. Neither
            # the tool message sent back to the model nor the turn log
            # needs a multi-KB base64 blob duplicated in text form, so both
            # get the stripped dict; the image itself goes out as a
            # separate image_url content block below.
            image_b64 = result.get("image_b64")
            image_mime = result.get("image_mime", "image/jpeg")
            result_for_logging = {k: v for k, v in result.items() if k not in ("image_b64", "image_mime")}
            all_tool_calls.append({"tool": call.name, "args": call.arguments})
            all_tool_results.append(result_for_logging)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result_for_logging)}
            )
            if image_b64:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Here is the screenshot from capture_screen:"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{image_mime};base64,{image_b64}"},
                            },
                        ],
                    }
                )

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
                raise LlmError("LLM rate limited repeatedly", rate_limited=True) from exc
        try:
            return await retry_with_fixed_backoff(
                _attempt, max_attempts=2, backoff_seconds=1.0, sleep_fn=sleep_fn
            )
        except RetryExhaustedError as exc:
            raise LlmError("LLM failed after retries") from exc


class RealGroqLlmClient:
    """Thin adapter around groq.AsyncGroq's chat.completions.create.

    Verify against console.groq.com/docs/api-reference at build time.

    Qwen and gpt-oss models on Groq default to an extended "thinking mode"
    that can add tens of seconds of reasoning-token latency per call
    (confirmed live: a single get_news turn took 45s of LLM time with
    qwen/qwen3.6-27b's default reasoning_effort). reasoning_effort="none"
    disables it -- per console.groq.com/docs/reasoning, this parameter is
    specific to Qwen/gpt-oss models, so it's only sent for those to avoid
    a 400 from models that don't recognize it.
    """

    def __init__(self, groq_async_client) -> None:
        self._client = groq_async_client

    async def complete(
        self, messages: list[dict], *, model: str, tools_schema: list[dict]
    ) -> LlmCompletion:
        extra_kwargs = {}
        if "qwen" in model.lower() or "gpt-oss" in model.lower():
            extra_kwargs["reasoning_effort"] = "none"
        response = await self._client.chat.completions.create(
            messages=messages,
            model=model,
            tools=tools_schema or None,
            tool_choice="auto" if tools_schema else None,
            **extra_kwargs,
        )
        message = response.choices[0].message
        raw_tool_calls = message.tool_calls or []
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
            for tc in raw_tool_calls
        ]
        return LlmCompletion(text=message.content, tool_calls=tool_calls)


class HuggingFaceLlmClient:
    """Thin adapter around huggingface_hub's AsyncInferenceClient.chat_completion,
    which is OpenAI-compatible (same messages/tools/tool_calls shapes as Groq's
    client) -- verified against the installed huggingface_hub SDK at build time.

    Not every model/provider combination on HF Inference supports tool calling
    reliably the way Groq's Llama models do -- verify with a real call before
    trusting this in the live voice loop, and be ready to swap llm.model in
    config.yaml if the configured model doesn't emit tool_calls correctly.
    """

    def __init__(self, client) -> None:
        self._client = client  # a huggingface_hub.AsyncInferenceClient

    async def complete(
        self, messages: list[dict], *, model: str, tools_schema: list[dict]
    ) -> LlmCompletion:
        response = await self._client.chat_completion(
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


def make_llm_client(config):
    """Picks the real LLM adapter based on config.llm_provider -- the seam
    that lets llm.provider in config.yaml switch between Groq and Hugging
    Face without touching any calling code."""
    if config.llm_provider == "huggingface":
        from huggingface_hub import AsyncInferenceClient

        # "hf-inference" (HF's own serverless infra, used for STT) does not
        # serve most chat/tool-calling models -- "auto" routes to whichever
        # of HF's partner Inference Providers (Together, Novita, Fireworks,
        # etc.) actually hosts config.llm_model. Verified against a real call
        # with meta-llama/Llama-3.3-70B-Instruct at build time.
        return HuggingFaceLlmClient(
            AsyncInferenceClient(provider="auto", api_key=config.huggingface_api_key)
        )
    if config.llm_provider == "groq":
        import groq

        return RealGroqLlmClient(groq.AsyncGroq(api_key=config.groq_api_key))
    raise ValueError(f"Unsupported llm_provider: {config.llm_provider!r}")
