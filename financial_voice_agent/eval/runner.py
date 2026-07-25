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
    skipped: bool = False
    note: str | None = None


def _wrap_tool_executor(base_tool_executor, mocked_screen_result: dict | None, recorded_calls: list[str]):
    async def executor(call: ToolCall) -> dict:
        recorded_calls.append(call.name)
        if call.name == "capture_screen":
            if mocked_screen_result is not None:
                return mocked_screen_result
            raise RuntimeError(
                "capture_screen was called but this eval case has no mocked_screen_result -- "
                "add one rather than letting the eval run touch real screen/window hardware."
            )
        return await base_tool_executor(call)

    return executor


async def run_eval_case(
    case: EvalCase,
    *,
    llm_client,
    base_tool_executor,
    model: str,
    tools_schema: list[dict],
    system_prompt: str | None = None,
) -> EvalResult:
    if case.skip_reason is not None:
        return EvalResult(
            case_name=case.name,
            passed=True,
            actual_tools=[],
            missing_tools=[],
            unexpected_forbidden_tools=[],
            skipped=True,
            note=case.skip_reason,
        )

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
            system_prompt=system_prompt,
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
        unexpected_forbidden_tools=list(dict.fromkeys(forbidden_hit + unexpected_when_none_expected)),
        error=error,
    )


async def run_eval_set(
    cases: list[EvalCase],
    *,
    llm_client,
    base_tool_executor,
    model: str,
    tools_schema: list[dict],
    system_prompt: str | None = None,
) -> list[EvalResult]:
    results = []
    for case in cases:
        result = await run_eval_case(
            case,
            llm_client=llm_client,
            base_tool_executor=base_tool_executor,
            model=model,
            tools_schema=tools_schema,
            system_prompt=system_prompt,
        )
        results.append(result)
    return results
