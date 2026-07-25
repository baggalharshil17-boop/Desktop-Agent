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
