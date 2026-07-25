from __future__ import annotations

from financial_voice_agent.eval.runner import EvalResult


def format_report(results: list[EvalResult]) -> str:
    total = len(results)
    skipped_count = sum(1 for r in results if r.skipped)
    evaluated = total - skipped_count
    passed = sum(1 for r in results if r.passed and not r.skipped)
    header = f"Eval results: {passed}/{evaluated} passed"
    if skipped_count:
        header += f" ({skipped_count} skipped)"
    lines = [header, ""]
    for r in results:
        if r.skipped:
            lines.append(f"[SKIP] {r.case_name}")
            if r.note:
                lines.append(f"    {r.note}")
            continue
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
