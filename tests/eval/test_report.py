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


def test_format_report_shows_skipped_cases_separately_from_pass_fail_counts():
    results = [
        EvalResult(case_name="a", passed=True, actual_tools=["get_quote"], missing_tools=[], unexpected_forbidden_tools=[]),
        EvalResult(
            case_name="b", passed=True, actual_tools=[], missing_tools=[], unexpected_forbidden_tools=[],
            skipped=True, note="vision not wired yet",
        ),
    ]

    report = format_report(results)

    assert "1/1 passed (1 skipped)" in report
    assert "[SKIP] b" in report
    assert "vision not wired yet" in report
