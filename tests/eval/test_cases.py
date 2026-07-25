import os

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
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cases_path = os.path.join(repo_root, "eval", "cases.json")
    cases = load_eval_cases(cases_path)

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


def test_load_eval_cases_supports_skip_reason(tmp_path):
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(
        '[{"name": "t4", "transcript": "x", "skip_reason": "not ready yet"}]'
    )

    cases = load_eval_cases(str(cases_file))

    assert cases[0].skip_reason == "not ready yet"


def test_repo_screen_instrument_rsi_case_is_marked_skipped():
    cases = load_eval_cases("eval/cases.json")
    case = next(c for c in cases if c.name == "screen_instrument_rsi")
    assert case.skip_reason is not None
