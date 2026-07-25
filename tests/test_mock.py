import json

import pytest

from financial_voice_agent.mock import FixtureNotFoundError, load_fixture


def test_load_fixture_reads_json(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "quote.json").write_text(json.dumps({"symbol": "NIFTY 50", "ltp": 24500}))

    result = load_fixture("quote", fixtures_dir=str(fixtures_dir))

    assert result == {"symbol": "NIFTY 50", "ltp": 24500}


def test_load_fixture_missing_file_raises(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()

    with pytest.raises(FixtureNotFoundError, match="does_not_exist"):
        load_fixture("does_not_exist", fixtures_dir=str(fixtures_dir))


def test_repo_fixtures_are_valid_json():
    # Guards the checked-in fixtures used by mode="mock" at runtime.
    for name in ("quote", "ohlc_history", "positions_holdings"):
        result = load_fixture(name, fixtures_dir="fixtures")
        assert isinstance(result, dict)
