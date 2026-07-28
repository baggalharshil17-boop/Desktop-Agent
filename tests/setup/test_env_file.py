from financial_voice_agent.setup.env_file import (
    merge_env_values,
    read_env_file,
    write_env_file,
)


def test_read_env_file_returns_empty_dict_when_file_missing(tmp_path):
    result = read_env_file(str(tmp_path / "does_not_exist.env"))

    assert result == {}


def test_read_env_file_parses_key_value_pairs(tmp_path):
    path = tmp_path / ".env"
    path.write_text("GROQ_API_KEY=abc123\nCARTESIA_API_KEY=xyz789\n")

    result = read_env_file(str(path))

    assert result == {"GROQ_API_KEY": "abc123", "CARTESIA_API_KEY": "xyz789"}


def test_merge_env_values_with_overwrite_true_prefers_new_values():
    existing = {"GROQ_API_KEY": "old", "TAVILY_API_KEY": "keep-me"}
    new_values = {"GROQ_API_KEY": "new"}

    result = merge_env_values(existing, new_values, overwrite=True)

    assert result == {"GROQ_API_KEY": "new", "TAVILY_API_KEY": "keep-me"}


def test_merge_env_values_with_overwrite_false_keeps_existing_values():
    # Re-running setup.py to add a previously-skipped key must not clobber
    # a value the user already has -- only fill in what's missing.
    existing = {"GROQ_API_KEY": "old"}
    new_values = {"GROQ_API_KEY": "new", "CARTESIA_API_KEY": "added"}

    result = merge_env_values(existing, new_values, overwrite=False)

    assert result == {"GROQ_API_KEY": "old", "CARTESIA_API_KEY": "added"}


def test_write_env_file_writes_sorted_key_value_lines(tmp_path):
    path = tmp_path / ".env"

    write_env_file(str(path), {"TAVILY_API_KEY": "b", "GROQ_API_KEY": "a"})

    # Sorted so re-running setup.py produces a stable diff, not a
    # randomly-reordered file every time.
    assert path.read_text() == "GROQ_API_KEY=a\nTAVILY_API_KEY=b\n"


def test_write_env_file_round_trips_through_read_env_file(tmp_path):
    path = tmp_path / ".env"
    values = {"GROQ_API_KEY": "abc", "HF_TOKEN": "def"}

    write_env_file(str(path), values)
    result = read_env_file(str(path))

    assert result == values
