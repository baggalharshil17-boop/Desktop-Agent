import os
import textwrap

import pytest

from financial_voice_agent.config import Config, ConfigError, load_config


def _write_yaml(tmp_path, contents):
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(contents))
    return str(path)


def _write_env(tmp_path, contents):
    path = tmp_path / ".env"
    path.write_text(textwrap.dedent(contents))
    return str(path)


VALID_YAML = """\
    vad:
      speech_threshold: 0.5
      silence_duration_ms: 600
      min_speech_duration_ms: 200
    audio:
      output_device_index: null
    input_mode: "always_on"
    tts:
      provider: "cartesia"
    llm:
      model: "test-model"
    storage:
      db_path: "./agent_turns.db"
    mode: "live"
    """


def test_load_config_reads_yaml_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    yaml_path = _write_yaml(tmp_path, VALID_YAML)
    env_path = _write_env(
        tmp_path,
        """\
        GROQ_API_KEY=groq-secret
        CARTESIA_API_KEY=cartesia-secret
        """,
    )

    config = load_config(config_path=yaml_path, env_path=env_path)

    assert isinstance(config, Config)
    assert config.vad_speech_threshold == 0.5
    assert config.vad_silence_duration_ms == 600
    assert config.vad_min_speech_duration_ms == 200
    assert config.audio_output_device_index is None
    assert config.input_mode == "always_on"
    assert config.tts_provider == "cartesia"
    assert config.llm_model == "test-model"
    assert config.storage_db_path == "./agent_turns.db"
    assert config.mode == "live"
    assert config.groq_api_key == "groq-secret"
    assert config.cartesia_api_key == "cartesia-secret"


def test_load_config_missing_groq_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    yaml_path = _write_yaml(tmp_path, VALID_YAML)
    env_path = _write_env(tmp_path, "CARTESIA_API_KEY=cartesia-secret\n")

    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        load_config(config_path=yaml_path, env_path=env_path)


def test_load_config_missing_provider_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    yaml_path = _write_yaml(tmp_path, VALID_YAML)  # tts.provider: cartesia
    env_path = _write_env(tmp_path, "GROQ_API_KEY=groq-secret\n")

    with pytest.raises(ConfigError, match="CARTESIA_API_KEY"):
        load_config(config_path=yaml_path, env_path=env_path)


def test_load_config_invalid_tts_provider_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    yaml_path = _write_yaml(
        tmp_path,
        """\
        vad:
          speech_threshold: 0.5
          silence_duration_ms: 600
          min_speech_duration_ms: 200
        audio:
          output_device_index: null
        input_mode: "always_on"
        tts:
          provider: "elevenlabs"
        llm:
          model: "test-model"
        storage:
          db_path: "./agent_turns.db"
        mode: "live"
        """,
    )
    env_path = _write_env(tmp_path, "GROQ_API_KEY=groq-secret\n")

    with pytest.raises(ConfigError, match="elevenlabs"):
        load_config(config_path=yaml_path, env_path=env_path)


def test_load_config_missing_yaml_file_raises_config_error(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    env_path = _write_env(tmp_path, "GROQ_API_KEY=groq-secret\n")
    missing_path = str(tmp_path / "does_not_exist.yaml")

    with pytest.raises(ConfigError):
        load_config(config_path=missing_path, env_path=env_path)


def test_load_config_malformed_yaml_raises_config_error(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    yaml_path = _write_yaml(
        tmp_path,
        """\
        audio:
          output_device_index: null
        input_mode: "always_on"
        tts:
          provider: "cartesia"
        llm:
          model: "test-model"
        storage:
          db_path: "./agent_turns.db"
        mode: "live"
        """,
    )
    env_path = _write_env(
        tmp_path,
        """\
        GROQ_API_KEY=groq-secret
        CARTESIA_API_KEY=cartesia-secret
        """,
    )

    with pytest.raises(ConfigError):
        load_config(config_path=yaml_path, env_path=env_path)


def test_load_config_placeholder_llm_model_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    yaml_path = _write_yaml(
        tmp_path,
        """\
        vad:
          speech_threshold: 0.5
          silence_duration_ms: 600
          min_speech_duration_ms: 200
        audio:
          output_device_index: null
        input_mode: "always_on"
        tts:
          provider: "cartesia"
        llm:
          model: "<check console.groq.com/docs/models>"
        storage:
          db_path: "./agent_turns.db"
        mode: "live"
        """,
    )
    env_path = _write_env(
        tmp_path,
        """\
        GROQ_API_KEY=groq-secret
        CARTESIA_API_KEY=cartesia-secret
        """,
    )

    with pytest.raises(ConfigError, match="placeholder"):
        load_config(config_path=yaml_path, env_path=env_path)
