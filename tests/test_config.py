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
      voice_id: "test-voice-id"
    stt:
      provider: "groq"
      model: "test-stt-model"
    llm:
      provider: "groq"
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
    assert config.barge_in_enabled is True  # default when vad.barge_in_enabled is absent
    assert config.audio_output_device_index is None
    assert config.audio_input_device_index is None
    assert config.input_mode == "always_on"
    assert config.tts_provider == "cartesia"
    assert config.stt_provider == "groq"
    assert config.stt_model == "test-stt-model"
    assert config.llm_provider == "groq"
    assert config.llm_model == "test-model"
    assert config.storage_db_path == "./agent_turns.db"
    assert config.mode == "live"
    assert config.groq_api_key == "groq-secret"
    assert config.cartesia_api_key == "cartesia-secret"


def test_load_config_reads_explicit_input_device_index(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    yaml_path = _write_yaml(
        tmp_path,
        """\
        vad:
          speech_threshold: 0.5
          silence_duration_ms: 600
          min_speech_duration_ms: 200
        audio:
          output_device_index: null
          input_device_index: 9
        input_mode: "always_on"
        tts:
          provider: "cartesia"
          voice_id: "test-voice-id"
        stt:
          provider: "groq"
          model: "test-stt-model"
        llm:
          provider: "groq"
          model: "test-model"
        storage:
          db_path: "./agent_turns.db"
        mode: "live"
        """,
    )
    env_path = _write_env(
        tmp_path, "GROQ_API_KEY=groq-secret\nCARTESIA_API_KEY=cartesia-secret\n"
    )

    config = load_config(config_path=yaml_path, env_path=env_path)

    assert config.audio_input_device_index == 9


def test_load_config_reads_explicit_barge_in_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    yaml_path = _write_yaml(
        tmp_path,
        """\
        vad:
          speech_threshold: 0.5
          silence_duration_ms: 600
          min_speech_duration_ms: 200
          barge_in_enabled: false
        audio:
          output_device_index: null
        input_mode: "always_on"
        tts:
          provider: "cartesia"
          voice_id: "test-voice-id"
        stt:
          provider: "groq"
          model: "test-stt-model"
        llm:
          provider: "groq"
          model: "test-model"
        storage:
          db_path: "./agent_turns.db"
        mode: "live"
        """,
    )
    env_path = _write_env(
        tmp_path, "GROQ_API_KEY=groq-secret\nCARTESIA_API_KEY=cartesia-secret\n"
    )

    config = load_config(config_path=yaml_path, env_path=env_path)

    assert config.barge_in_enabled is False


def test_load_config_missing_groq_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    yaml_path = _write_yaml(tmp_path, VALID_YAML)  # stt.provider and llm.provider are both "groq"
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


def test_load_config_missing_cartesia_voice_id_raises(tmp_path, monkeypatch):
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
          voice_id: "<pick a voice id from https://play.cartesia.ai/voices>"
        stt:
          provider: "groq"
          model: "test-stt-model"
        llm:
          provider: "groq"
          model: "test-model"
        storage:
          db_path: "./agent_turns.db"
        mode: "live"
        """,
    )
    env_path = _write_env(
        tmp_path, "GROQ_API_KEY=groq-secret\nCARTESIA_API_KEY=cartesia-secret\n"
    )

    with pytest.raises(ConfigError, match="voice_id"):
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
          provider: "groq"
          model: "test-model"
        storage:
          db_path: "./agent_turns.db"
        mode: "live"
        """,
    )
    env_path = _write_env(tmp_path, "GROQ_API_KEY=groq-secret\n")

    with pytest.raises(ConfigError, match="elevenlabs"):
        load_config(config_path=yaml_path, env_path=env_path)


def test_load_config_invalid_stt_provider_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
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
          voice_id: "test-voice-id"
        stt:
          provider: "whisper-cpp"
          model: "test-stt-model"
        llm:
          provider: "groq"
          model: "test-model"
        storage:
          db_path: "./agent_turns.db"
        mode: "live"
        """,
    )
    env_path = _write_env(tmp_path, "GROQ_API_KEY=groq-secret\nCARTESIA_API_KEY=cartesia-secret\n")

    with pytest.raises(ConfigError, match="whisper-cpp"):
        load_config(config_path=yaml_path, env_path=env_path)


def test_load_config_missing_huggingface_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
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
          voice_id: "test-voice-id"
        stt:
          provider: "huggingface"
          model: "openai/whisper-large-v3"
        llm:
          provider: "groq"
          model: "test-model"
        storage:
          db_path: "./agent_turns.db"
        mode: "live"
        """,
    )
    env_path = _write_env(tmp_path, "GROQ_API_KEY=groq-secret\nCARTESIA_API_KEY=cartesia-secret\n")

    with pytest.raises(ConfigError, match="HF_TOKEN"):
        load_config(config_path=yaml_path, env_path=env_path)


def test_load_config_huggingface_stt_provider_reads_key(tmp_path, monkeypatch):
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
          voice_id: "test-voice-id"
        stt:
          provider: "huggingface"
          model: "openai/whisper-large-v3"
        llm:
          provider: "groq"
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
        HF_TOKEN=hf-secret
        """,
    )

    config = load_config(config_path=yaml_path, env_path=env_path)

    assert config.stt_provider == "huggingface"
    assert config.stt_model == "openai/whisper-large-v3"
    assert config.huggingface_api_key == "hf-secret"


def test_load_config_invalid_llm_provider_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
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
          voice_id: "test-voice-id"
        stt:
          provider: "groq"
          model: "test-stt-model"
        llm:
          provider: "openai"
          model: "test-model"
        storage:
          db_path: "./agent_turns.db"
        mode: "live"
        """,
    )
    env_path = _write_env(tmp_path, "GROQ_API_KEY=groq-secret\nCARTESIA_API_KEY=cartesia-secret\n")

    with pytest.raises(ConfigError, match="openai"):
        load_config(config_path=yaml_path, env_path=env_path)


def test_load_config_huggingface_llm_provider_does_not_require_groq_key(tmp_path, monkeypatch):
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
          voice_id: "test-voice-id"
        stt:
          provider: "huggingface"
          model: "openai/whisper-large-v3-turbo"
        llm:
          provider: "huggingface"
          model: "test-hf-chat-model"
        storage:
          db_path: "./agent_turns.db"
        mode: "live"
        """,
    )
    env_path = _write_env(
        tmp_path,
        """\
        CARTESIA_API_KEY=cartesia-secret
        HF_TOKEN=hf-secret
        """,
    )

    config = load_config(config_path=yaml_path, env_path=env_path)

    assert config.groq_api_key is None
    assert config.llm_provider == "huggingface"
    assert config.llm_model == "test-hf-chat-model"


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
          voice_id: "test-voice-id"
        llm:
          provider: "groq"
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
          voice_id: "test-voice-id"
        stt:
          provider: "groq"
          model: "test-stt-model"
        llm:
          provider: "groq"
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
