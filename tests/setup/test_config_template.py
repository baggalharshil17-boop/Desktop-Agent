import yaml

from financial_voice_agent.setup.config_template import render_config_yaml


def test_render_config_yaml_produces_valid_yaml_with_chosen_values():
    text = render_config_yaml(
        stt_provider="groq",
        stt_model="whisper-large-v3-turbo",
        llm_provider="groq",
        llm_model="qwen/qwen3.6-27b",
        tts_provider="cartesia",
        cartesia_voice_id="db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
        mode="mock",
    )

    parsed = yaml.safe_load(text)

    assert parsed["stt"]["provider"] == "groq"
    assert parsed["stt"]["model"] == "whisper-large-v3-turbo"
    assert parsed["llm"]["provider"] == "groq"
    assert parsed["llm"]["model"] == "qwen/qwen3.6-27b"
    assert parsed["tts"]["provider"] == "cartesia"
    assert parsed["tts"]["voice_id"] == "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"
    assert parsed["mode"] == "mock"


def test_render_config_yaml_includes_required_top_level_sections():
    # These must be present with sane defaults since the wizard doesn't
    # prompt for them -- financial_voice_agent.config.load_config requires
    # all of these keys to exist or it raises ConfigError.
    text = render_config_yaml(
        stt_provider="huggingface",
        stt_model="openai/whisper-large-v3-turbo",
        llm_provider="huggingface",
        llm_model="qwen/qwen3.6-27b",
        tts_provider="cartesia",
        cartesia_voice_id="some-voice-id",
        mode="live",
    )

    parsed = yaml.safe_load(text)

    assert "vad" in parsed
    assert set(parsed["vad"].keys()) >= {"speech_threshold", "silence_duration_ms", "min_speech_duration_ms"}
    assert "audio" in parsed
    assert parsed["audio"]["output_device_index"] is None
    assert "storage" in parsed
    assert parsed["storage"]["db_path"] == "./agent_turns.db"


def test_render_config_yaml_is_loadable_by_the_real_config_loader(tmp_path):
    # The strongest possible check: the app's own loader must accept this
    # output without raising, using real provider/env combinations.
    from financial_voice_agent.config import load_config

    text = render_config_yaml(
        stt_provider="groq",
        stt_model="whisper-large-v3-turbo",
        llm_provider="groq",
        llm_model="qwen/qwen3.6-27b",
        tts_provider="cartesia",
        cartesia_voice_id="db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
        mode="mock",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(text)
    env_path = tmp_path / ".env"
    env_path.write_text("GROQ_API_KEY=x\nCARTESIA_API_KEY=y\n")

    config = load_config(config_path=str(config_path), env_path=str(env_path))

    assert config.stt_provider == "groq"
    assert config.llm_model == "qwen/qwen3.6-27b"


def test_render_config_yaml_renders_fish_audio_model_when_provider_is_fish_audio():
    text = render_config_yaml(
        stt_provider="groq",
        stt_model="whisper-large-v3-turbo",
        llm_provider="groq",
        llm_model="qwen/qwen3.6-27b",
        tts_provider="fish_audio",
        fish_audio_model="s2.1-pro-free",
        mode="mock",
    )

    parsed = yaml.safe_load(text)

    assert parsed["tts"]["provider"] == "fish_audio"
    assert parsed["tts"]["fish_audio_model"] == "s2.1-pro-free"
    assert "voice_id" not in parsed["tts"]
