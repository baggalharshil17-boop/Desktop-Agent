from __future__ import annotations

import os
import re
from dataclasses import dataclass

import yaml
from dotenv import dotenv_values

# Mirrors audio.echo.DEFAULT_MARGIN. Duplicated as a literal rather than
# imported so that loading config doesn't drag in the audio stack (numpy,
# noisereduce, torch) -- the eval harness and tests load config without
# touching audio at all.
_DEFAULT_ECHO_MARGIN = 2.0

_VALID_TTS_PROVIDERS = {"cartesia", "deepgram"}
_VALID_STT_PROVIDERS = {"groq", "huggingface"}
_VALID_LLM_PROVIDERS = {"groq", "huggingface"}
_PLACEHOLDER_RE = re.compile(r"^<.*>$")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    vad_speech_threshold: float
    vad_silence_duration_ms: int
    vad_min_speech_duration_ms: int
    barge_in_enabled: bool
    barge_in_min_speech_ms: float
    audio_output_device_index: int | None
    audio_input_device_index: int | None
    echo_suppression_enabled: bool
    echo_margin: float
    echo_gain: float | None
    input_mode: str
    tts_provider: str
    cartesia_voice_id: str | None
    stt_provider: str
    stt_model: str
    llm_provider: str
    llm_model: str
    storage_db_path: str
    mode: str
    processing_overlay_enabled: bool
    groq_api_key: str | None
    cartesia_api_key: str | None
    deepgram_api_key: str | None
    huggingface_api_key: str | None
    kite_api_key: str | None
    kite_access_token: str | None
    tavily_api_key: str | None


def _load_env(env_path: str) -> dict[str, str]:
    file_values = dotenv_values(env_path) if os.path.exists(env_path) else {}
    merged = dict(file_values)
    for key in (
        "GROQ_API_KEY",
        "CARTESIA_API_KEY",
        "DEEPGRAM_API_KEY",
        "HF_TOKEN",
        "KITE_API_KEY",
        "KITE_ACCESS_TOKEN",
        "TAVILY_API_KEY",
    ):
        if key in os.environ:
            merged[key] = os.environ[key]
    return merged


def load_config(config_path: str = "config.yaml", env_path: str = ".env") -> Config:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {config_path}")

    env = _load_env(env_path)
    groq_api_key = env.get("GROQ_API_KEY")
    huggingface_api_key = env.get("HF_TOKEN")

    try:
        tts_provider = raw["tts"]["provider"]

        if tts_provider not in _VALID_TTS_PROVIDERS:
            raise ConfigError(
                f"tts.provider must be 'cartesia' or 'deepgram', got {tts_provider!r}"
            )

        cartesia_api_key = env.get("CARTESIA_API_KEY")
        deepgram_api_key = env.get("DEEPGRAM_API_KEY")
        if tts_provider == "cartesia" and not cartesia_api_key:
            raise ConfigError("tts.provider is 'cartesia' but CARTESIA_API_KEY is not set")
        if tts_provider == "deepgram" and not deepgram_api_key:
            raise ConfigError("tts.provider is 'deepgram' but DEEPGRAM_API_KEY is not set")

        cartesia_voice_id = raw.get("tts", {}).get("voice_id")
        if tts_provider == "cartesia" and (
            not cartesia_voice_id or _PLACEHOLDER_RE.match(cartesia_voice_id)
        ):
            raise ConfigError(
                "tts.provider is 'cartesia' but tts.voice_id in config.yaml is not set "
                "(still a placeholder) — pick a voice id from Cartesia's voice library"
            )

        stt_provider = raw["stt"]["provider"]
        if stt_provider not in _VALID_STT_PROVIDERS:
            raise ConfigError(
                f"stt.provider must be 'groq' or 'huggingface', got {stt_provider!r}"
            )
        stt_model = raw["stt"]["model"]

        llm_provider = raw["llm"]["provider"]
        if llm_provider not in _VALID_LLM_PROVIDERS:
            raise ConfigError(
                f"llm.provider must be 'groq' or 'huggingface', got {llm_provider!r}"
            )
        llm_model = raw["llm"]["model"]
        if _PLACEHOLDER_RE.match(llm_model):
            raise ConfigError(
                f"llm.model in config.yaml is still a placeholder ({llm_model!r}) — "
                "set it to a real model id"
            )

        if (stt_provider == "groq" or llm_provider == "groq") and not groq_api_key:
            raise ConfigError("GROQ_API_KEY is not set, but stt.provider or llm.provider is 'groq'")
        if (stt_provider == "huggingface" or llm_provider == "huggingface") and not huggingface_api_key:
            raise ConfigError("HF_TOKEN is not set, but stt.provider or llm.provider is 'huggingface'")

        config = Config(
            vad_speech_threshold=raw["vad"]["speech_threshold"],
            vad_silence_duration_ms=raw["vad"]["silence_duration_ms"],
            vad_min_speech_duration_ms=raw["vad"]["min_speech_duration_ms"],
            barge_in_enabled=raw["vad"].get("barge_in_enabled", True),
            barge_in_min_speech_ms=raw["vad"].get("barge_in_min_speech_ms", 96.0),
            audio_output_device_index=raw["audio"]["output_device_index"],
            audio_input_device_index=raw["audio"].get("input_device_index"),
            echo_suppression_enabled=raw["audio"].get("echo_suppression", True),
            echo_margin=raw["audio"].get("echo_margin", _DEFAULT_ECHO_MARGIN),
            echo_gain=raw["audio"].get("echo_gain"),
            input_mode=raw["input_mode"],
            tts_provider=tts_provider,
            cartesia_voice_id=cartesia_voice_id,
            stt_provider=stt_provider,
            stt_model=stt_model,
            llm_provider=llm_provider,
            llm_model=llm_model,
            storage_db_path=raw["storage"]["db_path"],
            mode=raw["mode"],
            processing_overlay_enabled=raw.get("overlay", {}).get("enabled", True),
            groq_api_key=groq_api_key,
            cartesia_api_key=cartesia_api_key,
            deepgram_api_key=deepgram_api_key,
            huggingface_api_key=huggingface_api_key,
            kite_api_key=env.get("KITE_API_KEY"),
            kite_access_token=env.get("KITE_ACCESS_TOKEN"),
            tavily_api_key=env.get("TAVILY_API_KEY"),
        )
    except ConfigError:
        raise
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"config.yaml is missing or malformed: {exc}") from exc

    return config
