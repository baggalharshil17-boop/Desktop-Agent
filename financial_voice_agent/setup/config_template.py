from __future__ import annotations

_TEMPLATE = """\
vad:
  speech_threshold: 0.5
  silence_duration_ms: 600
  min_speech_duration_ms: 200
  barge_in_enabled: true
  barge_in_min_speech_ms: 200

audio:
  output_device_index: null
  input_device_index: null
  echo_suppression: true
  echo_margin: 1.75

input_mode: "always_on"  # or "ptt"
tts:
  provider: "{tts_provider}"  # or "cartesia"/"fish_audio"
{tts_provider_field}

stt:
  provider: "{stt_provider}"  # or "groq"/"huggingface"
  model: "{stt_model}"

llm:
  provider: "{llm_provider}"  # or "groq"/"huggingface"
  # Must support both tool calling and vision (capture_screen needs the
  # latter) -- check console.groq.com/docs/models and
  # console.groq.com/docs/vision before changing.
  model: "{llm_model}"

storage:
  db_path: "./agent_turns.db"

mode: "{mode}"  # or "live"/"mock"
"""

FISH_AUDIO_DEFAULT_MODEL = "s2.1-pro-free"


def render_config_yaml(
    *,
    stt_provider: str,
    stt_model: str,
    llm_provider: str,
    llm_model: str,
    tts_provider: str,
    cartesia_voice_id: str = "",
    fish_audio_model: str = FISH_AUDIO_DEFAULT_MODEL,
    fish_audio_voice_id: str = "",
    mode: str,
) -> str:
    if tts_provider == "cartesia":
        tts_provider_field = f'  voice_id: "{cartesia_voice_id}"'
    else:
        tts_provider_field = (
            f'  fish_audio_model: "{fish_audio_model}"\n'
            f'  fish_audio_voice_id: "{fish_audio_voice_id}"'
        )
    return _TEMPLATE.format(
        stt_provider=stt_provider,
        stt_model=stt_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
        tts_provider=tts_provider,
        tts_provider_field=tts_provider_field,
        mode=mode,
    )
