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
  provider: "cartesia"  # or "deepgram" -- deepgram has no adapter yet
  voice_id: "{cartesia_voice_id}"

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


def render_config_yaml(
    *,
    stt_provider: str,
    stt_model: str,
    llm_provider: str,
    llm_model: str,
    cartesia_voice_id: str,
    mode: str,
) -> str:
    return _TEMPLATE.format(
        stt_provider=stt_provider,
        stt_model=stt_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
        cartesia_voice_id=cartesia_voice_id,
        mode=mode,
    )
