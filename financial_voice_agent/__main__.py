"""Live entry point for the Financial Voice Agent.

Wires config.yaml's chosen providers to real microphone capture -> VAD
pipeline -> STT -> LLM tool loop -> TTS -> speaker playback, per PRD Section
5's data flow. Nothing else in this codebase runs this end-to-end; every
other module is a component exercised individually by tests or by the eval
harness (financial_voice_agent/eval). Run with:

    python -m financial_voice_agent

Talk after it prints "Listening...". Ctrl+C to stop -- capture, playback,
and HTTP clients are all closed cleanly on the way out.

Requires a real microphone and speakers, and (per config.yaml) whichever of
GROQ_API_KEY / HF_TOKEN / CARTESIA_API_KEY / KITE_API_KEY+KITE_ACCESS_TOKEN
your chosen providers need. With mode: "live", Kite tool calls hit your real
account (read-only) using today's KITE_ACCESS_TOKEN (see scripts/kite_login.py
-- Kite tokens expire daily).
"""

from __future__ import annotations

import asyncio

import janus
from cartesia import AsyncCartesia

from financial_voice_agent.audio.capture import AudioCapture
from financial_voice_agent.audio.pipeline import AudioPipeline
from financial_voice_agent.audio.vad import SileroVadScorer
from financial_voice_agent.config import load_config
from financial_voice_agent.db import init_db
from financial_voice_agent.http_clients import close_http_clients, create_http_clients
from financial_voice_agent.orchestrator.llm import make_llm_client, run_llm_turn
from financial_voice_agent.orchestrator.main_loop import play_with_barge_in, run_voice_loop
from financial_voice_agent.orchestrator.playback import AudioPlayback
from financial_voice_agent.orchestrator.stt import make_stt_client, transcribe_with_retry
from financial_voice_agent.orchestrator.system_prompt import SYSTEM_PROMPT
from financial_voice_agent.orchestrator.tts import RealCartesiaTtsClient, synthesize_with_fallback
from financial_voice_agent.tools.registry import TOOLS_SCHEMA, make_tool_executor


async def main() -> None:
    config = load_config()
    if config.tts_provider != "cartesia":
        raise NotImplementedError(
            f"Only tts.provider: 'cartesia' is wired up in the live loop right now, "
            f"got {config.tts_provider!r} -- no Deepgram TTS adapter exists yet"
        )

    init_db(config.storage_db_path)
    http_clients = await create_http_clients(config)

    capture_queue: janus.Queue = janus.Queue()
    capture = AudioCapture(capture_queue)
    pipeline = AudioPipeline(
        capture_queue.async_q,
        SileroVadScorer(),
        speech_threshold=config.vad_speech_threshold,
        silence_duration_ms=config.vad_silence_duration_ms,
        min_speech_duration_ms=config.vad_min_speech_duration_ms,
    )
    playback = AudioPlayback(output_device_index=config.audio_output_device_index)

    stt_client = make_stt_client(config)
    llm_client = make_llm_client(config)
    tool_executor = make_tool_executor(config, http_clients)
    tts_client = RealCartesiaTtsClient(
        AsyncCartesia(api_key=config.cartesia_api_key), voice_id=config.cartesia_voice_id
    )

    async def stt_fn(wav_bytes: bytes) -> str:
        return await transcribe_with_retry(stt_client, wav_bytes, model=config.stt_model)

    async def speak_ack() -> None:
        # Runs concurrently with the tool call so there's no dead air while
        # a slow tool (e.g. get_news's web search) is in flight. Swallow
        # failures -- a missed ack must never break or delay the real turn.
        try:
            audio = await synthesize_with_fallback(tts_client, "Let me check that for you.")
            await play_with_barge_in(playback, audio, pipeline)
        except Exception:  # noqa: BLE001
            pass

    async def llm_fn(transcript: str, history: list[dict]):
        return await run_llm_turn(
            llm_client,
            transcript,
            history,
            model=config.llm_model,
            tools_schema=TOOLS_SCHEMA,
            tool_executor=tool_executor,
            system_prompt=SYSTEM_PROMPT,
            on_tool_call_started=speak_ack,
        )

    async def tts_fn(text: str) -> bytes:
        return await synthesize_with_fallback(tts_client, text)

    playback.open()
    capture.start()
    print("Listening... Ctrl+C to stop.")
    try:
        await run_voice_loop(
            pipeline,
            playback,
            stt_fn=stt_fn,
            llm_fn=llm_fn,
            tts_fn=tts_fn,
            db_path=config.storage_db_path,
        )
    except asyncio.CancelledError:
        pass
    finally:
        capture.stop()
        playback.close()
        await close_http_clients(http_clients)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
