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
import pyaudio
from cartesia import AsyncCartesia

from financial_voice_agent.audio.capture import AudioCapture
from financial_voice_agent.audio.devices import resolve_input_device_index, resolve_output_device_index
from financial_voice_agent.audio.echo import (
    EchoGate,
    PlaybackReference,
    calibrate_echo_gain,
)
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

    # Resolved once via a throwaway PyAudio instance, using the WASAPI host
    # API's default device when config leaves output/input_device_index as
    # null -- WASAPI is what Windows Sound Settings actually controls, and
    # tracks live device changes that PyAudio's own "default" (a different,
    # less reliable host API) can silently miss. See audio/devices.py.
    device_probe = pyaudio.PyAudio()
    output_device_index = resolve_output_device_index(device_probe, config.audio_output_device_index)
    input_device_index = resolve_input_device_index(device_probe, config.audio_input_device_index)
    device_probe.terminate()

    capture_queue: janus.Queue = janus.Queue()
    capture = AudioCapture(capture_queue, input_device_index=input_device_index)

    # The reference is always recorded (it's cheap); only the gate that
    # consumes it is optional. Wiring it unconditionally keeps playback and
    # calibration identical whether or not suppression is enabled.
    playback_reference = PlaybackReference()
    playback = AudioPlayback(
        output_device_index=output_device_index, playback_reference=playback_reference
    )

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
            await play_with_barge_in(
                playback, audio, pipeline, barge_in_enabled=config.barge_in_enabled
            )
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
    try:
        # Calibration has to happen after capture is live (it listens) and
        # before the pipeline is built (the gate needs the measured gain),
        # so it sits inside this try -- a failure here must still close the
        # audio streams it was using.
        echo_gate = None
        if config.echo_suppression_enabled:
            if config.echo_gain is not None:
                # Fixed gain from config -- skips the audible calibration
                # burst entirely. Only as good as the number in config.yaml;
                # re-measure (see scripts/ or the earlier calibration path)
                # if the mic/speaker/volume setup changes.
                echo_gain = config.echo_gain
                print(f"Echo gain: {echo_gain:.3f} (fixed from config, no calibration)")
            else:
                print("Calibrating echo (brief noise burst -- stay quiet)...")
                echo_gain = await calibrate_echo_gain(playback, capture_queue.async_q)
                print(f"Echo gain: {echo_gain:.3f} (near 0 = isolated, higher = more speaker bleed)")
            echo_gate = EchoGate(
                playback_reference, echo_gain=echo_gain, margin=config.echo_margin
            )

        def log_echo_diagnostic(d: dict) -> None:
            # Fires for every VAD-flagged-speech chunk overlapping recent
            # playback -- both real barge-in and echo leaks land here,
            # whichever way they got classified. If barge-in misfires
            # again, this is what tells us whether it was a genuine close
            # call (numbers near the threshold) or something else.
            verdict = "SUPPRESSED (echo)" if d["is_echo"] else "PASSED (counted as speech)"
            print(
                f"  [echo] mic={d['mic_rms']:.5f} predicted={d['predicted_echo']:.5f} "
                f"threshold={d['threshold']:.5f} -> {verdict}"
            )

        pipeline = AudioPipeline(
            capture_queue.async_q,
            SileroVadScorer(),
            speech_threshold=config.vad_speech_threshold,
            silence_duration_ms=config.vad_silence_duration_ms,
            min_speech_duration_ms=config.vad_min_speech_duration_ms,
            echo_gate=echo_gate,
            min_barge_in_ms=config.barge_in_min_speech_ms,
            on_echo_diagnostic=log_echo_diagnostic if echo_gate is not None else None,
        )

        print("Listening... Ctrl+C to stop.")
        await run_voice_loop(
            pipeline,
            playback,
            stt_fn=stt_fn,
            llm_fn=llm_fn,
            tts_fn=tts_fn,
            db_path=config.storage_db_path,
            barge_in_enabled=config.barge_in_enabled,
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
