from __future__ import annotations

import asyncio
import random
import threading
from collections import deque
from typing import Callable

from financial_voice_agent.audio.wav import encode_wav
from financial_voice_agent.orchestrator.turn import run_turn, update_history

# Separate phrasing per failure stage -- "I didn't catch that" is only
# honest when STT itself is what failed (result.transcript is None).
# Saying it when STT succeeded but the LLM/tool call failed downstream
# wrongly implies a hearing problem. Several phrasings per category, picked
# via choice_fn and filtered against recently-used ones, so the agent
# doesn't parrot the identical line on repeated failures.
STT_FAILURE_MESSAGES = [
    "I didn't catch that, one moment.",
    "Sorry, didn't quite hear you there.",
    "That didn't come through clearly, could you say it again?",
    "Missed that one, one more time?",
    "Didn't quite get that, mind repeating?",
]

RATE_LIMIT_MESSAGES = [
    "I've hit a usage limit — give me a second.",
    "Running low on quota right now, hang tight a moment.",
    "Rate limited at the moment, one sec.",
    "Hitting a rate limit — give me just a second.",
]

GENERIC_FAILURE_MESSAGES = [
    "Something went wrong on my end, one moment.",
    "Hit a snag processing that, let me try again.",
    "Ran into an issue there, give me a second.",
    "That didn't go through on my end, one moment.",
]


async def drive_pipeline(pipeline, output_queue: asyncio.Queue) -> None:
    """Continuously pulls utterances from pipeline.run() and forwards them to
    output_queue. Must run as its own task, per AudioPipeline.speech_active's
    documented liveness contract (financial_voice_agent/audio/pipeline.py) --
    it must never be blocked waiting on turn processing, or speech_active
    (and therefore barge-in) freezes.

    AudioPipeline.run() yields raw PCM bytes (no container), so this is the
    one place that wraps each utterance in a WAV container before it's
    handed to STT -- Groq's Whisper endpoint is a file upload, not a raw PCM
    stream, and everything downstream refers to these bytes as
    "utterance_wav"."""
    async for utterance_pcm in pipeline.run():
        await output_queue.put(encode_wav(utterance_pcm))


async def play_with_barge_in(
    playback, tts_audio: bytes, pipeline, *, poll_interval: float = 0.02, barge_in_enabled: bool = True
) -> bool:
    """Runs playback.play() on a worker thread (asyncio.to_thread) rather
    than calling it directly on the event loop thread. playback.play() is a
    blocking, synchronous call (real PyAudio writes) -- calling it inline
    inside a coroutine would freeze the ENTIRE event loop for the full
    playback duration, not just this coroutine, which would stall
    drive_pipeline's task too and freeze pipeline.speech_active exactly the
    way that event's docstring warns against. While playback runs on its
    worker thread, this function polls pipeline.speech_active every
    poll_interval seconds and mirrors it into a threading.Event playback.play()
    checks between chunks -- a one-time snapshot before playback starts is
    not sufficient, since barge-in must be detected while the assistant is
    still talking, not only in the instant before it starts.

    barge_in_enabled=False ignores pipeline.speech_active entirely, playing
    the full response regardless. This is for setups where mic and speaker
    aren't acoustically isolated (e.g. a headset mic picking up bleed from
    laptop speakers) -- there, speech_active fires almost immediately from
    the assistant's own voice, cutting every response short after a word or
    two, with no way to distinguish that from real user barge-in using VAD
    alone. Confirmed live: this exact scenario reproduced on a laptop-
    speaker + headset-mic setup."""
    interrupt_event = threading.Event()
    playback_task = asyncio.ensure_future(
        asyncio.to_thread(playback.play, tts_audio, interrupt_event=interrupt_event)
    )
    while not playback_task.done():
        if barge_in_enabled and pipeline.speech_active.is_set():
            interrupt_event.set()
        await asyncio.sleep(poll_interval)
    return playback_task.result()


def _pick_fallback_message(
    result, recent_messages: deque, choice_fn: Callable[[list[str]], str]
) -> str:
    if result.rate_limited:
        candidates = RATE_LIMIT_MESSAGES
    elif result.transcript is None:
        # STT itself never produced a transcript -- "didn't catch that" is
        # accurate here.
        candidates = STT_FAILURE_MESSAGES
    else:
        # STT succeeded (we have a transcript) but something failed
        # downstream (LLM/tool) -- saying "didn't catch that" would be a lie.
        candidates = GENERIC_FAILURE_MESSAGES

    available = [m for m in candidates if m not in recent_messages] or candidates
    message = choice_fn(available)
    recent_messages.append(message)
    return message


async def run_voice_loop(
    pipeline,
    playback,
    *,
    stt_fn,
    llm_fn,
    tts_fn,
    db_path: str,
    max_turns_history: int = 8,
    choice_fn: Callable[[list[str]], str] = random.choice,
    recent_fallback_window: int = 5,
    barge_in_enabled: bool = True,
) -> None:
    output_queue: asyncio.Queue = asyncio.Queue()
    drive_task = asyncio.create_task(drive_pipeline(pipeline, output_queue))
    history: list[dict] = []
    recent_fallback_messages: deque = deque(maxlen=recent_fallback_window)

    try:
        while not drive_task.done() or not output_queue.empty():
            try:
                utterance_wav = await asyncio.wait_for(output_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                if drive_task.done():
                    exc = drive_task.exception()
                    if exc is not None:
                        raise exc
                continue

            result = await run_turn(
                utterance_wav, history, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
            )

            if result.transcript is not None and result.response_text is not None:
                history = update_history(
                    history, result.transcript, result.response_text, max_turns=max_turns_history
                )

            if result.error is not None:
                message = _pick_fallback_message(result, recent_fallback_messages, choice_fn)
                try:
                    fallback_audio = await tts_fn(message)
                except Exception:  # noqa: BLE001 -- fallback TTS failing must not crash the loop
                    fallback_audio = None
                if fallback_audio:
                    await play_with_barge_in(
                        playback, fallback_audio, pipeline, barge_in_enabled=barge_in_enabled
                    )
            elif result.tts_audio:
                await play_with_barge_in(
                    playback, result.tts_audio, pipeline, barge_in_enabled=barge_in_enabled
                )
    finally:
        drive_task.cancel()
