from __future__ import annotations

import asyncio
import random
import threading
from collections import deque
from typing import Callable

from financial_voice_agent.audio.wav import encode_wav
from financial_voice_agent.orchestrator.interrupt_keywords import is_interrupt_phrase
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


async def _watch_for_keyword_interrupt(
    output_queue: asyncio.Queue,
    stt_fn,
    interrupt_event: threading.Event,
    playback_task: "asyncio.Future",
    *,
    poll_interval: float = 0.05,
) -> None:
    """Best-effort second signal for barge-in, alongside VAD/echo-gate:
    while a response is still playing, transcribe any newly arrived
    utterance immediately (concurrently with playback, not blocking it) and
    force an interrupt if it opens with a deliberate redirect phrase --
    "wait", "stop", "hold on", etc. (interrupt_keywords.py). This exists
    because real-time audio-level barge-in can only ever judge loudness/
    duration; it can't know the words, since transcription needs the
    utterance already captured. Content-based detection closes that gap for
    the specific case of a clearly-spoken interruption word that VAD/echo
    suppression was too conservative to catch on level alone.

    The utterance is always handed back to output_queue (matched or not),
    so the normal turn loop still processes it once playback ends -- this
    function only ever peeks. Each item is speculatively transcribed at
    most once per playback (tracked by id()) to avoid re-transcribing the
    same handed-back item in a tight loop.

    Only reacts to utterances that arrive AFTER this playback starts.
    Whatever's already queued at entry is simply the next utterance in
    line (e.g. the user's normal follow-up question, queued while the
    previous turn was still being processed) -- not a live interruption --
    and must not be speculatively transcribed early, which would both
    waste an STT call on every routine turn and (confirmed by a test
    failure) call stt_fn on it before the real turn loop does, corrupting
    per-call state a caller might keep across STT calls."""
    checked_ids: set[int] = set()
    preexisting_ids: set[int] = set()
    preexisting: list = []
    while True:
        try:
            preexisting.append(output_queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    for wav in preexisting:
        preexisting_ids.add(id(wav))
        output_queue.put_nowait(wav)

    while not playback_task.done():
        try:
            wav = output_queue.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(poll_interval)
            continue
        if id(wav) in checked_ids or id(wav) in preexisting_ids:
            output_queue.put_nowait(wav)
            await asyncio.sleep(poll_interval)
            continue
        checked_ids.add(id(wav))
        try:
            transcript = await stt_fn(wav)
        except Exception:  # noqa: BLE001 -- a failed speculative transcribe must not crash playback
            transcript = None
        output_queue.put_nowait(wav)
        if is_interrupt_phrase(transcript):
            interrupt_event.set()
            return


async def play_with_barge_in(
    playback,
    tts_audio: bytes,
    pipeline,
    *,
    poll_interval: float = 0.02,
    barge_in_enabled: bool = True,
    output_queue: asyncio.Queue | None = None,
    stt_fn=None,
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
    speaker + headset-mic setup.

    output_queue and stt_fn (both optional, and independent of
    barge_in_enabled) additionally run _watch_for_keyword_interrupt as a
    content-based fallback -- see that function's docstring."""
    interrupt_event = threading.Event()
    playback_task = asyncio.ensure_future(
        asyncio.to_thread(playback.play, tts_audio, interrupt_event=interrupt_event)
    )
    keyword_watch_task = None
    if output_queue is not None and stt_fn is not None:
        keyword_watch_task = asyncio.ensure_future(
            _watch_for_keyword_interrupt(output_queue, stt_fn, interrupt_event, playback_task)
        )
    try:
        while not playback_task.done():
            if barge_in_enabled and pipeline.speech_active.is_set():
                interrupt_event.set()
            await asyncio.sleep(poll_interval)
        return playback_task.result()
    finally:
        if keyword_watch_task is not None:
            keyword_watch_task.cancel()


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
    on_processing_start: Callable[[], None] | None = None,
    on_processing_end: Callable[[], None] | None = None,
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

            if on_processing_start is not None:
                on_processing_start()
            try:
                result = await run_turn(
                    utterance_wav, history, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
                )
            finally:
                if on_processing_end is not None:
                    on_processing_end()

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
                        playback, fallback_audio, pipeline, barge_in_enabled=barge_in_enabled,
                        output_queue=output_queue, stt_fn=stt_fn,
                    )
            elif result.tts_audio:
                await play_with_barge_in(
                    playback, result.tts_audio, pipeline, barge_in_enabled=barge_in_enabled,
                    output_queue=output_queue, stt_fn=stt_fn,
                )
    finally:
        drive_task.cancel()
