from __future__ import annotations

import asyncio
import threading

from financial_voice_agent.orchestrator.turn import run_turn, update_history


async def drive_pipeline(pipeline, output_queue: asyncio.Queue) -> None:
    """Continuously pulls utterances from pipeline.run() and forwards them to
    output_queue. Must run as its own task, per AudioPipeline.speech_active's
    documented liveness contract (financial_voice_agent/audio/pipeline.py) --
    it must never be blocked waiting on turn processing, or speech_active
    (and therefore barge-in) freezes."""
    async for utterance_wav in pipeline.run():
        await output_queue.put(utterance_wav)


async def run_voice_loop(
    pipeline,
    playback,
    *,
    stt_fn,
    llm_fn,
    tts_fn,
    db_path: str,
    max_turns_history: int = 8,
) -> None:
    output_queue: asyncio.Queue = asyncio.Queue()
    drive_task = asyncio.create_task(drive_pipeline(pipeline, output_queue))
    history: list[dict] = []

    try:
        while not drive_task.done() or not output_queue.empty():
            try:
                utterance_wav = await asyncio.wait_for(output_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            result = await run_turn(
                utterance_wav, history, stt_fn=stt_fn, llm_fn=llm_fn, tts_fn=tts_fn, db_path=db_path
            )

            if result.transcript is not None and result.response_text is not None:
                history = update_history(
                    history, result.transcript, result.response_text, max_turns=max_turns_history
                )

            if result.tts_audio:
                interrupt_event = threading.Event()
                if pipeline.speech_active.is_set():
                    interrupt_event.set()
                playback.play(result.tts_audio, interrupt_event=interrupt_event)
    finally:
        drive_task.cancel()
