"""Phrases that mean the user is deliberately redirecting the assistant
mid-response, used as a second, content-based barge-in signal alongside
the audio-level VAD/echo-gate one in main_loop.py's play_with_barge_in.

Matched by prefix, not substring: "stop" must match "Stop, wait" but not
"what's the stop-loss level", which mentions the word without meaning it as
an interruption.
"""

from __future__ import annotations

INTERRUPT_PHRASES = [
    "wait",
    "wait wait",
    "no",
    "no no",
    "but",
    "not this",
    "not that",
    "i didn't ask that",
    "i didn't ask for that",
    "that's not what i asked",
    "that is not what i asked",
    "stop",
    "hold on",
    "hang on",
    "actually",
    "never mind",
    "nevermind",
    "cancel that",
    "sorry",
    "excuse me",
]


def is_interrupt_phrase(transcript: str | None) -> bool:
    if not transcript:
        return False
    normalized = transcript.strip().lower().strip(".,!?;: ")
    if not normalized:
        return False
    return any(
        normalized == phrase or normalized.startswith(phrase + " ") or normalized.startswith(phrase + ",")
        for phrase in INTERRUPT_PHRASES
    )
