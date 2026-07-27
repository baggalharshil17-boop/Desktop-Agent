"""Echo-aware barge-in support, so the user can interrupt the assistant
even when output goes to open speakers rather than headphones.

This is deliberately NOT full acoustic echo cancellation. AEC's hard job is
reconstructing clean user speech *during* double-talk (it needs adaptive
filters, double-talk detection, and precise reference alignment). We need
something much narrower: a binary "is this mic audio our own speaker, or a
person?" verdict. Once barge-in fires, playback stops and the mic is clean
for the actual utterance, so nothing downstream ever has to separate mixed
audio.

That reframing is what makes this tractable without a C-based AEC library
(neither speexdsp nor webrtc-audio-processing builds on this project's
Python version -- both are unmaintained bindings without current wheels).

Two design choices carry the robustness:

1. **Compare energy envelopes, not waveforms.** Waveform cross-correlation
   against the reference signal is the textbook approach, but it degrades
   badly with room reverb, speaker nonlinearity, and resampling -- all of
   which are present here (MME resamples 16kHz to the device rate). Energy
   is immune to all three.

2. **Compare against the peak reference energy over a delay window, not an
   instantaneous value.** Played audio reaches the mic tens to hundreds of
   ms late (device buffering plus acoustics), and on MME that latency is
   both large and variable. Taking the max over a window spanning plausible
   delays removes the need to estimate the delay at all. It biases toward
   over-predicting echo, which is the correct bias: a missed barge-in is a
   minor annoyance, a false barge-in cuts the assistant off mid-sentence.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from financial_voice_agent.audio import dsp

# Fraction of playback energy that typically returns to the mic. Only used
# when calibration can't produce a real measurement; the live value is
# measured per-device/per-volume at startup (see estimate_echo_gain).
DEFAULT_ECHO_GAIN = 0.35

# Mic energy must exceed predicted echo energy by this factor to count as a
# real person. Higher = harder to falsely interrupt, but requires the user
# to speak up more.
DEFAULT_MARGIN = 2.0

# How far back to look for playback energy that could still be arriving at
# the mic. Also sets how long after playback ends the gate keeps
# suppressing, so don't make it large.
DEFAULT_DELAY_WINDOW_SECONDS = 0.35


def rms(pcm_bytes: bytes) -> float:
    """Root-mean-square level of 16-bit PCM, normalized to 0.0-1.0."""
    if not pcm_bytes:
        return 0.0
    samples = dsp.pcm_to_float32(pcm_bytes)
    if samples.size == 0:
        return 0.0
    return float((samples**2).mean() ** 0.5)


def estimate_echo_gain(
    *, mic_rms: float, reference_rms: float, max_gain: float = 1.5
) -> float:
    """Echo gain = how much of what we play comes back into the mic.

    Measured at startup by playing a known signal and listening (see
    calibrate_echo_gain), which is what lets this adapt automatically to
    whatever output device, speaker volume, and mic the OS defaults picked
    -- rather than hardcoding a threshold that only works on one machine.
    """
    if reference_rms <= 0.0:
        # Calibration heard nothing on the reference side, so it learned
        # nothing. Returning 0.0 here would disable suppression entirely.
        return DEFAULT_ECHO_GAIN
    gain = mic_rms / reference_rms
    return min(gain, max_gain)


class PlaybackReference:
    """Rolling record of how loud recent playback was.

    Written from the playback worker thread and read from the asyncio event
    loop thread, hence the lock.
    """

    def __init__(
        self,
        *,
        history_seconds: float = 2.0,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._history_seconds = history_seconds
        self._clock_fn = clock_fn
        self._entries: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()

    def note_played(self, pcm_bytes: bytes) -> None:
        level = rms(pcm_bytes)
        now = self._clock_fn()
        with self._lock:
            self._entries.append((now, level))
            self._drop_expired(now)

    def peak_rms_since(self, seconds_ago: float) -> float:
        now = self._clock_fn()
        cutoff = now - seconds_ago
        with self._lock:
            self._drop_expired(now)
            recent = [level for ts, level in self._entries if ts >= cutoff]
        return max(recent) if recent else 0.0

    def _drop_expired(self, now: float) -> None:
        horizon = now - self._history_seconds
        while self._entries and self._entries[0][0] < horizon:
            self._entries.popleft()


class EchoGate:
    """Decides whether a mic chunk is explained by our own playback.

    Used to veto VAD's "this is speech" verdict, which fixes two problems at
    once: spurious barge-in cutting responses short, and the assistant
    transcribing its own voice as a new user utterance.
    """

    def __init__(
        self,
        reference: PlaybackReference,
        *,
        echo_gain: float = DEFAULT_ECHO_GAIN,
        margin: float = DEFAULT_MARGIN,
        delay_window_seconds: float = DEFAULT_DELAY_WINDOW_SECONDS,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._reference = reference
        self._echo_gain = echo_gain
        self._margin = margin
        self._delay_window_seconds = delay_window_seconds
        self._clock_fn = clock_fn

    def is_echo(self, pcm_bytes: bytes) -> bool:
        return self.diagnose(pcm_bytes)["is_echo"]

    def diagnose(self, pcm_bytes: bytes) -> dict:
        """Same verdict as is_echo, plus the numbers behind it.

        Exists so a caller can log what happened on a close call (e.g. real
        speech during real playback -- the case a false barge-in or a
        missed one both come from) without duplicating this arithmetic. A
        single scalar echo_gain is a broadband average of the speaker->mic
        path; real speech's spectral content can differ from the
        calibration tone's enough that an occasional chunk reads louder
        than the gain predicts, so these numbers are what actually explain
        a leak when one is reported.
        """
        mic_level = rms(pcm_bytes)
        reference_level = self._reference.peak_rms_since(self._delay_window_seconds)
        predicted_echo = reference_level * self._echo_gain
        threshold = predicted_echo * self._margin
        return {
            "mic_rms": mic_level,
            "reference_rms": reference_level,
            "predicted_echo": predicted_echo,
            "threshold": threshold,
            "is_echo": reference_level > 0.0 and mic_level < threshold,
        }


def make_calibration_tone(
    *, duration_seconds: float = 1.0, sample_rate: int = 16000, amplitude: float = 0.2, seed: int = 11
) -> bytes:
    """Short noise burst used to measure the speaker->mic path.

    Noise rather than a pure tone on purpose: a single frequency can land in
    a room null or a speaker response dip and under-measure the echo, while
    broadband noise better matches the spectrum of the speech this actually
    has to predict.
    """
    rng = np.random.default_rng(seed)
    samples = int(duration_seconds * sample_rate)
    return dsp.float32_to_pcm((amplitude * rng.standard_normal(samples)).astype(np.float32))


async def calibrate_echo_gain(
    playback,
    mic_queue,
    *,
    tone: bytes | None = None,
    poll_timeout: float = 0.05,
) -> float:
    """Plays a known signal and measures how much of it returns to the mic.

    This is what lets echo suppression work on an arbitrary machine without
    hand-tuned thresholds: the same code adapts to a quiet laptop speaker,
    a loud desk speaker, or headphones (where it measures near-zero echo and
    therefore barely suppresses anything).

    Stale mic audio is drained first so the measurement can't be polluted by
    whatever was captured before calibration started.
    """
    tone = tone if tone is not None else make_calibration_tone()

    while not mic_queue.empty():
        try:
            mic_queue.get_nowait()
        except Exception:  # noqa: BLE001 -- races with the capture thread are fine here
            break

    play_task = asyncio.ensure_future(asyncio.to_thread(playback.play, tone))
    captured: list[bytes] = []
    while not play_task.done():
        try:
            captured.append(await asyncio.wait_for(mic_queue.get(), timeout=poll_timeout))
        except asyncio.TimeoutError:
            continue
    await play_task

    if not captured:
        return estimate_echo_gain(mic_rms=0.0, reference_rms=rms(tone))

    # A percentile of per-chunk level, NOT the RMS of everything
    # concatenated. Measured on a laptop array mic: the OS's own echo
    # canceller lets a short burst through at playback onset while it
    # converges, ~30x the steady-state level. Averaging that burst in
    # overestimated the gain ~3x, which inflated the predicted echo enough
    # to block every real interruption. The percentile tracks the sustained
    # echo the gate actually has to discriminate against.
    #
    # p75 rather than p90 because the estimate has to be *stable* run to
    # run, not just representative: with a 1s tone (~30 chunks) p75 varied
    # 1.3x across repeats while p90 sat close enough to the maximum to
    # inherit its noise. An unstable gain is worse than a slightly low one,
    # since the derived threshold can drift above the user's speech level
    # and silently disable barge-in.
    levels = sorted(rms(chunk) for chunk in captured)
    mic_level = levels[min(int(len(levels) * 0.75), len(levels) - 1)]
    return estimate_echo_gain(mic_rms=mic_level, reference_rms=rms(tone))
