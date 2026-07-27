import math
import struct

import pytest

from financial_voice_agent.audio.echo import (
    calibrate_echo_gain,
    make_calibration_tone,
    DEFAULT_ECHO_GAIN,
    EchoGate,
    PlaybackReference,
    estimate_echo_gain,
    rms,
)


def _tone_pcm(amplitude: int, samples: int = 512, freq: int = 440, sample_rate: int = 16000) -> bytes:
    return b"".join(
        struct.pack("<h", int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate)))
        for i in range(samples)
    )


def _silence_pcm(samples: int = 512) -> bytes:
    return b"\x00\x00" * samples


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_rms_of_silence_is_zero():
    assert rms(_silence_pcm()) == 0.0


def test_rms_of_empty_input_is_zero():
    assert rms(b"") == 0.0


def test_rms_scales_with_amplitude():
    quiet = rms(_tone_pcm(2000))
    loud = rms(_tone_pcm(20000))

    assert loud > quiet
    # A 10x amplitude increase should give roughly 10x RMS.
    assert 9.0 < loud / quiet < 11.0


def test_playback_reference_with_no_history_reports_zero():
    reference = PlaybackReference(clock_fn=_FakeClock())

    assert reference.peak_rms_since(0.4) == 0.0


def test_playback_reference_reports_peak_of_recent_writes():
    clock = _FakeClock()
    reference = PlaybackReference(clock_fn=clock)

    reference.note_played(_tone_pcm(4000))
    clock.advance(0.05)
    reference.note_played(_tone_pcm(16000))
    clock.advance(0.05)
    reference.note_played(_tone_pcm(2000))

    peak = reference.peak_rms_since(0.4)

    assert peak == pytest.approx(rms(_tone_pcm(16000)), rel=1e-6)


def test_playback_reference_ignores_writes_older_than_the_window():
    clock = _FakeClock()
    reference = PlaybackReference(clock_fn=clock)

    reference.note_played(_tone_pcm(20000))
    clock.advance(1.0)  # well outside a 0.4s lookback
    reference.note_played(_tone_pcm(1000))

    peak = reference.peak_rms_since(0.4)

    assert peak == pytest.approx(rms(_tone_pcm(1000)), rel=1e-6)


def test_playback_reference_decays_to_zero_once_playback_stops():
    clock = _FakeClock()
    reference = PlaybackReference(clock_fn=clock)
    reference.note_played(_tone_pcm(20000))

    clock.advance(1.0)

    assert reference.peak_rms_since(0.4) == 0.0


def test_echo_gate_allows_barge_in_when_nothing_is_playing():
    # Nothing played -> no predicted echo -> even quiet mic audio must be
    # treated as genuine, or the agent could never hear anyone.
    clock = _FakeClock()
    reference = PlaybackReference(clock_fn=clock)
    gate = EchoGate(reference, echo_gain=0.5, clock_fn=clock)

    assert gate.is_echo(_tone_pcm(500)) is False


def test_echo_gate_suppresses_mic_audio_consistent_with_playback_echo():
    # Speaker plays loudly; mic hears something at roughly the calibrated
    # echo gain -> that's our own voice coming back, not the user.
    clock = _FakeClock()
    reference = PlaybackReference(clock_fn=clock)
    gate = EchoGate(reference, echo_gain=0.5, margin=2.0, clock_fn=clock)

    reference.note_played(_tone_pcm(20000))
    echo_like = _tone_pcm(10000)  # ~= 0.5 * playback amplitude

    assert gate.is_echo(echo_like) is True


def test_echo_gate_allows_barge_in_when_mic_is_much_louder_than_predicted_echo():
    # User talks over the assistant: mic energy far exceeds what the
    # speaker alone could account for.
    clock = _FakeClock()
    reference = PlaybackReference(clock_fn=clock)
    gate = EchoGate(reference, echo_gain=0.1, margin=2.0, clock_fn=clock)

    reference.note_played(_tone_pcm(8000))  # predicted echo ~= 800 amplitude
    loud_user_speech = _tone_pcm(20000)

    assert gate.is_echo(loud_user_speech) is False


def test_echo_gate_margin_controls_how_much_headroom_is_required():
    clock = _FakeClock()
    reference = PlaybackReference(clock_fn=clock)
    reference.note_played(_tone_pcm(10000))
    mic = _tone_pcm(6000)  # 1.2x the predicted echo of 5000

    lenient = EchoGate(reference, echo_gain=0.5, margin=1.1, clock_fn=clock)
    strict = EchoGate(reference, echo_gain=0.5, margin=3.0, clock_fn=clock)

    # Same audio, different verdicts purely from the safety margin.
    assert lenient.is_echo(mic) is False
    assert strict.is_echo(mic) is True


def test_echo_gate_uses_peak_over_the_delay_window_not_the_instantaneous_value():
    # Real playback reaches the mic tens to hundreds of ms late (device
    # buffering + acoustics). Comparing against the peak over a delay window
    # means the gate never needs an exact delay estimate -- confirmed
    # necessary because MME output latency is both large and variable.
    clock = _FakeClock()
    reference = PlaybackReference(clock_fn=clock)
    gate = EchoGate(reference, echo_gain=0.5, margin=2.0, delay_window_seconds=0.4, clock_fn=clock)

    reference.note_played(_tone_pcm(20000))  # loud burst
    clock.advance(0.2)
    reference.note_played(_silence_pcm())  # gone quiet just now

    # The loud burst is still in flight acoustically, so mic audio matching
    # it must still read as echo despite the instantaneous reference being 0.
    assert gate.is_echo(_tone_pcm(10000)) is True


def test_estimate_echo_gain_returns_ratio_of_mic_to_reference():
    assert estimate_echo_gain(mic_rms=0.25, reference_rms=0.5) == pytest.approx(0.5)


def test_estimate_echo_gain_falls_back_to_default_for_silent_reference():
    # A silent reference means calibration learned nothing (e.g. playback
    # failed) -- must not divide by zero or return a meaningless 0 gain that
    # would disable echo suppression entirely.
    assert estimate_echo_gain(mic_rms=0.1, reference_rms=0.0) == DEFAULT_ECHO_GAIN


def test_estimate_echo_gain_clamps_implausibly_large_ratios():
    # Mic RMS far above playback RMS means something other than echo
    # dominated calibration (someone talked, or the mic is on a different
    # device). Clamping keeps a bad calibration from making barge-in
    # impossible.
    gain = estimate_echo_gain(mic_rms=5.0, reference_rms=0.1, max_gain=1.5)

    assert gain == 1.5


def test_make_calibration_tone_is_broadband_and_nonsilent():
    tone = make_calibration_tone(duration_seconds=0.25, sample_rate=16000)

    assert len(tone) == int(0.25 * 16000) * 2  # 16-bit samples
    assert rms(tone) > 0.05


@pytest.mark.asyncio
async def test_calibrate_echo_gain_measures_ratio_of_captured_to_played():
    import asyncio
    import time

    import numpy as np

    from financial_voice_agent.audio import dsp

    played: list[bytes] = []
    tone = make_calibration_tone(duration_seconds=0.1)
    # Mic "hears" the tone back at half amplitude.
    echoed = dsp.float32_to_pcm((dsp.pcm_to_float32(tone) * 0.5).astype(np.float32))
    queue: asyncio.Queue = asyncio.Queue()

    class _FakePlayback:
        def play(self, pcm_bytes, *, chunk_size=1024, interrupt_event=None):
            # Blocking, like a real PyAudio write, so there's a real window
            # during which mic audio can arrive -- the echo only reaches the
            # mic while playback is actually happening.
            played.append(pcm_bytes)
            time.sleep(0.1)
            return True

    async def feed_mic_during_playback():
        await asyncio.sleep(0.02)
        await queue.put(echoed)

    feeder = asyncio.create_task(feed_mic_during_playback())
    gain = await calibrate_echo_gain(_FakePlayback(), queue, tone=tone, poll_timeout=0.01)
    await feeder

    assert played == [tone]
    assert gain == pytest.approx(0.5, abs=0.05)


@pytest.mark.asyncio
async def test_calibrate_echo_gain_drains_stale_mic_audio_first():
    # Audio captured before calibration started must not count toward the
    # measurement, or a user talking just beforehand inflates the gain and
    # makes barge-in impossible afterwards.
    import asyncio

    class _FakePlayback:
        def play(self, pcm_bytes, *, chunk_size=1024, interrupt_event=None):
            return True

    tone = make_calibration_tone(duration_seconds=0.1)
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put(b"\x00\x7f" * 4000)  # loud stale audio

    gain = await calibrate_echo_gain(_FakePlayback(), queue, tone=tone, poll_timeout=0.01)

    # Nothing captured during playback -> near-zero measured echo.
    assert gain == pytest.approx(0.0, abs=0.05)


@pytest.mark.asyncio
async def test_calibrate_echo_gain_on_headphones_measures_near_zero():
    # Headphones leak almost nothing into the mic, so calibration should
    # produce a tiny gain -- meaning the gate barely suppresses and barge-in
    # stays maximally responsive. Same code path, no config change.
    import asyncio

    class _FakePlayback:
        def play(self, pcm_bytes, *, chunk_size=1024, interrupt_event=None):
            return True

    tone = make_calibration_tone(duration_seconds=0.1)
    queue: asyncio.Queue = asyncio.Queue()

    gain = await calibrate_echo_gain(_FakePlayback(), queue, tone=tone, poll_timeout=0.01)

    reference = PlaybackReference()
    gate = EchoGate(reference, echo_gain=gain)
    reference.note_played(tone)
    assert gate.is_echo(_tone_pcm(3000)) is False


def test_echo_gate_diagnose_returns_the_numbers_behind_the_verdict():
    clock = _FakeClock()
    reference = PlaybackReference(clock_fn=clock)
    reference.note_played(_tone_pcm(20000))
    gate = EchoGate(reference, echo_gain=0.5, margin=2.0, clock_fn=clock)

    diagnostics = gate.diagnose(_tone_pcm(10000))

    assert diagnostics["is_echo"] == gate.is_echo(_tone_pcm(10000))
    assert diagnostics["reference_rms"] == pytest.approx(rms(_tone_pcm(20000)), rel=1e-6)
    assert diagnostics["predicted_echo"] == pytest.approx(diagnostics["reference_rms"] * 0.5, rel=1e-6)
    assert diagnostics["threshold"] == pytest.approx(diagnostics["predicted_echo"] * 2.0, rel=1e-6)
    assert diagnostics["mic_rms"] == pytest.approx(rms(_tone_pcm(10000)), rel=1e-6)


def test_echo_gate_diagnose_reports_zero_reference_when_nothing_playing():
    reference = PlaybackReference(clock_fn=_FakeClock())
    gate = EchoGate(reference, echo_gain=0.5, clock_fn=_FakeClock())

    diagnostics = gate.diagnose(_tone_pcm(500))

    assert diagnostics["reference_rms"] == 0.0
    assert diagnostics["is_echo"] is False
