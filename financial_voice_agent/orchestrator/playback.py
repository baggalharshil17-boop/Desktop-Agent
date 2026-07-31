from __future__ import annotations

import pyaudio


class AudioPlayback:
    """PyAudio output stream matching Cartesia's PCM format exactly (16kHz,
    16-bit, mono) -- a mismatch here is PRD Section 10.2's "chipmunk voice"
    bug. Interruptible mid-stream via a threading/asyncio Event
    (barge-in, Section 11.3).
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        output_device_index: int | None = None,
        pyaudio_factory=pyaudio.PyAudio,
        playback_reference=None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._output_device_index = output_device_index
        self._pyaudio_factory = pyaudio_factory
        # Optional PlaybackReference (financial_voice_agent/audio/echo.py).
        # Recording what we send to the speaker is what lets the mic side
        # tell our own voice apart from a user interrupting us, which is
        # what makes barge-in work on open speakers rather than only on
        # headphones.
        self._playback_reference = playback_reference
        self._pa = None
        self._stream = None

    def open(self) -> None:
        self._pa = self._pyaudio_factory()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._sample_rate,
            output=True,
            output_device_index=self._output_device_index,
        )

    def play(self, pcm_bytes: bytes, *, chunk_size: int = 1024, interrupt_event=None) -> bool:
        for offset in range(0, len(pcm_bytes), chunk_size):
            if interrupt_event is not None and interrupt_event.is_set():
                return False
            chunk = pcm_bytes[offset : offset + chunk_size]
            # Recorded per chunk rather than once up front: stream.write()
            # blocks until the device has buffer space, so this loop is
            # paced roughly with real playback, giving the reference
            # sensible timestamps to compare mic audio against.
            if self._playback_reference is not None:
                self._playback_reference.note_played(chunk)
            self._stream.write(chunk)
        return True

    def close(self) -> None:
        # Every step is best-effort and independent. close() runs in the
        # shutdown path, including when the stream has already failed (e.g.
        # PortAudio's "[Errno -9999] Unanticipated host error" after the audio
        # endpoint changes mid-playback). stop_stream() raises on an
        # already-broken stream, which would otherwise mask the original error
        # and leave the PyAudio instance un-terminated.
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:  # noqa: BLE001
                pass
            self._pa = None
