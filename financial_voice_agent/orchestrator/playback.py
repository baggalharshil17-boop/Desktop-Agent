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
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._output_device_index = output_device_index
        self._pyaudio_factory = pyaudio_factory
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
            self._stream.write(pcm_bytes[offset : offset + chunk_size])
        return True

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
