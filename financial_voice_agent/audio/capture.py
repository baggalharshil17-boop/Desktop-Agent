from __future__ import annotations

from typing import Callable

import pyaudio


def make_capture_callback(sync_queue) -> Callable[[bytes, int, dict, int], tuple]:
    def _callback(in_data, frame_count, time_info, status):
        sync_queue.put(bytes(in_data))
        return (None, pyaudio.paContinue)

    return _callback


class AudioCapture:
    def __init__(
        self,
        queue,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 512,
        input_device_index: int | None = None,
        pyaudio_factory=pyaudio.PyAudio,
    ) -> None:
        self._queue = queue
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._input_device_index = input_device_index
        self._pyaudio_factory = pyaudio_factory
        self._pa = None
        self._stream = None

    def start(self) -> None:
        self._pa = self._pyaudio_factory()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._sample_rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            input_device_index=self._input_device_index,
            stream_callback=make_capture_callback(self._queue.sync_q),
        )
        self._stream.start_stream()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
