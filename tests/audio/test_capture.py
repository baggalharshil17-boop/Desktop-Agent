import janus
import pyaudio
import pytest

from financial_voice_agent.audio.capture import AudioCapture, make_capture_callback


def test_capture_callback_pushes_bytes_to_sync_queue():
    queue = janus.Queue()
    callback = make_capture_callback(queue.sync_q)

    result = callback(b"\x01\x02" * 256, 256, {}, 0)

    assert result == (None, pyaudio.paContinue)
    assert queue.sync_q.get_nowait() == b"\x01\x02" * 256
    queue.close()


class _FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.closed = False

    def start_stream(self):
        self.started = True

    def stop_stream(self):
        self.started = False

    def close(self):
        self.closed = True


class _FakePyAudio:
    def __init__(self):
        self.terminated = False
        self.opened_stream = None

    def open(self, **kwargs):
        self.opened_stream = _FakeStream(**kwargs)
        return self.opened_stream

    def terminate(self):
        self.terminated = True


def test_start_opens_stream_with_documented_data_contract():
    queue = janus.Queue()
    fake_pa = _FakePyAudio()
    capture = AudioCapture(queue, pyaudio_factory=lambda: fake_pa)

    capture.start()

    stream = fake_pa.opened_stream
    assert stream.kwargs["format"] == pyaudio.paInt16
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["rate"] == 16000
    assert stream.kwargs["frames_per_buffer"] == 512
    assert stream.kwargs["input"] is True
    assert callable(stream.kwargs["stream_callback"])
    assert stream.started is True
    queue.close()


def test_start_respects_custom_chunk_size_and_device_index():
    queue = janus.Queue()
    fake_pa = _FakePyAudio()
    capture = AudioCapture(
        queue, chunk_size=1024, input_device_index=3, pyaudio_factory=lambda: fake_pa
    )

    capture.start()

    stream = fake_pa.opened_stream
    assert stream.kwargs["frames_per_buffer"] == 1024
    assert stream.kwargs["input_device_index"] == 3
    queue.close()


def test_stop_closes_stream_and_terminates_pyaudio():
    queue = janus.Queue()
    fake_pa = _FakePyAudio()
    capture = AudioCapture(queue, pyaudio_factory=lambda: fake_pa)
    capture.start()
    stream = fake_pa.opened_stream

    capture.stop()

    assert stream.started is False
    assert stream.closed is True
    assert fake_pa.terminated is True
    queue.close()


def test_stop_before_start_does_not_raise():
    queue = janus.Queue()
    capture = AudioCapture(queue, pyaudio_factory=_FakePyAudio)

    capture.stop()  # must not raise
    queue.close()
