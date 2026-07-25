from financial_voice_agent.orchestrator.playback import AudioPlayback


class _FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.written_chunks: list[bytes] = []
        self.stopped = False
        self.closed = False

    def write(self, chunk: bytes) -> None:
        self.written_chunks.append(chunk)

    def stop_stream(self):
        self.stopped = True

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


def test_open_uses_documented_output_data_contract():
    import pyaudio

    fake_pa = _FakePyAudio()
    playback = AudioPlayback(pyaudio_factory=lambda: fake_pa)

    playback.open()

    stream = fake_pa.opened_stream
    assert stream.kwargs["format"] == pyaudio.paInt16
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["rate"] == 16000
    assert stream.kwargs["output"] is True


def test_play_writes_all_chunks_when_not_interrupted():
    fake_pa = _FakePyAudio()
    playback = AudioPlayback(pyaudio_factory=lambda: fake_pa)
    playback.open()
    pcm_bytes = b"\x00\x01" * 3000  # spans multiple 1024-byte chunks

    completed = playback.play(pcm_bytes, chunk_size=1024)

    stream = fake_pa.opened_stream
    assert completed is True
    assert b"".join(stream.written_chunks) == pcm_bytes


def test_play_stops_early_when_interrupt_event_is_set():
    import threading

    fake_pa = _FakePyAudio()
    playback = AudioPlayback(pyaudio_factory=lambda: fake_pa)
    playback.open()
    pcm_bytes = b"\x00\x01" * 3000
    interrupt_event = threading.Event()

    stream = fake_pa.opened_stream
    original_write = stream.write

    def _write_and_interrupt_after_first_chunk(chunk):
        original_write(chunk)
        if len(stream.written_chunks) == 1:
            interrupt_event.set()

    stream.write = _write_and_interrupt_after_first_chunk

    completed = playback.play(pcm_bytes, chunk_size=1024, interrupt_event=interrupt_event)

    assert completed is False
    assert len(stream.written_chunks) == 1  # stopped after the first chunk, not all 3


def test_close_stops_and_closes_stream_and_terminates_pyaudio():
    fake_pa = _FakePyAudio()
    playback = AudioPlayback(pyaudio_factory=lambda: fake_pa)
    playback.open()
    stream = fake_pa.opened_stream

    playback.close()

    assert stream.stopped is True
    assert stream.closed is True
    assert fake_pa.terminated is True
