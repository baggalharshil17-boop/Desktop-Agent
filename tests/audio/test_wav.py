import wave
import io

from financial_voice_agent.audio.wav import encode_wav


def test_encode_wav_round_trips_pcm_bytes():
    pcm = bytes(range(0, 256)) * 4  # 1024 bytes of synthetic PCM

    wav_bytes = encode_wav(pcm, sample_rate=16000, channels=1, sample_width=2)

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        frames = wav_file.readframes(wav_file.getnframes())
    assert frames == pcm


def test_encode_wav_uses_default_params():
    pcm = b"\x00\x01" * 100

    wav_bytes = encode_wav(pcm)

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.readframes(wav_file.getnframes()) == pcm


def test_encode_wav_empty_pcm_produces_valid_empty_wav():
    wav_bytes = encode_wav(b"")

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnframes() == 0
