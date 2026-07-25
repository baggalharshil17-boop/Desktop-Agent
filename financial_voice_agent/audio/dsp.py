from __future__ import annotations

import numpy as np
import noisereduce as nr


def pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    ints = np.frombuffer(pcm_bytes, dtype=np.int16)
    return (ints.astype(np.float32)) / 32768.0


def float32_to_pcm(arr: np.ndarray) -> bytes:
    clipped = np.clip(arr, -1.0, 1.0)
    ints = (clipped * 32767.0).astype(np.int16)
    return ints.tobytes()


def reduce_noise(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    float_arr = pcm_to_float32(pcm_bytes)
    reduced = nr.reduce_noise(y=float_arr, sr=sample_rate, stationary=True)
    return float32_to_pcm(reduced.astype(np.float32))
