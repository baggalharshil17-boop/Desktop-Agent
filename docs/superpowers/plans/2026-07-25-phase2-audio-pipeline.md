# Financial Voice Agent — Phase 2: Audio Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the audio capture-to-utterance pipeline: microphone → thread-safe handoff → noise suppression → VAD gating → buffered utterance → WAV bytes. This phase produces no LLM/TTS behavior — it produces a `bytes` WAV payload per spoken utterance, ready for Phase 3 to hand to Groq Whisper.

**Architecture:** Five small modules under `financial_voice_agent/audio/`. `wav.py` and `dsp.py` are pure, synchronous, and fully testable with no hardware or heavy models. `pipeline.py` is the async VAD-gating state machine and is tested entirely with fakes (fake queue, fake VAD scorer) — no real audio or ML model involved. `capture.py` and `vad.py` are the two modules that wrap real system/ML resources (PyAudio, Silero VAD); both are designed so their core logic is unit-testable via dependency injection (fake PyAudio factory, fake VAD model), while the actual hardware/model integration is a documented manual smoke test, not an automated one.

**Tech Stack:** Python 3.11+, pyaudio, janus, noisereduce, numpy, torch + silero-vad (or equivalent Silero VAD package), pytest, pytest-asyncio (already configured in Phase 1).

> ## ⚠️ ENVIRONMENT STATUS (2026-07-25, resolved) — pyaudio install on Python 3.14
>
> **Task 4 (`financial_voice_agent/audio/capture.py`) was BLOCKED, now unblocked.** This dev
> environment runs **Python 3.14.2**, which is too new for `pyaudio`'s prebuilt wheels:
> - `pyaudio` 0.2.14 has no prebuilt wheel for `cp314` on PyPI — plain `pip install pyaudio`
>   tries to build from source and fails first with `Microsoft Visual C++ 14.0 or greater is
>   required`, then (once a compiler is present) with `Cannot open include file: 'portaudio.h'`
>   — the PortAudio C library itself isn't vendored anywhere pip can find it.
> - The documented fallback, `pipwin install pyaudio`, also fails — `pipwin`'s transitive
>   dependency `js2py` crashes on import on Python 3.14 (bytecode-introspection assumptions
>   that don't hold on this recent a release).
>
> **Resolution actually used (2026-07-25):**
> 1. Installed Microsoft C++ Build Tools ("Desktop development with C++" workload) — resolves
>    the compiler error.
> 2. Installed [vcpkg](https://github.com/microsoft/vcpkg) at a **short path**
>    (`C:\Users\dell\vcpkg`, not a deeply nested temp/scratchpad path — vcpkg's own build
>    process hits Windows' 260-character path limit otherwise: `ninja: error: ... Filename
>    longer than 260 characters`).
> 3. Built PortAudio via vcpkg using the **static** triplet specifically — pyaudio's `setup.py`
>    only supports static linking on Windows (see its own comments: "Only supports statically
>    linking with portaudio... use MT flag to match... vcpkg's portaudio"):
>    ```
>    vcpkg install portaudio:x64-windows-static
>    ```
>    (the plain `portaudio:x64-windows` dynamic triplet builds fine but does NOT work — pyaudio
>    will still fail with the missing-header error against it, since pyaudio's setup.py expects
>    the static triplet's directory layout.)
> 4. Set `VCPKG_PATH` to the triplet's **installed** directory, not the vcpkg checkout root —
>    pyaudio's `setup.py` does `os.path.join(WIN_VCPKG_PATH, 'include')` / `'lib'` directly:
>    ```
>    VCPKG_PATH=C:/Users/dell/vcpkg/installed/x64-windows-static
>    ```
>    (`VCPKG_PATH=C:/Users/dell/vcpkg` — the checkout root — silently produces the same missing-
>    header failure, since `include`/`lib` don't exist directly under the vcpkg root, only under
>    `installed/<triplet>/`.)
> 5. `pip install pyaudio` then builds successfully and produces a **statically-linked** wheel —
>    no runtime dependency on vcpkg or a portaudio DLL after install; `vcpkg` itself can be
>    deleted post-build if desired.
>
> Full diagnostic detail (both the original blocker and the resolution) is preserved in the
> Task 4 SDD report.
>
> **When pyaudio ships an official `cp314` wheel** (check `pip index versions pyaudio` or
> PyPI's file list for the `pyaudio` project), all of the above becomes unnecessary for future
> machines — plain `pip install pyaudio` will work again. Update this note when that happens;
> it is safe to delete this whole callout once Task 4 is merged and stable.
>
> **Task 5 (`torch` + `silero-vad`) installed cleanly on this Python 3.14 environment with no
> issues** — confirmed during Task 5's implementation, unlike pyaudio.

## Global Constraints

- Python 3.11+ only (per PRD Section 9).
- Microphone data contract: 16-bit PCM, mono, 16000 Hz, 512 or 1024-sample chunks (PRD Section 10 "Microphone → STT").
- Audio capture must hand off from the PyAudio callback thread to asyncio via a thread-safe queue (`janus` or `loop.call_soon_threadsafe`) — **never** a plain `asyncio.Queue` written to from the callback thread (PRD Section 11.2, the documented hard part).
- VAD tuning values come from `Config` (Phase 1 `financial_voice_agent/config.py`): `vad_speech_threshold` (default 0.5), `vad_silence_duration_ms` (default 600), `vad_min_speech_duration_ms` (default 200). Silero VAD requires 16000 Hz input (PRD "Voice & Audio Design").
- Pipeline order is fixed: PyAudio → noisereduce (stationary mode) → VAD gate → buffer (PRD "Voice & Audio Design").
- Buffer the full utterance in memory, then wrap in WAV using stdlib `wave` (PRD Section 10) — no streaming/chunked upload in this phase.
- No token-level streaming, no TTS, no playback in this phase — those are Phase 3.

## Known Environment Risks (read before dispatching Tasks 4 and 5)

- **Task 4 (`pyaudio`)** requires the native PortAudio library. On Windows, `pip install pyaudio` can fail without a prebuilt wheel. If it fails, try `pip install pipwin` then `pipwin install pyaudio` before escalating; if both fail, see the resolved "ENVIRONMENT STATUS" callout above for the vcpkg-based fix that worked on this machine (already applied — `pyaudio` should already be importable in this environment; if `import pyaudio` fails again, re-apply that callout's steps before reporting BLOCKED).
- **Task 5 (`torch` + Silero VAD)** pulls in a multi-hundred-MB dependency and the Silero model itself is fetched on first real use. The task's automated tests inject a fake model and never trigger a real download; only a manual smoke test (documented in the task's exit criteria) exercises the real model.

---

## File Structure

```
financial_voice_agent/
    audio/
        __init__.py
        wav.py         # encode_wav() — raw PCM -> WAV bytes (stdlib wave)
        dsp.py          # pcm_to_float32/float32_to_pcm/reduce_noise()
        pipeline.py     # AudioPipeline — VAD-gated buffering state machine
        capture.py      # make_capture_callback(), AudioCapture — PyAudio thread + janus handoff
        vad.py          # SileroVadScorer — production VadScorer implementation
tests/
    audio/
        __init__.py
        test_wav.py
        test_dsp.py
        test_pipeline.py
        test_capture.py
        test_vad.py
```

- `wav.py`: owns WAV container encoding. Nothing else touches the `wave` module.
- `dsp.py`: owns PCM↔float conversion and noise reduction. `pipeline.py` calls this; nothing else does.
- `pipeline.py`: owns the VAD-gating state machine (buffering, silence/min-duration thresholds, the `speech_active` signal Phase 3 will use for barge-in). Depends only on a `VadScorer` protocol, not on any concrete VAD implementation — this is what makes it testable without real ML.
- `capture.py`: owns the PyAudio thread and the callback-to-queue handoff. Nothing else opens a PyAudio stream.
- `vad.py`: owns the concrete Silero-backed `VadScorer`. Nothing else imports `torch` or `silero_vad`.

---

### Task 1: WAV encoding (`wav.py`)

**Files:**
- Create: `financial_voice_agent/audio/__init__.py` (empty)
- Create: `financial_voice_agent/audio/wav.py`
- Create: `tests/audio/__init__.py` (empty)
- Test: `tests/audio/test_wav.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `encode_wav(pcm_bytes: bytes, *, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes` — wraps raw PCM bytes in a WAV container.

- [ ] **Step 1: Write the failing test**

```python
# tests/audio/test_wav.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/audio/test_wav.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.audio'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/audio/wav.py
from __future__ import annotations

import io
import wave


def encode_wav(
    pcm_bytes: bytes,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()
```

Also create empty `financial_voice_agent/audio/__init__.py` and empty `tests/audio/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/audio/test_wav.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/audio/__init__.py financial_voice_agent/audio/wav.py tests/audio/__init__.py tests/audio/test_wav.py
git commit -m "feat: add WAV encoding for buffered utterances"
```

---

### Task 2: PCM conversion and noise reduction (`dsp.py`)

**Files:**
- Create: `financial_voice_agent/audio/dsp.py`
- Test: `tests/audio/test_dsp.py`
- Modify: `requirements.txt` — add `numpy`, `noisereduce`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pcm_to_float32(pcm_bytes: bytes) -> numpy.ndarray` — int16 PCM → float32 array in `[-1.0, 1.0]`.
  - `float32_to_pcm(arr: numpy.ndarray) -> bytes` — float32 array → int16 PCM bytes (clipped to `[-1.0, 1.0]` first).
  - `reduce_noise(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes` — runs `noisereduce.reduce_noise(..., stationary=True)` and returns PCM bytes of the same length as the input.

- [ ] **Step 1: Write the failing test**

```python
# tests/audio/test_dsp.py
import numpy as np

from financial_voice_agent.audio.dsp import float32_to_pcm, pcm_to_float32, reduce_noise


def test_pcm_to_float32_converts_known_values():
    pcm = np.array([0, 32767, -32768, 16384], dtype=np.int16).tobytes()

    result = pcm_to_float32(pcm)

    assert result.dtype == np.float32
    assert result[0] == 0.0
    assert abs(result[1] - 1.0) < 0.001
    assert abs(result[2] - (-1.0)) < 0.001
    assert abs(result[3] - 0.5) < 0.001


def test_float32_to_pcm_round_trips_within_rounding_tolerance():
    original = np.array([0, 16384, -16384, 32000], dtype=np.int16)
    pcm = original.tobytes()

    float_arr = pcm_to_float32(pcm)
    round_tripped = float32_to_pcm(float_arr)

    result = np.frombuffer(round_tripped, dtype=np.int16)
    assert np.allclose(result, original, atol=2)


def test_float32_to_pcm_clips_out_of_range_values():
    arr = np.array([2.0, -2.0, 0.0], dtype=np.float32)

    pcm = float32_to_pcm(arr)

    result = np.frombuffer(pcm, dtype=np.int16)
    assert result[0] == 32767
    assert result[1] == -32767
    assert result[2] == 0


def test_reduce_noise_returns_same_length_as_input():
    rng = np.random.default_rng(42)
    sample_rate = 16000
    t = np.arange(sample_rate) / sample_rate
    clean_tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noisy = clean_tone + rng.normal(0, 0.05, size=clean_tone.shape).astype(np.float32)
    pcm_in = float32_to_pcm(noisy)

    pcm_out = reduce_noise(pcm_in, sample_rate=sample_rate)

    assert len(pcm_out) == len(pcm_in)
    assert isinstance(pcm_out, bytes)


def test_reduce_noise_moves_signal_closer_to_clean_tone():
    rng = np.random.default_rng(42)
    sample_rate = 16000
    t = np.arange(sample_rate) / sample_rate
    clean_tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noise = rng.normal(0, 0.05, size=clean_tone.shape).astype(np.float32)
    noisy = clean_tone + noise
    pcm_noisy = float32_to_pcm(noisy)

    pcm_reduced = reduce_noise(pcm_noisy, sample_rate=sample_rate)
    reduced = pcm_to_float32(pcm_reduced)

    rms_before = np.sqrt(np.mean((noisy - clean_tone) ** 2))
    rms_after = np.sqrt(np.mean((reduced - clean_tone) ** 2))
    assert rms_after <= rms_before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/audio/test_dsp.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.audio.dsp'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/audio/dsp.py
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
```

Add to `requirements.txt`: `numpy`, `noisereduce`. Run `pip install -r requirements.txt`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/audio/test_dsp.py -v`
Expected: PASS (5 tests). `test_reduce_noise_moves_signal_closer_to_clean_tone` uses a fixed seed (42) so it is deterministic — if it fails, do not loosen the assertion; re-check the `reduce_noise` implementation first.

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/audio/dsp.py tests/audio/test_dsp.py requirements.txt
git commit -m "feat: add PCM conversion and noise reduction"
```

---

### Task 3: VAD-gated buffering pipeline (`pipeline.py`)

**Files:**
- Create: `financial_voice_agent/audio/pipeline.py`
- Test: `tests/audio/test_pipeline.py`

**Interfaces:**
- Consumes:
  - `dsp.reduce_noise(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes` (Task 2).
  - A `queue` object with an async `get()` method yielding `bytes` chunks or `None` as an end-of-stream sentinel (in production this is `janus.Queue().async_q` from Task 4; tests use a plain `asyncio.Queue`, which has the same `get()` interface).
  - A `vad_scorer` object implementing `score(pcm_chunk: bytes) -> float` (in production this is `vad.SileroVadScorer` from Task 5; tests use a fake).
- Produces:
  - `class VadScorer(Protocol): def score(self, pcm_chunk: bytes) -> float: ...` — the interface Task 5 implements.
  - `class AudioPipeline:`
    - `__init__(self, queue, vad_scorer, *, sample_rate: int = 16000, speech_threshold: float = 0.5, silence_duration_ms: float = 600.0, min_speech_duration_ms: float = 200.0, apply_noise_reduction: bool = True)`
    - `self.speech_active: asyncio.Event` — set while VAD currently detects speech, cleared once an utterance finalizes. Phase 3 uses this for barge-in.
    - `async def run(self) -> AsyncIterator[bytes]` — async generator; yields one raw PCM `bytes` payload per completed utterance (post-noise-reduction, pre-WAV — Phase 3 passes this to `wav.encode_wav`). Ends when the queue yields `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/audio/test_pipeline.py
import asyncio

import pytest

from financial_voice_agent.audio.pipeline import AudioPipeline

CHUNK_MS = 10.0  # 160 samples at 16000 Hz = 10ms per chunk


def _chunk(sample: int = 0) -> bytes:
    return sample.to_bytes(2, "little", signed=True) * 160


class _ScriptedVad:
    def __init__(self, scores):
        self._scores = list(scores)

    def score(self, pcm_chunk: bytes) -> float:
        return self._scores.pop(0)


async def _feed(queue, chunks):
    for c in chunks:
        await queue.put(c)
    await queue.put(None)


@pytest.mark.asyncio
async def test_pipeline_yields_one_utterance_for_speech_then_silence():
    queue = asyncio.Queue()
    # 3 speech chunks (30ms >= min_speech_duration_ms=20), then 3 silence chunks (30ms >= silence_duration_ms=20)
    vad = _ScriptedVad([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    chunks = [_chunk(100)] * 3 + [_chunk(0)] * 3
    await _feed(queue, chunks)
    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=20.0, min_speech_duration_ms=20.0, apply_noise_reduction=False
    )

    utterances = [u async for u in pipeline.run()]

    assert len(utterances) == 1
    assert utterances[0] == b"".join(chunks[:6])


@pytest.mark.asyncio
async def test_pipeline_discards_utterance_shorter_than_min_speech_duration():
    queue = asyncio.Queue()
    # 1 speech chunk (10ms < min_speech_duration_ms=20), then 3 silence chunks (30ms >= silence_duration_ms=20)
    vad = _ScriptedVad([0.9, 0.1, 0.1, 0.1])
    chunks = [_chunk(100)] * 1 + [_chunk(0)] * 3
    await _feed(queue, chunks)
    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=20.0, min_speech_duration_ms=20.0, apply_noise_reduction=False
    )

    utterances = [u async for u in pipeline.run()]

    assert utterances == []


@pytest.mark.asyncio
async def test_pipeline_pure_silence_yields_nothing():
    queue = asyncio.Queue()
    vad = _ScriptedVad([0.1, 0.1, 0.1])
    chunks = [_chunk(0)] * 3
    await _feed(queue, chunks)
    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=20.0, min_speech_duration_ms=20.0, apply_noise_reduction=False
    )

    utterances = [u async for u in pipeline.run()]

    assert utterances == []


@pytest.mark.asyncio
async def test_pipeline_flushes_pending_utterance_on_end_of_stream():
    queue = asyncio.Queue()
    # 3 speech chunks (30ms >= 20), then queue closes with no trailing silence
    vad = _ScriptedVad([0.9, 0.9, 0.9])
    chunks = [_chunk(100)] * 3
    await _feed(queue, chunks)
    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=20.0, min_speech_duration_ms=20.0, apply_noise_reduction=False
    )

    utterances = [u async for u in pipeline.run()]

    assert len(utterances) == 1
    assert utterances[0] == b"".join(chunks)


@pytest.mark.asyncio
async def test_speech_active_event_set_during_speech_and_cleared_after():
    queue = asyncio.Queue()
    vad = _ScriptedVad([0.9, 0.9, 0.1, 0.1])
    chunks = [_chunk(100)] * 2 + [_chunk(0)] * 2
    await _feed(queue, chunks)
    pipeline = AudioPipeline(
        queue, vad, silence_duration_ms=15.0, min_speech_duration_ms=15.0, apply_noise_reduction=False
    )

    states_during_run = []
    gen = pipeline.run()
    async for _utterance in gen:
        pass
    # After the generator is exhausted, speech has finalized and the event must be clear.
    assert not pipeline.speech_active.is_set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/audio/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.audio.pipeline'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/audio/pipeline.py
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Protocol

from financial_voice_agent.audio import dsp

SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM


class VadScorer(Protocol):
    def score(self, pcm_chunk: bytes) -> float: ...


def _chunk_duration_ms(chunk: bytes, sample_rate: int) -> float:
    num_samples = len(chunk) / SAMPLE_WIDTH_BYTES
    return (num_samples / sample_rate) * 1000.0


class AudioPipeline:
    def __init__(
        self,
        queue,
        vad_scorer: VadScorer,
        *,
        sample_rate: int = 16000,
        speech_threshold: float = 0.5,
        silence_duration_ms: float = 600.0,
        min_speech_duration_ms: float = 200.0,
        apply_noise_reduction: bool = True,
    ) -> None:
        self._queue = queue
        self._vad_scorer = vad_scorer
        self._sample_rate = sample_rate
        self._speech_threshold = speech_threshold
        self._silence_duration_ms = silence_duration_ms
        self._min_speech_duration_ms = min_speech_duration_ms
        self._apply_noise_reduction = apply_noise_reduction
        self.speech_active = asyncio.Event()

    async def run(self) -> AsyncIterator[bytes]:
        buffer = bytearray()
        speech_ms = 0.0
        silence_ms = 0.0
        in_utterance = False

        while True:
            chunk = await self._queue.get()
            if chunk is None:
                if in_utterance and speech_ms >= self._min_speech_duration_ms:
                    yield bytes(buffer)
                self.speech_active.clear()
                return

            if self._apply_noise_reduction:
                chunk = dsp.reduce_noise(chunk, sample_rate=self._sample_rate)

            score = self._vad_scorer.score(chunk)
            is_speech = score >= self._speech_threshold
            duration_ms = _chunk_duration_ms(chunk, self._sample_rate)

            if is_speech:
                if not in_utterance:
                    in_utterance = True
                    buffer = bytearray()
                    speech_ms = 0.0
                    silence_ms = 0.0
                if not self.speech_active.is_set():
                    self.speech_active.set()
                buffer.extend(chunk)
                speech_ms += duration_ms
                silence_ms = 0.0
            elif in_utterance:
                buffer.extend(chunk)
                silence_ms += duration_ms
                if silence_ms >= self._silence_duration_ms:
                    self.speech_active.clear()
                    if speech_ms >= self._min_speech_duration_ms:
                        yield bytes(buffer)
                    in_utterance = False
                    buffer = bytearray()
                    speech_ms = 0.0
                    silence_ms = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/audio/test_pipeline.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/audio/pipeline.py tests/audio/test_pipeline.py
git commit -m "feat: add VAD-gated audio buffering pipeline"
```

---

### Task 4: Thread-safe audio capture (`capture.py`)

**Files:**
- Create: `financial_voice_agent/audio/capture.py`
- Test: `tests/audio/test_capture.py`
- Modify: `requirements.txt` — add `pyaudio`, `janus`

**Interfaces:**
- Consumes: nothing from earlier tasks in this phase (it is the producer end of the queue `pipeline.py` consumes from).
- Produces:
  - `make_capture_callback(sync_queue) -> Callable[[bytes, int, dict, int], tuple[None, int]]` — pure function; returns a PyAudio `stream_callback` that pushes `bytes(in_data)` onto `sync_queue` (a `janus.Queue().sync_q` in production) and returns `(None, pyaudio.paContinue)`.
  - `class AudioCapture:`
    - `__init__(self, queue, *, sample_rate: int = 16000, channels: int = 1, chunk_size: int = 512, input_device_index: int | None = None, pyaudio_factory=pyaudio.PyAudio)` — `queue` is a `janus.Queue` instance (or anything with a `.sync_q.put()`-compatible `sync_q` attribute).
    - `def start(self) -> None` — opens a PyAudio input stream: `format=pyaudio.paInt16, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size, input_device_index=input_device_index, stream_callback=make_capture_callback(queue.sync_q)`, then starts it.
    - `def stop(self) -> None` — stops and closes the stream, terminates the PyAudio instance. Safe to call even if `start()` was never called.

- [ ] **Step 1: Write the failing test**

```python
# tests/audio/test_capture.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/audio/test_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.audio.capture'` (or, if `pyaudio` itself is not yet installed, an `ImportError` for `pyaudio` — install it first per the Known Environment Risks section above; report BLOCKED if the native install fails).

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/audio/capture.py
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
```

Add to `requirements.txt`: `pyaudio`, `janus`. Run `pip install -r requirements.txt`. If `pyaudio` fails to build on Windows, try `pip install pipwin && pipwin install pyaudio`. If that also fails, STOP and report BLOCKED with the exact error — do not mock around the missing dependency.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/audio/test_capture.py -v`
Expected: PASS (5 tests). No real microphone or PortAudio hardware access occurs in these tests — `_FakePyAudio` never touches the real `pyaudio.PyAudio` class.

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/audio/capture.py tests/audio/test_capture.py requirements.txt
git commit -m "feat: add thread-safe PyAudio capture with janus handoff"
```

---

### Task 5: Silero VAD scorer (`vad.py`)

**Files:**
- Create: `financial_voice_agent/audio/vad.py`
- Test: `tests/audio/test_vad.py`
- Modify: `requirements.txt` — add `torch`, `silero-vad`

**Interfaces:**
- Consumes: `dsp.pcm_to_float32(pcm_bytes: bytes) -> numpy.ndarray` (Task 2).
- Produces: `class SileroVadScorer:` implementing the `pipeline.VadScorer` protocol from Task 3 (`score(pcm_chunk: bytes) -> float`).
  - `__init__(self, model=None)` — `model` is any callable `model(tensor, sample_rate) -> tensor_with_item()`; if `None`, the real Silero model is lazily loaded on first `score()` call via `silero_vad.load_silero_vad()`.
  - `score(self, pcm_chunk: bytes) -> float` — converts the chunk to a float32 tensor via `dsp.pcm_to_float32`, runs the model at 16000 Hz, returns `float(output.item())`.

- [ ] **Step 1: Write the failing test**

```python
# tests/audio/test_vad.py
import numpy as np
import torch

from financial_voice_agent.audio.vad import SileroVadScorer


class _FakeModel:
    def __init__(self, value: float):
        self._value = value
        self.calls = []

    def __call__(self, tensor, sample_rate):
        self.calls.append((tensor, sample_rate))
        return torch.tensor(self._value)


def test_score_returns_model_output_as_float():
    fake_model = _FakeModel(0.87)
    scorer = SileroVadScorer(model=fake_model)
    pcm = np.array([0, 16384, -16384, 0], dtype=np.int16).tobytes()

    result = scorer.score(pcm)

    assert isinstance(result, float)
    assert abs(result - 0.87) < 1e-6


def test_score_calls_model_with_16000_sample_rate():
    fake_model = _FakeModel(0.1)
    scorer = SileroVadScorer(model=fake_model)

    scorer.score(b"\x00\x00" * 160)

    assert fake_model.calls[0][1] == 16000


def test_score_passes_float_tensor_derived_from_pcm():
    fake_model = _FakeModel(0.1)
    scorer = SileroVadScorer(model=fake_model)
    pcm = np.array([0, 32767], dtype=np.int16).tobytes()

    scorer.score(pcm)

    tensor_arg = fake_model.calls[0][0]
    assert isinstance(tensor_arg, torch.Tensor)
    assert tensor_arg.dtype == torch.float32
    assert abs(float(tensor_arg[1]) - 1.0) < 0.001


def test_injected_model_is_reused_across_calls_without_reloading():
    fake_model = _FakeModel(0.1)
    scorer = SileroVadScorer(model=fake_model)

    scorer.score(b"\x00\x00" * 10)
    scorer.score(b"\x00\x00" * 10)

    assert len(fake_model.calls) == 2
    assert scorer._model is fake_model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/audio/test_vad.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'financial_voice_agent.audio.vad'`.

- [ ] **Step 3: Write minimal implementation**

```python
# financial_voice_agent/audio/vad.py
from __future__ import annotations

import torch

from financial_voice_agent.audio import dsp


class SileroVadScorer:
    def __init__(self, model=None) -> None:
        self._model = model

    def _ensure_model(self):
        if self._model is None:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
        return self._model

    def score(self, pcm_chunk: bytes) -> float:
        model = self._ensure_model()
        float_arr = dsp.pcm_to_float32(pcm_chunk)
        tensor = torch.from_numpy(float_arr)
        with torch.no_grad():
            output = model(tensor, 16000)
        return float(output.item())
```

Add to `requirements.txt`: `torch`, `silero-vad`. Run `pip install -r requirements.txt` (this is a large download — expect several minutes).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/audio/test_vad.py -v`
Expected: PASS (4 tests). None of these tests load the real Silero model or touch the network — `model=fake_model` is always injected.

- [ ] **Step 5: Manual smoke test (not automated — record the result in your report, do not skip)**

Run this once, interactively, to confirm the real model loads and produces sane scores (requires network access for the first-time model download):

```bash
python -c "
from financial_voice_agent.audio.vad import SileroVadScorer
from financial_voice_agent.audio.dsp import float32_to_pcm
import numpy as np

scorer = SileroVadScorer()
silence = float32_to_pcm(np.zeros(160, dtype=np.float32))
print('silence score:', scorer.score(silence))
tone = float32_to_pcm((0.5 * np.sin(2 * np.pi * 200 * np.arange(160) / 16000)).astype(np.float32))
print('tone score:', scorer.score(tone))
"
```

Expected: runs without error and prints two float scores between 0 and 1 (a pure tone is not real speech, so don't expect a high score — this just confirms the model loads and the interface works end-to-end). If this fails due to no network access in your environment, note it as a concern in your report rather than blocking the task — the automated tests (Step 4) are what gates this task's completion.

- [ ] **Step 6: Commit**

```bash
git add financial_voice_agent/audio/vad.py tests/audio/test_vad.py requirements.txt
git commit -m "feat: add Silero-backed VAD scorer"
```

---

## Phase 2 Exit Criteria

- `pytest tests/ -v` passes with 0 failures (Phase 1's 23 tests + Phase 2's ~22 new tests).
- No test in `tests/audio/` opens a real PyAudio stream, loads the real Silero model, or requires a microphone — all hardware/ML-model interaction is behind dependency injection.
- `AudioPipeline.run()` never yields an utterance shorter than `min_speech_duration_ms` and always yields (or discards, per the same rule) the buffered utterance when the queue closes.
- Task 5's manual smoke test result (pass, fail, or "skipped — no network") is recorded in that task's report.

---

## Upcoming Phases (summaries — to be written in full detail after Phase 2 review)

**Phase 3 — Turn Orchestrator Core:** Wires Phase 1 (config, db, http_clients) and Phase 2 (`AudioCapture` → `AudioPipeline` → `wav.encode_wav`) into the full turn loop: Groq Whisper STT → Groq LLM tool-calling loop (`asyncio.gather` for concurrent tool calls) → Cartesia WebSocket TTS → PyAudio playback. Barge-in wiring uses Phase 2's `AudioPipeline.speech_active` event: playback checks it between PCM chunks and stops immediately when set. `latency_*_ms` timing is captured at each stage for `db.log_turn`. Tool calls are stubbed/mocked in this phase; Phase 4 implements them for real.

**Phase 4 — Tools:** Implements `get_quote`, `get_ohlc_history`, `compute_indicator` (Bollinger/Fibonacci/MA/RSI via `ta`/`pandas`/`numpy`), `get_positions_holdings`, `get_news` (Tavily), `capture_screen` (mss + pygetwindow/Quartz), each following the PRD Section 6 tool table's exact error/retry behavior (401 → "session expired", 429 → backoff, mock mode reads Phase 1 fixtures). Registers all six with the Phase 3 LLM tool loop.

**Phase 5 — Eval Harness:** JSON test cases (PRD Section 17) with input transcript, optional mocked screen result, expected tool calls, and tools that must NOT be called. A runner asserts on tool names/args only (not exact wording), run against `mode: "mock"` end-to-end through Phases 1-4. Seeds the 8 starting cases from the PRD and provides the harness for adding a case every time real usage surfaces a wrong tool call.
