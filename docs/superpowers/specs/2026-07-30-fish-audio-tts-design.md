# Add Fish Audio as a TTS Provider Option

**Status:** Approved design, not yet implemented.
**Date:** 2026-07-30

## Goal

Add Fish Audio as a selectable `tts.provider` option alongside Cartesia, replacing the Deepgram slot that was configured but never actually implemented. Also let a fresh clone choose it during setup, via `scripts/setup.py`.

## Scope

**In scope:**
- A real `RealFishAudioTtsClient` adapter (`financial_voice_agent/orchestrator/tts.py`), implementing the existing `TtsClient` Protocol.
- Removing Deepgram's placeholder entirely (unused config validation and an unused `http_clients.py` branch — no adapter was ever built for it).
- Config, `http_clients.py`, and `__main__.py` wiring so `tts.provider: "fish_audio"` actually runs.
- `scripts/setup.py` prompting for TTS provider choice (currently hardcoded to Cartesia) and conditionally collecting/validating a Fish Audio key, plus a new `validate_fish_audio_key()` in `financial_voice_agent/setup/validators.py`.
- `README.md`'s provider table gets a Fish Audio row.

**Explicitly out of scope:**
- Fish Audio's ASR/speech-to-text endpoint (`/v1/asr`) — this plan is TTS only. ASR requires paid API credit with no free tier (confirmed by live testing), unlike TTS's free `s2.1-pro-free` model, and the user's original ask was TTS-specific. A separate future effort if/when there's a reason to revisit it.
- A live automatic TTS fallback mechanism (Cartesia primary + Fish Audio auto-fallback on failure). `tts.provider` stays a single mutually-exclusive choice, same shape as `stt.provider`/`llm.provider` today — no new runtime fallback behavior. `synthesize_with_fallback()`'s existing fallback parameter remains unwired (as it already is today for Cartesia alone).
- Fish Audio voice cloning / custom reference voices (`reference_id`/`references` fields) — the default voice (no `reference_id`) is used, matching how this project doesn't currently expose Cartesia voice cloning either, just a `voice_id` picker.

## Context

`financial_voice_agent/orchestrator/tts.py` already has a `TtsClient` Protocol (`async def synthesize(self, text: str) -> bytes`) and a working `RealCartesiaTtsClient` adapter. `financial_voice_agent/config.py`'s `_VALID_TTS_PROVIDERS = {"cartesia", "deepgram"}` and `http_clients.py`'s Deepgram branch have existed since Phase 1, but no `RealDeepgramTtsClient` was ever written — `__main__.py` has a hard `NotImplementedError` guard: `"Only tts.provider: 'cartesia' is wired up in the live loop right now... no Deepgram TTS adapter exists yet"`. Deepgram was always a placeholder, not a real second option.

Fish Audio's TTS API was verified with real live calls (not just documentation): `POST https://api.fish.audio/v1/tts`, `Authorization: Bearer <key>` header, model selection via a `model` **request header** (not a JSON body field — this was the one non-obvious detail, discovered from a real code sample, not the OpenAPI schema summary, which didn't mention it), JSON body `{"text": ..., "format": "wav", "sample_rate": 16000}`, and the response is raw audio bytes (`audio/wav` content-type) — no streaming, no websocket, no SDK needed. The `s2.1-pro-free` model works with zero API credit balance; other models (`s2.1-pro`, `s2-pro`, `s1`) return `402 Insufficient API credit` without a funded account (confirmed directly — this is a real account-balance gate, not a bug or misconfiguration).

`scripts/setup.py` (from the v3.0 onboarding wizard) currently never prompts for TTS provider at all — it unconditionally prompts for and validates `CARTESIA_API_KEY`. `financial_voice_agent/setup/validators.py` already has `validate_cartesia_key()` as a template to follow for the new `validate_fish_audio_key()`.

## Design

### Core TTS adapter

`financial_voice_agent/orchestrator/tts.py` gains:
```python
class RealFishAudioTtsClient:
    def __init__(self, http_client, *, model: str = "s2.1-pro-free") -> None:
        self._client = http_client
        self._model = model

    async def synthesize(self, text: str) -> bytes:
        response = await self._client.post(
            "/v1/tts", headers={"model": self._model},
            json={"text": text, "format": "wav", "sample_rate": 16000},
        )
        response.raise_for_status()
        return response.content
```
`http_client` is an `httpx.AsyncClient` already carrying the `Authorization: Bearer` header, base URL, and timeout, built in `http_clients.py` (matching how the Kite/Tavily/Indian-Stock clients are already constructed) — no new dependency, no vendor SDK.

A new `make_tts_client(config, http_clients) -> TtsClient` factory (also in `tts.py`) replaces `__main__.py`'s current inline `RealCartesiaTtsClient(...)` construction and hard `NotImplementedError` guard, branching on `config.tts_provider` — matching the existing `make_stt_client`/`make_llm_client` factory pattern in `stt.py`/`llm.py`.

### Configuration

- `financial_voice_agent/config.py`: `_VALID_TTS_PROVIDERS` becomes `{"cartesia", "fish_audio"}`. `deepgram_api_key` field removed (with its validation branch); new `fish_audio_api_key: str | None`, read from `.env`'s `FISH_AUDIO_API_KEY`, required (validated, same pattern as Cartesia's key) when `tts.provider` is `"fish_audio"`.
- `financial_voice_agent/http_clients.py`: `DEEPGRAM_BASE_URL` removed; new `FISH_AUDIO_BASE_URL = "https://api.fish.audio"`, client built with the `Authorization: Bearer` header.
- `config.yaml`: new optional `tts.fish_audio_model` field (default `"s2.1-pro-free"` when absent), Fish-Audio-specific like `tts.voice_id` is Cartesia-specific — not a shared/generic key.
- `.env.example`: `DEEPGRAM_API_KEY=` line removed, `FISH_AUDIO_API_KEY=` added.

### Setup wizard

`scripts/setup.py` gains a `tts.provider` prompt (`_ask_choice("TTS provider", ["cartesia", "fish_audio"], default="cartesia")`), currently absent. `CARTESIA_API_KEY` collection becomes conditional on choosing Cartesia (currently unconditional); `FISH_AUDIO_API_KEY` collection is conditional on choosing Fish Audio, validated via a new `validate_fish_audio_key()` in `financial_voice_agent/setup/validators.py` (a cheap live call — the verified `POST /v1/tts` with short text and the free `s2.1-pro-free` model). `financial_voice_agent/setup/config_template.py`'s rendered template gets the new `tts.fish_audio_model` field. `README.md`'s provider table gets a Fish Audio row (env var, where to get a key, free-tier note).

## Error Handling

`RealFishAudioTtsClient.synthesize()` raises on any HTTP error via `response.raise_for_status()`, letting `synthesize_with_fallback()`'s existing retry logic (2 attempts, 1s fixed backoff) handle transient failures — identical to how Cartesia failures are already retried today. No new error-handling code needed; this reuses the existing `TtsError` path unchanged. `validate_fish_audio_key()` follows `validate_cartesia_key()`'s existing pattern: catches errors and returns `ValidationResult(ok=False, message=...)` rather than raising, so a bad/unfunded key during setup shows a clear message and offers retry-or-skip, never crashing the wizard.

## Testing

- `RealFishAudioTtsClient.synthesize()`: unit tests via `httpx.MockTransport` (matching `tests/tools/test_fundamentals.py`'s pattern for the Indian Stock API) — verify the `model` header is sent, the JSON body shape is correct, and raw response bytes are returned unchanged.
- `make_tts_client()`: unit tests with DI (fake `http_client`/fake Cartesia client), matching `make_stt_client`/`make_llm_client`'s existing test pattern.
- `validate_fish_audio_key()`: unit tests with a fake client factory, matching `validate_cartesia_key()`'s existing tests.
- One manual verification: run the real agent with `tts.provider: "fish_audio"` in `config.yaml` and confirm real synthesized audio plays back correctly — consistent with how every other vendor integration in this project has been confirmed against the real API before being considered done.

## Explicitly Deferred / Not Addressed

- Fish Audio ASR (speech-to-text) — no free tier found; a separate future effort.
- Live automatic TTS fallback (Cartesia + Fish Audio both wired, auto-switching on failure) — `tts.provider` stays single-choice.
- Fish Audio voice cloning / custom reference voices.
