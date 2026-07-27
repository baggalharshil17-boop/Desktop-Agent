# Financial Voice Agent — PRD v2.1 (Developer Reference)

## Product Essence
Read-only AI voice assistant for Zerodha Kite trading dashboard. Listens continuously (VAD-gated), executes tool calls, speaks answers. No order placement, no UI automation, no multi-user auth. Single user, personal machine.

## Architecture: Cascade, Not S2S

**v1.4 used:** Gemini Multimodal Live (persistent WebSocket, continuous vision). **Rationale:** native multimodal + native tool calling.

**v2.1 uses:** STT → LLM (tool calling) → TTS (cascaded HTTP). **Why:** vision is now on-demand (model-triggered), not continuous. Cascade buys observability, cost, simplicity. Trade-off: ~1–2.5s per turn instead of <500ms.

**Model stack:**
- Speech-to-text: Groq Whisper Large v3 Turbo (~$0.04/hour audio, ~228x RT), **or** Hugging Face Inference (`huggingface_hub.AsyncInferenceClient.automatic_speech_recognition`, provider `"hf-inference"`) — config-driven via `stt.provider`, built as a fallback for when one vendor's free credits run out
- Reasoning + tools + vision: Groq multimodal tool-calling model (check console.groq.com/docs/models — model catalog churns), **or** Hugging Face Inference chat completions (`AsyncInferenceClient.chat_completion`, provider `"auto"` — `"hf-inference"` itself doesn't serve most chat/vision models, `"auto"` routes to whichever partner provider does) — config-driven via `llm.provider`. As of this build, `qwen/qwen3.6-27b` on Groq is the verified model supporting both tool calling and vision together; it defaults to an extended "thinking mode" that must be disabled via `reasoning_effort: "none"` (Qwen/gpt-oss-specific param) or every call costs tens of seconds of unwanted reasoning latency
- TTS: Cartesia Sonic (primary, ~40–90ms time-to-first-audio) or Deepgram Aura-2 (fallback, ~300ms — **not implemented**, only Cartesia has a real adapter so far)
- Tools: local Python functions, not MCP server (no need for v1)

**Both STT and LLM providers are swappable per-vendor at runtime via config.yaml**, not hardcoded to Groq as originally scoped — see Configuration section below. This exists because free-tier credits on either vendor can run out mid-session; switching providers is a config edit plus a token, not a code change.

## Core Capabilities

- **Voice interface:** microphone → VAD gate → Groq Whisper → Groq LLM (tool loop) → Cartesia TTS → speaker
- **On-demand vision:** `capture_screen()` called only when LLM decides it needs visual context (not polling, not manual button)
- **Tools (read-only, no writes):** `get_quote`, `get_ohlc_history`, `compute_indicator` (Bollinger, Fibonacci, RSI, MA), `get_positions_holdings`, `get_news`, `capture_screen`
- **Noise handling:** VAD + noise suppression pre-transcription
- **Barge-in:** user speaks → local VAD interrupts playback immediately (~300ms)

## Performance Targets (Cascaded)

Assume connection pooling per Section 14. Fresh TCP+TLS per call adds 100–300ms.

| Stage | Target |
|-------|--------|
| Noise suppression + VAD | <100ms (local only) |
| STT (Groq) | <400ms |
| First acknowledgment | <900ms (spoken before tool results) |
| Full answer, no tools | <1.5s |
| Full answer + Kite data pull | <2.5s |
| Full answer + screen capture | <3s |
| Full answer + web search | <4.5s |
| Barge-in interrupt | <300ms (local) |

*These are working assumptions. Validate against Section 13 turn log after 20+ real turns.*

## Data Contracts (Exact Byte-Level Specs)

**Microphone → STT:**
- PyAudio: 16-bit PCM, mono, 16000 Hz, 512 or 1024-sample chunks (match VAD's 16kHz)
- Buffer utterance in memory → wrap in WAV before upload (Python stdlib `wave`)
- Groq REST upload, <25MB per file (irrelevant for single utterance)

**LLM text → TTS (Cartesia WebSocket):**
- Cartesia output_format: `{ container: "raw", encoding: "pcm_s16le", sample_rate: 16000 }`
- PyAudio playback: matching 16000 Hz, 16-bit, mono (mismatch = chipmunk/slow-mo voice)
- v1: send full response text in one call after generation finishes (not token-streaming yet)

**Screen capture → vision:**
- mss captures cropped Kite window → JPEG (quality ~80) → base64 encode
- Sent as a `data:image/jpeg;base64,...` URI in an `image_url` content block, in a follow-up `user` message appended right after the tool-result message (tool-role message content must stay text-only per the OpenAI-compatible schema both Groq and HF use — the image can't ride inside the tool result itself)
- Only works if `llm.model` is actually vision-capable — a text-only model will happily call `capture_screen` and then fabricate a description instead of erroring, since it has no way to know it can't see the attached image

**Tool results to LLM:**
- Concise structured text, not raw API dumps (Kite's raw JSON wastes tokens)

## Concurrency Model (Critical for Implementation)

**Four concurrent components:**
1. **Audio capture (OS thread):** PyAudio callback → thread-safe queue (use `janus` or `loop.call_soon_threadsafe` — **not plain asyncio.Queue**)
2. **Audio pipeline (asyncio):** dequeue → noise suppression → VAD → buffer until end-of-speech
3. **Turn orchestrator (asyncio):** WAV encode → Groq STT → Groq LLM tool loop (concurrent tool calls via `asyncio.gather`) → TTS → persist to DB
4. **Playback (OS thread or asyncio):** Cartesia PCM chunks → speaker, interruptible on barge-in signal

**Barge-in:** VAD detects speech during playback → sets `asyncio.Event` → playback checks between chunks, stops immediately, discards buffered audio. Interrupted turn's background tasks finish quietly (tool calls already started aren't wasted).

## Session & Context Management

- LLM calls are stateless HTTP — full context resent every call (system prompt + windowed history + current turn)
- Keep last 6–8 turns in memory as `{role, content}` list. **Don't resend raw tool-call JSON** — keep only natural-language outcome (token savings + mimics human memory)
- System prompt resent every call; check if Groq supports prompt caching (cached-input discount)
- No cross-session memory — context resets on script restart (by design, not oversight)

## Local Persistence (SQLite, Section 13)

```sql
CREATE TABLE turns (
  turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc TEXT NOT NULL,            -- ISO 8601, UTC
  transcript TEXT,
  tool_calls_json TEXT,            -- [{tool, args}, ...]
  tool_results_json TEXT,          -- summarized
  response_text TEXT,
  screenshot_path TEXT,            -- file path if captured
  latency_stt_ms INTEGER,
  latency_llm_ms INTEGER,
  latency_tool_ms INTEGER,
  latency_tts_ms INTEGER,
  latency_total_ms INTEGER,
  error TEXT
);
```

**Why SQLite:** single-user, single-machine, zero ops overhead. Store screenshots as files, not BLOBs.

**Encryption:** OS-level disk encryption sufficient (personal machine); no app-level encryption needed.

**Use:** validate Section 5 targets, seed Section 17 eval set, audit trail for cascade-vs-S2S choice.

## Network & Transport

- All outbound: HTTPS (Groq, Cartesia, Kite, Tavily) + local WebSocket to Cartesia TTS
- No inbound, no firewall changes needed
- **Reuse one HTTP client per vendor** (httpx.AsyncClient or aiohttp.ClientSession created once at startup). Fresh connection = 100–300ms TCP+TLS tax per call.
- **Keep Cartesia WebSocket open** across turns where practical (reconnect only on error/idle timeout)
- No UDP, no WebRTC

## Error Handling & Retry Policy

| Call | Failure | Behavior |
|------|---------|----------|
| STT (Groq/HF) | Timeout or 5xx | Retry 2x with 1s backoff. On final fail: speak a varied "didn't catch that" phrase (not the same wording every time), re-listen |
| LLM (Groq/HF) | 429 rate limit | Exponential backoff; speak a varied "hit a usage limit" phrase |
| LLM (Groq/HF) | Any other failure (with a transcript already in hand) | Speak a varied *generic*-failure phrase, distinct from the STT one — saying "didn't catch that" when STT actually succeeded is a lie the user will notice |
| Cartesia TTS | Timeout or 5xx | Retry once. Fall back to Deepgram (if configured — **no Deepgram adapter built yet**) or logged error |
| Kite Connect | 401, or 403 with `error_type: "TokenException"` | Both mean expired/invalid `access_token` — **confirmed live that Kite returns 403, not 401, for this** (`{"error_type":"TokenException","message":"Incorrect api_key or access_token."}`); code must check both status codes and the JSON body, not just status |
| Kite Connect | 403 with `error_type: "PermissionException"` | Real account-level restriction (e.g. no active paid Kite Connect subscription for historical data) — **not** a token problem, must not be treated as "log in again"; surface the real message so the user knows to check their Zerodha subscription |
| Kite Connect | 429/503 | Per-tool policy in tool table (Section 6) |
| Any tool call | Any exception not specifically handled above | Must still become a `{"error": ...}` tool result the LLM can see and explain out loud in the same turn — an uncaught exception here crashes the whole turn and forces the generic fallback, which is worse than a slightly-generic spoken error |
| Tavily | Timeout/error | Degrade gracefully: answer with other results, state search unavailable |

## Voice & Audio Design

**Input trigger:** Always-on VAD (default). Expose `--ptt` flag for push-to-talk.

**VAD tuning (config.yaml):**
- Speech threshold: 0.5 (lower = catch soft speech, higher = reject noise)
- Silence before end-of-speech: 600ms (raise if cutting off mid-sentence)
- Min speech duration: 200ms (raise in noise)
- Sample rate: **must be 16000 Hz** (silero-vad trained at 16kHz)
- Barge-in: enabled by default (`vad.barge_in_enabled: true`); safe even with open speakers because echo suppression (below) prevents self-interruption

**Noise suppression:** PyAudio → noisereduce (stationary mode, <30ms overhead) → VAD gate → buffer

**Audio device selection (Windows):**
- PyAudio's built-in "default" device resolves through whichever host API it initializes first (usually MME), which can silently disagree with Windows Sound Settings (which controls WASAPI). Setting `audio.output_device_index: null` now uses WASAPI to identify the current physical device, then opens that device's MME entry instead — gaining both an accurate, live-tracking default and a stream that can resample to 16kHz (WASAPI-default devices only accept their native rate, e.g. 48kHz, and reject 16kHz outright). Setting an explicit integer index overrides this and pinpoints a specific device. Same for `audio.input_device_index`.

**Echo suppression (barge-in on open speakers):**
- Enabled by default (`audio.echo_suppression: true`). Startup calibration plays a short noise burst, measures what the mic hears back, and tunes an echo suppressor to ignore mic audio quiet enough to be explained by that echo. This is NOT full acoustic echo cancellation — AEC libraries (speexdsp, webrtc-audio-processing) don't build on Python 3.14 and solving full double-talk is unnecessary; we only need a binary "is this the speaker or a person?" verdict, since barge-in stops playback and cleans the mic. The gate vetoes VAD speech verdicts during playback, which fixes two bugs: the assistant can no longer interrupt itself, nor transcribe its own voice as a user utterance. Measured threshold (`audio.echo_margin`) is a fixed safety margin (default 1.75, tuned from real hardware) above the peak reference echo energy, removing the need to estimate speaker→mic delay.
- Set `audio.echo_gain` to a fixed value (e.g., 0.025 on this hardware) to skip the audible calibration burst at startup, trading a little re-adaptiveness for silence. Leave it unset (null) to auto-calibrate every run.

## System Prompt (Current — financial_voice_agent/orchestrator/system_prompt.py)

```
You are a read-only voice assistant for a personal trading desk running Zerodha Kite.
You observe market data, positions, and news — you never place, modify, or cancel orders,
and you have no tool capable of doing so. Speak in short, natural sentences meant to be
heard, not read. When you need to call a tool that will take a moment, say a brief natural
acknowledgment first — e.g. "let me check that" or "one sec, pulling the chart" — and
vary the phrasing so it doesn't sound scripted. If you cannot see the relevant instrument
on screen and the query depends on what's currently visible, call capture_screen rather
than guessing. If a query is ambiguous about which instrument or timeframe is meant, ask a
short clarifying question instead of assuming. Never speculate about figures you have not
retrieved from a tool in this turn. If a tool call returns an error, say so plainly and in
plain language — e.g. "I couldn't pull that up, looks like a permissions issue on the Kite
side" — rather than staying silent, giving up without explanation, or pretending it worked.
Briefly suggest what the user could check or do next if it's obvious from the error.
```

Note: the system prompt asking the model to speak an acknowledgment doesn't fully cover it —
`run_llm_turn` has its own `on_tool_call_started` hook (see Concurrency Model) that fires a
*separate*, code-driven ack independent of what the model says, and only for tools slower
than `ack_delay_seconds` (default 0.6s). Fast tools (`get_quote`, `capture_screen`, <100ms
live) never trigger it — acking every tool call made the agent sound repetitive in practice.

**Persona constraints:**
- Rotate acknowledgment/fallback phrases (tracked against the last 5 spoken, not just a
  fixed pool — applies to both the tool-call ack and the error-fallback messages)
- No unsolicited trading advice
- State unavailable data plainly, never invent numbers — **in practice, weaker open models
  will still sometimes fabricate specific figures (RSI values, moving averages, momentum %)
  without calling a tool, despite this instruction; this is a known model-compliance gap,
  not a deterministic bug, and is worse on some models/providers than others**

## Tool Inventory (All Read-Only)

| Tool | Source | Returns | Notes |
|------|--------|---------|-------|
| `get_quote(symbol)` | Kite REST `GET /quote` | LTP, day OHLC, volume | **`i` param must be `exchange:tradingsymbol` (e.g. `NSE:RELIANCE`), not a bare symbol — confirmed live, a bare symbol 403s.** The LLM only ever passes a bare symbol per the tool schema, so `get_quote` prepends `NSE:` itself when there's no colon. 401/403-TokenException → speak "session expired," exit. 429 → backoff. |
| `get_ohlc_history(symbol, interval, from, to)` | Kite REST `GET /instruments/historical/:instrument_token/:interval` | OHLC series | **Takes a numeric `instrument_token` in the URL, not a trading symbol — confirmed live, there's no per-symbol lookup endpoint.** Resolved via `tools/instruments.py` against Kite's full instrument dump (`GET /instruments/:exchange`, gzipped CSV, one row per tradable instrument); cached in-process (created once in `make_tool_executor`, shared across calls) since Kite's own docs say the dump should be fetched at most once a day. **Historical data requires an active paid Kite Connect subscription — a 403 `PermissionException` ("Insufficient permission for that call") means the account isn't subscribed, not a code bug**; this is separate from a TokenException and must not be reported as "log in again." |
| `compute_indicator(symbol, indicator, params)` | Local (pandas/numpy/ta) | Bollinger, Fibonacci, MA, RSI | Fails if insufficient candles — state this. Inherits `get_ohlc_history`'s instrument_token/subscription constraints above. |
| `get_positions_holdings()` | Kite REST | Current holdings/positions | Never pass to `get_news` |
| `get_news(query)` | Tavily API | Headlines + summaries | Degrade gracefully; never include account data in query |
| `capture_screen()` | Local (mss + pygetwindow/AppKit) | base64 JPEG + file path | "window not found" error if can't locate Kite. The base64 is what actually lets the LLM see the screen (see Data Contracts) — the file path alone (the original v1 shape) only lets the model confirm a screenshot was taken, not describe it. |

**Screen capture trigger:** Model calls `capture_screen` when it decides it needs visual context — no button, no polling, no staleness cap.

**Quote as REST, not WebSocket ticker:** Kite offers both. Agent answers discrete questions, not watching continuous ticks. Matches "no persistent connections" design choice in Section 2. Revisit if future version adds independent on-screen ticking display.

## Tech Stack & Dependencies

**Runtime:** Python 3.11+

**APIs & SDKs:** groq (STT+LLM) and/or huggingface_hub (config-driven alternative for both), cartesia (TTS) or deepgram-sdk (**not implemented**), pyaudio, silero-vad, noisereduce, janus (thread-safe async queue — **critical for Section 11.2 bug**), mss, pygetwindow (Windows) / pyobjc-framework-Quartz (macOS), pandas, numpy, ta, httpx (reused per Section 14), sqlite3 (stdlib), pyyaml or python-dotenv, kiteconnect (only used by `scripts/kite_login.py`'s one-off daily token exchange, not by the live agent itself, which speaks raw Kite REST via httpx)

**HTTP client:** httpx.AsyncClient (one per vendor, created at startup, reused)

## Configuration (config.yaml + env vars)

**Environment variables (see `.env.example`):**
- `GROQ_API_KEY` — required only if `stt.provider` or `llm.provider` is `"groq"`
- `HF_TOKEN` — required only if `stt.provider` or `llm.provider` is `"huggingface"` (matches `huggingface_hub`'s own conventional env var name)
- `CARTESIA_API_KEY` (or `DEEPGRAM_API_KEY` for fallback — **no Deepgram adapter built yet**, so this isn't currently usable)
- `KITE_API_KEY` / `KITE_API_SECRET` / `KITE_ACCESS_TOKEN` (confirm read-only scope at Kite Connect app registration). `KITE_ACCESS_TOKEN` expires daily — regenerate it with `python scripts/kite_login.py`, which walks through the login flow and writes the fresh token into `.env` for you (requires `KITE_API_KEY`/`KITE_API_SECRET` already set)
- `TAVILY_API_KEY`

**config.yaml required keys (current shape):**
```yaml
vad:
  speech_threshold: 0.5
  silence_duration_ms: 600
  min_speech_duration_ms: 200
  barge_in_enabled: true  # Interrupt playback when user speaks. Safe with open speakers
                           # because audio.echo_suppression stops the assistant's own voice
                           # from counting as a user interruption.
  barge_in_min_speech_ms: 200  # Sustained-speech requirement to guard against brief echo
                                # spikes. Raised from 96 to 200 after false mid-word
                                # barge-in; still short enough for real interruptions.

audio:
  output_device_index: null  # null = follow Windows' current default (WASAPI, most reliable).
                              # Set an explicit integer index only to override.
  input_device_index: null   # Allows mic and speaker to independently follow OS defaults.

  echo_suppression: true     # Lets barge-in work on open speakers, not just headphones.
                              # Startup calibration measures speaker->mic echo and ignores
                              # mic audio quiet enough to be that echo. Disables the
                              # assistant from interrupting itself.
  echo_margin: 1.75          # How much louder than predicted echo the mic must be to count
                              # as real speech. Chosen from measured levels on deployed
                              # hardware (speech ~0.0141 RMS, echo ~0.0068 RMS).
  echo_gain: 0.025           # Fixed echo gain -- skips audible calibration noise burst at
                              # startup. Set from live calibration on this machine; remove
                              # (set to null) to auto-calibrate on every run instead.

input_mode: "always_on"  # or "ptt"
tts:
  provider: "cartesia"  # or "deepgram" -- deepgram has no adapter yet
  voice_id: "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"

stt:
  provider: "groq"  # or "huggingface"
  model: "whisper-large-v3-turbo"  # Groq model id, or a Hugging Face model id (e.g.
                                    # "openai/whisper-large-v3-turbo") when provider is "huggingface"

llm:
  provider: "groq"  # or "huggingface"
  model: "qwen/qwen3.6-27b"  # Must support both tool calling and vision (capture_screen
                              # needs the latter). Check console.groq.com/docs/models and
                              # console.groq.com/docs/vision before changing.

storage:
  db_path: "./agent_turns.db"

mode: "live"  # or "mock"
```

**Mock mode:** when `mode: "mock"`, `get_quote`, `get_ohlc_history`, `get_positions_holdings` load from `fixtures/` directory (recorded real responses). Enables dev/test at any hour without live market state or valid Kite session.

**Running it:** `python -m financial_voice_agent` is the live entry point — the only thing in this codebase that actually wires mic capture → VAD → STT → LLM tool loop → TTS → speaker into a runnable loop end to end. (The eval harness, `python -m financial_voice_agent.eval`, exercises the LLM+tools loop with scripted text inputs instead of a mic, and doesn't play audio.)

## Local Evaluation Set (Section 17)

JSON test cases: input transcript, optional mocked screen result, expected tool calls, tools that must NOT be called. Runner asserts on tool names/args, not exact wording.

**Starting cases:**
- "What's the Nifty level?" → `get_quote(NIFTY 50)` (not `capture_screen`, no writes)
- "What is this instrument and RSI?" → `capture_screen`, then `compute_indicator` (no news)
- "Latest news on Nifty" → `get_news("Nifty")` (never `get_positions_holdings`)
- "My holdings?" → `get_positions_holdings()` (no news)
- "Buy 10 shares" → none (decline, explain read-only)
- "Bollinger Bands, 15-min Reliance" → `get_ohlc_history`, `compute_indicator`
- (Barge-in test) User speaks mid-response → interrupt playback, queue new input
- "What's on screen?" → `capture_screen` (no Kite/Tavily unless screen implies it)

**Add a case every time real usage (turn log) surfaces a wrong tool call or hallucinated figure.**

## Success Metrics

- Run eval set after every prompt/tool/orchestration change
- Spot-check agent responses vs. actual Kite + news (accuracy)
- Confirm agent declines/clarifies when it can't see relevant data (no guessing)
- Confirm no tool call reaches write endpoint (code review Section 6)
- Acknowledgment phrases don't repeat (Section 4.2)
- Indicator accuracy vs. Kite charts (before trusting for real)
- VAD: <1 false trigger per 10 min idle background noise in your environment
- Barge-in: playback interrupts within 300ms of speech start
- Latency: query turn log after 20+ real turns, compare `latency_total_ms` vs. Section 5 targets

## Out of Scope / Non-Goals

- Placing, modifying, cancelling orders (no risk review, no build)
- Mouse/keyboard automation of Kite UI
- Multi-user auth or shared access
- Continuous screen streaming (replaced by on-demand model-triggered capture)
- Cross-session memory (reset on restart by design)
- Token-level streaming to TTS (deferred until simpler path measured insufficient)
- Production SLAs (all targets are prototype working assumptions)

## Cost Estimate (Personal Use)

| Service | Rate | Notes |
|---------|------|-------|
| Groq Whisper v3 Turbo | ~$0.04/hour audio | Billed on duration, not tokens |
| Groq multimodal LLM | <$1/M input tokens | Confirm current rate; check model before pinning |
| Cartesia Sonic | Credit-based free tier | Confirm allowances; pricing not flat per-char |
| Zerodha Kite Connect | ₹500/month | Required for live read-only access |
| Tavily | Free to 1k credits/month | Personal use should stay in free tier |
| **Estimated monthly** | **~₹500 + few dollars incidental** | Assumes Groq/Cartesia stay in low-usage tiers |

*v1.4 Gemini Live (token-metered audio at connection-time scale) would have been more expensive for always-listening prototype. Verify all rates before treating as current.*

## Vendor Risk

| Vendor | Risk | Mitigation |
|--------|------|-----------|
| Groq (STT+LLM) | Model catalog churns; may deprecate multimodal model | Treat model string as config (Section 18.2); check console.groq.com/docs/models at build time |
| Groq — rate limits | Free/dev tier can throttle on sustained use | Confirm current limits before treating as personal prototype; Section 15.1 retry policy is immediate mitigation |
| Cartesia | Credit-based pricing (harder to predict than token-metered) | Confirm plan allowances before assuming cost estimate holds |
| Zerodha Kite Connect | Retail algo/API faces regulatory attention in India; terms can shift | Confirm API key scoped read-only at app-registration level, not by convention |
| Tavily | Free tier → pay-as-you-go; no stated fallback for outages | Low risk (read-only, non-critical); degrade gracefully per Section 15.1 |

## Privacy & Compliance (Single-User Assumption)

- **Leaves device:** audio (to Groq), screenshots (to Groq when `capture_screen` called), TTS text (to Cartesia), search queries (to Tavily, never account-specific)
- **Confirm vendor data-retention & training policies** before relying beyond personal prototype
- **Local logging:** Section 13 turn log (SQLite) + screenshots (disk files) behind OS disk encryption
- **No third-party sharing** beyond: Groq, Cartesia/Deepgram, Kite, Tavily
- **Revisit explicitly if shared with anyone else** (no multi-user auth planned)

---

**Build checklist:**
1. Spin up isolated Python 3.11+ env, install deps from Section 9
2. Create `config.yaml`, populate env vars (Section 18)
3. Scaffold concurrency model: PyAudio callback → janus queue → asyncio pipeline (Section 11.2 gotcha)
4. Implement Section 10 data contracts (byte-level specs) — most bugs are here
5. Build Section 13 SQLite schema, start logging from turn one
6. Seed Section 17 eval set, run before every orchestration change
7. Validate Section 5 latency targets after 20+ real turns (don't trust estimates)
8. Mock mode (Section 18.4) for dev outside market hours

**Known hard parts:**
- Thread-safe audio callback → asyncio handoff (Section 11.2 — janus or `call_soon_threadsafe`)
- Barge-in signal path and playback interruption without losing tool results (Section 11.3)
- Groq model name is not a constant (Section 2.3, 20 — check at build time)
- Kite session expires at 6 AM IST, token expires daily (mock mode buys dev bandwidth)
- Cartesia output_format must match PyAudio playback stream exactly (Section 10.2 — chipmunk voice is a subtle mismatch)
- **Kite's `/quote` needs `exchange:tradingsymbol`, and `/instruments/historical` needs a numeric `instrument_token` looked up from a separate instrument dump — neither is obvious from the tool schema's bare-symbol shape, and both silently 403 otherwise**
- **Kite's real auth failures (expired/invalid token) return HTTP 403 with `error_type: "TokenException"`, not 401 as the naive reading of Kite's docs suggests** — code that only checks for 401 will misreport every real session expiry as a raw, unhandled error
- **A 403 isn't always a token problem** — `error_type: "PermissionException"` means an account-level restriction (e.g. no active paid subscription for historical data), and must be surfaced as such, not as "log in again"
- **Any tool exception not specifically anticipated must still become a spoken, explainable error, not a crash** — the turn orchestrator's top-level catch-all produces a generic, misleading fallback ("didn't catch that") for failures that have nothing to do with STT; the tool executor needs its own catch-all so the LLM gets a chance to explain the real failure in the same turn
- **Free-tier LLM/STT credits (Hugging Face Inference in particular) run out mid-project** — this is why both providers ended up config-switchable rather than Groq-only as originally scoped; budget for occasionally needing to flip `stt.provider`/`llm.provider` and re-verify against the real API before trusting a "it worked yesterday" assumption
- **A vision-capable, tool-calling-capable model is a much smaller set than either capability alone** — `qwen/qwen3.6-27b` on Groq was the one verified to support both together; picking a model for one capability without checking the other silently breaks `capture_screen`'s value (the model calls it, gets an image, but the image goes nowhere if the model can't actually see it)
- **Some models default to an expensive "thinking"/extended-reasoning mode** (Qwen 3.6, gpt-oss on Groq) that can add tens of seconds of latency per call with no visible symptom besides "it's slow" — check `reasoning_effort` support for whatever model you pick
- **PyAudio device indices are not stable across reconnects, and "OS default" resolution differs by host API.** Confirmed live: PyAudio's own `get_default_output_device_info()` resolves through whichever host API it initializes first (often MME), which silently disagrees with Windows Sound Settings (which controls WASAPI). Built `audio/devices.py` to use WASAPI only to identify the currently-selected physical device (by name, matched via prefix since MME truncates names), then open that device's MME entry instead — getting both an accurate, live-tracking default and a stream capable of resampling to 16kHz (WASAPI-default devices only accept their native rate, e.g. 48kHz, and reject 16kHz outright, breaking the entire pipeline).
- **AEC libraries (speexdsp, webrtc-audio-processing) don't build on Python 3.14** — unmaintained bindings, no current wheels. This project framed echo suppression not as full acoustic echo cancellation (solving hard double-talk reconstruction) but as a much narrower binary classifier: "is this the speaker or a person?" Since barge-in stops playback and cleans the mic on match, we never need to separate speech during double-talk.
- **Comparing waveforms (cross-correlation) doesn't work in a reverberant room with speaker nonlinearity and resampling.** Chose energy-envelope comparison (RMS per 20ms chunk) instead — immune to reverb, speaker distortion, and the resampling MME does. Peak reference energy over a delay window (not instantaneous) removes any need to estimate speaker→mic propagation delay.
- **Startup calibration measures speaker→mic echo gain on whatever the OS chose for mic/speaker/volume.** A 0.5s tone measured only the onset burst (inflated by the OS echo canceller's convergence spike), overestimating gain 3x. Fixed by taking a percentile of per-chunk levels and doubling tone to 1s — stability matters more than representativeness here. Measured on this hardware (headset mic + laptop speakers): speech ~0.0141 RMS, echo peak ~0.0068 RMS, tuned echo_margin to 1.75, landing threshold near ~0.0094.
- **Scalar-gain echo prediction is inherently broadband, but real TTS speech has different spectral content.** Calibration against noise burst averages the acoustic path, but moments of real speech can momentarily produce louder echo than the average predicts, causing false barge-ins (cut mid-word). Mitigated with `vad.barge_in_min_speech_ms` (raised from 96 to 200) — real interruption speech sustains a full word (150ms+), but brief spectral-mismatch spikes are too short. Added `EchoGate.diagnose()` and `on_echo_diagnostic` hook for live debugging of ambiguous cases; logged to console so the next false positive has real mic/predicted/threshold numbers instead of needing reproduction.
- **Echo gate must veto VAD inside AudioPipeline, not at the playback check.** Placing it in pipeline fixes two bugs: assistant can't interrupt itself (would see its own playback spike as speech), and can't transcribe its own voice as a user utterance (the source of bogus "Thank you." turns appearing in logs when the user never said them).
- **Real-time audio-level barge-in (VAD + echo gate) can only judge loudness/duration — transcription needs the utterance already captured.** Added keyword-triggered barge-in as a second independent signal: `play_with_barge_in` watches `output_queue` while playback runs, transcribes any newly-arrived utterance in parallel (non-blocking), and forces interrupt on a keyword match against `INTERRUPT_PHRASES` ("wait", "stop", "hold on", "actually", "never mind", etc., prefix-matched to avoid false positives like "what's the stop-loss level"). Critical bug fix caught by pre-existing test: must NOT speculatively transcribe items already queued at playback start (those are normal next-turn-in-line, not live interruption) — fixed by snapshotting pre-existing queue contents at entry and only reacting to new arrivals.
- **Cartesia can run out of API credits (account-level, not code bugs).** Confirmed live: a reported "something went wrong on my end" TTS error was a transient Cartesia API failure from credit exhaustion, unrelated to device switching. Similar to Groq/Hugging Face credit limits — worth budgeting vendor quota separately from code reliability.

