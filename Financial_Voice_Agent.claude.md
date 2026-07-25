# Financial Voice Agent — PRD v2.1 (Developer Reference)

## Product Essence
Read-only AI voice assistant for Zerodha Kite trading dashboard. Listens continuously (VAD-gated), executes tool calls, speaks answers. No order placement, no UI automation, no multi-user auth. Single user, personal machine.

## Architecture: Cascade, Not S2S

**v1.4 used:** Gemini Multimodal Live (persistent WebSocket, continuous vision). **Rationale:** native multimodal + native tool calling.

**v2.1 uses:** STT → LLM (tool calling) → TTS (cascaded HTTP). **Why:** vision is now on-demand (model-triggered), not continuous. Cascade buys observability, cost, simplicity. Trade-off: ~1–2.5s per turn instead of <500ms.

**Model stack:**
- Speech-to-text: Groq Whisper Large v3 Turbo (~$0.04/hour audio, ~228x RT)
- Reasoning + tools + vision: Groq multimodal tool-calling model (check console.groq.com/docs/models — model catalog churns)
- TTS: Cartesia Sonic (primary, ~40–90ms time-to-first-audio) or Deepgram Aura-2 (fallback, ~300ms)
- Tools: local Python functions, not MCP server (no need for v1)

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
- Groq multimodal model: confirm inline base64 vs data-URI at build time

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
| Groq STT/LLM | Timeout or 5xx | Retry 2x with 1s backoff. On final fail: speak "I didn't catch that, one moment", re-listen |
| Groq LLM | 429 rate limit | Exponential backoff; speak "I've hit a usage limit — give me a second" |
| Cartesia TTS | Timeout or 5xx | Retry once. Fall back to Deepgram (if configured) or logged error |
| Kite Connect | 401/429/503 | Per-tool policy in tool table (Section 6) — spoken response differs by failure type |
| Tavily | Timeout/error | Degrade gracefully: answer with other results, state search unavailable |

## Voice & Audio Design

**Input trigger:** Always-on VAD (default). Expose `--ptt` flag for push-to-talk.

**VAD tuning (config.yaml):**
- Speech threshold: 0.5 (lower = catch soft speech, higher = reject noise)
- Silence before end-of-speech: 600ms (raise if cutting off mid-sentence)
- Min speech duration: 200ms (raise in noise)
- Sample rate: **must be 16000 Hz** (silero-vad trained at 16kHz)

**Noise suppression:** PyAudio → noisereduce (stationary mode, <30ms overhead) → VAD gate → buffer

**Output device:** enumerate with `pyaudio.get_device_info_by_index()`, accept config key `audio.output_device_index`, fall back to OS default with warning.

## System Prompt (Draft — Iterate After Real Usage)

```
You are a read-only voice assistant for a personal trading desk running Zerodha Kite. 
Observe market data, positions, news — never place, modify, or cancel orders. 
Speak in short, natural sentences meant to be heard. When calling a tool that takes time, 
say brief acknowledgment first (e.g., "let me check that," "one sec, pulling the chart") 
and vary phrasing. If query depends on screen content and you can't see it, call 
capture_screen rather than guessing. If ambiguous about instrument/timeframe, ask. 
Never speculate — retrieve from tools.
```

**Persona constraints:**
- Rotate acknowledgment phrases (track last 5 used, avoid repeats)
- No unsolicited trading advice
- State unavailable data plainly, never invent numbers

## Tool Inventory (All Read-Only)

| Tool | Source | Returns | Notes |
|------|--------|---------|-------|
| `get_quote(symbol)` | Kite REST | LTP, day OHLC, volume | 401 → speak "session expired," exit. 429 → backoff. Stale → "data unavailable" |
| `get_ohlc_history(symbol, interval, from, to)` | Kite REST | OHLC series | Expired options unsupported — state explicitly |
| `compute_indicator(symbol, indicator, params)` | Local (pandas/numpy/ta) | Bollinger, Fibonacci, MA, RSI | Fails if insufficient candles — state this |
| `get_positions_holdings()` | Kite REST | Current holdings/positions | Never pass to `get_news` |
| `get_news(query)` | Tavily API | Headlines + summaries | Degrade gracefully; never include account data in query |
| `capture_screen()` | Local (mss + pygetwindow/AppKit) | base64 JPEG | "window not found" error if can't locate Kite |

**Screen capture trigger:** Model calls `capture_screen` when it decides it needs visual context — no button, no polling, no staleness cap.

**Quote as REST, not WebSocket ticker:** Kite offers both. Agent answers discrete questions, not watching continuous ticks. Matches "no persistent connections" design choice in Section 2. Revisit if future version adds independent on-screen ticking display.

## Tech Stack & Dependencies

**Runtime:** Python 3.11+

**APIs & SDKs:** groq (STT+LLM), cartesia (TTS) or deepgram-sdk, pyaudio, silero-vad, noisereduce, janus (thread-safe async queue — **critical for Section 11.2 bug**), mss, pygetwindow (Windows) / pyobjc-framework-Quartz (macOS), pandas, numpy, ta, httpx (reused per Section 14), sqlite3 (stdlib), pyyaml or python-dotenv

**HTTP client:** httpx.AsyncClient (one per vendor, created at startup, reused)

## Configuration (config.yaml + env vars)

**Environment variables:**
- `GROQ_API_KEY`
- `CARTESIA_API_KEY` (or `DEEPGRAM_API_KEY` for fallback)
- `KITE_API_KEY` / `KITE_ACCESS_TOKEN` (confirm read-only scope at Kite Connect app registration)
- `TAVILY_API_KEY`

**config.yaml required keys:**
```yaml
vad:
  speech_threshold: 0.5
  silence_duration_ms: 600
  min_speech_duration_ms: 200

audio:
  output_device_index: null  # null = OS default

input_mode: "always_on"  # or "ppt"
tts:
  provider: "cartesia"  # or "deepgram"

llm:
  model: "<check console.groq.com/docs/models>"  # treat as config, not constant

storage:
  db_path: "./agent_turns.db"

mode: "live"  # or "mock"
```

**Mock mode:** when `mode: "mock"`, `get_quote`, `get_ohlc_history`, `get_positions_holdings` load from `fixtures/` directory (recorded real responses). Enables dev/test at any hour without live market state or valid Kite session.

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

