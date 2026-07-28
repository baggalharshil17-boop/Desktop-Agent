# v4.0: Screen Overlay, Technical Charts, and Investment-Analyst Tools

**Status:** Approved design, not yet implemented.
**Date:** 2026-07-29

## Goal

Two related additions, brainstormed together because they share the same on-screen overlay infrastructure:

1. **Processing feedback** — a subtle visual glow while the agent is silently working (STT + LLM + TTS generation), since today there's no feedback during that gap.
2. **Technical charts + investment-analyst behavior** — the agent can render a candlestick chart with technical indicators as an on-screen overlay panel, and can pull stock fundamentals (P/E, market cap, etc.) from a new external API to answer analysis-style questions like an analyst would — synthesizing data, not issuing recommendations.

## Scope

**In scope:**
- A new always-on-top, click-through screen overlay process (`financial_voice_agent/overlay/screen_overlay.py`) that shows a thin glow around the primary monitor's edges while the agent is processing.
- The same overlay process also displays a slide-in/out chart panel (not click-through — needs a close button) when the agent renders a chart.
- Extending `financial_voice_agent/tools/indicators.py` to expose full indicator series (not just the latest value) for charting, reusing its existing math.
- A new `show_chart` tool that renders a candlestick chart (via `mplfinance`) with optional moving average / Bollinger bands / RSI / Fibonacci overlays, using OHLC data from the existing `get_ohlc_history`.
- A new `get_stock_fundamentals` tool backed by the Indian Stock API (`indianapi.in`) for P/E, market cap, 52-week range, etc. — including fuzzy company-name-to-ticker resolution via that API's `/stock?name=` endpoint.
- System prompt updates so the agent synthesizes fundamentals + technicals + news into analyst-style explanations, without issuing buy/sell/hold recommendations or price targets.
- New config: `processing_overlay_enabled` (bool, default true) and `INDIAN_STOCK_API_KEY` (new secret).

**Explicitly out of scope (not deferred, just not part of this effort):**
- Multi-monitor overlay support — glow and chart panel target the primary monitor only.
- An interactive/scrubbable chart (slider control on the chart itself) — the "slider" is purely the panel's slide-in/out entrance animation.
- Distinguishing "processing" from "speaking" states in the glow — one state (processing) is either on or off.
- Generalizing `capture_screen` beyond Kite windows (a separate, already-identified follow-up, not part of this design).
- Any enforcement mechanism beyond the system prompt for the no-recommendations guardrail — this is prompt-level guidance, not a technical filter.
- Real-time/streaming chart updates — each `show_chart` call renders one static snapshot.

## Context

The voice pipeline (`orchestrator/main_loop.py`) currently gives no visual signal during `run_turn()` (STT → LLM tool loop → TTS generation) — only console `print()`s like "Listening...". This silent gap is the actual problem the glow addresses.

Separately, `financial_voice_agent/tools/indicators.py` already computes moving average, Bollinger bands, RSI, and Fibonacci retracement — but only returns each indicator's *latest* value (used today for spoken answers like "what's the RSI"). Charting needs the full series across the candle range, which the same rolling-window math already computes internally before slicing to `.iloc[-1]`.

`financial_voice_agent/tools/instruments.py`'s `get_instrument_token` only does exact `tradingsymbol` matching against Kite's daily instrument dump (confirmed: Kite has no fuzzy-name lookup). The Indian Stock API's `/stock?name=` endpoint does fuzzy company-name resolution *and* returns fundamentals in the same call, so it's used for both needs rather than adding a third mechanism.

`financial_voice_agent/tools/news.py`'s `get_news` is already a free-text query tool (via Tavily) — no changes needed there; the LLM can already ask for company-specific news by crafting the query.

## Design

### Screen overlay process

`financial_voice_agent/overlay/screen_overlay.py` is a standalone script, launched by `__main__.py` via `subprocess.Popen` at startup (gated by `config.processing_overlay_enabled`, default true) and terminated in the same `finally:` block that already closes `playback` on shutdown.

It's a `tkinter`-based, always-on-top, borderless, transparent window sized to the primary monitor (`winfo_screenwidth`/`winfo_screenheight`). The glow layer is click-through (Windows layered-window styles via `ctypes`, `WS_EX_LAYERED | WS_EX_TRANSPARENT`); the chart panel layer is not click-through, so its close button is clickable.

It listens on a fixed localhost UDP port for single-packet text commands from the agent process:
- `processing_on` / `processing_off` — toggles the edge glow (soft blue, gentle pulse; easy to retune later).
- `show_chart:<absolute path to PNG>` — slides the chart panel in, loading the given image.
- (No `close_chart` message needed from the agent side — closing is a local UI action via the panel's own close button.)

UDP is used because these are fire-and-forget state signals — an occasional dropped packet self-corrects on the next transition, and no connection handshake is needed for same-machine loopback traffic.

### Agent-side wiring

In `run_voice_loop` (`orchestrator/main_loop.py`), a `processing_on` signal is sent immediately before `run_turn(...)` is called, and `processing_off` immediately after it returns (success or error path) — this precisely brackets the STT+LLM+TTS-generation window, before playback starts. All sends are wrapped in a best-effort try/except so a missing or crashed overlay process never blocks or crashes the voice loop.

### Chart rendering

`financial_voice_agent/tools/indicators.py` is extended with series-returning counterparts to its existing functions (reusing the same rolling-window computations, without the final `.iloc[-1]` slice).

New `financial_voice_agent/tools/charting.py`:
```
render_chart(symbol: str, indicators: list[str], *, history_fn, output_dir="charts") -> str
```
Fetches OHLC via the existing `get_ohlc_history`, computes the requested indicator series, renders a candlestick chart via `mplfinance` (new dependency) with moving average / Bollinger bands drawn on the price panel, RSI in a small sub-panel, and Fibonacci levels as horizontal reference lines, saves a timestamped PNG to `charts/` (mirroring `capture_screen`'s `screenshots/` convention), and returns the file path.

New `show_chart` tool in `registry.py`'s `TOOLS_SCHEMA` calls `render_chart`, then sends `show_chart:<path>` to the overlay process. The LLM decides when to call it — either because the user asked directly ("show me a chart") or because it judges a chart would help answer a question — same standard tool-calling behavior as `capture_screen` today, no new trigger mechanism.

### Stock fundamentals

New `financial_voice_agent/tools/fundamentals.py`:
```
get_stock_fundamentals(name: str, *, http_client, mode="live", fixtures_dir="fixtures") -> dict
```
Calls the Indian Stock API's `/stock?name=` endpoint (fuzzy company-name lookup, e.g. "Reliance" or "HDFC Bank"), returning fundamentals (P/E, market cap, 52-week high/low, and whatever else the response includes). Follows the same `mode="mock"/"live"` + injected-`http_client` + `fixtures_dir` pattern as `get_quote`/`get_news`. New secret `INDIAN_STOCK_API_KEY`, added to `.env`/`config.yaml`/`http_clients.py`'s per-vendor client setup alongside the existing Groq/Cartesia/Kite/Tavily clients.

The exact request format (auth header name, query params beyond `name`, and full response JSON shape) needs to be verified against a real API call during implementation — the vendor's public docs page didn't expose complete details, consistent with how every other vendor integration in this project (Groq, Cartesia, HuggingFace) was verified against the live API rather than trusted from docs alone.

### Investment-analyst behavior

`orchestrator/system_prompt.py` gets new guidance: for analysis-style questions, synthesize fundamentals (`get_stock_fundamentals`) + technicals (the indicator tools, optionally shown via `show_chart`) + news (`get_news`) into a clear explanation of what the numbers mean — e.g. "P/E is above the sector average, RSI suggests overbought conditions, recent news is mixed." The prompt explicitly instructs the model to avoid buy/sell/hold calls, price targets, or "you should" language. This is prompt-level guidance shaping model behavior, not a technical enforcement filter — SEBI has rules around who can register as an investment adviser, and while this is a personal tool rather than a public product, staying descriptive rather than prescriptive is the safer default.

## Error Handling

- `get_stock_fundamentals`: unknown company name or API error returns `{"error": ...}` (never raises), matching `get_news`'s degrade-gracefully convention — the LLM gets a clear signal to relay rather than a crash.
- `show_chart`: insufficient candle history reuses `indicators.py`'s existing `InsufficientDataError` pattern. If the overlay process isn't running or its port is unavailable, the socket send is best-effort (try/except, same as the `processing_on`/`processing_off` signals) — a chart still gets rendered and its path returned to the LLM even if on-screen display silently no-ops.
- The overlay process itself launching or crashing must never block or crash the voice agent — `subprocess.Popen` is fire-and-forget at startup, and its absence is only ever detected indirectly via failed sends, never checked synchronously.

## Testing

- `indicators.py`'s new series functions, `charting.py`'s data-prep logic (indicator series → chart-ready structure, kept separate from the actual `mplfinance` render call), and `fundamentals.py` all get unit tests with injected fakes (`history_fn`, `http_client`), matching this project's established DI testing pattern.
- The actual `mplfinance` rendering call and `screen_overlay.py`'s `tkinter` GUI process are not unit tested, matching the existing convention for interactive/GUI scripts in this project (`scripts/kite_login.py`, `scripts/setup.py`) — verified instead by one real manual run: launching the agent, confirming the glow appears during a real turn and disappears before playback, and confirming a real `show_chart` call renders and displays a chart correctly.

## Explicitly Deferred / Not Addressed

- Multi-monitor glow/chart placement.
- An interactive chart slider/scrubber.
- A three-state (idle/processing/speaking) overlay, versus the current two-state (processing on/off) design.
- Generalizing `capture_screen` beyond Kite windows.
- Any hard technical enforcement of the no-recommendations guardrail beyond the system prompt.
