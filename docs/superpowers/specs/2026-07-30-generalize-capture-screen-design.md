# Generalize capture_screen Beyond Kite Windows

**Status:** Approved design, not yet implemented.
**Date:** 2026-07-30

## Goal

`capture_screen` currently only works when a window titled "Kite" is open — it can't see anything else on screen, even when a question depends on it (a browser tab, a PDF, an error dialog). This generalizes it to capture whatever window currently has focus, so the agent can look at anything the user is actually looking at, not just Kite.

## Scope

**In scope:**
- Replace the Kite-specific window lookup with an active-window lookup.
- Update the `capture_screen` tool's schema description and the system prompt's guidance to reflect the broader capability.

**Explicitly out of scope:**
- Changing the assistant's core persona/identity — it remains a read-only trading-desk assistant. Its other tools (quotes, positions, charts, fundamentals) stay Kite/market-focused; only screen capture stops being artificially restricted to Kite windows.
- Any general-purpose "answer questions about anything" behavior beyond what capture_screen already enables via the existing vision-capable LLM pipeline.
- Multi-monitor handling, window-picking by name, or any new prompting UX — out of scope for this narrow fix (see Context below for why "whatever's focused" was chosen over these).

## Context

`financial_voice_agent/tools/screen.py`'s `capture_screen()` function is already fully generic — it takes an injected `window_finder` callable and has no Kite-specific logic itself. The only Kite-specific piece is `find_kite_window()`, which filters `pygetwindow.getAllWindows()` for `"kite" in title.lower()`.

Verified on this machine: `pygetwindow.getActiveWindow()` returns the currently focused window (or `None` if nothing is focused, e.g. the desktop) with the same `left`/`top`/`width`/`height` shape `find_kite_window()` already returns — so this is a drop-in replacement requiring no changes to `capture_screen()` itself.

One known, pre-existing-style imprecision (not introduced by this change): maximized windows can report coordinates a few pixels outside the visible screen area, due to Windows' invisible DWM border padding (observed directly: a maximized window reported `left=-7, top=-7`). This affects `find_kite_window()` today exactly the same way for a maximized Kite window, so it's not a new problem — not addressed here.

## Design

`financial_voice_agent/tools/screen.py`:
- `find_kite_window()` is replaced by `find_active_window()`, calling `pygetwindow.getActiveWindow()` and returning the same `{"left": ..., "top": ..., "width": ..., "height": ...}` shape, or `None` if there's no active window — preserving `capture_screen`'s existing `WindowNotFoundError` behavior unchanged.
- `WindowNotFoundError`'s message updates from `"Kite window not found"` to a generic message like `"No active window found"`.

`financial_voice_agent/tools/registry.py`:
- The `capture_screen` dispatch branch's `window_finder=find_kite_window` becomes `window_finder=find_active_window`.
- `TOOLS_SCHEMA`'s `capture_screen` entry description changes from "Capture the current Kite window as an image..." to reflect capturing whatever's currently active (e.g. "Capture the currently active window as an image, for questions about what's visible on screen").

`financial_voice_agent/orchestrator/system_prompt.py`:
- The existing guidance ("If you cannot see the relevant instrument on screen and the query depends on what's currently visible, call `capture_screen` rather than guessing") is broadened to not be instrument-specific, so it naturally covers any on-screen question, not just Kite instrument lookups.

## Error Handling

No new error paths. `capture_screen`'s existing `WindowNotFoundError` (raised when `window_finder()` returns `None`) already covers "nothing to capture" — this now fires when no window has focus (e.g. desktop clicked) instead of "no Kite window open," with an updated message.

## Testing

`tests/tools/test_screen.py`'s existing tests need no changes — they already exercise `capture_screen()` via an injected fake `window_finder`, independent of which real adapter is wired in. `find_active_window()` itself, like `find_kite_window()` before it, is a "real adapter" not covered by DI unit tests (matches this project's existing convention — `capture_region()`'s `mss` call is the same). Verification is one manual run: ask the agent a question about something visible in a non-Kite window (e.g. a browser tab) and confirm it correctly describes it.
