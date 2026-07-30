# Generalize capture_screen Beyond Kite Windows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `capture_screen` capture whatever window currently has focus, instead of only windows titled "Kite", so the agent can look at anything the user is actually looking at.

**Architecture:** Replace `financial_voice_agent/tools/screen.py`'s Kite-title-filtered `find_kite_window()` with a generic `find_active_window()` using `pygetwindow.getActiveWindow()` (verified to return the same `left`/`top`/`width`/`height` shape). `capture_screen()` itself is already generic via its injected `window_finder` parameter, so only the adapter, its one call site in `registry.py`, and two description/prompt strings need to change.

**Tech Stack:** `pygetwindow` (already a dependency, already imported in `screen.py`).

## Global Constraints

- Windows only.
- No new dependencies.
- The assistant's core persona/identity does not change — it remains a read-only trading-desk assistant; only screen-capture's window selection generalizes.
- `find_active_window()`, like the `find_kite_window()` it replaces, is a "real adapter" not covered by DI unit tests (matches this project's existing convention for `capture_region()`'s `mss` call) — verified by one manual run instead.

---

### Task 1: Replace find_kite_window with find_active_window

**Files:**
- Modify: `financial_voice_agent/tools/screen.py`
- Modify: `financial_voice_agent/tools/registry.py`
- Modify: `financial_voice_agent/orchestrator/system_prompt.py`
- Test: `tests/tools/test_screen.py` (no changes needed — see Step 4; included here only to confirm it still passes)

**Interfaces:**
- Consumes: nothing new.
- Produces: `find_active_window() -> dict | None` in `financial_voice_agent/tools/screen.py`, replacing `find_kite_window`. Same return shape as before: `{"left": int, "top": int, "width": int, "height": int}` or `None`.

- [ ] **Step 1: Replace `find_kite_window` with `find_active_window` in `screen.py`**

In `financial_voice_agent/tools/screen.py`, replace the `find_kite_window` function (currently lines 43-54) with:

```python
def find_active_window() -> dict | None:
    """Real adapter: locates the currently focused window using pygetwindow.
    Returns None if nothing is focused (e.g. the desktop itself is active).
    Note: pygetwindow can report coordinates a few pixels outside the visible
    screen for a maximized window, due to Windows' invisible DWM border
    padding -- a pre-existing imprecision, not something this function
    corrects for."""
    import pygetwindow as gw

    window = gw.getActiveWindow()
    if window is None:
        return None
    return {"left": window.left, "top": window.top, "width": window.width, "height": window.height}
```

Also update `capture_screen`'s hardcoded error message on line 21 — change:
```python
        raise WindowNotFoundError("Kite window not found")
```
to:
```python
        raise WindowNotFoundError("No active window found")
```

- [ ] **Step 2: Update `registry.py`'s import and dispatch to use the new function**

In `financial_voice_agent/tools/registry.py`:

1. Change the import (around line 18-23) from:
```python
from financial_voice_agent.tools.screen import (
    WindowNotFoundError,
    capture_region,
    capture_screen,
    find_kite_window,
)
```
to:
```python
from financial_voice_agent.tools.screen import (
    WindowNotFoundError,
    capture_region,
    capture_screen,
    find_active_window,
)
```

2. Change the `TOOLS_SCHEMA` entry's description (currently line 115) from:
```python
            "description": "Capture the current Kite window as an image, for questions about what's visible on screen.",
```
to:
```python
            "description": "Capture the currently active window as an image, for questions about what's visible on screen.",
```

3. Change the dispatch call (currently line 199) from:
```python
        return await capture_screen(window_finder=find_kite_window, screenshot_fn=capture_region)
```
to:
```python
        return await capture_screen(window_finder=find_active_window, screenshot_fn=capture_region)
```

4. `except WindowNotFoundError:` (currently line 242) and its returned message `{"error": "Could not find the Kite window on screen"}` — update that error message to match the broader capability:
```python
        except WindowNotFoundError:
            return {"error": "Could not find an active window to capture"}
```

- [ ] **Step 3: Broaden the system prompt's capture_screen guidance**

In `financial_voice_agent/orchestrator/system_prompt.py`, change this sentence within `SYSTEM_PROMPT`:
```python
    "If you cannot see the relevant instrument "
    "on screen and the query depends on what's currently visible, call capture_screen rather "
    "than guessing. "
```
to:
```python
    "If a query depends on what's currently visible on screen -- an instrument, a chart, "
    "an error message, anything -- call capture_screen rather than guessing. "
```

(This is a same-sentence rewording, not a structural change — everything else in `SYSTEM_PROMPT` before and after this sentence stays exactly as it already is.)

- [ ] **Step 4: Run the existing test suite to confirm nothing broke**

Run: `python -m pytest tests/tools/test_screen.py tests/tools/test_registry.py -v`
Expected: PASS. `test_screen.py`'s tests inject a fake `window_finder` and don't reference `find_kite_window`/`find_active_window` by name, so they should need no changes. `test_registry.py` has a `test_tools_schema_names_include_all_expected_tools` test asserting tool *names* (not descriptions), so it should also still pass unmodified.

Then run the full suite:
Run: `python -m pytest -q`
Expected: PASS, no failures (expect the same count as before this change — no tests added or removed).

- [ ] **Step 5: Commit**

```bash
git add financial_voice_agent/tools/screen.py financial_voice_agent/tools/registry.py financial_voice_agent/orchestrator/system_prompt.py
git commit -m "Generalize capture_screen to the active window instead of only Kite"
```

---

### Task 2: Manual verification

**Files:**
- None (verification only).

**Interfaces:**
- Consumes: Task 1's changes, running end-to-end.

- [ ] **Step 1: Run the full test suite one more time**

Run: `python -m pytest -q`
Expected: PASS, all tests.

- [ ] **Step 2: Manually verify capture_screen works on a non-Kite window**

Run: `python -m financial_voice_agent`

With some non-Kite window focused (a browser tab, a text file, anything with visible text or a distinctive image), ask the agent a question that depends on what's on screen — e.g. "what does this page say" or "what's the title of this window." Confirm the agent calls `capture_screen` and correctly describes what's actually on screen, not a stale Kite-window capture or an error.

- [ ] **Step 3: Manually verify it still works when Kite is focused**

Switch focus to the Kite window (or the mock/live Kite session you use), ask a question that depends on what's visible there, and confirm `capture_screen` still works correctly — since Kite, when focused, is now just "the active window" like anything else.

- [ ] **Step 4: Manually verify the no-window-focused case**

Click on the desktop (so no application window has focus), ask a question that would trigger `capture_screen`, and confirm the agent reports a clear "couldn't find anything to capture" style error rather than crashing or hanging.

- [ ] **Step 5: Commit any manual-verification fixes**

If Steps 2-4 required code changes, commit them now with a message describing what was fixed. If no changes were needed, there is nothing to commit for this task.
