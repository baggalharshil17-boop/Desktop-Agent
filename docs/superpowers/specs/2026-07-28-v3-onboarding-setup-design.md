# v3.0: Onboarding Setup for Sharing with a Friend

**Status:** Approved design, not yet implemented.
**Date:** 2026-07-28

## Goal

The user wants to share this project with a technical friend so they can run it on their own machine with their own API keys. This spec covers making that clone-to-running path smooth, not a distributable installer.

## Scope

**In scope:**
- `scripts/setup.py` — an interactive wizard that collects API keys, validates each one live, and generates `.env` / `config.yaml`.
- A new top-level `README.md` with a "Getting Started (Fresh Clone)" section.
- Small `requirements.txt` / doc fixes (PyAudio wheel fallback, CPU-only torch install) that remove friction on a fresh clone.

**Explicitly out of scope (not deferred, just not part of this effort):**
- PyInstaller / standalone executable packaging.
- Any GUI.
- macOS/Linux support — Windows only, matching both the current dev machine and the friend's machine.
- Vendoring/pinning a specific PyAudio wheel — documented as a fallback command instead.
- A Chrome extension port. Discussed separately as a much larger, later effort (would require rewriting the audio/VAD/echo-suppression pipeline in JavaScript — none of the current Python code ports directly). Not part of v3.0.

## Context

The project currently assumes a developer runs `python -m financial_voice_agent` from the repo root, with `config.yaml` and `.env` read via relative paths (`financial_voice_agent/config.py`'s `load_config()` defaults). This is fine for a git-clone distribution model (the friend is a developer, per clarification) and does not need to change.

Two concrete friction points exist today for a first-time clone:
- `torch` (used by Silero VAD) is ~485MB installed, and a plain `pip install torch` can pull a large CUDA build unnecessarily on a machine without a GPU.
- `pyaudio` frequently fails to build from source on Windows without a prebuilt wheel or Visual C++ Build Tools.

Manually setting up `.env` and `config.yaml` today requires reading through the PRD doc and inline comments to know which of ~10 env vars are actually required for a given provider combination, and Cartesia's `voice_id` currently requires visiting `play.cartesia.ai/voices` outside the app.

## Design

### `scripts/setup.py`

Run once after cloning. Flow:

1. If `.env` or `config.yaml` already exist, ask once whether to overwrite from scratch or fill in only what's missing — re-running setup.py later to add a previously-skipped key must not destroy existing values.
2. Copy `.env.example` → `.env` if it doesn't exist yet.
3. Prompt for provider choices: `stt.provider` (`groq`/`huggingface`), `llm.provider` (`groq`/`huggingface`, with an inline note that it must be vision-capable for `capture_screen` to work), `tts.provider` (`cartesia` only — no Deepgram adapter exists).
4. For each API key required by those choices, prompt with hidden input (`getpass`), write it to `.env`, then validate it live before moving on:
   - Groq → a cheap `models.list()` call.
   - Hugging Face → `huggingface_hub.whoami()`.
   - Cartesia → fetch the real voice list, which also lets the user **pick a `voice_id` interactively** from real data instead of visiting the website.
   - Tavily → best-effort validation if a cheap endpoint exists; otherwise skipped and documented as untested until first real use.
   - Kite is **not** collected here. It's a separate OAuth-style browser flow already handled by `scripts/kite_login.py`. setup.py just points to it at the end, and mentions `mode: "mock"` as a way to try the agent without a Zerodha account.
5. A failed live validation (bad key, network error) shows the real error and offers retry-this-key or skip-and-fix-later — it does not crash or restart the whole wizard.
6. Write a fresh `config.yaml` from a template with the collected choices substituted in (not an in-place edit of the existing file — plain YAML round-tripping via pyyaml loses comments, and the file is meant to stay human-readable/hand-editable afterward).
7. Print next steps: `python scripts/kite_login.py` (only if live mode was chosen) and `python -m financial_voice_agent`.

### README.md

New file (none exists today). "Getting Started (Fresh Clone)" section, in order:
1. Prerequisites, including the PyAudio wheel fallback command if the plain install fails.
2. `pip install -r requirements.txt`, with a note to install torch's CPU-only build first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) unless they have and want a GPU.
3. `python scripts/setup.py`.
4. `python -m financial_voice_agent`.
5. A short table of what each provider needs (key, where to get it, rough free-tier/cost), consolidating what's currently scattered through the PRD's Configuration section.

## Error Handling

Covered inline above per-step: overwrite-or-merge prompt for existing `.env`/`config.yaml`, and retry-or-skip on a failed live key validation. No other new error paths — the wizard is a thin layer over existing, already-tested config/client code.

## Testing

- `scripts/setup.py`'s core logic (writing `.env`, generating `config.yaml` from chosen values) gets unit tests with injected fake stdin and fake API clients, matching the dependency-injection pattern used throughout the rest of this codebase (`sleep_fn`, `clock_fn`, fake HTTP clients, etc.) — not literally scripting interactive prompts in CI.
- Manually run the script once for real against live APIs as part of implementation verification, consistent with how every other feature in this project has been confirmed against the real vendor APIs before being considered done (not just unit-tested).

## Explicitly Deferred / Not Addressed

- Making `config.yaml`/`.env` paths configurable beyond the repo root (not needed for a git-clone workflow where running from the repo root is the expected developer norm).
- Any automated dependency-installation smoothing beyond documentation (e.g., a `Makefile` or install script) — the friend is a developer and can run `pip install` themselves given clear instructions.
