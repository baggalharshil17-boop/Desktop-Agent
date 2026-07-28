"""Interactive first-run setup wizard.

Walks through choosing providers (STT/LLM/TTS), collecting the API keys
those choices need, validating each one live, and writing .env and
config.yaml. Run once after cloning:

    python scripts/setup.py

Does not collect Kite credentials -- that's a separate OAuth-style browser
flow, see scripts/kite_login.py. If you don't want to deal with a live
Zerodha account yet, choose mode "mock" when prompted below; the agent will
use canned fixture data for Kite-backed tools instead.
"""

from __future__ import annotations

import getpass
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from financial_voice_agent.setup.config_template import render_config_yaml
from financial_voice_agent.setup.env_file import merge_env_values, read_env_file, write_env_file
from financial_voice_agent.setup.validators import (
    ValidationResult,
    validate_cartesia_key,
    validate_groq_key,
    validate_huggingface_key,
    validate_tavily_key,
)

ENV_PATH = REPO_ROOT / ".env"
CONFIG_PATH = REPO_ROOT / "config.yaml"


def _ask_choice(prompt: str, options: list[str], default: str) -> str:
    options_str = "/".join(o if o != default else o.upper() for o in options)
    while True:
        raw = input(f"{prompt} [{options_str}]: ").strip().lower()
        if not raw:
            return default
        if raw in options:
            return raw
        print(f"  Please enter one of: {', '.join(options)}")


def _ask_yes_no(prompt: str, *, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def _collect_and_validate(
    env_var: str, prompt: str, validate_fn
) -> tuple[str, ValidationResult]:
    while True:
        value = getpass.getpass(f"{prompt} ({env_var}): ").strip()
        if not value:
            print("  Skipped -- you can add this later by re-running this script or editing .env directly.")
            return "", ValidationResult(ok=False, message="skipped by user")
        print("  Checking...")
        result = validate_fn(value)
        if result.ok:
            print(f"  OK: {result.message}")
            return value, result
        print(f"  Failed: {result.message}")
        if not _ask_yes_no("  Try this key again?", default=True):
            return value, result


def main() -> None:
    print("Financial Voice Agent -- setup wizard\n")

    existing_env = read_env_file(str(ENV_PATH))
    overwrite_env = True
    if existing_env:
        overwrite_env = _ask_yes_no(
            "Found an existing .env. Overwrite everything (choose No to only fill in what's missing)?",
            default=False,
        )

    config_exists = CONFIG_PATH.exists()
    write_config = True
    if config_exists:
        write_config = _ask_yes_no(
            "Found an existing config.yaml. Regenerate it from your choices below?", default=False
        )

    new_env: dict[str, str] = {}
    cartesia_voice_id = ""

    stt_provider = _ask_choice("STT provider", ["groq", "huggingface"], default="groq")
    llm_provider = _ask_choice("LLM provider", ["groq", "huggingface"], default="groq")
    print(
        "  Note: whichever LLM model you use must support BOTH tool calling and vision "
        "for capture_screen results to actually be described, not just confirmed."
    )
    mode = _ask_choice(
        "Mode -- 'mock' uses canned data and needs no Zerodha account; 'live' needs Kite Connect set up separately",
        ["mock", "live"],
        default="mock",
    )

    if stt_provider == "groq" or llm_provider == "groq":
        value, _ = _collect_and_validate(
            "GROQ_API_KEY", "Groq API key (console.groq.com)", validate_groq_key
        )
        if value:
            new_env["GROQ_API_KEY"] = value

    if stt_provider == "huggingface" or llm_provider == "huggingface":
        value, _ = _collect_and_validate(
            "HF_TOKEN", "Hugging Face token (huggingface.co/settings/tokens)", validate_huggingface_key
        )
        if value:
            new_env["HF_TOKEN"] = value

    cartesia_key, cartesia_result = _collect_and_validate(
        "CARTESIA_API_KEY", "Cartesia API key (play.cartesia.ai)", validate_cartesia_key
    )
    if cartesia_key:
        new_env["CARTESIA_API_KEY"] = cartesia_key
        voices = (cartesia_result.data or {}).get("voices", [])
        if voices:
            print("\n  Available voices:")
            for i, (voice_id, name) in enumerate(voices[:15], start=1):
                print(f"    {i}. {name} ({voice_id})")
            choice = input("  Pick a voice number (or press Enter to type an id manually): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(voices[:15]):
                cartesia_voice_id = voices[int(choice) - 1][0]
            else:
                cartesia_voice_id = input("  Paste a voice_id: ").strip()

    tavily_value, _ = _collect_and_validate(
        "TAVILY_API_KEY", "Tavily API key (tavily.com) -- for news search", validate_tavily_key
    )
    if tavily_value:
        new_env["TAVILY_API_KEY"] = tavily_value

    merged_env = merge_env_values(existing_env, new_env, overwrite=overwrite_env)
    write_env_file(str(ENV_PATH), merged_env)
    print(f"\nWrote {ENV_PATH}")

    if write_config:
        stt_model = "whisper-large-v3-turbo" if stt_provider == "groq" else "openai/whisper-large-v3-turbo"
        llm_model = "qwen/qwen3.6-27b"
        config_text = render_config_yaml(
            stt_provider=stt_provider,
            stt_model=stt_model,
            llm_provider=llm_provider,
            llm_model=llm_model,
            cartesia_voice_id=cartesia_voice_id or "<pick a voice id from https://play.cartesia.ai/voices>",
            mode=mode,
        )
        CONFIG_PATH.write_text(config_text)
        print(f"Wrote {CONFIG_PATH}")

    print("\nNext steps:")
    if mode == "live":
        print("  1. python scripts/kite_login.py   (get today's Kite access token)")
        print("  2. python -m financial_voice_agent")
    else:
        print("  1. python -m financial_voice_agent")


if __name__ == "__main__":
    main()
