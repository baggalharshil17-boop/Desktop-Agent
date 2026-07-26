"""One-off helper to generate today's Kite Connect access token.

Kite Connect access tokens expire daily, so this needs to be re-run each
trading day before the voice agent is used with mode: "live" in config.yaml.

Usage:
    python scripts/kite_login.py

Requires KITE_API_KEY and KITE_API_SECRET to already be set in .env.
Prints a login URL -- open it, log in with your normal Zerodha credentials,
and paste the resulting redirect URL (or just the request_token) back here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import dotenv_values
from kiteconnect import KiteConnect

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _extract_request_token(pasted: str) -> str:
    pasted = pasted.strip()
    if "request_token" in pasted:
        query = urlparse(pasted).query or pasted.split("?", 1)[-1]
        token = parse_qs(query).get("request_token", [None])[0]
        if not token:
            raise SystemExit("Could not find request_token in the pasted URL.")
        return token
    return pasted


def _write_access_token(access_token: str) -> None:
    lines = ENV_PATH.read_text().splitlines()
    pattern = re.compile(r"^KITE_ACCESS_TOKEN=.*$")
    replaced = False
    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = f"KITE_ACCESS_TOKEN={access_token}"
            replaced = True
            break
    if not replaced:
        lines.append(f"KITE_ACCESS_TOKEN={access_token}")
    ENV_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    env = dotenv_values(ENV_PATH)
    api_key = env.get("KITE_API_KEY")
    api_secret = env.get("KITE_API_SECRET")
    if not api_key or not api_secret:
        raise SystemExit("Set KITE_API_KEY and KITE_API_SECRET in .env before running this.")

    kite = KiteConnect(api_key=api_key)
    print(f"1. Open this URL and log in with your normal Zerodha credentials:\n\n   {kite.login_url()}\n")
    print("2. After login, you'll land on an error page (that's expected) --")
    print("   copy the full URL from your browser's address bar and paste it below.\n")

    pasted = input("Paste the redirect URL (or just the request_token): ").strip()
    if not pasted:
        raise SystemExit("No input given.")
    request_token = _extract_request_token(pasted)

    session = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session["access_token"]

    _write_access_token(access_token)
    print(f"\nKITE_ACCESS_TOKEN updated in {ENV_PATH} (expires ~6am IST tomorrow).")


if __name__ == "__main__":
    sys.exit(main())
