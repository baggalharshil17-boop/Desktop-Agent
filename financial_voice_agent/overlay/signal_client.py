from __future__ import annotations

import os
import secrets
import socket
from typing import Callable

# Arbitrary fixed high port for localhost-only agent<->overlay signaling.
# Chosen to be unlikely to collide with common dev-server ports.
DEFAULT_OVERLAY_PORT = 47765

# The agent generates a token at startup and passes it to the overlay
# subprocess through this environment variable. Every datagram must carry it,
# so a datagram from any other local process is ignored. Without this, the
# port is a fixed published constant that any local process can write to --
# enough to render an arbitrary image in a borderless always-on-top panel the
# user did not initiate (a UI-spoofing primitive).
OVERLAY_TOKEN_ENV_VAR = "FVA_OVERLAY_TOKEN"

_TOKEN_SEPARATOR = " "


def generate_overlay_token() -> str:
    return secrets.token_urlsafe(32)


def read_overlay_token() -> str | None:
    return os.environ.get(OVERLAY_TOKEN_ENV_VAR) or None


def encode_message(token: str, message: str) -> str:
    return f"{token}{_TOKEN_SEPARATOR}{message}"


def decode_message(expected_token: str | None, datagram: str) -> str | None:
    """Returns the message body if the datagram carries the expected token,
    else None. Returns None for every datagram when no token is configured --
    failing closed, so a missing token never silently downgrades to the
    unauthenticated behaviour this replaced."""
    if not expected_token:
        return None
    token, separator, message = datagram.partition(_TOKEN_SEPARATOR)
    if not separator:
        return None
    # compare_digest to keep the check constant-time.
    if not secrets.compare_digest(token, expected_token):
        return None
    return message


def make_overlay_sender(
    port: int = DEFAULT_OVERLAY_PORT,
    *,
    token: str | None = None,
    socket_factory: Callable[[], socket.socket] | None = None,
) -> Callable[[str], None]:
    sock = (socket_factory or (lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)))()
    resolved_token = token if token is not None else read_overlay_token()

    def send(message: str) -> None:
        # Fire-and-forget: the overlay process may not be running (disabled
        # via config, crashed, or not yet started), and a missing overlay
        # must never block or crash the voice agent.
        try:
            payload = encode_message(resolved_token or "", message)
            sock.sendto(payload.encode("utf-8"), ("127.0.0.1", port))
        except OSError:
            pass

    return send
