from __future__ import annotations

import socket
from typing import Callable

# Arbitrary fixed high port for localhost-only agent<->overlay signaling.
# Chosen to be unlikely to collide with common dev-server ports.
DEFAULT_OVERLAY_PORT = 47765


def make_overlay_sender(
    port: int = DEFAULT_OVERLAY_PORT,
    *,
    socket_factory: Callable[[], socket.socket] | None = None,
) -> Callable[[str], None]:
    sock = (socket_factory or (lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)))()

    def send(message: str) -> None:
        # Fire-and-forget: the overlay process may not be running (disabled
        # via config, crashed, or not yet started), and a missing overlay
        # must never block or crash the voice agent.
        try:
            sock.sendto(message.encode("utf-8"), ("127.0.0.1", port))
        except OSError:
            pass

    return send
