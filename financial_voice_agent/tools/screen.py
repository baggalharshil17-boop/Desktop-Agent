from __future__ import annotations

import base64
from typing import Callable


class WindowNotFoundError(Exception):
    pass


async def capture_screen(
    *,
    window_finder: Callable[[], dict | None],
    screenshot_fn: Callable[[dict], bytes],
    jpeg_quality: int = 80,
) -> dict:
    region = window_finder()
    if region is None:
        raise WindowNotFoundError("Kite window not found")
    jpeg_bytes = screenshot_fn(region)
    return {"image_base64": base64.b64encode(jpeg_bytes).decode("ascii")}


def find_kite_window() -> dict | None:
    """Real adapter: locates a window whose title contains "Kite" using
    pygetwindow. Verify this title-matching heuristic against your actual
    browser/window setup at build time -- exact title varies by browser and
    by whether Kite is a PWA, browser tab, or dedicated window."""
    import pygetwindow as gw

    matches = [w for w in gw.getAllWindows() if "kite" in w.title.lower()]
    if not matches:
        return None
    window = matches[0]
    return {"left": window.left, "top": window.top, "width": window.width, "height": window.height}


def capture_region(region: dict, *, jpeg_quality: int = 80) -> bytes:
    """Real adapter: grabs `region` via mss and JPEG-encodes it. Verify
    against your actual display/DPI setup at build time -- mss's raw BGRA
    buffer needs PIL to re-encode as JPEG, matching PRD Section 10.3."""
    import io

    import mss
    from PIL import Image

    with mss.mss() as sct:
        raw = sct.grab(region)
    image = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()
