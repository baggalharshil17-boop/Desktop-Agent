from __future__ import annotations

import base64
import os
import time
from typing import Callable


class WindowNotFoundError(Exception):
    pass


async def capture_screen(
    *,
    window_finder: Callable[[], dict | None],
    screenshot_fn: Callable[[dict], bytes],
    screenshot_dir: str = "screenshots",
) -> dict:
    region = window_finder()
    if region is None:
        raise WindowNotFoundError("Kite window not found")
    jpeg_bytes = screenshot_fn(region)
    os.makedirs(screenshot_dir, exist_ok=True)
    filename = f"capture_{int(time.time() * 1000)}.jpg"
    path = os.path.join(screenshot_dir, filename)
    with open(path, "wb") as f:
        f.write(jpeg_bytes)
    return {
        "screenshot_path": path,
        "width": region.get("width"),
        "height": region.get("height"),
        # Consumed by run_llm_turn to attach the actual image to the next
        # message so a vision-capable model can describe it -- not just log
        # that a screenshot exists. Stripped before this dict is JSON-
        # serialized into the tool message or persisted to the turns DB
        # (financial_voice_agent/orchestrator/llm.py), since neither needs a
        # multi-KB base64 blob.
        "image_b64": base64.b64encode(jpeg_bytes).decode("ascii"),
        "image_mime": "image/jpeg",
    }


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
