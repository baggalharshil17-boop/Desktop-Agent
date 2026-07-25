import base64
import io

import pytest
from PIL import Image

from financial_voice_agent.tools.screen import WindowNotFoundError, capture_screen


def _make_test_image_bytes(width: int = 10, height: int = 10) -> bytes:
    image = Image.new("RGB", (width, height), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_capture_screen_returns_base64_jpeg_when_window_found():
    def window_finder():
        return {"left": 0, "top": 0, "width": 10, "height": 10}

    def screenshot_fn(region):
        assert region == {"left": 0, "top": 0, "width": 10, "height": 10}
        return _make_test_image_bytes()

    result = await capture_screen(window_finder=window_finder, screenshot_fn=screenshot_fn)

    decoded = base64.b64decode(result["image_base64"])
    image = Image.open(io.BytesIO(decoded))
    assert image.format == "JPEG"


@pytest.mark.asyncio
async def test_capture_screen_raises_window_not_found_when_window_finder_returns_none():
    def window_finder():
        return None

    def screenshot_fn(region):
        raise AssertionError("must not be called if window not found")

    with pytest.raises(WindowNotFoundError):
        await capture_screen(window_finder=window_finder, screenshot_fn=screenshot_fn)
