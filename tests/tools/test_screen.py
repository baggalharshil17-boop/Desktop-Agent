import base64
import io
import os

import pytest
from PIL import Image

from financial_voice_agent.tools.screen import WindowNotFoundError, capture_screen


def _make_test_image_bytes(width: int = 10, height: int = 10) -> bytes:
    image = Image.new("RGB", (width, height), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_capture_screen_writes_jpeg_to_disk_and_returns_path(tmp_path):
    def window_finder():
        return {"left": 0, "top": 0, "width": 10, "height": 10}

    def screenshot_fn(region):
        assert region == {"left": 0, "top": 0, "width": 10, "height": 10}
        return _make_test_image_bytes()

    screenshot_dir = str(tmp_path / "screenshots")
    result = await capture_screen(
        window_finder=window_finder, screenshot_fn=screenshot_fn, screenshot_dir=screenshot_dir
    )

    assert os.path.exists(result["screenshot_path"])
    assert result["width"] == 10
    assert result["height"] == 10
    with Image.open(result["screenshot_path"]) as image:
        assert image.format == "JPEG"
    assert result["image_mime"] == "image/jpeg"
    with open(result["screenshot_path"], "rb") as f:
        assert base64.b64decode(result["image_b64"]) == f.read()


@pytest.mark.asyncio
async def test_capture_screen_raises_window_not_found_when_window_finder_returns_none():
    def window_finder():
        return None

    def screenshot_fn(region):
        raise AssertionError("must not be called if window not found")

    with pytest.raises(WindowNotFoundError):
        await capture_screen(window_finder=window_finder, screenshot_fn=screenshot_fn)
