import os

from financial_voice_agent.overlay.screen_overlay import is_allowed_chart_path


def _make_png(path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    return str(path)


def test_is_allowed_chart_path_accepts_a_png_inside_the_charts_dir(tmp_path):
    charts_dir = tmp_path / "charts"
    chart = _make_png(charts_dir / "chart_RELIANCE_1.png")

    assert is_allowed_chart_path(chart, str(charts_dir)) is True


def test_is_allowed_chart_path_rejects_a_file_outside_the_charts_dir(tmp_path):
    # The sender is authenticated but still shouldn't be able to point the
    # always-on-top panel at arbitrary files on disk.
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir()
    outsider = _make_png(tmp_path / "private.png")

    assert is_allowed_chart_path(outsider, str(charts_dir)) is False


def test_is_allowed_chart_path_rejects_traversal_out_of_the_charts_dir(tmp_path):
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir()
    _make_png(tmp_path / "private.png")
    traversal = os.path.join(str(charts_dir), "..", "private.png")

    assert is_allowed_chart_path(traversal, str(charts_dir)) is False


def test_is_allowed_chart_path_rejects_non_png_files(tmp_path):
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir()
    script = charts_dir / "payload.bat"
    script.write_text("echo pwned")

    assert is_allowed_chart_path(str(script), str(charts_dir)) is False


def test_is_allowed_chart_path_rejects_a_missing_file(tmp_path):
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir()

    assert is_allowed_chart_path(str(charts_dir / "nope.png"), str(charts_dir)) is False


def test_is_allowed_chart_path_rejects_garbage_without_raising(tmp_path):
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir()

    assert is_allowed_chart_path("", str(charts_dir)) is False
    assert is_allowed_chart_path("\x00bad", str(charts_dir)) is False
