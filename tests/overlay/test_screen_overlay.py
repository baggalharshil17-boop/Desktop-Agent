import os
import socket

from financial_voice_agent.overlay.screen_overlay import (
    bind_listener_socket,
    is_allowed_chart_path,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_bind_listener_socket_returns_a_socket_when_the_port_is_free():
    port = _free_port()

    sock = bind_listener_socket(port)

    try:
        assert sock is not None
        assert sock.getsockname()[1] == port
    finally:
        if sock is not None:
            sock.close()


def test_bind_listener_socket_returns_none_when_the_port_is_taken():
    # An orphaned overlay from a previous run still owning the port is the
    # real-world case: main() must be able to detect it and exit rather than
    # showing a window that can never receive anything.
    port = _free_port()
    holder = bind_listener_socket(port)

    try:
        assert holder is not None
        assert bind_listener_socket(port) is None
    finally:
        if holder is not None:
            holder.close()


def test_bind_listener_socket_does_not_leak_the_socket_on_failure():
    port = _free_port()
    holder = bind_listener_socket(port)

    try:
        # Repeated failures must not pile up unclosed file descriptors.
        for _ in range(50):
            assert bind_listener_socket(port) is None
    finally:
        if holder is not None:
            holder.close()


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
