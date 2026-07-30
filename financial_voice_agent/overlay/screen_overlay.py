"""Standalone always-on-top screen overlay process for the Financial Voice
Agent. Shows a click-through edge glow while the agent is processing, and a
sliding chart panel when a chart is rendered. Launched by __main__.py as a
subprocess (see financial_voice_agent/overlay/signal_client.py for the
message protocol this listens for). Run directly for manual testing:

    python -m financial_voice_agent.overlay.screen_overlay
"""

from __future__ import annotations

import ctypes
import queue
import socket
import threading
import tkinter as tk

from financial_voice_agent.overlay.signal_client import DEFAULT_OVERLAY_PORT

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020

_GLOW_COLOR = "#3399ff"
_GLOW_BORDER_PX = 8
_TRANSPARENT_COLOR = "black"  # chosen as the -transparentcolor key; never drawn otherwise
_CHART_PANEL_WIDTH = 520
_CHART_PANEL_HEIGHT = 420
_SLIDE_STEP_PX = 40
_SLIDE_INTERVAL_MS = 15


def _make_click_through(root: tk.Tk) -> None:
    root.update_idletasks()
    hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
    if hwnd == 0:
        hwnd = root.winfo_id()
    ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style | WS_EX_LAYERED | WS_EX_TRANSPARENT)


def _listen(inbox: "queue.Queue[str]", port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    while True:
        data, _addr = sock.recvfrom(2048)
        inbox.put(data.decode("utf-8"))


class ChartPanel:
    """A non-click-through Toplevel that slides in from the right edge of
    the screen when shown, and slides back out when closed."""

    def __init__(self, root: tk.Tk):
        self._root = root
        self._window = tk.Toplevel(root)
        self._window.overrideredirect(True)
        self._window.attributes("-topmost", True)
        self._window.withdraw()
        self._screen_w = root.winfo_screenwidth()
        self._screen_h = root.winfo_screenheight()
        self._target_x = self._screen_w - _CHART_PANEL_WIDTH - 20
        self._hidden_x = self._screen_w
        self._y = (self._screen_h - _CHART_PANEL_HEIGHT) // 2

        close_bar = tk.Frame(self._window, bg="#222222", height=24)
        close_bar.pack(fill="x", side="top")
        close_button = tk.Button(
            close_bar, text="×", command=self.hide, bg="#222222", fg="white",
            bd=0, activebackground="#444444", activeforeground="white",
        )
        close_button.pack(side="right")

        self._image_label = tk.Label(self._window)
        self._image_label.pack(fill="both", expand=True)
        self._current_image = None  # keep a reference so tkinter doesn't garbage-collect it

    def show(self, image_path: str) -> None:
        self._current_image = tk.PhotoImage(file=image_path)
        self._image_label.configure(image=self._current_image)
        self._window.geometry(
            f"{_CHART_PANEL_WIDTH}x{_CHART_PANEL_HEIGHT}+{self._hidden_x}+{self._y}"
        )
        self._window.deiconify()
        self._animate_to(self._target_x)

    def hide(self) -> None:
        self._animate_to(self._hidden_x, on_done=self._window.withdraw)

    def _animate_to(self, target_x: int, *, on_done=None) -> None:
        current_x = self._window.winfo_x()
        step = _SLIDE_STEP_PX if target_x > current_x else -_SLIDE_STEP_PX

        def tick() -> None:
            nonlocal current_x
            current_x += step
            reached = (step > 0 and current_x >= target_x) or (step < 0 and current_x <= target_x)
            if reached:
                current_x = target_x
            self._window.geometry(f"{_CHART_PANEL_WIDTH}x{_CHART_PANEL_HEIGHT}+{current_x}+{self._y}")
            if not reached:
                self._window.after(_SLIDE_INTERVAL_MS, tick)
            elif on_done is not None:
                on_done()

        tick()


def main() -> None:
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg=_TRANSPARENT_COLOR)
    root.attributes("-transparentcolor", _TRANSPARENT_COLOR)

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.geometry(f"{screen_w}x{screen_h}+0+0")

    canvas = tk.Canvas(root, width=screen_w, height=screen_h, bg=_TRANSPARENT_COLOR, highlightthickness=0)
    canvas.pack()
    glow_id = canvas.create_rectangle(
        _GLOW_BORDER_PX // 2, _GLOW_BORDER_PX // 2,
        screen_w - _GLOW_BORDER_PX // 2, screen_h - _GLOW_BORDER_PX // 2,
        outline=_GLOW_COLOR, width=_GLOW_BORDER_PX, state="hidden",
    )

    _make_click_through(root)

    chart_panel = ChartPanel(root)

    inbox: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_listen, args=(inbox, DEFAULT_OVERLAY_PORT), daemon=True).start()

    def poll() -> None:
        try:
            while True:
                message = inbox.get_nowait()
                if message == "processing_on":
                    canvas.itemconfigure(glow_id, state="normal")
                elif message == "processing_off":
                    canvas.itemconfigure(glow_id, state="hidden")
                elif message.startswith("show_chart:"):
                    chart_panel.show(message[len("show_chart:") :])
        except queue.Empty:
            pass
        root.after(30, poll)

    root.after(30, poll)
    root.mainloop()


if __name__ == "__main__":
    main()
