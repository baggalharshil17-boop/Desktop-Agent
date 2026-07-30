from financial_voice_agent.overlay.signal_client import DEFAULT_OVERLAY_PORT, make_overlay_sender


class _FakeSocket:
    def __init__(self, *, raise_on_send: bool = False):
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self._raise_on_send = raise_on_send

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        if self._raise_on_send:
            raise OSError("no listener")
        self.sent.append((data, addr))


def test_make_overlay_sender_sends_utf8_encoded_message_to_loopback():
    fake_socket = _FakeSocket()
    send = make_overlay_sender(47765, socket_factory=lambda: fake_socket)

    send("processing_on")

    assert fake_socket.sent == [(b"processing_on", ("127.0.0.1", 47765))]


def test_make_overlay_sender_uses_default_port_when_not_specified():
    fake_socket = _FakeSocket()
    send = make_overlay_sender(socket_factory=lambda: fake_socket)

    send("processing_off")

    assert fake_socket.sent[0][1] == ("127.0.0.1", DEFAULT_OVERLAY_PORT)


def test_make_overlay_sender_swallows_os_error_when_overlay_not_running():
    fake_socket = _FakeSocket(raise_on_send=True)
    send = make_overlay_sender(47765, socket_factory=lambda: fake_socket)

    send("processing_on")  # must not raise


def test_make_overlay_sender_reuses_same_socket_across_calls():
    fake_socket = _FakeSocket()
    factory_calls = []

    def factory():
        factory_calls.append(1)
        return fake_socket

    send = make_overlay_sender(47765, socket_factory=factory)
    send("processing_on")
    send("processing_off")

    assert len(factory_calls) == 1
    assert len(fake_socket.sent) == 2
