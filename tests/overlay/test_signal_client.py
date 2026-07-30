from financial_voice_agent.overlay.signal_client import (
    DEFAULT_OVERLAY_PORT,
    decode_message,
    encode_message,
    generate_overlay_token,
    make_overlay_sender,
)


class _FakeSocket:
    def __init__(self, *, raise_on_send: bool = False):
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self._raise_on_send = raise_on_send

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        if self._raise_on_send:
            raise OSError("no listener")
        self.sent.append((data, addr))


def test_make_overlay_sender_sends_token_prefixed_message_to_loopback():
    fake_socket = _FakeSocket()
    send = make_overlay_sender(47765, token="tok123", socket_factory=lambda: fake_socket)

    send("processing_on")

    assert fake_socket.sent == [(b"tok123 processing_on", ("127.0.0.1", 47765))]


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


def test_decode_message_accepts_a_datagram_carrying_the_expected_token():
    assert decode_message("tok123", encode_message("tok123", "processing_on")) == "processing_on"


def test_decode_message_rejects_a_datagram_with_a_wrong_token():
    # The core of the fix: another local process guessing the (published)
    # port still can't drive the overlay without the per-run token.
    assert decode_message("tok123", encode_message("wrong", "processing_on")) is None


def test_decode_message_rejects_an_unauthenticated_datagram():
    assert decode_message("tok123", "processing_on") is None
    assert decode_message("tok123", "show_chart:C:/Users/dell/secret.png") is None


def test_decode_message_fails_closed_when_no_token_is_configured():
    # Must not silently fall back to the old unauthenticated behaviour.
    assert decode_message(None, encode_message("anything", "processing_on")) is None
    assert decode_message("", "processing_on") is None


def test_decode_message_preserves_a_message_body_containing_separators():
    body = "show_chart:C:/Users/dell/Desktop-Agent/charts/chart_RELIANCE_1.png"
    assert decode_message("tok123", encode_message("tok123", body)) == body


def test_decode_message_rejects_non_ascii_token_without_raising():
    # secrets.compare_digest raises TypeError on non-ASCII str. The token
    # prefix is attacker-controlled, so a single such datagram would otherwise
    # propagate out of decode_message and kill the listener thread for good,
    # leaving the overlay permanently deaf.
    assert decode_message("tok123", "é processing_on") is None
    assert decode_message("tok123", "你好 processing_on") is None


def test_generate_overlay_token_is_unguessable_and_unique():
    tokens = {generate_overlay_token() for _ in range(50)}

    assert len(tokens) == 50
    assert all(len(token) >= 32 for token in tokens)
