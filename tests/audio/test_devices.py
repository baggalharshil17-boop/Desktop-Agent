import pytest

from financial_voice_agent.audio.devices import (
    resolve_input_device_index,
    resolve_output_device_index,
)


class _FakePyaudioModule:
    paWASAPI = 13


class _FakePyaudioInstance:
    def __init__(self, *, wasapi_info=None, wasapi_error=None, default_output=None, default_input=None):
        self._wasapi_info = wasapi_info
        self._wasapi_error = wasapi_error
        self._default_output = default_output
        self._default_input = default_input

    def get_host_api_info_by_type(self, host_api_type):
        if self._wasapi_error is not None:
            raise self._wasapi_error
        return self._wasapi_info

    def get_default_output_device_info(self):
        return self._default_output

    def get_default_input_device_info(self):
        return self._default_input


def test_resolve_output_device_index_returns_configured_value_when_set():
    pa = _FakePyaudioInstance(wasapi_info={"defaultOutputDevice": 8})

    result = resolve_output_device_index(pa, 4, pyaudio_module=_FakePyaudioModule())

    assert result == 4


def test_resolve_output_device_index_prefers_wasapi_default_over_plain_default():
    pa = _FakePyaudioInstance(
        wasapi_info={"defaultOutputDevice": 8}, default_output={"index": 3}
    )

    result = resolve_output_device_index(pa, None, pyaudio_module=_FakePyaudioModule())

    assert result == 8


def test_resolve_output_device_index_falls_back_to_plain_default_when_wasapi_unavailable():
    pa = _FakePyaudioInstance(wasapi_error=OSError("no WASAPI"), default_output={"index": 3})

    result = resolve_output_device_index(pa, None, pyaudio_module=_FakePyaudioModule())

    assert result == 3


def test_resolve_output_device_index_returns_none_when_nothing_available():
    class _AlwaysFailingPyaudioInstance:
        def get_host_api_info_by_type(self, host_api_type):
            raise OSError("no WASAPI")

        def get_default_output_device_info(self):
            raise OSError("no default either")

    result = resolve_output_device_index(
        _AlwaysFailingPyaudioInstance(), None, pyaudio_module=_FakePyaudioModule()
    )

    assert result is None


def test_resolve_input_device_index_returns_configured_value_when_set():
    pa = _FakePyaudioInstance(wasapi_info={"defaultInputDevice": 9})

    result = resolve_input_device_index(pa, 1, pyaudio_module=_FakePyaudioModule())

    assert result == 1


def test_resolve_input_device_index_prefers_wasapi_default_over_plain_default():
    pa = _FakePyaudioInstance(
        wasapi_info={"defaultInputDevice": 9}, default_input={"index": 1}
    )

    result = resolve_input_device_index(pa, None, pyaudio_module=_FakePyaudioModule())

    assert result == 9


def test_resolve_input_device_index_falls_back_to_plain_default_when_wasapi_unavailable():
    pa = _FakePyaudioInstance(wasapi_error=OSError("no WASAPI"), default_input={"index": 1})

    result = resolve_input_device_index(pa, None, pyaudio_module=_FakePyaudioModule())

    assert result == 1
