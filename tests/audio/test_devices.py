import pytest

from financial_voice_agent.audio.devices import (
    resolve_input_device_index,
    resolve_output_device_index,
)


class _FakePyaudioModule:
    paWASAPI = "WASAPI"
    paMME = "MME"


class _FakePyaudioInstance:
    """Models enough of PyAudio's API surface for the WASAPI-identify /
    MME-open resolution: two host APIs (WASAPI at host_api index 2, MME at
    host_api index 0) and a device table."""

    def __init__(self, *, devices, wasapi_default_output=None, wasapi_default_input=None,
                 wasapi_error=None, default_output=None, default_input=None):
        self._devices = devices  # {index: {"name", "hostApi", "maxOutputChannels", "maxInputChannels"}}
        self._wasapi_default_output = wasapi_default_output
        self._wasapi_default_input = wasapi_default_input
        self._wasapi_error = wasapi_error
        self._default_output = default_output
        self._default_input = default_input

    def get_host_api_info_by_type(self, host_api_type):
        if host_api_type == "WASAPI":
            if self._wasapi_error is not None:
                raise self._wasapi_error
            return {
                "index": 2,
                "defaultOutputDevice": self._wasapi_default_output,
                "defaultInputDevice": self._wasapi_default_input,
            }
        if host_api_type == "MME":
            return {"index": 0}
        raise ValueError(f"unknown host api type: {host_api_type}")

    def get_device_count(self):
        return max(self._devices) + 1

    def get_device_info_by_index(self, index):
        return self._devices[index]

    def get_default_output_device_info(self):
        return self._default_output

    def get_default_input_device_info(self):
        return self._default_input


def test_resolve_output_device_index_returns_configured_value_when_set():
    pa = _FakePyaudioInstance(devices={0: {"name": "x", "hostApi": 0, "maxOutputChannels": 2}})

    result = resolve_output_device_index(pa, 4, pyaudio_module=_FakePyaudioModule())

    assert result == 4


def test_resolve_output_device_index_opens_mme_counterpart_of_wasapi_default():
    # WASAPI's default is device 8 ("Speakers ..."), but the same physical
    # device also has an MME entry (device 4) -- that's the one that must
    # actually be opened, since WASAPI's own entry only accepts its native
    # 48kHz rate while MME can resample to 16kHz.
    devices = {
        4: {"name": "Speakers (2- Realtek(R) Audio)", "hostApi": 0, "maxOutputChannels": 2},
        8: {"name": "Speakers (2- Realtek(R) Audio)", "hostApi": 2, "maxOutputChannels": 2},
    }
    pa = _FakePyaudioInstance(devices=devices, wasapi_default_output=8)

    result = resolve_output_device_index(pa, None, pyaudio_module=_FakePyaudioModule())

    assert result == 4


def test_resolve_output_device_index_matches_mme_counterpart_despite_name_truncation():
    # MME truncates names to ~32 chars -- confirmed live, WASAPI's full
    # "Headset Microphone (Jabra Evolve2 40 SE)" becomes MME's truncated
    # "Headset Microphone (Jabra Evolv". Must still match as the same device.
    devices = {
        1: {"name": "Headset Microphone (Jabra Evolv", "hostApi": 0, "maxOutputChannels": 0, "maxInputChannels": 1},
        15: {"name": "Headset Microphone (Jabra Evolve2 40 SE)", "hostApi": 2, "maxInputChannels": 2},
    }
    pa = _FakePyaudioInstance(devices=devices, wasapi_default_input=15)

    result = resolve_input_device_index(pa, None, pyaudio_module=_FakePyaudioModule())

    assert result == 1


def test_resolve_output_device_index_falls_back_to_wasapi_index_when_no_mme_counterpart():
    devices = {8: {"name": "Some USB Device", "hostApi": 2, "maxOutputChannels": 2}}
    pa = _FakePyaudioInstance(devices=devices, wasapi_default_output=8)

    result = resolve_output_device_index(pa, None, pyaudio_module=_FakePyaudioModule())

    assert result == 8


def test_resolve_output_device_index_ignores_same_named_input_only_mme_device():
    # An MME device with a matching name but zero output channels (e.g. a
    # same-named input device) must not be selected for output resolution.
    devices = {
        2: {"name": "Speakers (2- Realtek(R) Audio)", "hostApi": 0, "maxOutputChannels": 0, "maxInputChannels": 2},
        4: {"name": "Speakers (2- Realtek(R) Audio)", "hostApi": 0, "maxOutputChannels": 2},
        8: {"name": "Speakers (2- Realtek(R) Audio)", "hostApi": 2, "maxOutputChannels": 2},
    }
    pa = _FakePyaudioInstance(devices=devices, wasapi_default_output=8)

    result = resolve_output_device_index(pa, None, pyaudio_module=_FakePyaudioModule())

    assert result == 4


def test_resolve_output_device_index_falls_back_to_plain_default_when_wasapi_unavailable():
    pa = _FakePyaudioInstance(
        devices={3: {"name": "x", "hostApi": 0, "maxOutputChannels": 2}},
        wasapi_error=OSError("no WASAPI"),
        default_output={"index": 3},
    )

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
    pa = _FakePyaudioInstance(devices={0: {"name": "x", "hostApi": 0, "maxInputChannels": 2}})

    result = resolve_input_device_index(pa, 1, pyaudio_module=_FakePyaudioModule())

    assert result == 1


def test_resolve_input_device_index_opens_mme_counterpart_of_wasapi_default():
    devices = {
        2: {"name": "Microphone Array (2- Realtek(R) Audio)", "hostApi": 0, "maxInputChannels": 4},
        9: {"name": "Microphone Array (2- Realtek(R) Audio)", "hostApi": 2, "maxInputChannels": 2},
    }
    pa = _FakePyaudioInstance(devices=devices, wasapi_default_input=9)

    result = resolve_input_device_index(pa, None, pyaudio_module=_FakePyaudioModule())

    assert result == 2


def test_resolve_input_device_index_falls_back_to_plain_default_when_wasapi_unavailable():
    pa = _FakePyaudioInstance(
        devices={1: {"name": "x", "hostApi": 0, "maxInputChannels": 2}},
        wasapi_error=OSError("no WASAPI"),
        default_input={"index": 1},
    )

    result = resolve_input_device_index(pa, None, pyaudio_module=_FakePyaudioModule())

    assert result == 1
