from __future__ import annotations

import pyaudio as _default_pyaudio_module


def resolve_output_device_index(
    pyaudio_instance, configured_index: int | None, *, pyaudio_module=_default_pyaudio_module
) -> int | None:
    """Resolves which output device index to actually open.

    An explicit config value always wins. Otherwise, prefer the WASAPI host
    API's default output device over PyAudio's own
    get_default_output_device_info() -- the latter resolves through
    whichever host API PortAudio initializes first (typically MME on
    Windows), which can silently disagree with whatever the user actually
    has selected in Windows Sound Settings. WASAPI is the API Windows
    Settings itself controls, so it's the one that actually tracks live
    changes (e.g. plugging in a new default playback device) -- confirmed
    live: MME's default stayed on the built-in speakers' old index while
    WASAPI's default correctly reflected the currently-connected device.

    Falls back to PyAudio's own default, then None (let PortAudio pick), if
    WASAPI isn't available on this platform.
    """
    if configured_index is not None:
        return configured_index
    try:
        wasapi_info = pyaudio_instance.get_host_api_info_by_type(pyaudio_module.paWASAPI)
        return wasapi_info["defaultOutputDevice"]
    except Exception:  # noqa: BLE001 -- any lookup failure just means "no WASAPI info available"
        pass
    try:
        return pyaudio_instance.get_default_output_device_info()["index"]
    except Exception:  # noqa: BLE001
        return None


def resolve_input_device_index(
    pyaudio_instance, configured_index: int | None, *, pyaudio_module=_default_pyaudio_module
) -> int | None:
    """Input-side counterpart to resolve_output_device_index -- same
    WASAPI-first resolution order and same rationale."""
    if configured_index is not None:
        return configured_index
    try:
        wasapi_info = pyaudio_instance.get_host_api_info_by_type(pyaudio_module.paWASAPI)
        return wasapi_info["defaultInputDevice"]
    except Exception:  # noqa: BLE001
        pass
    try:
        return pyaudio_instance.get_default_input_device_info()["index"]
    except Exception:  # noqa: BLE001
        return None
