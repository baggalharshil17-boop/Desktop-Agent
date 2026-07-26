from __future__ import annotations

import pyaudio as _default_pyaudio_module


def _find_mme_counterpart(
    pyaudio_instance, pyaudio_module, target_name: str, *, is_output: bool
) -> int | None:
    """Finds the MME entry for the same physical device WASAPI named
    target_name.

    MME devices accept arbitrary sample-rate resampling (this app needs
    16kHz throughout, for Silero VAD and Cartesia's output format); WASAPI
    devices generally only accept their native mix-format rate (48kHz,
    confirmed live) and reject 16kHz/44100kHz outright with "Invalid sample
    rate". So WASAPI's default is used only to identify which physical
    device is actually selected in Windows Sound Settings -- the stream
    itself is opened via that same device's MME entry, which can resample.

    MME truncates device names to ~32 chars, so this matches by prefix, not
    exact string equality -- confirmed live: WASAPI's full "Headset
    Microphone (Jabra Evolve2 40 SE)" becomes MME's truncated "Headset
    Microphone (Jabra Evolv".
    """
    try:
        mme_info = pyaudio_instance.get_host_api_info_by_type(pyaudio_module.paMME)
    except Exception:  # noqa: BLE001
        return None
    host_api_index = mme_info["index"]
    try:
        device_count = pyaudio_instance.get_device_count()
    except Exception:  # noqa: BLE001
        return None
    for i in range(device_count):
        try:
            info = pyaudio_instance.get_device_info_by_index(i)
        except Exception:  # noqa: BLE001
            continue
        if info.get("hostApi") != host_api_index:
            continue
        channel_key = "maxOutputChannels" if is_output else "maxInputChannels"
        if info.get(channel_key, 0) <= 0:
            continue
        name = info.get("name", "")
        if target_name == name or target_name.startswith(name) or name.startswith(target_name):
            return i
    return None


def _resolve_via_wasapi_then_mme(
    pyaudio_instance, pyaudio_module, *, is_output: bool
) -> int | None:
    wasapi_key = "defaultOutputDevice" if is_output else "defaultInputDevice"
    wasapi_info = pyaudio_instance.get_host_api_info_by_type(pyaudio_module.paWASAPI)
    wasapi_index = wasapi_info[wasapi_key]
    wasapi_name = pyaudio_instance.get_device_info_by_index(wasapi_index)["name"]
    mme_index = _find_mme_counterpart(pyaudio_instance, pyaudio_module, wasapi_name, is_output=is_output)
    return mme_index if mme_index is not None else wasapi_index


def resolve_output_device_index(
    pyaudio_instance, configured_index: int | None, *, pyaudio_module=_default_pyaudio_module
) -> int | None:
    """Resolves which output device index to actually open.

    An explicit config value always wins. Otherwise, identifies the
    currently-selected device via the WASAPI host API (the one Windows
    Sound Settings actually controls -- PyAudio's own
    get_default_output_device_info() resolves through a different, less
    reliable host API that can silently disagree with what's actually
    selected), then opens that same physical device via its MME entry so
    it can be opened at the 16kHz rate this app requires -- see
    _find_mme_counterpart for why WASAPI's own entry can't be opened
    directly at 16kHz.

    Falls back to PyAudio's own default, then None (let PortAudio pick), if
    WASAPI isn't available on this platform.
    """
    if configured_index is not None:
        return configured_index
    try:
        return _resolve_via_wasapi_then_mme(pyaudio_instance, pyaudio_module, is_output=True)
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
    WASAPI-identify-then-MME-open resolution order and same rationale."""
    if configured_index is not None:
        return configured_index
    try:
        return _resolve_via_wasapi_then_mme(pyaudio_instance, pyaudio_module, is_output=False)
    except Exception:  # noqa: BLE001
        pass
    try:
        return pyaudio_instance.get_default_input_device_info()["index"]
    except Exception:  # noqa: BLE001
        return None
