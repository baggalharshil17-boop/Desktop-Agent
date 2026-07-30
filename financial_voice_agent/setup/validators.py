from __future__ import annotations

from dataclasses import dataclass

import cartesia
import groq
import httpx
import huggingface_hub
import huggingface_hub.errors


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    message: str
    data: dict | None = None


def validate_groq_key(api_key: str, *, client_factory=groq.Groq) -> ValidationResult:
    try:
        client = client_factory(api_key=api_key)
        client.models.list()
    except groq.GroqError as exc:
        return ValidationResult(ok=False, message=f"Groq key rejected: {exc}")
    except Exception as exc:  # noqa: BLE001 -- any other failure (network, etc.) is still "not validated"
        return ValidationResult(ok=False, message=f"Could not reach Groq: {exc}")
    return ValidationResult(ok=True, message="Groq key OK")


def validate_huggingface_key(token: str, *, whoami_fn=huggingface_hub.whoami) -> ValidationResult:
    try:
        whoami_fn(token=token)
    except huggingface_hub.errors.HfHubHTTPError as exc:
        return ValidationResult(ok=False, message=f"Hugging Face token rejected: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(ok=False, message=f"Could not reach Hugging Face: {exc}")
    return ValidationResult(ok=True, message="Hugging Face token OK")


def validate_cartesia_key(api_key: str, *, client_factory=cartesia.Cartesia) -> ValidationResult:
    try:
        client = client_factory(api_key=api_key)
        voices = [(v.id, v.name) for v in client.voices.list(limit=50)]
    except cartesia.CartesiaError as exc:
        return ValidationResult(ok=False, message=f"Cartesia key rejected: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(ok=False, message=f"Could not reach Cartesia: {exc}")
    return ValidationResult(ok=True, message="Cartesia key OK", data={"voices": voices})


def validate_tavily_key(api_key: str, *, http_client: httpx.Client | None = None) -> ValidationResult:
    owns_client = http_client is None
    client = http_client or httpx.Client(base_url="https://api.tavily.com", timeout=15.0)
    try:
        response = client.post(
            "/search", json={"api_key": api_key, "query": "test", "max_results": 1}
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return ValidationResult(ok=False, message=f"Tavily key rejected: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(ok=False, message=f"Could not reach Tavily: {exc}")
    finally:
        if owns_client:
            client.close()
    return ValidationResult(ok=True, message="Tavily key OK")


def validate_fish_audio_key(api_key: str, *, http_client: httpx.Client | None = None) -> ValidationResult:
    owns_client = http_client is None
    client = http_client or httpx.Client(base_url="https://api.fish.audio", timeout=15.0)
    try:
        response = client.post(
            "/v1/tts",
            headers={"Authorization": f"Bearer {api_key}", "model": "s2.1-pro-free"},
            json={"text": "Test.", "format": "pcm", "sample_rate": 16000},
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return ValidationResult(ok=False, message=f"Fish Audio key rejected: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(ok=False, message=f"Could not reach Fish Audio: {exc}")
    finally:
        if owns_client:
            client.close()
    return ValidationResult(ok=True, message="Fish Audio key OK")
