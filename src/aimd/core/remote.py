"""Shared configuration for opt-in remote inference backends."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal
from urllib.parse import urlparse

from .errors import BackendUnavailableError

RemoteModality = Literal["asr", "ocr"]


@dataclass(frozen=True, slots=True)
class RemoteBackendConfig:
    """Resolved OpenAI-compatible backend settings."""

    base_url: str
    model: str
    api_key: str


def resolve_remote_backend(
    modality: RemoteModality,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> RemoteBackendConfig | None:
    """Resolve explicit settings over environment variables for one modality."""
    prefix = f"AIMD_{modality.upper()}"
    resolved_url = base_url or os.getenv(f"{prefix}_BASE_URL")
    if not resolved_url:
        return None

    resolved_url = _normalize_base_url(resolved_url)
    default_model = "Qwen3-ASR-1.7B" if modality == "asr" else "Unlimited-OCR"
    return RemoteBackendConfig(
        base_url=resolved_url,
        model=model or os.getenv(f"{prefix}_MODEL") or default_model,
        api_key=api_key or os.getenv(f"{prefix}_API_KEY") or "not-needed",
    )


def _normalize_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BackendUnavailableError(
            f"Invalid remote backend URL {value!r}; expected an http(s) URL."
        )
    if not parsed.path or parsed.path == "/":
        base_url += "/v1"
    return base_url
