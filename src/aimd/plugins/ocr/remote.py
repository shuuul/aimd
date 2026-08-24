"""OpenAI-compatible remote Unlimited-OCR backend."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aimd.core.errors import BackendUnavailableError, ProcessingFailedError
from aimd.core.remote import RemoteBackendConfig

from .models.mlx import UNLIMITED_OCR_NGRAM_SIZE, UNLIMITED_OCR_NGRAM_WINDOW
from .models.unlimited import normalize_unlimited_ocr_markdown


class RemoteOCRClient:
    """Recognize one rendered image through vLLM chat completions."""

    def __init__(self, config: RemoteBackendConfig) -> None:
        self.config = config

    def recognize_image(self, image_path: Path, *, multi_page: bool = False) -> str:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode()
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<image>\ndocument parsing."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                        },
                    ],
                }
            ],
            "skip_special_tokens": False,
            "ngram_size": UNLIMITED_OCR_NGRAM_SIZE,
            "window_size": 1024 if multi_page else UNLIMITED_OCR_NGRAM_WINDOW,
        }
        request = Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=600) as response:  # noqa: S310
                response_payload = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            if exc.code >= 500:
                raise BackendUnavailableError(
                    f"Remote OCR backend returned HTTP {exc.code}: {detail}"
                ) from exc
            raise ProcessingFailedError(
                f"Remote OCR request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise BackendUnavailableError(
                f"Remote OCR backend is unavailable at {self.config.base_url}: {exc}"
            ) from exc
        except (json.JSONDecodeError, TypeError) as exc:
            raise ProcessingFailedError(
                "Remote OCR backend returned an invalid JSON response"
            ) from exc

        try:
            text = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProcessingFailedError(
                "Remote OCR backend returned an invalid chat completion"
            ) from exc
        if not isinstance(text, str) or not text.strip():
            raise ProcessingFailedError("Remote OCR produced empty content")
        return normalize_unlimited_ocr_markdown(text).strip()
