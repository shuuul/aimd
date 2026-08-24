"""OpenAI-compatible remote ASR model adapter."""

from __future__ import annotations

import asyncio
import json
import mimetypes
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

from aimd.core.errors import BackendUnavailableError, ProcessingFailedError
from aimd.core.remote import RemoteBackendConfig

from ..audio_utils import convert_to_wav_if_needed


class RemoteASRModel:
    """Transcribe files through an OpenAI-compatible HTTP endpoint."""

    def __init__(self, config: RemoteBackendConfig) -> None:
        self.config = config
        self.model_id = config.model

    async def transcribe(
        self,
        file_path: Path,
        *,
        language: str | None = None,
        temp_dir: Path | None = None,
        context: str | None = None,
    ) -> str:
        converted_path = convert_to_wav_if_needed(file_path, temp_dir=temp_dir)
        request_path = converted_path or file_path
        try:
            return await asyncio.to_thread(
                self._transcribe_sync,
                request_path,
                language=language,
                context=context,
            )
        finally:
            if converted_path is not None:
                converted_path.unlink(missing_ok=True)

    def _transcribe_sync(
        self,
        file_path: Path,
        *,
        language: str | None,
        context: str | None,
    ) -> str:
        fields = {"model": self.model_id}
        if language:
            fields["language"] = language
        if context:
            fields["prompt"] = context
        body, content_type = _multipart_body(file_path, fields)
        request = Request(
            f"{self.config.base_url}/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": content_type,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=600) as response:  # noqa: S310
                payload = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            if exc.code >= 500:
                raise BackendUnavailableError(
                    f"Remote ASR backend returned HTTP {exc.code}: {detail}"
                ) from exc
            raise ProcessingFailedError(
                f"Remote ASR request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise BackendUnavailableError(
                f"Remote ASR backend is unavailable at {self.config.base_url}: {exc}"
            ) from exc
        except (json.JSONDecodeError, TypeError) as exc:
            raise ProcessingFailedError(
                "Remote ASR backend returned an invalid JSON response"
            ) from exc

        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ProcessingFailedError("Remote ASR produced empty transcription")
        return text.strip()


def _multipart_body(file_path: Path, fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = f"----aimd-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{file_path.name}"\r\n'
            ).encode(),
            f"Content-Type: {mime_type}\r\n\r\n".encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
