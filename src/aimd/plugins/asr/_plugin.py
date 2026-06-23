"""MarkItDown plugin for local audio/video transcription."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, BinaryIO

from markitdown import (
    DocumentConverter,
    DocumentConverterResult,
    MarkItDown,
    StreamInfo,
)

from .const import AUDIO_EXTENSIONS
from .processor import transcribe_file

__plugin_interface_version__ = 1


def _transcribe_file_sync(
    file_path: Path,
    *,
    language: str | None = None,
    model: str | None = None,
    temp_dir: Path | None = None,
) -> str:
    """Synchronous MarkItDown boundary for ASR transcription."""
    return asyncio.run(
        transcribe_file(
            file_path,
            language=language,
            model=model,
            temp_dir=temp_dir,
        )
    )


def register_converters(markitdown: MarkItDown, **kwargs: Any) -> None:
    """Register the ASR converter with MarkItDown."""
    markitdown.register_converter(AimdASRConverter(), priority=-1.0)


class AimdASRConverter(DocumentConverter):
    """Convert local audio/video files to markdown transcripts."""

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        return (stream_info.extension or "").lower() in AUDIO_EXTENSIONS

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        input_path = Path(stream_info.local_path)
        transcript = _transcribe_file_sync(
            input_path,
            language=kwargs.get("language"),
            model=kwargs.get("model"),
            temp_dir=kwargs.get("temp_dir"),
        )

        return DocumentConverterResult(
            title=input_path.stem,
            markdown=transcript,
        )
