"""MarkItDown plugin for local audio/video transcription."""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

from markitdown import (
    DocumentConverter,
    DocumentConverterResult,
    FailedConversionAttempt,
    MarkItDown,
    StreamInfo,
)

from .const import AUDIO_EXTENSIONS
from .processor import transcribe_file_sync

__plugin_interface_version__ = 1


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
        extension = (stream_info.extension or "").lower()
        return extension in AUDIO_EXTENSIONS

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        if not stream_info.local_path:
            raise FailedConversionAttempt("aimd.plugins.asr requires a local file path")

        input_path = Path(stream_info.local_path)
        try:
            transcript = transcribe_file_sync(
                input_path,
                language=kwargs.get("language"),
                model=kwargs.get("model"),
                temp_dir=kwargs.get("temp_dir"),
            )
        except Exception as exc:
            raise FailedConversionAttempt(f"ASR conversion failed: {exc}") from exc

        return DocumentConverterResult(
            title=input_path.stem,
            markdown=transcript,
        )
