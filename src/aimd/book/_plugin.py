"""MarkItDown plugin for ebook conversion."""

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

from .processor import process_book_with_images

BOOK_EXTENSIONS = {".epub", ".mobi", ".azw3"}

__plugin_interface_version__ = 1


def register_converters(markitdown: MarkItDown, **kwargs: Any) -> None:
    """Register the ebook converter with MarkItDown."""
    markitdown.register_converter(AimdBookConverter(), priority=10.0)


class AimdBookConverter(DocumentConverter):
    """Convert EPUB-like ebooks to markdown with image extraction."""

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        extension = (stream_info.extension or "").lower()
        return extension in BOOK_EXTENSIONS

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        if not stream_info.local_path:
            raise FailedConversionAttempt("aimd.book requires a local file path")

        try:
            result = process_book_with_images(
                Path(stream_info.local_path),
                output_dir=kwargs.get("output_dir"),
                temp_dir=kwargs.get("temp_dir"),
            )
        except Exception as exc:
            raise FailedConversionAttempt(f"Book conversion failed: {exc}") from exc

        return DocumentConverterResult(
            title=result.title,
            markdown=result.markdown,
        )
