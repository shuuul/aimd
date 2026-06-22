"""MarkItDown plugin for Pandoc-backed document conversion."""

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

from .processor import PANDOC_DOCUMENT_EXTENSIONS, process_doc_with_assets

__plugin_interface_version__ = 1


def register_converters(markitdown: MarkItDown, **kwargs: Any) -> None:
    """Register the Pandoc-backed document converter with MarkItDown."""
    markitdown.register_converter(AimdDocConverter(), priority=10.0)


class AimdDocConverter(DocumentConverter):
    """Convert Pandoc-supported local documents to Markdown."""

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        extension = (stream_info.extension or "").lower()
        return extension in PANDOC_DOCUMENT_EXTENSIONS

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        if not stream_info.local_path:
            raise FailedConversionAttempt("aimd.plugins.doc requires a local file path")

        try:
            result = process_doc_with_assets(
                Path(stream_info.local_path),
                output_dir=kwargs.get("output_dir"),
                temp_dir=kwargs.get("temp_dir"),
            )
        except Exception as exc:
            raise FailedConversionAttempt(f"Document conversion failed: {exc}") from exc

        return DocumentConverterResult(
            title=result.title,
            markdown=result.markdown,
        )
