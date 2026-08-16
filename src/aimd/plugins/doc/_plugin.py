"""MarkItDown plugin for document conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

from markitdown import (
    DocumentConverter,
    DocumentConverterResult,
    MarkItDown,
    StreamInfo,
)

from .conversion import PANDOC_DOCUMENT_EXTENSIONS, process_doc_with_assets
from .pdf import process_pdf_text

__plugin_interface_version__ = 1


def register_converters(markitdown: MarkItDown, **kwargs: Any) -> None:
    """Register the document converters with MarkItDown."""
    markitdown.register_converter(AimdPdfConverter(), priority=-1.0)
    markitdown.register_converter(AimdDocConverter(), priority=10.0)


class AimdPdfConverter(DocumentConverter):
    """Convert text-layer PDFs to Markdown with pdf-inspector."""

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        return (
            kwargs.get("task_type") != "ocr"
            and (stream_info.extension or "").lower() == ".pdf"
            and bool(stream_info.local_path)
        )

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        title, markdown = process_pdf_text(Path(stream_info.local_path))
        return DocumentConverterResult(title=title, markdown=markdown)


class AimdDocConverter(DocumentConverter):
    """Convert Pandoc-supported local documents to Markdown."""

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        return (stream_info.extension or "").lower() in PANDOC_DOCUMENT_EXTENSIONS

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        result = process_doc_with_assets(
            Path(stream_info.local_path),
            output_dir=kwargs.get("output_dir"),
            temp_dir=kwargs.get("temp_dir"),
            cancellation_check=kwargs.get("cancellation_check"),
            progress_reporter=kwargs.get("progress_reporter"),
        )

        return DocumentConverterResult(
            title=result.title,
            markdown=result.markdown,
        )
