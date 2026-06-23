"""MarkItDown plugin for OCR conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

from markitdown import (
    DocumentConverter,
    DocumentConverterResult,
    MarkItDown,
    StreamInfo,
)

from .processor import process_ocr_sync

OCR_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

__plugin_interface_version__ = 1


def register_converters(markitdown: MarkItDown, **kwargs: Any) -> None:
    """Register the OCR converter with MarkItDown."""
    markitdown.register_converter(AimdOCRConverter(), priority=-1.0)


class AimdOCRConverter(DocumentConverter):
    """Convert images and scanned PDFs to markdown with OCR."""

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        return (
            kwargs.get("task_type") == "ocr"
            and (stream_info.extension or "").lower() in OCR_EXTENSIONS
        )

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        result = process_ocr_sync(
            Path(stream_info.local_path),
            model=kwargs.get("model"),
            language=kwargs.get("language"),
            start=kwargs.get("start"),
            end=kwargs.get("end"),
            temp_dir=kwargs.get("temp_dir"),
        )

        return DocumentConverterResult(
            title=result.title,
            markdown="\n\n".join(result.chunk_list),
        )
