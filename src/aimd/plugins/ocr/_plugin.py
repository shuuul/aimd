"""MarkItDown plugin for OCR conversion."""

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

from aimd.core.errors import InputNotFoundError, ProcessingFailedError
from aimd.core.models import TextContext

from .backends import OCRPage, OCRResult, create_ocr_backend
from .const import OCR_EXTENSIONS

__plugin_interface_version__ = 1


def _page_to_markdown(page: OCRPage, total_pages: int) -> str:
    text = page.text.strip()
    if total_pages <= 1 or page.page_index is None:
        return text
    return f"## Page {page.page_index + 1}\n\n{text}".strip()


def _ocr_result_chunks(result: OCRResult) -> list[str]:
    """Build non-empty page-oriented markdown chunks from an OCR result."""
    chunks = [_page_to_markdown(page, len(result.pages)) for page in result.pages]
    return [chunk for chunk in chunks if chunk]


def _ocr_result_to_markdown(result: OCRResult) -> tuple[str, str]:
    """Build title + markdown from a backend OCR result."""
    chunks = _ocr_result_chunks(result)
    if not chunks:
        raise ProcessingFailedError("OCR returned empty content")
    return result.title, "\n\n".join(chunks)


async def _recognize_ocr_result(
    input_path: str | Path,
    model: str | None = None,
    language: str | None = None,
    start: int | None = None,
    end: int | None = None,
    temp_dir: Path | None = None,
) -> OCRResult:
    """Validate an OCR request and return the backend result."""
    path = Path(input_path)
    if not path.exists():
        raise InputNotFoundError(f"Input file not found: {input_path}")
    if start is not None and start < 0:
        raise ProcessingFailedError(
            "OCR start page must be greater than or equal to 0."
        )
    if end is not None and end < 0:
        raise ProcessingFailedError("OCR end page must be greater than or equal to 0.")
    if start is not None and end is not None and start > end:
        raise ProcessingFailedError(
            "OCR start page must be less than or equal to end page."
        )

    backend = create_ocr_backend()
    result = await asyncio.to_thread(
        backend.recognize,
        path,
        model=model,
        language=language,
        start=start,
        end=end,
        temp_dir=temp_dir,
    )
    if not any(page.text.strip() for page in result.pages):
        raise ProcessingFailedError("OCR returned empty content")
    return result


def _recognize_ocr_sync(
    input_path: str | Path,
    model: str | None = None,
    language: str | None = None,
    start: int | None = None,
    end: int | None = None,
    temp_dir: Path | None = None,
) -> DocumentConverterResult:
    """Synchronous MarkItDown boundary: title + markdown only."""
    result = asyncio.run(
        _recognize_ocr_result(
            input_path,
            model=model,
            language=language,
            start=start,
            end=end,
            temp_dir=temp_dir,
        )
    )
    title, markdown = _ocr_result_to_markdown(result)
    return DocumentConverterResult(title=title, markdown=markdown)


async def process_ocr(
    input_path: str | Path,
    model: str | None = None,
    language: str | None = None,
    start: int | None = None,
    end: int | None = None,
    temp_dir: Path | None = None,
) -> TextContext:
    """Compatibility API returning the original page-oriented TextContext."""
    result = await _recognize_ocr_result(
        input_path,
        model=model,
        language=language,
        start=start,
        end=end,
        temp_dir=temp_dir,
    )
    return TextContext(
        title=result.title,
        chunk_list=_ocr_result_chunks(result),
        split_header_level=2,
    )


def process_ocr_sync(
    input_path: str | Path,
    model: str | None = None,
    language: str | None = None,
    start: int | None = None,
    end: int | None = None,
    temp_dir: Path | None = None,
) -> TextContext:
    """Synchronous compatibility wrapper around process_ocr."""
    return asyncio.run(
        process_ocr(
            input_path,
            model=model,
            language=language,
            start=start,
            end=end,
            temp_dir=temp_dir,
        )
    )


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
        return _recognize_ocr_sync(
            Path(stream_info.local_path),
            model=kwargs.get("model"),
            language=kwargs.get("language"),
            start=kwargs.get("start"),
            end=kwargs.get("end"),
            temp_dir=kwargs.get("temp_dir"),
        )
