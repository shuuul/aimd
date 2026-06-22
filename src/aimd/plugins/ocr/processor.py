"""OCR processing entrypoint."""

import asyncio
from pathlib import Path

from aimd.core.errors import InputNotFoundError, ProcessingFailedError
from aimd.core.models import TextContext

from .backends import OCRPage, create_ocr_backend


def _page_to_markdown(page: OCRPage, total_pages: int) -> str:
    text = page.text.strip()
    if total_pages <= 1 or page.page_index is None:
        return text
    return f"## Page {page.page_index + 1}\n\n{text}".strip()


async def process_ocr(
    input_path: str | Path,
    model: str | None = None,
    language: str | None = None,
    start: int | None = None,
    end: int | None = None,
    temp_dir: Path | None = None,
) -> TextContext:
    """Process a scanned PDF or image with OCR and return TextContext."""
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
    chunks = [_page_to_markdown(page, len(result.pages)) for page in result.pages]
    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        raise ProcessingFailedError("OCR returned empty content")
    return TextContext(title=result.title, chunk_list=chunks, split_header_level=2)


def process_ocr_sync(
    input_path: str | Path,
    model: str | None = None,
    language: str | None = None,
    start: int | None = None,
    end: int | None = None,
    temp_dir: Path | None = None,
) -> TextContext:
    """Synchronous MarkItDown boundary for OCR conversion."""
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
