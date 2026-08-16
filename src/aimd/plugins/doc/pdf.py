"""Text-layer PDF conversion through pdf-inspector.

Text-layer PDFs are routed as ordinary document conversions; pdf-inspector
extracts position-aware Markdown locally without OCR. Scanned PDFs produce no
Markdown here and surface a domain error so MarkItDown can fall through to the
next converter.
"""

from __future__ import annotations

from pathlib import Path

from aimd.core.errors import (
    BackendUnavailableError,
    InputNotFoundError,
    ProcessingFailedError,
)


def process_pdf_text(input_path: str | Path) -> tuple[str | None, str]:
    """Extract a title and Markdown from a text-layer PDF via pdf-inspector."""
    path = Path(input_path)
    if not path.exists():
        raise InputNotFoundError(f"Document file not found: {input_path}")
    try:
        import pdf_inspector
    except ImportError as exc:
        raise BackendUnavailableError(
            "pdf-inspector is required for text-layer PDF conversion."
        ) from exc
    try:
        result = pdf_inspector.process_pdf(str(path))
    except Exception as exc:
        raise ProcessingFailedError(
            f"pdf-inspector failed to process PDF: {path.name}"
        ) from exc
    markdown = (result.markdown or "").strip()
    if not markdown:
        raise ProcessingFailedError(
            f"pdf-inspector extracted no text from PDF: {path.name}"
        )
    return result.title, markdown
