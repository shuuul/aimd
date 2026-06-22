"""MarkItDown-backed local file conversion."""

import asyncio
from functools import partial
from pathlib import Path

from markitdown import MarkItDown

from ..const import BOOK_EXTENSIONS, MARKITDOWN_FILE_EXTENSIONS
from ..errors import InputNotFoundError, UnsupportedInputError
from ..types import TextContext
from .documents.chunking import (
    combine_sections_for_processing,
    split_markdown_by_headers,
)
from .documents.title_extractor import extract_title_from_content


def is_supported_file(file_path: str | Path) -> bool:
    """Return whether a local file extension should be offered to MarkItDown."""
    if isinstance(file_path, str) and file_path.startswith(("http://", "https://")):
        return False
    return Path(file_path).suffix.lower() in MARKITDOWN_FILE_EXTENSIONS


def _text_context_from_markdown(
    markdown: str,
    fallback_title: str,
    title: str | None,
    max_chunk_size: int,
) -> TextContext:
    """Convert MarkItDown markdown output into aimd's TextContext shape."""
    resolved_title = title or extract_title_from_content(markdown, fallback_title)
    stripped = markdown.strip()
    if len(stripped) <= max_chunk_size:
        return TextContext(
            title=resolved_title,
            chunk_list=[stripped] if stripped else [],
            split_header_level=None,
        )

    sections, header_level = split_markdown_by_headers(
        stripped,
        max_chunk_size=max_chunk_size,
    )
    section_data = [
        (section_title, section_content.strip())
        for section_title, section_content in sections
        if section_content.strip()
    ]
    chunks = combine_sections_for_processing(section_data, max_chunk_size)
    return TextContext(
        title=resolved_title,
        chunk_list=chunks,
        split_header_level=header_level,
    )


async def convert_file_with_markitdown(
    file_path: str | Path,
    transcribe_engine: str = "auto",
    language: str | None = None,
    model: str | None = None,
    temp_dir: Path | None = None,
    *,
    max_chunk_size: int = 40000,
) -> tuple[TextContext, Path | None]:
    """Convert a local file through MarkItDown and installed aimd plugins."""
    input_path = Path(file_path)
    if not input_path.exists():
        raise InputNotFoundError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise UnsupportedInputError(f"Path is not a file: {input_path}")

    suffix = input_path.suffix.lower()
    output_dir = (
        input_path.parent / input_path.stem if suffix in BOOK_EXTENSIONS else None
    )

    md = MarkItDown(enable_plugins=True)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        partial(
            md.convert,
            input_path,
            transcribe_engine=transcribe_engine,
            language=language,
            model=model,
            temp_dir=temp_dir,
            output_dir=output_dir,
        ),
    )
    markdown = result.markdown
    return (
        _text_context_from_markdown(
            markdown,
            fallback_title=input_path.stem,
            title=result.title,
            max_chunk_size=max_chunk_size,
        ),
        output_dir,
    )
