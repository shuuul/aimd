"""Pandoc-backed document conversion."""

import asyncio
import warnings
from pathlib import Path

import pandoc
from logly import logger

from ...errors import ProcessingFailedError
from .chunking import combine_sections_for_processing, split_markdown_by_headers
from .title_extractor import extract_title_from_content

warnings.filterwarnings("ignore", category=UserWarning, module="pandoc.utils")


async def process_file_with_splitting(
    file_path: Path,
    max_chunk_size: int = 40000,
) -> tuple[str, list[str], int | None]:
    """Convert a document with pandoc and split into processing chunks."""
    file_extension = file_path.suffix.lower()

    def _read_and_process() -> str:
        pandoc_format = pandoc._ext_to_file_format.get(file_extension)
        if not pandoc_format:
            raise ProcessingFailedError(
                f"No pandoc format found for extension: {file_extension}"
            )

        try:
            doc = pandoc.read(file=str(file_path), format=pandoc_format)
            if doc is None:
                raise ProcessingFailedError(
                    f"Pandoc failed to read {pandoc_format}: {file_path}"
                )
        except Exception as e:
            raise ProcessingFailedError(
                f"Failed to read {file_extension} file with pandoc: {e}"
            ) from e

        markdown_content = pandoc.write(doc, format="markdown")
        if not markdown_content:
            raise ProcessingFailedError(f"Pandoc produced empty markdown from: {file_path}")

        return markdown_content

    try:
        loop = asyncio.get_event_loop()
        full_markdown = await loop.run_in_executor(None, _read_and_process)

        if len(full_markdown) <= max_chunk_size:
            logger.info(
                f"Document under {max_chunk_size} chars, no splitting needed: {len(full_markdown)} characters"
            )
            extracted_title = extract_title_from_content(full_markdown, file_path.stem)
            chunk_list = [full_markdown.strip()] if full_markdown.strip() else []
            return extracted_title, chunk_list, None

        sections, header_level = split_markdown_by_headers(
            full_markdown,
            max_chunk_size=max_chunk_size,
        )
        section_data = [
            (title, section_content.strip())
            for title, section_content in sections
            if section_content.strip()
        ]

        if not section_data:
            raise ProcessingFailedError(
                f"Document could not be split into meaningful sections under {max_chunk_size} characters"
            )

        logger.info(f"Successfully split document into {len(section_data)} sections")
        combined_chunks = combine_sections_for_processing(section_data, max_chunk_size)
        logger.info(
            f"Combined {len(section_data)} sections into {len(combined_chunks)} chunks for processing"
        )

        return file_path.stem, combined_chunks, header_level
    except Exception as e:
        if isinstance(e, ProcessingFailedError):
            raise
        logger.error(
            f"Failed to process file {file_path}: {e}",
            extra={
                "file_path": str(file_path),
                "file_extension": file_extension,
                "file_size": file_path.stat().st_size if file_path.exists() else "unknown",
                "max_chunk_size": max_chunk_size,
                "error_type": type(e).__name__,
                "error_details": str(e),
            },
        )
        raise ProcessingFailedError(f"Document processing failed: {e}") from e


def is_supported_file(file_path: str | Path) -> bool:
    """Check if file extension is supported by pandoc."""
    if isinstance(file_path, str):
        if file_path.startswith(("http://", "https://")):
            return False
        file_path = Path(file_path)
    else:
        file_path = Path(file_path)

    supported_extensions = set(pandoc._ext_to_file_format.keys())
    return file_path.suffix.lower() in supported_extensions
