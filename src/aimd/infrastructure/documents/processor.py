"""Document processing orchestration."""

from pathlib import Path

from logly import logger

from ...errors import InputNotFoundError, UnsupportedInputError
from ...types import TextContext
from .epub_processor import process_epub_with_images
from .pandoc_reader import is_supported_file, process_file_with_splitting


async def get_text_from_file(
    file_path: str | Path, max_chunk_size: int = 40000
) -> TextContext:
    """Extract text from file using pandoc for various formats."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise InputNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise UnsupportedInputError(f"Path is not a file: {file_path}")

    file_extension = file_path.suffix.lower()
    logger.info(f"Processing {file_extension} file: {file_path}")

    title, chunk_list, split_header_level = await process_file_with_splitting(
        file_path,
        max_chunk_size=max_chunk_size,
    )
    return TextContext(
        title=title,
        chunk_list=chunk_list,
        split_header_level=split_header_level,
    )


__all__ = ["get_text_from_file", "is_supported_file", "process_epub_with_images"]
