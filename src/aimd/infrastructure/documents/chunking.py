"""Chunking helpers for markdown document processing."""

import re

from logly import logger


def combine_sections_for_processing(
    section_data: list[tuple[str | None, str]], max_chunk_size: int = 40000
) -> list[str]:
    """Combine multiple sections into larger chunks to reduce API calls."""
    combined_chunks = []
    current_chunk_parts = []
    current_chunk_size = 0
    section_separator = "\n\n" + "=" * 80 + "\n\n"
    separator_size = len(section_separator)

    for _, content in section_data:
        section_size = len(content)
        would_exceed = (
            current_chunk_size
            + section_size
            + (separator_size if current_chunk_parts else 0)
        ) > max_chunk_size

        if would_exceed and current_chunk_parts:
            combined_chunks.append(section_separator.join(current_chunk_parts))
            current_chunk_parts = []
            current_chunk_size = 0

        current_chunk_parts.append(content)
        current_chunk_size += section_size
        if len(current_chunk_parts) > 1:
            current_chunk_size += separator_size

    if current_chunk_parts:
        combined_chunks.append(section_separator.join(current_chunk_parts))

    return combined_chunks


def split_text_by_paragraphs(text: str, max_chunk_size: int) -> list[str]:
    """Split text by paragraph boundaries and then hard-wrap oversized blocks."""
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not paragraphs:
        stripped = text.strip()
        return [stripped] if stripped else []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_size = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        separator = "\n\n" if current_parts else ""
        projected = current_size + len(separator) + paragraph_len
        if current_parts and projected > max_chunk_size:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_size = 0

        if paragraph_len <= max_chunk_size:
            current_parts.append(paragraph)
            current_size += (2 if current_size else 0) + paragraph_len
            continue

        if current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_size = 0

        for idx in range(0, paragraph_len, max_chunk_size):
            piece = paragraph[idx : idx + max_chunk_size].strip()
            if piece:
                chunks.append(piece)

    if current_parts:
        chunks.append("\n\n".join(current_parts))
    return chunks


def split_markdown_by_header_level(
    markdown_content: str,
    header_level: int,
) -> list[tuple[str | None, str]]:
    """Split markdown content by a specific header level."""
    header_pattern = f"^{'#' * header_level}\\s+(.+)$"
    lines = markdown_content.split("\n")

    sections = []
    current_lines = []
    current_title = None

    def _save_current_section() -> None:
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((current_title, content))

    for line in lines:
        header_match = re.match(header_pattern, line)

        if header_match:
            _save_current_section()
            current_title = header_match.group(1).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    _save_current_section()
    return sections


def split_markdown_by_headers(
    markdown_content: str,
    max_chunk_size: int = 40000,
) -> tuple[list[tuple[str | None, str]], int | None]:
    """Split markdown by best-fit header level with paragraph fallback."""
    for split_level in range(1, 7):
        sections = split_markdown_by_header_level(markdown_content, split_level)
        if len(sections) <= 1:
            continue

        all_under_limit = True
        max_section_size = 0
        for _, section_content in sections:
            section_size = len(section_content)
            max_section_size = max(max_section_size, section_size)
            if section_size > max_chunk_size:
                all_under_limit = False
                break

        if all_under_limit:
            logger.info(
                f"Using split level {split_level} - all chunks under {max_chunk_size} chars (max: {max_section_size})"
            )
            return sections, split_level

    fallback_chunks = split_text_by_paragraphs(markdown_content, max_chunk_size)
    return [(None, chunk) for chunk in fallback_chunks], None


def split_processed_chunk_into_chapters(
    processed_content: str,
    header_level: int | None = None,
) -> list[tuple[str, str]]:
    """Split processed content back into chapters using stored header level."""
    from .title_extractor import extract_title_from_content

    if header_level is not None:
        sections = split_markdown_by_header_level(processed_content, header_level)
        chapters = []
        for title, content in sections:
            if content.strip():
                clean_title = extract_title_from_content(
                    content,
                    title or "Chapter",
                    for_filename=True,
                )
                chapters.append((clean_title, content.strip()))

        if chapters:
            return chapters

    title = extract_title_from_content(processed_content, "Chapter", for_filename=True)
    return [(title, processed_content.strip())]
