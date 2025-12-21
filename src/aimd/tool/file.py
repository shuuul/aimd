"""File processing tools for extracting text from various document formats."""

import asyncio
import re
import shutil
import tempfile
import warnings
import zipfile
from pathlib import Path

import pandoc
from logly import logger

from ..types import TextContext


def extract_title_from_content(
    content: str, fallback_title: str = "Untitled", for_filename: bool = False
) -> str:
    """Extract and clean title from content with unified logic.

    Args:
        content: Text content to extract title from
        fallback_title: Title to use if none found
        for_filename: Apply additional filename-safe cleaning

    Returns:
        Clean title string
    """
    if not content or not content.strip():
        return fallback_title

    lines = content.strip().split("\n")
    extracted_title = None

    # Look for first H1 heading (# Title)
    for line in lines:
        line = line.strip()
        if line.startswith("# "):
            extracted_title = line[2:].strip()
            break

    # Look for title in frontmatter (YAML) if no H1 found
    if not extracted_title and content.strip().startswith("---"):
        in_frontmatter = False
        for line in lines:
            if line.strip() == "---":
                if in_frontmatter:
                    break
                in_frontmatter = True
                continue
            if in_frontmatter and line.strip().startswith("title:"):
                title_match = re.match(r'title:\s*["\']?([^"\']+)["\']?', line.strip())
                if title_match:
                    extracted_title = title_match.group(1).strip()
                    break

    # Look for setext-style headings if still no title
    if not extracted_title:
        for i, line in enumerate(lines):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and (
                    all(c == "=" for c in next_line) or all(c == "-" for c in next_line)
                ):
                    extracted_title = line.strip()
                    break

    # Look for first meaningful line if still no title
    if not extracted_title:
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if not line:
                continue
            # Skip image lines, links, and other markdown artifacts
            if (
                line.startswith("![")
                or line.startswith("[]{")
                or line.startswith(":::")
                or line.startswith("<div")
                or line.startswith("</div")
                or "calibre" in line.lower()
                or "kindle-cn" in line.lower()
            ):
                continue
            if (
                len(line) >= 2
                and len(line) <= 100
                and not line.lower().startswith("http")
            ):
                extracted_title = line
                break

    # Use fallback if no title found
    if not extracted_title:
        return fallback_title

    # Apply consistent cleaning - optimized regex patterns
    # 1. Remove markdown headers
    clean_text = re.sub(r"^#+\s*", "", extracted_title)

    # 2. Remove bold/italic and links (preserve inner text)
    clean_text = re.sub(r"\*+([^*]+)\*+", r"\1", clean_text)  # bold/italic
    clean_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean_text)  # links

    # 3. Remove all curly brace patterns in one go
    clean_text = re.sub(r"\{[^}]*\}", "", clean_text)

    # 4. Remove all footnote patterns comprehensively
    clean_text = re.sub(
        r"\[\^[^\]]*\](?:\([^)]*\))?", "", clean_text
    )  # [^...] with optional (...)
    clean_text = re.sub(r"\^[^\]]*\]", "", clean_text)  # remaining ^...]

    # 5. Clean up anchors, quotes, and remaining artifacts
    clean_text = re.sub(r"#[a-zA-Z0-9_.-]+", "", clean_text)  # anchor links
    clean_text = re.sub(
        r"\([^)]*\)", "", clean_text
    )  # remove any remaining parentheses content
    clean_text = re.sub(
        r'^["""\'\']+|["""\'\']+$', "", clean_text
    )  # quotes at start/end

    # 6. Final cleanup: normalize whitespace and remove trailing punctuation
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    clean_text = re.sub(r"[。，、；：！？]+$", "", clean_text)

    # Additional filename-safe cleaning if requested
    if for_filename:
        clean_text = re.sub(r'[<>:"/\\|?*]', "", clean_text)  # Remove unsafe chars
        clean_text = re.sub(
            r"\s+", "_", clean_text.strip()
        )  # Replace spaces with underscores
        # Truncate if too long
        if len(clean_text) > 50:
            clean_text = clean_text[:50].rstrip("_")

    return clean_text or fallback_title


# Suppress pandoc version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandoc.utils")


async def get_text_from_file(
    file_path: str | Path, max_chunk_size: int = 40000
) -> TextContext:
    """Extract text from file using pandoc for various formats.

    Automatically splits documents into chunks under max_chunk_size characters.
    Uses dynamic header-based splitting when possible, otherwise reports error.

    Args:
        file_path: Path to the document file

    Returns:
        TextContext with title and chunk_list

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If pandoc conversion fails or document cannot be split properly
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    file_extension = file_path.suffix.lower()
    logger.info(f"Processing {file_extension} file: {file_path}")

    # Process all file types with unified logic
    return await _process_file_with_splitting(file_path, max_chunk_size)


async def _process_file_with_splitting(
    file_path: Path, max_chunk_size: int = 40000
) -> TextContext:
    """Process any supported file with dynamic splitting to keep chunks under max_chunk_size characters.

    Args:
        file_path: Path to document file
        max_chunk_size: Maximum characters per chunk

    Returns:
        TextContext with title and text sections

    Raises:
        RuntimeError: If document cannot be split properly under the size limit
    """
    file_extension = file_path.suffix.lower()

    try:
        # Run pandoc processing in thread pool to avoid blocking
        def _read_and_process():
            # Get pandoc format from internal mapping
            pandoc_format = pandoc._ext_to_file_format.get(file_extension)
            if not pandoc_format:
                raise RuntimeError(
                    f"No pandoc format found for extension: {file_extension}"
                )

            # Read document using pandoc consistently for all file types
            try:
                doc = pandoc.read(file=str(file_path), format=pandoc_format)
                if doc is None:
                    raise RuntimeError(
                        f"Pandoc failed to read {pandoc_format}: {file_path}"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read {file_extension} file with pandoc: {e}"
                ) from e

            # Convert to markdown
            markdown_content = pandoc.write(doc, format="markdown")
            if not markdown_content:
                raise RuntimeError(f"Pandoc produced empty markdown from: {file_path}")

            return markdown_content

        loop = asyncio.get_event_loop()
        full_markdown = await loop.run_in_executor(None, _read_and_process)

        # Check if content is under the limit
        if len(full_markdown) <= max_chunk_size:
            # Document is small enough, no splitting needed
            logger.info(
                f"Document under {max_chunk_size} chars, no splitting needed: {len(full_markdown)} characters"
            )
            extracted_title = extract_title_from_content(full_markdown, file_path.stem)
            return TextContext(
                title=extracted_title,
                chunk_list=[full_markdown.strip()] if full_markdown.strip() else [],
            )

        # Document needs splitting

        # Split the markdown by headers
        sections, header_level = _split_markdown_by_headers(
            full_markdown, max_chunk_size=max_chunk_size
        )

        # Collect all sections without filtering
        section_data = []
        for title, section_content in sections:
            if section_content.strip():
                section_data.append((title, section_content.strip()))

        if not section_data:
            raise RuntimeError(
                f"Document could not be split into meaningful sections under {max_chunk_size} characters"
            )

        logger.info(f"Successfully split document into {len(section_data)} sections")

        # Combine sections into larger chunks to reduce API calls and respect token limits
        combined_chunks = _combine_sections_for_processing(section_data, max_chunk_size)

        logger.info(
            f"Combined {len(section_data)} sections into {len(combined_chunks)} chunks for processing"
        )

        # Use file name as title (titles will be extracted after AI processing)
        extracted_title = file_path.stem

        return TextContext(
            title=extracted_title,
            chunk_list=combined_chunks,
            split_header_level=header_level,
        )

    except Exception as e:
        # Enhanced error logging with debugging information
        logger.error(
            f"Failed to process file {file_path}: {e}",
            extra={
                "file_path": str(file_path),
                "file_extension": file_extension,
                "file_size": file_path.stat().st_size
                if file_path.exists()
                else "unknown",
                "max_chunk_size": max_chunk_size,
                "error_type": type(e).__name__,
                "error_details": str(e),
            },
        )
        raise RuntimeError(f"Document processing failed: {e}") from e


def _combine_sections_for_processing(
    section_data: list[tuple[str | None, str]], max_chunk_size: int = 40000
) -> list[str]:
    """Combine multiple sections into larger chunks to reduce API calls and respect token limits.

    Args:
        section_data: List of (title, content) tuples
        max_chunk_size: Maximum characters per combined chunk (considers token limits)

    Returns:
        List of combined text chunks ready for processing
    """
    combined_chunks = []
    current_chunk_parts = []
    current_chunk_size = 0

    # Add separator between sections
    section_separator = "\n\n" + "=" * 80 + "\n\n"
    separator_size = len(section_separator)

    for _, content in section_data:
        section_size = len(content)

        # Check if adding this section would exceed the limit
        would_exceed = (
            current_chunk_size
            + section_size
            + (separator_size if current_chunk_parts else 0)
        ) > max_chunk_size

        if would_exceed and current_chunk_parts:
            # Save current chunk and start a new one
            combined_chunk = section_separator.join(current_chunk_parts)
            combined_chunks.append(combined_chunk)

            # Reset for new chunk
            current_chunk_parts = []
            current_chunk_size = 0

        # Add section to current chunk
        current_chunk_parts.append(content)
        current_chunk_size += section_size
        if len(current_chunk_parts) > 1:
            current_chunk_size += separator_size

    # Don't forget the last chunk
    if current_chunk_parts:
        combined_chunk = section_separator.join(current_chunk_parts)
        combined_chunks.append(combined_chunk)

    return combined_chunks


def _split_markdown_by_headers(
    markdown_content: str, max_chunk_size: int = 40000
) -> tuple[list[tuple[str | None, str]], int | None]:
    """Split markdown content by headers to keep chunks under max_chunk_size.

    Tries different header levels (1-6) until chunks are small enough.
    If no split level works, raises an error.

    Args:
        markdown_content: Markdown text to split
        max_chunk_size: Maximum characters per chunk

    Returns:
        Tuple of (list of (section_title, section_content), header_level_used)

    Raises:
        RuntimeError: If no split level produces chunks under the size limit
    """
    # Try different header levels, starting with H1 (level 1)
    for split_level in range(1, 7):  # Try H1 through H6
        sections = _split_markdown_by_header_level(markdown_content, split_level)

        # Skip if no sections found or only one section
        if len(sections) <= 1:
            continue

        # Check if all sections are under the size limit
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

    # If no header level works, raise an error
    raise RuntimeError(
        f"Document cannot be split into chunks under {max_chunk_size} characters using any header level (H1-H6). "
        f"The document structure may not be suitable for automatic splitting."
    )


def _split_markdown_by_header_level(
    markdown_content: str, header_level: int
) -> list[tuple[str | None, str]]:
    """Split markdown content by a specific header level.

    Args:
        markdown_content: Markdown text to split
        header_level: Header level to split on (1-6)

    Returns:
        List of tuples: (section_title, section_content)
    """
    # Create regex pattern for the specified header level
    header_pattern = f"^{'#' * header_level}\\s+(.+)$"
    lines = markdown_content.split("\n")

    sections = []
    current_lines = []
    current_title = None

    def _save_current_section():
        """Save current section if it has content."""
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((current_title, content))

    for line in lines:
        header_match = re.match(header_pattern, line)

        if header_match:
            # Save previous section and start new one
            _save_current_section()
            current_title = header_match.group(1).strip()
            current_lines = [line]  # Include header in section
        else:
            current_lines.append(line)

    # Save the final section
    _save_current_section()

    return sections


def is_supported_file(file_path: str | Path) -> bool:
    """Check if file is supported by pandoc.

    Args:
        file_path: Path to check

    Returns:
        True if file has supported extension
    """
    file_path = Path(file_path)

    if isinstance(file_path, str):
        if file_path.startswith(("http://", "https://")):
            return False

    # Get all supported extensions from pandoc's internal mapping
    supported_extensions = set(pandoc._ext_to_file_format.keys())
    return file_path.suffix.lower() in supported_extensions


async def process_epub_with_images(
    file_path: str | Path, output_dir: Path | None = None
) -> tuple[TextContext, Path]:
    """Process EPUB file with image extraction.

    Extracts images and converts each chapter to individual markdown files.
    Creates an output directory with the following structure:
        output_dir/
            chapter_001.md
            chapter_002.md
            ...
            images/
                *.jpg, *.png, etc.
            full.txt

    Args:
        file_path: Path to the EPUB file
        output_dir: Directory to save extracted content (defaults to EPUB location)

    Returns:
        Tuple of (TextContext with title and chunks, path to output directory)

    Raises:
        FileNotFoundError: If EPUB file doesn't exist
        RuntimeError: If EPUB processing fails
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"EPUB file not found: {file_path}")

    if output_dir is None:
        output_dir = file_path.parent / file_path.stem

    # Create output directory structure
    chapters_dir = output_dir / "chapters"
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    chapters_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # EPUB is a ZIP file - copy and unzip
        zip_path = temp_path / "epub.zip"
        shutil.copy(file_path, zip_path)

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_path)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Invalid EPUB file (not a valid ZIP): {e}") from e

        # Find OEBPS folder (standard EPUB structure)
        oebps_dir = None
        for item in temp_path.iterdir():
            if item.is_dir() and item.name in ("OEBPS", "OPS"):
                oebps_dir = item
                break

        # Fallback: search for any directory containing xhtml/html files
        if oebps_dir is None:
            for item in temp_path.rglob("*.xhtml"):
                oebps_dir = item.parent
                break
            for item in temp_path.rglob("*.html"):
                oebps_dir = item.parent
                break

        if oebps_dir is None:
            raise RuntimeError("Could not find OEBPS/OPS folder in EPUB")

        # Find HTML/XHTML files
        html_files = []
        for ext in ("*.xhtml", "*.html"):
            html_files.extend(oebps_dir.glob(ext))

        if not html_files:
            raise RuntimeError("No HTML/XHTML files found in EPUB")

        # Copy images from OEBPS and its subdirectories
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp"}
        for ext in image_extensions:
            for img_path in oebps_dir.rglob(ext):
                rel_path = img_path.relative_to(oebps_dir)
                dest_path = images_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_path, dest_path)

        # Convert each HTML file to markdown and save in output directory root
        chapter_files = []
        for html_file in sorted(html_files):
            try:
                doc = pandoc.read(file=str(html_file), format="html")
                markdown_content = pandoc.write(doc, format="markdown")

                # Create chapter filename
                chapter_idx = len(chapter_files) + 1
                chapter_name = f"chapter_{chapter_idx:03d}"
                md_path = chapters_dir / f"{chapter_name}.md"

                # Write markdown file
                md_path.write_text(markdown_content, encoding="utf-8")
                chapter_files.append((chapter_name, markdown_content))

            except Exception as e:
                logger.warning(f"Failed to convert {html_file}: {e}")
                continue

        if not chapter_files:
            raise RuntimeError("Failed to convert any HTML files to markdown")

        # Extract title from first chapter
        first_chapter_content = chapter_files[0][1]
        title = extract_title_from_content(first_chapter_content, file_path.stem)

        # Combine all chapters into single content for TextContext
        combined_content = "\n\n".join(
            f"# {name}\n\n{content}" for name, content in chapter_files
        )

        # Create combined markdown file
        full_md_path = output_dir / f"{file_path.stem}.md"
        full_md_path.write_text(combined_content, encoding="utf-8")

        logger.info(
            f"EPUB extracted to {output_dir}: {len(chapter_files)} chapters, "
            f"{len(list(images_dir.rglob('*')))} images"
        )

        # Return TextContext with combined content
        text_context = TextContext(
            title=title,
            chunk_list=[combined_content] if len(combined_content) <= 40000 else [],
        )

        return text_context, output_dir


def split_processed_chunk_into_chapters(
    processed_content: str, header_level: int | None = None
) -> list[tuple[str, str]]:
    """Split processed AI content back into individual chapters using a specific header level.

    Args:
        processed_content: AI-processed text content
        header_level: Specific header level to use for splitting (1-6), None for auto-detection

    Returns:
        List of (chapter_title, chapter_content) tuples
    """
    if header_level is not None:
        # Use the specific header level from pre-processing
        sections = _split_markdown_by_header_level(processed_content, header_level)

        # Convert sections to chapters with clean titles
        chapters = []
        for title, content in sections:
            if content.strip():
                # Clean the title for filename use
                clean_title = extract_title_from_content(
                    content, title or "Chapter", for_filename=True
                )
                chapters.append((clean_title, content.strip()))

        if chapters:
            return chapters

    # Fallback: if no header level specified or no sections found, return as single chapter
    title = extract_title_from_content(processed_content, "Chapter", for_filename=True)
    return [(title, processed_content.strip())]
