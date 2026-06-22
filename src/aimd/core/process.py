"""Core input processing orchestration."""

import asyncio
from functools import partial
import io
from pathlib import Path
import re
from typing import Awaitable, Callable

from logly import logger
from markitdown import MarkItDown, StreamInfo

from aimd.plugins.asr.const import AUDIO_EXTENSIONS
from .errors import InputNotFoundError, ProcessingFailedError, UnsupportedInputError
from .models import InputRoute, ProcessInput, ProcessResult, TaskType, TextContext
from .router import FileSupportChecker, ensure_supported_input as ensure_supported_route

from aimd.plugins.asr import resolve_engine_with_preflight
from aimd.plugins.url import detect_platform
from aimd.plugins.doc import PANDOC_DOCUMENT_EXTENSIONS

_DOCUMENT_ASSET_EXTENSIONS = {".docx", ".epub", ".odt"}
_MARKITDOWN_FILE_EXTENSIONS = AUDIO_EXTENSIONS | PANDOC_DOCUMENT_EXTENSIONS | {
    ".doc",
    ".pdf",
    ".ppt",
    ".pptx",
    ".txt",
    ".xls",
    ".xlsx",
}

UrlProcessor = Callable[
    [
        str,
        str,
        str | None,
        str | None,
        Path | None,
        str | None,
        str | None,
        Path | None,
        bool,
    ],
    Awaitable[tuple[TextContext, str]],
]

LocalFileProcessor = Callable[
    [str, str, str | None, str | None, Path | None, str | None, int | None, int | None],
    Awaitable[tuple[TextContext, Path | None]],
]


def is_supported_file(file_path: str | Path) -> bool:
    """Return whether a local file extension should be offered to MarkItDown."""
    if isinstance(file_path, str) and file_path.startswith(("http://", "https://")):
        return False
    return Path(file_path).suffix.lower() in _MARKITDOWN_FILE_EXTENSIONS


def _extract_title_from_content(
    content: str, fallback_title: str = "Untitled", for_filename: bool = False
) -> str:
    """Extract and clean title from markdown text."""
    if not content or not content.strip():
        return fallback_title

    lines = content.strip().split("\n")
    extracted_title = None

    for line in lines:
        line = line.strip()
        if line.startswith("# "):
            extracted_title = line[2:].strip()
            break

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

    if not extracted_title:
        for i, line in enumerate(lines):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and (
                    all(c == "=" for c in next_line) or all(c == "-" for c in next_line)
                ):
                    extracted_title = line.strip()
                    break

    if not extracted_title:
        for line in lines[:5]:
            line = line.strip()
            if not line:
                continue
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
            if 2 <= len(line) <= 100 and not line.lower().startswith("http"):
                extracted_title = line
                break

    if not extracted_title:
        return fallback_title

    clean_text = re.sub(r"^#+\s*", "", extracted_title)
    clean_text = re.sub(r"\*+([^*]+)\*+", r"\1", clean_text)
    clean_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean_text)
    clean_text = re.sub(r"\{[^}]*\}", "", clean_text)
    clean_text = re.sub(r"\[\^[^\]]*\](?:\([^)]*\))?", "", clean_text)
    clean_text = re.sub(r"\^[^\]]*\]", "", clean_text)
    clean_text = re.sub(r"#[a-zA-Z0-9_.-]+", "", clean_text)
    clean_text = re.sub(r"\([^)]*\)", "", clean_text)
    clean_text = re.sub(r'^["""\'\']+|["""\'\']+$', "", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    clean_text = re.sub(r"[。，、；：！？]+$", "", clean_text)

    if for_filename:
        clean_text = re.sub(r'[<>:"/\\|?*]', "", clean_text)
        clean_text = re.sub(r"\s+", "_", clean_text.strip())
        if len(clean_text) > 50:
            clean_text = clean_text[:50].rstrip("_")

    return clean_text or fallback_title


def _combine_sections_for_processing(
    section_data: list[tuple[str | None, str]], max_chunk_size: int = 40000
) -> list[str]:
    """Combine multiple sections into larger chunks."""
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


def _split_text_by_paragraphs(text: str, max_chunk_size: int) -> list[str]:
    """Split text by paragraph boundaries and then hard-wrap oversized blocks."""
    paragraphs = [
        block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()
    ]
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


def _split_markdown_by_header_level(
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


def _split_markdown_by_headers(
    markdown_content: str,
    max_chunk_size: int = 40000,
) -> tuple[list[tuple[str | None, str]], int | None]:
    """Split markdown by best-fit header level with paragraph fallback."""
    for split_level in range(1, 7):
        sections = _split_markdown_by_header_level(markdown_content, split_level)
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

    fallback_chunks = _split_text_by_paragraphs(markdown_content, max_chunk_size)
    return [(None, chunk) for chunk in fallback_chunks], None


def _text_context_from_markdown(
    markdown: str,
    fallback_title: str,
    title: str | None,
    max_chunk_size: int,
) -> TextContext:
    """Convert markdown output into aimd's TextContext shape."""
    resolved_title = title or _extract_title_from_content(markdown, fallback_title)
    stripped = markdown.strip()
    if len(stripped) <= max_chunk_size:
        return TextContext(
            title=resolved_title,
            chunk_list=[stripped] if stripped else [],
            split_header_level=None,
        )

    sections, header_level = _split_markdown_by_headers(
        stripped,
        max_chunk_size=max_chunk_size,
    )
    section_data = [
        (section_title, section_content.strip())
        for section_title, section_content in sections
        if section_content.strip()
    ]
    chunks = _combine_sections_for_processing(section_data, max_chunk_size)
    return TextContext(
        title=resolved_title,
        chunk_list=chunks,
        split_header_level=header_level,
    )


async def convert_url_with_markitdown(
    url: str,
    transcribe_engine: str = "auto",
    language: str | None = None,
    model: str | None = None,
    save_original_path: Path | None = None,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    temp_dir: Path | None = None,
    raw_transcript: bool = False,
) -> tuple[TextContext, str]:
    """Convert a URL through MarkItDown and AIMD's bundled URL plugin."""
    md = MarkItDown(enable_plugins=True)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        partial(
            md.convert_stream,
            io.BytesIO(),
            stream_info=StreamInfo(url=url),
            task_type="transcript",
            transcribe_engine=transcribe_engine,
            language=language,
            model=model,
            save_original_path=save_original_path,
            cookies_file=cookies_file,
            cookies_from_browser=cookies_from_browser,
            temp_dir=temp_dir,
            raw_transcript=raw_transcript,
        ),
    )

    return (
        _text_context_from_markdown(
            result.markdown,
            fallback_title=result.title,
            title=result.title,
            max_chunk_size=40000,
        ),
        detect_platform(url),
    )


async def convert_file_with_markitdown(
    file_path: str | Path,
    transcribe_engine: str = "auto",
    language: str | None = None,
    model: str | None = None,
    temp_dir: Path | None = None,
    task_type: str | None = None,
    start: int | None = None,
    end: int | None = None,
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
    output_dir = input_path.parent / input_path.stem if suffix in _DOCUMENT_ASSET_EXTENSIONS else None

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
            task_type=task_type,
            start=start,
            end=end,
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


def ensure_supported_input(
    input_source: str,
    task_type: TaskType | None = None,
    *,
    is_supported_file_fn: FileSupportChecker = is_supported_file,
) -> InputRoute:
    """Validate and return the source/task route for a source."""
    return ensure_supported_route(input_source, is_supported_file_fn, task_type)


async def process_input(
    request: ProcessInput,
    *,
    process_url: UrlProcessor = convert_url_with_markitdown,
    process_file: LocalFileProcessor = convert_file_with_markitdown,
    resolve_engine: Callable[[str], str] = resolve_engine_with_preflight,
    is_supported_file_fn: FileSupportChecker = is_supported_file,
) -> ProcessResult:
    """Process a routed input request."""
    route = ensure_supported_input(
        request.input_source,
        request.task_type,
        is_supported_file_fn=is_supported_file_fn,
    )
    task_type = route.task_type
    if task_type is None:
        raise UnsupportedInputError("Unsupported input source.")

    try:
        if route.source_kind == "url":
            return await _process_url(request, process_url, resolve_engine)
        return await _process_local_file(request, task_type, process_file, resolve_engine)
    except (InputNotFoundError, UnsupportedInputError, ProcessingFailedError):
        raise
    except Exception as exc:
        raise ProcessingFailedError(str(exc)) from exc


async def _process_url(
    request: ProcessInput,
    process_url: UrlProcessor,
    resolve_engine: Callable[[str], str],
) -> ProcessResult:
    if request.transcribe_engine != "auto":
        resolve_engine(request.transcribe_engine)

    text_context, platform = await process_url(
        request.input_source,
        request.transcribe_engine,
        request.language,
        request.model,
        request.save_original,
        str(request.cookies) if request.cookies else None,
        request.cookies_from_browser,
        request.temp_dir,
        request.raw_transcript,
    )
    return ProcessResult(
        task_type="transcript",
        text_context=text_context,
        platform=platform,
    )


async def _process_local_file(
    request: ProcessInput,
    task_type: TaskType,
    process_file: LocalFileProcessor,
    resolve_engine: Callable[[str], str],
) -> ProcessResult:
    input_path = Path(request.input_source)
    if not input_path.exists():
        raise InputNotFoundError(f"Input file not found: {request.input_source}")

    engine = request.transcribe_engine
    if task_type == "transcript":
        engine = resolve_engine(request.transcribe_engine)

    text_context, output_dir = await process_file(
        input_path.as_posix(),
        engine,
        request.language,
        request.model,
        request.temp_dir,
        task_type,
        request.start,
        request.end,
    )
    return ProcessResult(
        task_type=task_type,
        text_context=text_context,
        output_dir=output_dir,
    )
