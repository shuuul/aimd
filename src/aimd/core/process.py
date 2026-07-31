"""Core input processing orchestration."""

import asyncio
from functools import partial
import io
from pathlib import Path
import re
from typing import Awaitable, Callable

from logly import logger
from markitdown import FileConversionException, MarkItDown, StreamInfo

from aimd.plugins.asr.const import AUDIO_EXTENSIONS
from .errors import (
    AimdError,
    InputNotFoundError,
    ProcessingFailedError,
    UnsupportedInputError,
)
from .models import InputRoute, ProcessInput, ProcessResult, TaskType, TextContext

from aimd.plugins.asr.const import AUDIO_FILE_EXTENSIONS, VIDEO_FILE_EXTENSIONS
from aimd.plugins.url import detect_platform, is_url
from aimd.plugins.doc import PANDOC_DOCUMENT_EXTENSIONS
from aimd.plugins.ocr.const import IMAGE_FILE_EXTENSIONS, OCR_DOCUMENT_EXTENSIONS

# --- Routing logic merged from router.py (Phase 5) ---
FileSupportChecker = Callable[[str | Path], bool]

_VALID_TASK_TYPES: set[TaskType] = {"transcript", "convert", "ocr"}


def _pdf_has_extractable_text(file_path: Path) -> bool:
    """Best-effort check for text-layer PDFs before falling back to OCR."""
    try:
        import pymupdf
    except ImportError:
        return True
    try:
        with pymupdf.open(file_path) as document:
            for page in document:
                if page.get_text("text").strip():
                    return True
    except Exception:
        return True
    return False


def get_input_route(
    input_source: str,
    is_supported_file: FileSupportChecker,
    requested_task_type: TaskType | None = None,
) -> InputRoute:
    """Classify a source and select the processing task."""
    if is_url(input_source):
        if requested_task_type and requested_task_type != "transcript":
            return InputRoute(source_kind="url", task_type=None)
        return InputRoute(source_kind="url", task_type="transcript")

    try:
        file_path = Path(input_source)
        if file_path.exists():
            suffix = file_path.suffix.lower()
            if requested_task_type == "ocr":
                if suffix in IMAGE_FILE_EXTENSIONS:
                    return InputRoute(source_kind="image_file", task_type="ocr")
                if suffix in OCR_DOCUMENT_EXTENSIONS:
                    return InputRoute(source_kind="document_file", task_type="ocr")
                return InputRoute(source_kind="unknown", task_type=None)
            if suffix in IMAGE_FILE_EXTENSIONS:
                return InputRoute(source_kind="image_file", task_type="ocr")
            if suffix in OCR_DOCUMENT_EXTENSIONS and not _pdf_has_extractable_text(
                file_path
            ):
                return InputRoute(source_kind="document_file", task_type="ocr")
            if suffix in AUDIO_FILE_EXTENSIONS:
                if requested_task_type and requested_task_type != "transcript":
                    return InputRoute(source_kind="audio_file", task_type=None)
                return InputRoute(source_kind="audio_file", task_type="transcript")
            if suffix in VIDEO_FILE_EXTENSIONS:
                if requested_task_type and requested_task_type != "transcript":
                    return InputRoute(source_kind="video_file", task_type=None)
                return InputRoute(source_kind="video_file", task_type="transcript")
            if is_supported_file(file_path):
                if requested_task_type and requested_task_type != "convert":
                    return InputRoute(source_kind="document_file", task_type=None)
                return InputRoute(source_kind="document_file", task_type="convert")
    except (OSError, ValueError):
        pass

    return InputRoute(source_kind="unknown", task_type=None)


def ensure_supported_input(
    input_source: str,
    requested_task_type: TaskType | None = None,
    *,
    is_supported_file_fn: FileSupportChecker | None = None,
) -> InputRoute:
    """Validate and return supported input route, else raise domain error."""
    if is_supported_file_fn is None:
        is_supported_file_fn = is_supported_file
    if requested_task_type is not None and requested_task_type not in _VALID_TASK_TYPES:
        raise UnsupportedInputError(
            "Unsupported task. Supported tasks: transcript, convert, ocr."
        )

    route = get_input_route(input_source, is_supported_file_fn, requested_task_type)
    if route.task_type is None:
        input_path = Path(input_source)
        if not is_url(input_source) and input_path.suffix and not input_path.exists():
            raise InputNotFoundError(f"Input file not found: {input_source}")
        if requested_task_type == "ocr":
            raise UnsupportedInputError(
                "Unsupported OCR input. OCR supports image files "
                "(.png, .jpg, .jpeg, .webp, .tif, .tiff) and PDF files."
            )
        raise UnsupportedInputError(
            "Unsupported input source. Supported inputs: audio/video files, "
            "video URLs, supported document files, and OCR-capable images/PDFs."
        )
    return route


_DOCUMENT_ASSET_EXTENSIONS = {".docx", ".epub", ".odt"}
_MARKITDOWN_FILE_EXTENSIONS = (
    AUDIO_EXTENSIONS
    | PANDOC_DOCUMENT_EXTENSIONS
    | {
        ".doc",
        ".pdf",
        ".ppt",
        ".pptx",
        ".txt",
        ".xls",
        ".xlsx",
    }
)

UrlProcessor = Callable[
    [
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
    [str, str | None, str | None, Path | None, str | None, int | None, int | None],
    Awaitable[tuple[TextContext, Path | None]],
]


def is_supported_file(file_path: str | Path) -> bool:
    """Return whether a local file extension should be offered to MarkItDown."""
    if isinstance(file_path, str) and file_path.startswith(("http://", "https://")):
        return False
    return Path(file_path).suffix.lower() in _MARKITDOWN_FILE_EXTENSIONS


def _extract_title_from_content(content: str, fallback_title: str = "Untitled") -> str:
    """Extract and clean title from markdown text (simplified)."""
    if not content or not content.strip():
        return fallback_title

    lines = [ln.strip() for ln in content.strip().split("\n") if ln.strip()]

    extracted = None
    for line in lines:
        if line.startswith("# "):
            extracted = line[2:].strip()
            break
    if not extracted:
        for i, line in enumerate(lines):
            if i + 1 < len(lines) and (
                all(c == "=" for c in lines[i + 1])
                or all(c == "-" for c in lines[i + 1])
            ):
                extracted = line
                break
    if not extracted:
        for line in lines[:5]:
            if (
                line
                and not line.startswith(("#", "!", "[", "<", "{", ":::", "---"))
                and not line.lower().startswith("http")
            ):
                extracted = line
                break

    if not extracted:
        return fallback_title

    clean = re.sub(r"^#+\s*", "", extracted)
    clean = re.sub(r"\*+([^*]+)\*+", r"\1", clean)
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
    clean = re.sub(r"\{[^}]*\}|\([^)]*\)|#\S+", "", clean)
    clean = re.sub(r'^["\'\']+|["\'\']+$', "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = re.sub(r"[。，、；：！？]+$", "", clean)

    return clean or fallback_title


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


def _iter_exception_chain(exc: BaseException):
    """Yield an exception and its cause/context chain without cycles."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = current.__cause__ or current.__context__


def _extract_aimd_error_from_file_conversion(
    exc: FileConversionException,
) -> tuple[AimdError, object | None] | None:
    """Restore the first domain error from MarkItDown's ordered attempts."""
    for attempt in exc.attempts or []:
        exc_info = attempt.exc_info
        if not exc_info or exc_info[1] is None:
            continue
        for candidate in _iter_exception_chain(exc_info[1]):
            if isinstance(candidate, AimdError):
                return candidate, candidate.__traceback__
    return None


def _raise_from_markitdown_failure(exc: BaseException) -> None:
    """Re-raise domain errors hidden in MarkItDown aggregates; wrap unknowns."""
    if isinstance(exc, AimdError):
        raise exc
    if isinstance(exc, FileConversionException):
        restored = _extract_aimd_error_from_file_conversion(exc)
        if restored is not None:
            aimd_error, traceback = restored
            if traceback is not None:
                raise aimd_error.with_traceback(traceback)
            raise aimd_error
    raise ProcessingFailedError(str(exc)) from exc


async def _run_markitdown(fn, /, *args, **kwargs):
    """Run MarkItDown off the event loop and normalize conversion failures."""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, partial(fn, *args, **kwargs))
    except Exception as exc:
        _raise_from_markitdown_failure(exc)


async def convert_url_with_markitdown(
    url: str,
    language: str | None = None,
    model: str | None = None,
    save_original_path: Path | None = None,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    temp_dir: Path | None = None,
    raw_transcript: bool = False,
) -> tuple[TextContext, str]:
    """Convert a URL through MarkItDown and AIMD's bundled URL plugin."""
    md = MarkItDown(enable_builtins=False, enable_plugins=True)
    result = await _run_markitdown(
        md.convert_stream,
        io.BytesIO(),
        stream_info=StreamInfo(url=url),
        task_type="transcript",
        language=language,
        model=model,
        save_original_path=save_original_path,
        cookies_file=cookies_file,
        cookies_from_browser=cookies_from_browser,
        temp_dir=temp_dir,
        raw_transcript=raw_transcript,
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
    output_dir = (
        input_path.parent / input_path.stem
        if suffix in _DOCUMENT_ASSET_EXTENSIONS
        else None
    )

    aimd_owned_route = task_type in {"transcript", "ocr"} or (
        task_type == "convert" and suffix in PANDOC_DOCUMENT_EXTENSIONS
    )
    md = MarkItDown(enable_builtins=not aimd_owned_route, enable_plugins=True)
    result = await _run_markitdown(
        md.convert,
        input_path,
        language=language,
        model=model,
        temp_dir=temp_dir,
        output_dir=output_dir,
        task_type=task_type,
        start=start,
        end=end,
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


async def process_input(
    request: ProcessInput,
    *,
    process_url: UrlProcessor = convert_url_with_markitdown,
    process_file: LocalFileProcessor = convert_file_with_markitdown,
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
            return await _process_url(request, process_url)
        return await _process_local_file(request, task_type, process_file)
    except AimdError:
        raise
    except Exception as exc:
        raise ProcessingFailedError(str(exc)) from exc


async def _process_url(
    request: ProcessInput,
    process_url: UrlProcessor,
) -> ProcessResult:
    text_context, platform = await process_url(
        request.input_source,
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
) -> ProcessResult:
    input_path = Path(request.input_source)

    text_context, output_dir = await process_file(
        input_path.as_posix(),
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
