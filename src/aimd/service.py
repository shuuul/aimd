"""Shared service layer for CLI and API entry points."""

from pathlib import Path
from typing import Literal

from .capabilities import resolve_engine_with_preflight
from .const import AUDIO_EXTENSIONS, EPUB_EXTENSIONS
from .errors import InputNotFoundError, ProcessingFailedError, UnsupportedInputError
from .tool.audio import get_text_from_audio
from .tool.file import get_text_from_file, is_supported_file, process_epub_with_images
from .tool.url import get_text_from_url
from .types import TextContext
from .utils import is_url

TaskType = Literal["transcript", "convert", "unknown"]


def get_task_type(input_source: str) -> TaskType:
    """Determine task type based on the input source."""
    if is_url(input_source):
        return "transcript"

    try:
        file_path = Path(input_source)
        if file_path.exists():
            if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                return "transcript"
            if is_supported_file(file_path):
                return "convert"
    except (OSError, ValueError):
        pass

    return "unknown"


def ensure_supported_input(input_source: str) -> TaskType:
    """Validate and return supported task type, else raise domain error."""
    task_type = get_task_type(input_source)
    if task_type == "unknown":
        input_path = Path(input_source)
        if not is_url(input_source) and input_path.suffix and not input_path.exists():
            raise InputNotFoundError(f"Input file not found: {input_source}")
        raise UnsupportedInputError(
            "Unsupported input source. Supported inputs: audio/video files, "
            "video URLs, and supported document files."
        )
    return task_type


async def process_transcript_input(
    input_source: str,
    engine: str = "auto",
    language: str | None = None,
    save_original: Path | None = None,
    cookies: Path | None = None,
    cookies_from_browser: str | None = None,
) -> TextContext:
    """Process transcription input from URL or local audio/video file."""
    if is_url(input_source):
        try:
            if engine != "auto":
                resolve_engine_with_preflight(engine)
            return await get_text_from_url(
                input_source,
                transcribe_engine=engine,
                language=language,
                save_original_path=save_original,
                cookies_file=str(cookies) if cookies else None,
                cookies_from_browser=cookies_from_browser,
            )
        except (InputNotFoundError, UnsupportedInputError, ProcessingFailedError):
            raise
        except Exception as exc:
            raise ProcessingFailedError(str(exc)) from exc

    input_path = Path(input_source)
    if not input_path.exists():
        raise InputNotFoundError(f"Input file not found: {input_source}")

    try:
        resolved_engine = resolve_engine_with_preflight(engine)
        return await get_text_from_audio(input_path, resolved_engine, language)
    except (InputNotFoundError, UnsupportedInputError, ProcessingFailedError):
        raise
    except Exception as exc:
        raise ProcessingFailedError(str(exc)) from exc


async def process_convert_input(
    input_file: str,
) -> tuple[TextContext, Path | None]:
    """Process document conversion input.

    Returns:
        A tuple of (text_context, epub_output_dir). `epub_output_dir` is only set
        for EPUB-family conversions.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise InputNotFoundError(f"Input file not found: {input_file}")

    file_extension = input_path.suffix.lower()

    try:
        if file_extension in EPUB_EXTENSIONS:
            text_context, output_dir = await process_epub_with_images(input_path)
            return text_context, output_dir

        text_context = await get_text_from_file(input_path)
        return text_context, None
    except (InputNotFoundError, UnsupportedInputError, ProcessingFailedError):
        raise
    except Exception as exc:
        raise ProcessingFailedError(str(exc)) from exc
