"""Use-case for input processing orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from ...const import AUDIO_EXTENSIONS, EPUB_EXTENSIONS
from ...errors import InputNotFoundError, ProcessingFailedError, UnsupportedInputError
from ...types import TextContext
from ...utils import is_url
from ..models import ProcessInput, ProcessResult, TaskType

TranscriptProcessor = Callable[
    [
        str,
        str,
        str | None,
        str | None,
        Path | None,
        Path | None,
        str | None,
        Path | None,
        bool,
    ],
    Awaitable[tuple[TextContext, str | None]],
]
ConvertProcessor = Callable[
    [str, Path | None], Awaitable[tuple[TextContext, Path | None]]
]
FileSupportChecker = Callable[[str | Path], bool]


def get_task_type(input_source: str, is_supported_file: FileSupportChecker) -> TaskType:
    """Determine task type based on source path/URL."""
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


def ensure_supported_input(
    input_source: str, is_supported_file: FileSupportChecker
) -> TaskType:
    """Validate and return supported task type, else raise domain error."""
    task_type = get_task_type(input_source, is_supported_file)
    if task_type == "unknown":
        input_path = Path(input_source)
        if not is_url(input_source) and input_path.suffix and not input_path.exists():
            raise InputNotFoundError(f"Input file not found: {input_source}")
        raise UnsupportedInputError(
            "Unsupported input source. Supported inputs: audio/video files, "
            "video URLs, and supported document files."
        )
    return task_type


@dataclass(slots=True)
class ProcessInputUseCase:
    """Single orchestration use-case for transcript/convert processing."""

    transcript_processor: TranscriptProcessor
    convert_processor: ConvertProcessor
    is_supported_file: FileSupportChecker

    async def execute(self, request: ProcessInput) -> ProcessResult:
        task_type = ensure_supported_input(request.input_source, self.is_supported_file)

        if task_type == "transcript":
            try:
                text_context, platform = await self.transcript_processor(
                    request.input_source,
                    request.transcribe_engine,
                    request.language,
                    request.model,
                    request.save_original,
                    request.cookies,
                    request.cookies_from_browser,
                    request.temp_dir,
                    request.raw_transcript,
                )
            except (InputNotFoundError, UnsupportedInputError, ProcessingFailedError):
                raise
            except Exception as exc:
                raise ProcessingFailedError(str(exc)) from exc

            return ProcessResult(
                task_type="transcript", text_context=text_context, platform=platform
            )

        try:
            text_context, output_dir = await self.convert_processor(
                request.input_source,
                request.temp_dir,
            )
        except (InputNotFoundError, UnsupportedInputError, ProcessingFailedError):
            raise
        except Exception as exc:
            raise ProcessingFailedError(str(exc)) from exc

        return ProcessResult(
            task_type="convert",
            text_context=text_context,
            output_dir=output_dir,
        )


async def process_transcript_input(
    input_source: str,
    engine: str,
    language: str | None,
    model: str | None,
    save_original: Path | None,
    cookies: Path | None,
    cookies_from_browser: str | None,
    temp_dir: Path | None,
    raw_transcript: bool,
    process_url: Callable[
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
    ],
    process_audio: Callable[
        [Path, str, str | None, str | None, Path | None], Awaitable[TextContext]
    ],
    resolve_engine: Callable[[str], str],
) -> tuple[TextContext, str | None]:
    """Transcript pipeline used by the process use-case."""
    if is_url(input_source):
        if engine != "auto":
            resolve_engine(engine)
        return await process_url(
            input_source,
            engine,
            language,
            model,
            save_original,
            str(cookies) if cookies else None,
            cookies_from_browser,
            temp_dir,
            raw_transcript,
        )

    input_path = Path(input_source)
    if not input_path.exists():
        raise InputNotFoundError(f"Input file not found: {input_source}")

    resolved_engine = resolve_engine(engine)
    text_context = await process_audio(
        input_path, resolved_engine, language, model, temp_dir
    )
    return text_context, None


async def process_convert_input(
    input_file: str,
    temp_dir: Path | None,
    process_epub: Callable[[Path, Path | None], Awaitable[tuple[TextContext, Path]]],
    process_file: Callable[[Path], Awaitable[TextContext]],
) -> tuple[TextContext, Path | None]:
    """Convert pipeline used by the process use-case."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise InputNotFoundError(f"Input file not found: {input_file}")

    if input_path.suffix.lower() in EPUB_EXTENSIONS:
        return await process_epub(input_path, temp_dir)

    return await process_file(input_path), None
