"""Application bootstrap and explicit dependency wiring."""

from dataclasses import dataclass
from pathlib import Path

from ..infrastructure.capabilities.detector import (
    get_engine_capabilities,
    resolve_engine_with_preflight,
)
from ..infrastructure.documents.epub_processor import process_epub_with_images
from ..infrastructure.documents.pandoc_reader import is_supported_file
from ..infrastructure.documents.processor import get_text_from_file
from ..infrastructure.transcription.processor import get_text_from_audio
from ..infrastructure.url.processor import get_text_from_url
from .use_cases.list_engines import ListEnginesUseCase
from .use_cases.process_input import (
    ProcessInputUseCase,
    process_convert_input,
    process_transcript_input,
)


@dataclass(slots=True)
class AppContainer:
    """Resolved dependencies and use-cases for all adapters."""

    process_input_use_case: ProcessInputUseCase
    list_engines_use_case: ListEnginesUseCase


async def _transcript_processor(
    input_source: str,
    engine: str,
    language: str | None,
    model: str | None,
    save_original: Path | None,
    cookies: Path | None,
    cookies_from_browser: str | None,
    temp_dir: Path | None = None,
    raw_transcript: bool = False,
):
    return await process_transcript_input(
        input_source=input_source,
        engine=engine,
        language=language,
        model=model,
        save_original=save_original,
        cookies=cookies,
        cookies_from_browser=cookies_from_browser,
        temp_dir=temp_dir,
        raw_transcript=raw_transcript,
        process_url=get_text_from_url,
        process_audio=get_text_from_audio,
        resolve_engine=resolve_engine_with_preflight,
    )


async def _convert_processor(input_file: str, temp_dir: Path | None = None):
    return await process_convert_input(
        input_file,
        temp_dir=temp_dir,
        process_epub=process_epub_with_images,
        process_file=get_text_from_file,
    )


def build_container() -> AppContainer:
    """Build app container with explicit dependency wiring."""
    return AppContainer(
        process_input_use_case=ProcessInputUseCase(
            transcript_processor=_transcript_processor,
            convert_processor=_convert_processor,
            is_supported_file=is_supported_file,
        ),
        list_engines_use_case=ListEnginesUseCase(
            get_capabilities=get_engine_capabilities,
            resolve_engine=resolve_engine_with_preflight,
        ),
    )
