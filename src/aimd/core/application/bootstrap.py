"""Application bootstrap and explicit dependency wiring."""

from dataclasses import dataclass

from aimd.media import (
    get_engine_capabilities,
    resolve_engine_with_preflight,
)

from ..infrastructure.markitdown_processor import (
    convert_file_with_markitdown,
    is_supported_file,
)
from ..infrastructure.media_processor import get_text_context_from_media_url
from .use_cases.list_engines import ListEnginesUseCase
from .use_cases.process_input import ProcessInputUseCase
from .use_cases.processors import ConvertTaskProcessor, TranscriptTaskProcessor


@dataclass(slots=True)
class AppContainer:
    """Resolved dependencies and use-cases for all adapters."""

    process_input_use_case: ProcessInputUseCase
    list_engines_use_case: ListEnginesUseCase


def build_container() -> AppContainer:
    """Build app container with explicit dependency wiring."""
    transcript_processor = TranscriptTaskProcessor(
        process_url=get_text_context_from_media_url,
        process_file=convert_file_with_markitdown,
        resolve_engine=resolve_engine_with_preflight,
    )
    convert_processor = ConvertTaskProcessor(
        process_file=convert_file_with_markitdown,
    )

    return AppContainer(
        process_input_use_case=ProcessInputUseCase(
            processors={
                "transcript": transcript_processor,
                "convert": convert_processor,
            },
            is_supported_file=is_supported_file,
        ),
        list_engines_use_case=ListEnginesUseCase(
            get_capabilities=get_engine_capabilities,
            resolve_engine=resolve_engine_with_preflight,
        ),
    )
