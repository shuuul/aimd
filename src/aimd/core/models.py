"""Core request/response models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

SourceKind = Literal[
    "url", "audio_file", "video_file", "document_file", "image_file", "unknown"
]
TaskType = Literal["transcript", "convert", "ocr"]


class TextContext(BaseModel):
    """Context for text processing with title and content."""

    title: str = Field(..., description="Title of the text")
    chunk_list: list[str] = Field(..., description="List of combined text chunks")
    split_header_level: int | None = Field(
        default=None,
        description="Header level used for splitting (1-6), None if no splitting was done",
    )


@dataclass(slots=True, frozen=True)
class InputRoute:
    """Classified input source and selected processing task."""

    source_kind: SourceKind
    task_type: TaskType | None


@dataclass(slots=True)
class ProcessInput:
    """Canonical process request model consumed by use-cases."""

    input_source: str
    task_type: TaskType | None = None
    model: str | None = None
    language: str | None = None
    start: int | None = None
    end: int | None = None
    save_original: Path | None = None
    cookies: Path | None = None
    cookies_from_browser: str | None = None
    temp_dir: Path | None = None
    raw_transcript: bool = False
    precision: str | None = None
    cancellation_check: Callable[[], bool] | None = None
    progress_reporter: (
        Callable[[str, int | None, int | None, str | None], None] | None
    ) = None
    context: str | None = None
    metadata_context: bool = True
    asr_base_url: str | None = None
    asr_model: str | None = None
    asr_api_key: str | None = None
    ocr_base_url: str | None = None
    ocr_model: str | None = None
    ocr_api_key: str | None = None


@dataclass(slots=True)
class ProcessResult:
    """Canonical process response model produced by use-cases."""

    task_type: TaskType
    text_context: TextContext
    markdown: str = ""
    asset_base_uri: str | None = None
    output_dir: Path | None = None
    platform: str | None = None
