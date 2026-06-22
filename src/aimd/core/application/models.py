"""Application-level request/response models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..types import TextContext

SourceKind = Literal["url", "audio_file", "video_file", "document_file", "unknown"]
TaskType = Literal["transcript", "convert"]


@dataclass(slots=True, frozen=True)
class InputRoute:
    """Classified input source and selected processing task."""

    source_kind: SourceKind
    task_type: TaskType | None


@dataclass(slots=True)
class ProcessInput:
    """Canonical process request model consumed by use-cases."""

    input_source: str
    transcribe_engine: str = "auto"
    model: str | None = None
    language: str | None = None
    save_original: Path | None = None
    cookies: Path | None = None
    cookies_from_browser: str | None = None
    temp_dir: Path | None = None
    raw_transcript: bool = False


@dataclass(slots=True)
class ProcessResult:
    """Canonical process response model produced by use-cases."""

    task_type: TaskType
    text_context: TextContext
    output_dir: Path | None = None
    platform: str | None = None
