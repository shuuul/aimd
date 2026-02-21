"""Application-level request/response models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..types import TextContext

TaskType = Literal["transcript", "convert", "unknown"]


@dataclass(slots=True)
class ProcessInput:
    """Canonical process request model consumed by use-cases."""

    input_source: str
    transcribe_engine: str = "auto"
    language: str | None = None
    output_file: Path | None = None
    save_original: Path | None = None
    cookies: Path | None = None
    cookies_from_browser: str | None = None


@dataclass(slots=True)
class ProcessResult:
    """Canonical process response model produced by use-cases."""

    task_type: Literal["transcript", "convert"]
    text_context: TextContext
    output_file: Path | None = None
    output_dir: Path | None = None
