"""Versioned HTTP schemas shared by synchronous and job APIs."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aimd.core.models import TaskType
from aimd.interfaces.output import (
    CONTEXT_HELP_TEXT,
    MODEL_HELP_TEXT,
    PRECISION_HELP_TEXT,
)

JobState = Literal["queued", "running", "completed", "failed", "cancelled"]
JobStage = Literal[
    "downloading",
    "extracting",
    "transcribing",
    "ocr",
    "converting",
    "saving",
]
CancellationStatus = Literal[
    "none", "requested", "cancelled", "completed_after_request"
]


class HealthResponse(BaseModel):
    """Sidecar health response."""

    status: str = "ok"


class BlobUploadResponse(BaseModel):
    """Response returned after storing sidecar-owned blob bytes."""

    blob_id: str
    bytes: int
    filename: str | None = None


class ProcessRequest(BaseModel):
    """AIMD processing request accepted by synchronous and job APIs."""

    input_source: str | None = Field(
        default=None,
        description="Audio/video file path, video URL, document path, image path, or scanned PDF path.",
    )
    blob_id: str | None = Field(
        default=None,
        description="Sidecar-owned blob identifier from POST /v1/blobs.",
    )
    task_type: TaskType | None = Field(
        default=None,
        description="Optional explicit task: transcript, convert, or ocr. Defaults to auto-routing.",
    )
    output_file: str | None = Field(
        default=None, description="Optional path to write resulting markdown output."
    )
    model: str | None = Field(default=None, description=MODEL_HELP_TEXT)
    precision: str | None = Field(default=None, description=PRECISION_HELP_TEXT)
    asr_base_url: str | None = Field(
        default=None, description="OpenAI-compatible remote ASR base URL."
    )
    asr_model: str | None = Field(
        default=None, description="Model ID served by the remote ASR endpoint."
    )
    asr_api_key: str | None = Field(
        default=None, description="Bearer token for the remote ASR endpoint."
    )
    ocr_base_url: str | None = Field(
        default=None, description="OpenAI-compatible remote OCR base URL."
    )
    ocr_model: str | None = Field(
        default=None, description="Model ID served by the remote OCR endpoint."
    )
    ocr_api_key: str | None = Field(
        default=None, description="Bearer token for the remote OCR endpoint."
    )
    context: str | None = Field(default=None, description=CONTEXT_HELP_TEXT)
    metadata_context: bool = Field(
        default=True,
        description="Automatically build ASR context from URL metadata "
        "(title/description/tags) to bias transcription.",
    )
    language: str | None = Field(
        default=None,
        description="Language code/hint for transcription or OCR, e.g. zh, en, ja.",
    )
    start: int | None = Field(
        default=None, description="0-based inclusive start page for OCR PDF inputs."
    )
    end: int | None = Field(
        default=None, description="0-based inclusive end page for OCR PDF inputs."
    )
    save_original: str | None = Field(
        default=None,
        description="Optional path to persist downloaded audio from URL processing.",
    )
    cookies: str | None = Field(
        default=None,
        description="Path to Netscape-format cookies file for URL extraction.",
    )
    cookies_from_browser: str | None = Field(
        default=None,
        description="Browser cookie source for URL extraction, e.g. chrome:default.",
    )
    raw_transcript: bool = Field(
        default=False,
        description="Preserve original subtitle formatting (SRT/VTT timestamps).",
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "ProcessRequest":
        has_input = bool(self.input_source)
        has_blob = bool(self.blob_id)
        if has_input == has_blob:
            raise ValueError("Exactly one of input_source or blob_id must be set")
        return self


class ProcessResponse(BaseModel):
    """Backward-compatible synchronous processing response."""

    task_type: TaskType
    title: str
    markdown: str
    asset_base_uri: str | None = None
    chunk_list: list[str]
    split_header_level: int | None = None
    platform: str | None = None
    output_file: str | None = None
    output_dir: str | None = None


class ProcessArtifact(BaseModel):
    """Lossless editor artifact returned by a completed job."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_type": "convert",
                "title": "Example document",
                "markdown": "# Example document\n\nExact AIMD output.\n",
                "source_uri": "file:///Users/example/source.docx",
                "asset_base_uri": "file:///tmp/aimd-output/assets/",
                "platform": None,
                "output_file": None,
                "output_dir": "/tmp/aimd-output",
            }
        }
    )

    task_type: TaskType = Field(description="Processing route selected by AIMD.")
    title: str = Field(description="Display title inferred from the source.")
    markdown: str = Field(
        description="Exact Markdown produced before context chunking."
    )
    source_uri: str = Field(description="Original URL or canonical local file URI.")
    asset_base_uri: str | None = Field(
        default=None, description="Base URI for relative assets referenced by Markdown."
    )
    platform: str | None = Field(default=None, description="Detected source platform.")
    output_file: str | None = Field(
        default=None, description="Canonical persisted Markdown path, when requested."
    )
    output_dir: str | None = Field(
        default=None, description="Canonical asset-producing output directory, if any."
    )


class JobEvent(BaseModel):
    """Monotonic event emitted by a processing job."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "8a41c856-eef8-4c40-99c4-ce3d49760c71",
                "state": "running",
                "stage": "transcribing",
                "current": 2,
                "total": 5,
                "message": "Transcribing segment 2 of 5",
                "cancellation_status": "none",
                "sequence": 3,
                "created_at": "2026-08-15T12:00:00Z",
            }
        }
    )

    job_id: str
    state: JobState
    stage: JobStage | None = None
    current: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)
    message: str | None = None
    cancellation_status: CancellationStatus = "none"
    sequence: int = Field(ge=1, description="Per-job SSE event identifier.")
    created_at: datetime


class JobSnapshot(BaseModel):
    """Current state and terminal result of a job."""

    job_id: str
    state: JobState
    request: ProcessRequest
    artifact: ProcessArtifact | None = None
    error: str | None = None
    stage: JobStage | None = None
    current: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)
    cancellation_status: CancellationStatus = "none"
    created_at: datetime
    updated_at: datetime


class JobCreated(BaseModel):
    """Links returned when a job is accepted."""

    job_id: str
    state: JobState
    status_url: str
    events_url: str
