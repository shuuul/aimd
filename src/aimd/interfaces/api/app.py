"""FastAPI-backed HTTP API for aimd."""

from importlib.metadata import version
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from logly import logger
from pydantic import BaseModel, Field

from aimd.interfaces.output import (
    MODEL_HELP_TEXT,
    get_request_temp_dir,
    persist_result_output_if_requested,
)
from aimd.core.models import ProcessInput, ProcessResult, TaskType
from aimd.core.process import process_input as process_core_input
from aimd.core.errors import AimdError


class HealthResponse(BaseModel):
    status: str = "ok"


class ProcessRequest(BaseModel):
    input_source: str = Field(
        ...,
        description="Audio/video file path, video URL, document path, image path, or scanned PDF path.",
    )
    task_type: TaskType | None = Field(
        default=None,
        description="Optional explicit task: transcript, convert, or ocr. Defaults to auto-routing.",
    )
    output_file: str | None = Field(
        default=None, description="Optional path to write resulting markdown output."
    )
    model: str | None = Field(
        default=None,
        description=MODEL_HELP_TEXT,
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
        description="Preserve original subtitle formatting (SRT/VTT timestamps). "
        "By default, subtitles are simplified to plain text.",
    )


class ProcessResponse(BaseModel):
    task_type: TaskType
    title: str
    chunk_list: list[str]
    split_header_level: int | None = None
    platform: str | None = None
    output_file: str | None = None
    output_dir: str | None = None


def _build_process_input(
    request: ProcessRequest, temp_dir: Path | None
) -> ProcessInput:
    return ProcessInput(
        input_source=request.input_source,
        task_type=request.task_type,
        model=request.model,
        language=request.language,
        start=request.start,
        end=request.end,
        save_original=Path(request.save_original) if request.save_original else None,
        cookies=Path(request.cookies) if request.cookies else None,
        cookies_from_browser=request.cookies_from_browser,
        temp_dir=temp_dir,
        raw_transcript=request.raw_transcript,
    )


def _process_response(
    result: ProcessResult,
    *,
    output_file: str | None,
    output_dir: str | None,
) -> ProcessResponse:
    return ProcessResponse(
        task_type=result.task_type,
        title=result.text_context.title,
        chunk_list=result.text_context.chunk_list,
        split_header_level=result.text_context.split_header_level,
        platform=result.platform,
        output_file=output_file,
        output_dir=output_dir,
    )


def create_app() -> FastAPI:
    """Build FastAPI app instance."""
    app = FastAPI(
        title="aimd HTTP",
        description="Context preparation HTTP API for LLM workflows",
        version=version("aimd-tool"),
    )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse()

    @app.post("/v1/process", response_model=ProcessResponse)
    async def process(request: ProcessRequest) -> ProcessResponse:
        try:
            temp_dir = get_request_temp_dir()

            result = await process_core_input(_build_process_input(request, temp_dir))

            persisted = persist_result_output_if_requested(result, request.output_file)
            if persisted.ignored_output_file:
                logger.warning(
                    "Ignoring output_file for document asset conversions; output is a directory."
                )

            return _process_response(
                result,
                output_file=persisted.output_file,
                output_dir=persisted.output_dir,
            )
        except AimdError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error(f"HTTP processing failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()


def main() -> None:
    """Run FastAPI service via uvicorn."""
    import uvicorn

    host = os.getenv("AIMD_HTTP_HOST", "127.0.0.1")
    port = int(os.getenv("AIMD_HTTP_PORT", "8000"))
    uvicorn.run("aimd.interfaces.api.app:app", host=host, port=port, reload=False)
