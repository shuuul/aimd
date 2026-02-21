"""FastAPI application for exposing aimd as an HTTP service."""

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from logly import logger
from pydantic import BaseModel, Field

from .capabilities import get_engine_capabilities, resolve_engine_with_preflight
from .errors import AimdError, EngineUnavailableError
from .service import (
    ensure_supported_input,
    process_convert_input,
    process_transcript_input,
)


class HealthResponse(BaseModel):
    status: str = "ok"


class EngineCapabilityResponse(BaseModel):
    name: str
    available: bool
    reason: str | None = None
    fix_hint: str | None = None
    selected_by_auto: bool = False


class EnginesResponse(BaseModel):
    auto_selected_engine: str | None = None
    engines: list[EngineCapabilityResponse]


class ProcessRequest(BaseModel):
    input_source: str = Field(
        ...,
        description="Audio/video file path, video URL, or document file path.",
    )
    output_file: str | None = Field(
        default=None,
        description="Optional path to write resulting markdown output.",
    )
    transcribe_engine: str = Field(
        default="auto",
        description="Transcription engine: auto, yap, mlx, cuda, cpu.",
    )
    language: str | None = Field(
        default=None,
        description="Whisper language code, e.g. zh, en, ja.",
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
        description=(
            "Browser cookie source for URL extraction, "
            "e.g. chrome, chrome:default, chrome+gnomekeyring:default."
        ),
    )


class ProcessResponse(BaseModel):
    task_type: Literal["transcript", "convert"]
    title: str
    chunk_list: list[str]
    split_header_level: int | None = None
    output_file: str | None = None
    output_dir: str | None = None


def _persist_output(
    output_file: Path,
    task_type: Literal["transcript", "convert"],
    chunk_list: list[str],
) -> None:
    """Persist processed content to a markdown file path."""
    if task_type == "transcript":
        text = chunk_list[0] if chunk_list else ""
        if not text:
            raise ValueError("Transcription returned empty content")
    else:
        text = "\n\n".join(chunk_list)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")


def create_app() -> FastAPI:
    """Build FastAPI app instance."""
    app = FastAPI(
        title="aimd API",
        description="Context preparation API for LLM workflows",
        version="0.0.10",
    )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse()

    @app.get("/v1/engines", response_model=EnginesResponse)
    async def engines() -> EnginesResponse:
        capabilities = get_engine_capabilities()
        auto_selected_engine: str | None = None
        try:
            auto_selected_engine = resolve_engine_with_preflight("auto")
        except EngineUnavailableError:
            auto_selected_engine = None

        ordered_engines = ("yap", "mlx", "cuda", "cpu")
        response_engines = [
            EngineCapabilityResponse(
                name=engine,
                available=capabilities[engine].available,
                reason=capabilities[engine].reason,
                fix_hint=capabilities[engine].fix_hint,
                selected_by_auto=engine == auto_selected_engine,
            )
            for engine in ordered_engines
        ]
        return EnginesResponse(
            auto_selected_engine=auto_selected_engine,
            engines=response_engines,
        )

    @app.post("/v1/process", response_model=ProcessResponse)
    async def process(request: ProcessRequest) -> ProcessResponse:
        try:
            task_type = ensure_supported_input(request.input_source)
            output_file: str | None = None
            output_dir: str | None = None

            if task_type == "transcript":
                text_context = await process_transcript_input(
                    input_source=request.input_source,
                    engine=request.transcribe_engine,
                    language=request.language,
                    save_original=Path(request.save_original)
                    if request.save_original
                    else None,
                    cookies=Path(request.cookies) if request.cookies else None,
                    cookies_from_browser=request.cookies_from_browser,
                )
            else:
                text_context, epub_output_dir = await process_convert_input(
                    request.input_source
                )
                if epub_output_dir is not None:
                    output_dir = str(epub_output_dir.resolve())

            if request.output_file and output_dir is None:
                output_path = Path(request.output_file)
                _persist_output(output_path, task_type, text_context.chunk_list)
                output_file = str(output_path.resolve())
            elif request.output_file and output_dir is not None:
                logger.warning(
                    "Ignoring output_file for EPUB conversions; output is a directory."
                )

            return ProcessResponse(
                task_type=task_type,
                title=text_context.title,
                chunk_list=text_context.chunk_list,
                split_header_level=text_context.split_header_level,
                output_file=output_file,
                output_dir=output_dir,
            )
        except AimdError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error(f"API processing failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()


def main() -> None:
    """Run the FastAPI service via uvicorn."""
    import uvicorn

    host = os.getenv("AIMD_API_HOST", "127.0.0.1")
    port = int(os.getenv("AIMD_API_PORT", "8000"))
    uvicorn.run("aimd.api:app", host=host, port=port, reload=False)
