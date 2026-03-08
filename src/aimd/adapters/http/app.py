"""FastAPI adapter for aimd."""

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from logly import logger
from pydantic import BaseModel, Field

from ...application.bootstrap import build_container
from ...application.models import ProcessInput
from ...application.services.output_writer import persist_output
from ...errors import AimdError


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
        ..., description="Audio/video file path, video URL, or document file path."
    )
    output_file: str | None = Field(
        default=None, description="Optional path to write resulting markdown output."
    )
    transcribe_engine: str = Field(
        default="auto",
        description="Transcription engine: auto, mlx, qwen, funasr, yap.",
    )
    model: str | None = Field(
        default=None,
        description="Model for transcription. For mlx: HuggingFace model path. "
        "For qwen: Qwen3-ASR model. For funasr: FunAudioLLM/SenseVoiceSmall (default) "
        "or FunAudioLLM/Fun-ASR-Nano-2512.",
    )
    language: str | None = Field(
        default=None, description="Language code for transcription, e.g. zh, en, ja."
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
    task_type: Literal["transcript", "convert"]
    title: str
    chunk_list: list[str]
    split_header_level: int | None = None
    platform: str | None = None
    output_file: str | None = None
    output_dir: str | None = None


def create_app() -> FastAPI:
    """Build FastAPI app instance."""
    app = FastAPI(
        title="aimd API",
        description="Context preparation API for LLM workflows",
        version="0.7.0",
    )
    container = build_container()

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse()

    @app.get("/v1/engines", response_model=EnginesResponse)
    async def engines() -> EnginesResponse:
        result = container.list_engines_use_case.execute()
        ordered_engines = ("mlx", "yap", "qwen", "funasr")
        response_engines = [
            EngineCapabilityResponse(
                name=engine,
                available=result.engines[engine].available,
                reason=result.engines[engine].reason,
                fix_hint=result.engines[engine].fix_hint,
                selected_by_auto=engine == result.auto_selected_engine,
            )
            for engine in ordered_engines
        ]
        return EnginesResponse(
            auto_selected_engine=result.auto_selected_engine,
            engines=response_engines,
        )

    @app.post("/v1/process", response_model=ProcessResponse)
    async def process(request: ProcessRequest) -> ProcessResponse:
        try:
            env_temp_dir = os.environ.get("AIMD_TEMP_DIR")
            temp_dir = Path(env_temp_dir) if env_temp_dir else None
            if temp_dir is not None:
                temp_dir.mkdir(parents=True, exist_ok=True)

            result = await container.process_input_use_case.execute(
                ProcessInput(
                    input_source=request.input_source,
                    output_file=Path(request.output_file)
                    if request.output_file
                    else None,
                    transcribe_engine=request.transcribe_engine,
                    model=request.model,
                    language=request.language,
                    save_original=Path(request.save_original)
                    if request.save_original
                    else None,
                    cookies=Path(request.cookies) if request.cookies else None,
                    cookies_from_browser=request.cookies_from_browser,
                    temp_dir=temp_dir,
                    raw_transcript=request.raw_transcript,
                )
            )

            output_file: str | None = None
            output_dir: str | None = None
            if result.output_dir is not None:
                output_dir = str(result.output_dir.resolve())

            if request.output_file and result.output_dir is None:
                resolved = persist_output(
                    Path(request.output_file),
                    result.task_type,
                    result.text_context.chunk_list,
                )
                output_file = str(resolved)
            elif request.output_file and result.output_dir is not None:
                logger.warning(
                    "Ignoring output_file for EPUB conversions; output is a directory."
                )

            return ProcessResponse(
                task_type=result.task_type,
                title=result.text_context.title,
                chunk_list=result.text_context.chunk_list,
                split_header_level=result.text_context.split_header_level,
                platform=result.platform,
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
