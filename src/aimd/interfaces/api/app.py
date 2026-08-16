"""FastAPI-backed HTTP API for aimd."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
import hmac
import ipaddress
from importlib.metadata import version
import os
from pathlib import Path
from urllib.parse import urlparse

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.openapi.utils import get_openapi
from fastapi.responses import StreamingResponse
from logly import logger

from aimd.interfaces.output import (
    get_request_temp_dir,
    persist_result_output_if_requested,
)
from aimd.core.models import ProcessInput, ProcessResult
from aimd.core.process import process_input as process_core_input
from aimd.core.errors import AimdError
from aimd.interfaces.api.blobs import (
    BlobIdError,
    BlobNotFoundError,
    BlobStore,
)
from aimd.interfaces.api.jobs import JobCancelledError, JobManager, JobNotFoundError
from aimd.interfaces.api.paths import PathAccessError, PathPolicy
from aimd.interfaces.api.schemas import (
    BlobUploadResponse,
    HealthResponse,
    JobCreated,
    JobEvent,
    JobStage,
    JobSnapshot,
    ProcessArtifact,
    ProcessRequest,
    ProcessResponse,
)


def _resolved_input_source(
    request: ProcessRequest, blob_store: BlobStore | None
) -> str:
    if request.blob_id:
        if blob_store is None:
            raise BlobNotFoundError(request.blob_id)
        return str(blob_store.resolve(request.blob_id).path)
    if not request.input_source:
        raise ValueError("ProcessRequest.input_source is required after resolution")
    return request.input_source


def _build_process_input(
    request: ProcessRequest,
    temp_dir: Path | None,
    cancellation_check: Callable[[], bool] | None = None,
    progress_reporter: Callable[[str, int | None, int | None, str | None], None]
    | None = None,
    blob_store: BlobStore | None = None,
) -> ProcessInput:
    return ProcessInput(
        input_source=_resolved_input_source(request, blob_store),
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
        precision=request.precision,
        context=request.context,
        metadata_context=request.metadata_context,
        cancellation_check=cancellation_check,
        progress_reporter=progress_reporter,
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
        markdown=result.markdown,
        asset_base_uri=result.asset_base_uri,
        chunk_list=result.text_context.chunk_list,
        split_header_level=result.text_context.split_header_level,
        platform=result.platform,
        output_file=output_file,
        output_dir=output_dir,
    )


def _source_uri(request: ProcessRequest) -> str:
    if request.blob_id:
        return f"blob:{request.blob_id}"
    input_source = request.input_source or ""
    parsed = urlparse(input_source)
    if parsed.scheme in {"http", "https"}:
        return input_source
    return Path(input_source).expanduser().resolve().as_uri()


async def _run_job(
    request: ProcessRequest,
    cancellation_requested: asyncio.Event,
    report_progress: Callable[
        [JobStage, int | None, int | None, str | None], Awaitable[None]
    ]
    | None = None,
    blob_store: BlobStore | None = None,
) -> ProcessArtifact:
    temp_dir = get_request_temp_dir()
    if cancellation_requested.is_set():
        raise JobCancelledError

    progress_queue: asyncio.Queue[
        tuple[JobStage, int | None, int | None, str | None]
    ] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def enqueue_progress(
        stage: str,
        current: int | None,
        total: int | None,
        message: str | None,
    ) -> None:
        loop.call_soon_threadsafe(
            progress_queue.put_nowait,
            (stage, current, total, message),
        )

    process_task = asyncio.create_task(
        process_core_input(
            _build_process_input(
                request,
                temp_dir,
                cancellation_requested.is_set,
                enqueue_progress,
                blob_store=blob_store,
            )
        )
    )
    while not process_task.done():
        progress_task = asyncio.create_task(progress_queue.get())
        done, _ = await asyncio.wait(
            {process_task, progress_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if progress_task in done:
            if report_progress is not None:
                await report_progress(*progress_task.result())
        else:
            progress_task.cancel()
            await asyncio.gather(progress_task, return_exceptions=True)
    while not progress_queue.empty():
        if report_progress is not None:
            await report_progress(*progress_queue.get_nowait())
        else:
            progress_queue.get_nowait()
    result = await process_task

    persisted = persist_result_output_if_requested(result, request.output_file)
    if persisted.ignored_output_file:
        logger.warning(
            "Ignoring output_file for document asset conversions; output is a directory."
        )
    return ProcessArtifact(
        task_type=result.task_type,
        title=result.text_context.title,
        markdown=result.markdown,
        source_uri=_source_uri(request),
        asset_base_uri=result.asset_base_uri,
        platform=result.platform,
        output_file=persisted.output_file,
        output_dir=persisted.output_dir,
    )


def create_app(
    *,
    allowed_roots: tuple[str | Path, ...] | None = None,
    bearer_token: str | None = None,
    blob_dir: str | Path | None = None,
) -> FastAPI:
    """Build FastAPI app instance."""
    blob_store = (
        BlobStore(blob_dir) if blob_dir is not None else BlobStore.from_environment()
    )

    async def run_job(
        request: ProcessRequest,
        cancellation_requested: asyncio.Event,
        report_progress: Callable[
            [JobStage, int | None, int | None, str | None], Awaitable[None]
        ]
        | None = None,
    ) -> ProcessArtifact:
        return await _run_job(
            request,
            cancellation_requested,
            report_progress,
            blob_store=blob_store,
        )

    jobs = JobManager(run_job)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await jobs.shutdown()

    app = FastAPI(
        title="aimd HTTP",
        description="Context preparation HTTP API for LLM workflows",
        version=version("aimd-tool"),
        lifespan=lifespan,
    )
    app.state.job_manager = jobs
    app.state.blob_store = blob_store
    path_policy = (
        PathPolicy.from_environment()
        if allowed_roots is None
        else PathPolicy.from_roots(allowed_roots)
    )
    token = bearer_token if bearer_token is not None else os.getenv("AIMD_API_TOKEN")

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if token:
            supplied = request.headers.get("authorization", "")
            expected = f"Bearer {token}"
            if not hmac.compare_digest(supplied, expected):
                from fastapi.responses import JSONResponse

                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)

    def normalize_request(request: ProcessRequest) -> ProcessRequest:
        try:
            return path_policy.normalize_request(request)
        except PathAccessError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    def prepare_request(request: ProcessRequest) -> ProcessRequest:
        request = normalize_request(request)
        if not request.blob_id:
            return request
        try:
            blob_store.resolve(request.blob_id)
        except BlobIdError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except BlobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Blob not found") from exc
        return request

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse()

    @app.get("/readyz", response_model=HealthResponse)
    async def readyz() -> HealthResponse:
        return HealthResponse()

    @app.post("/v1/blobs", response_model=BlobUploadResponse)
    async def upload_blob(
        file: UploadFile = File(...),
        blob_id: str | None = Form(default=None),
    ) -> BlobUploadResponse:
        data = await file.read()
        try:
            stored = blob_store.put(data, filename=file.filename, blob_id=blob_id)
        except BlobIdError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return BlobUploadResponse(
            blob_id=stored.blob_id,
            bytes=stored.size,
            filename=file.filename or stored.filename,
        )

    @app.post("/v1/process", response_model=ProcessResponse)
    async def process(request: ProcessRequest) -> ProcessResponse:
        request = prepare_request(request)
        try:
            temp_dir = get_request_temp_dir()

            result = await process_core_input(
                _build_process_input(request, temp_dir, blob_store=blob_store)
            )

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
        except Exception as exc:
            logger.error(f"HTTP processing failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(
        "/v1/jobs",
        response_model=JobCreated,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_job(request: ProcessRequest) -> JobCreated:
        request = prepare_request(request)
        snapshot = await jobs.create(request)
        return JobCreated(
            job_id=snapshot.job_id,
            state=snapshot.state,
            status_url=f"/v1/jobs/{snapshot.job_id}",
            events_url=f"/v1/jobs/{snapshot.job_id}/events",
        )

    @app.get("/v1/jobs/{job_id}", response_model=JobSnapshot)
    async def get_job(job_id: str) -> JobSnapshot:
        try:
            return jobs.get(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @app.get(
        "/v1/jobs/{job_id}/events",
        responses={
            200: {
                "description": (
                    "Server-sent JobEvent payloads. Use Last-Event-ID to resume after "
                    "the last received sequence."
                ),
                "content": {
                    "text/event-stream": {
                        "schema": {
                            "$ref": "#/components/schemas/JobEvent",
                        }
                    }
                },
            }
        },
    )
    async def get_job_events(
        job_id: str,
        request: Request,
        last_event_id: int | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        try:
            jobs.get(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

        async def stream_events():
            async for event in jobs.events(job_id, after_sequence=last_event_id or 0):
                if await request.is_disconnected():
                    return
                yield (
                    f"id: {event.sequence}\n"
                    "event: job\n"
                    f"data: {event.model_dump_json()}\n\n"
                )

        return StreamingResponse(
            stream_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete(
        "/v1/jobs/{job_id}",
        response_model=JobSnapshot,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def cancel_job(job_id: str) -> JobSnapshot:
        try:
            return await jobs.cancel(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    def openapi() -> dict:
        if app.openapi_schema is None:
            schema = get_openapi(
                title=app.title,
                version=app.version,
                description=app.description,
                routes=app.routes,
            )
            schema.setdefault("components", {}).setdefault("schemas", {})[
                "JobEvent"
            ] = JobEvent.model_json_schema(ref_template="#/components/schemas/{model}")
            if token:
                schema["components"].setdefault("securitySchemes", {})["bearerAuth"] = {
                    "type": "http",
                    "scheme": "bearer",
                }
                schema["security"] = [{"bearerAuth": []}]
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi
    return app


app = create_app()


def main() -> None:
    """Run FastAPI service via uvicorn."""
    import uvicorn

    host = os.getenv("AIMD_HTTP_HOST", "127.0.0.1")
    port = int(os.getenv("AIMD_HTTP_PORT", "8000"))
    _validate_sidecar_bind(host, port)
    uvicorn.run("aimd.interfaces.api.app:app", host=host, port=port, reload=False)


def _validate_sidecar_bind(host: str, port: int) -> None:
    """Reject non-loopback or invalid sidecar bind settings."""
    if host != "localhost":
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError(
                "AIMD_HTTP_HOST must be a loopback IP or localhost"
            ) from exc
        if not address.is_loopback:
            raise ValueError("AIMD_HTTP_HOST must be loopback-only")
    if not 1 <= port <= 65535:
        raise ValueError("AIMD_HTTP_PORT must be between 1 and 65535")
