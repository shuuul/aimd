"""In-memory job lifecycle and SSE event storage for the local sidecar."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from aimd.core.errors import ProcessingCancelledError
from aimd.interfaces.api.schemas import (
    CancellationStatus,
    JobEvent,
    JobSnapshot,
    JobStage,
    JobState,
    ProcessArtifact,
    ProcessRequest,
)

ProgressReporter = Callable[
    [JobStage, int | None, int | None, str | None], Awaitable[None]
]
JobRunner = Callable[..., Awaitable[ProcessArtifact]]
TERMINAL_STATES: frozenset[JobState] = frozenset({"completed", "failed", "cancelled"})


class JobNotFoundError(LookupError):
    """Raised when a job identifier is unknown or expired."""


class JobCancelledError(Exception):
    """Raised by a runner at a cooperative cancellation checkpoint."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class _JobRecord:
    job_id: str
    request: ProcessRequest
    state: JobState = "queued"
    stage: JobStage | None = None
    artifact: ProcessArtifact | None = None
    error: str | None = None
    cancellation_status: CancellationStatus = "none"
    current: int | None = None
    total: int | None = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    events: list[JobEvent] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None


class JobManager:
    """Own local jobs, bounded terminal history, and resumable event streams."""

    def __init__(self, runner: JobRunner, *, max_terminal_jobs: int = 64) -> None:
        if max_terminal_jobs < 1:
            raise ValueError("max_terminal_jobs must be at least 1")
        self._runner = runner
        self._max_terminal_jobs = max_terminal_jobs
        self._jobs: dict[str, _JobRecord] = {}

    async def create(self, request: ProcessRequest) -> JobSnapshot:
        self._prune_terminal_jobs()
        record = _JobRecord(job_id=str(uuid4()), request=request)
        self._jobs[record.job_id] = record
        await self._emit(record, message="Job queued")
        record.task = asyncio.create_task(
            self._run(record), name=f"aimd-job-{record.job_id}"
        )
        return self._snapshot(record)

    def get(self, job_id: str) -> JobSnapshot:
        return self._snapshot(self._record(job_id))

    async def cancel(self, job_id: str) -> JobSnapshot:
        record = self._record(job_id)
        if record.state in TERMINAL_STATES:
            return self._snapshot(record)

        record.cancel_event.set()
        record.cancellation_status = "requested"
        if record.state == "queued":
            record.state = "cancelled"
            record.cancellation_status = "cancelled"
            if record.task is not None:
                record.task.cancel()
            await self._emit(record, message="Job cancelled before processing started")
        else:
            await self._emit(
                record,
                message="Cancellation requested; waiting for a safe processor checkpoint",
            )
        return self._snapshot(record)

    async def shutdown(self, *, timeout: float = 5.0) -> None:
        """Request cooperative stops, then cancel tasks that exceed the grace period."""
        active = [
            record
            for record in self._jobs.values()
            if record.state not in TERMINAL_STATES
        ]
        for record in active:
            await self.cancel(record.job_id)
        tasks = [record.task for record in active if record.task is not None]
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def events(
        self, job_id: str, *, after_sequence: int = 0
    ) -> AsyncIterator[JobEvent]:
        record = self._record(job_id)
        cursor = max(after_sequence, 0)
        while True:
            pending = [event for event in record.events if event.sequence > cursor]
            for event in pending:
                cursor = event.sequence
                yield event
            if record.state in TERMINAL_STATES:
                return
            async with record.condition:
                if not any(event.sequence > cursor for event in record.events):
                    await record.condition.wait()

    async def _run(self, record: _JobRecord) -> None:
        if record.state == "cancelled":
            return
        try:
            record.state = "running"
            record.stage = _initial_stage(record.request)
            await self._emit(record, message="Processing started")
            if _accepts_progress_reporter(self._runner):
                artifact = await self._runner(
                    record.request,
                    record.cancel_event,
                    lambda stage, current, total, message: self._report_progress(
                        record, stage, current, total, message
                    ),
                )
            else:
                artifact = await self._runner(record.request, record.cancel_event)
            record.artifact = artifact
            record.state = "completed"
            if record.cancel_event.is_set():
                record.cancellation_status = "completed_after_request"
            record.stage = "saving" if artifact.output_file else record.stage
            await self._emit(record, message="Processing completed")
        except (JobCancelledError, ProcessingCancelledError):
            record.state = "cancelled"
            record.cancellation_status = "cancelled"
            await self._emit(record, message="Job stopped at a cancellation checkpoint")
        except asyncio.CancelledError:
            if record.state != "cancelled":
                record.state = "cancelled"
                record.cancellation_status = "cancelled"
                await self._emit(record, message="Job task cancelled")
        except Exception as exc:
            record.state = "failed"
            record.error = str(exc)
            await self._emit(record, message="Processing failed")
        finally:
            if record.state in TERMINAL_STATES:
                self._prune_terminal_jobs()

    async def _emit(self, record: _JobRecord, *, message: str) -> None:
        record.updated_at = _utcnow()
        event = JobEvent(
            job_id=record.job_id,
            state=record.state,
            stage=record.stage,
            current=record.current,
            total=record.total,
            message=message,
            cancellation_status=record.cancellation_status,
            sequence=len(record.events) + 1,
            created_at=record.updated_at,
        )
        record.events.append(event)
        async with record.condition:
            record.condition.notify_all()

    def _record(self, job_id: str) -> _JobRecord:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise JobNotFoundError(job_id) from exc

    @staticmethod
    def _snapshot(record: _JobRecord) -> JobSnapshot:
        return JobSnapshot(
            job_id=record.job_id,
            state=record.state,
            request=record.request,
            artifact=record.artifact,
            error=record.error,
            stage=record.stage,
            current=record.current,
            total=record.total,
            cancellation_status=record.cancellation_status,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    async def _report_progress(
        self,
        record: _JobRecord,
        stage: JobStage,
        current: int | None,
        total: int | None,
        message: str | None,
    ) -> None:
        if record.state != "running":
            return
        record.stage = stage
        record.current = current
        record.total = total
        await self._emit(record, message=message or "Processing progress")

    def _prune_terminal_jobs(self) -> None:
        terminal = sorted(
            (
                record
                for record in self._jobs.values()
                if record.state in TERMINAL_STATES
            ),
            key=lambda record: record.updated_at,
        )
        for record in terminal[: -self._max_terminal_jobs]:
            del self._jobs[record.job_id]


def _initial_stage(request: ProcessRequest) -> JobStage:
    source = request.input_source or ""
    if not request.blob_id and source.startswith(("http://", "https://")):
        return "downloading"
    if request.task_type == "transcript":
        return "transcribing"
    if request.task_type == "ocr":
        return "ocr"
    if request.task_type == "convert":
        return "converting"
    return "extracting"


def _accepts_progress_reporter(runner: JobRunner) -> bool:
    """Keep two-argument job runners compatible with the progress-aware contract."""
    import inspect

    try:
        parameters = inspect.signature(runner).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    return len(positional) >= 3 or any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    )
