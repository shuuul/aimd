import asyncio
import json
import threading
import time

from fastapi.testclient import TestClient
import pytest

from aimd.core.errors import ProcessingCancelledError
from aimd.core.models import ProcessResult, TextContext
from aimd.interfaces.api.app import _run_job, create_app
from aimd.interfaces.api.jobs import (
    JobCancelledError,
    JobManager,
    JobNotFoundError,
)
from aimd.interfaces.api.schemas import ProcessArtifact, ProcessRequest


def _artifact(markdown: str = "# Result\n") -> ProcessArtifact:
    return ProcessArtifact(
        task_type="convert",
        title="Result",
        markdown=markdown,
        source_uri="file:///tmp/source.md",
    )


async def _wait_for_terminal(manager: JobManager, job_id: str):
    for _ in range(1000):
        snapshot = manager.get(job_id)
        if snapshot.state in {"completed", "failed", "cancelled"}:
            return snapshot
        await asyncio.sleep(0.001)
    raise AssertionError("job did not reach a terminal state")


@pytest.mark.asyncio
async def test_job_lifecycle_and_resumable_events() -> None:
    async def runner(request, cancellation_requested, report_progress):  # noqa: ARG001
        await report_progress("converting", 1, 2, "Converting chapter 1 of 2")
        await report_progress("converting", 2, 2, "Converting chapter 2 of 2")
        return _artifact()

    manager = JobManager(runner)
    created = await manager.create(
        ProcessRequest(input_source="document.md", task_type="convert")
    )
    assert created.state == "queued"

    terminal = await _wait_for_terminal(manager, created.job_id)
    assert terminal.state == "completed"
    assert terminal.stage == "converting"
    assert terminal.current == 2
    assert terminal.total == 2
    assert terminal.artifact == _artifact()

    all_events = [event async for event in manager.events(created.job_id)]
    assert [event.state for event in all_events] == [
        "queued",
        "running",
        "running",
        "running",
        "completed",
    ]
    assert [event.sequence for event in all_events] == [1, 2, 3, 4, 5]
    assert (all_events[2].current, all_events[2].total) == (1, 2)
    assert all_events[3].message == "Converting chapter 2 of 2"

    resumed = [
        event async for event in manager.events(created.job_id, after_sequence=1)
    ]
    assert [event.sequence for event in resumed] == [2, 3, 4, 5]


@pytest.mark.asyncio
async def test_event_consumer_can_disconnect_and_resume() -> None:
    release = asyncio.Event()

    async def runner(request, cancellation_requested):  # noqa: ARG001
        await release.wait()
        return _artifact()

    manager = JobManager(runner)
    created = await manager.create(ProcessRequest(input_source="document.md"))
    stream = manager.events(created.job_id)
    first = await anext(stream)
    assert first.sequence == 1
    await stream.aclose()

    release.set()
    await _wait_for_terminal(manager, created.job_id)
    resumed = [
        event async for event in manager.events(created.job_id, after_sequence=1)
    ]
    assert [event.sequence for event in resumed] == [2, 3]


@pytest.mark.asyncio
async def test_job_failure_is_stored_and_emitted() -> None:
    async def runner(request, cancellation_requested):  # noqa: ARG001
        raise RuntimeError("processor failed")

    manager = JobManager(runner)
    created = await manager.create(ProcessRequest(input_source="broken.md"))
    terminal = await _wait_for_terminal(manager, created.job_id)

    assert terminal.state == "failed"
    assert terminal.error == "processor failed"
    assert terminal.artifact is None
    events = [event async for event in manager.events(created.job_id)]
    assert events[-1].state == "failed"
    assert events[-1].message == "Processing failed"


@pytest.mark.asyncio
async def test_cancel_before_start_never_calls_runner() -> None:
    called = False

    async def runner(request, cancellation_requested):  # noqa: ARG001
        nonlocal called
        called = True
        return _artifact()

    manager = JobManager(runner)
    created = await manager.create(ProcessRequest(input_source="document.md"))
    cancelled = await manager.cancel(created.job_id)
    await asyncio.sleep(0)

    assert cancelled.state == "cancelled"
    assert cancelled.cancellation_status == "cancelled"
    assert called is False
    events = [event async for event in manager.events(created.job_id)]
    assert [event.state for event in events] == ["queued", "cancelled"]


@pytest.mark.asyncio
async def test_cancel_during_cooperative_work_stops_at_checkpoint() -> None:
    started = asyncio.Event()

    async def runner(request, cancellation_requested):  # noqa: ARG001
        started.set()
        await cancellation_requested.wait()
        raise JobCancelledError

    manager = JobManager(runner)
    created = await manager.create(ProcessRequest(input_source="audio.wav"))
    await started.wait()

    requested = await manager.cancel(created.job_id)
    assert requested.state == "running"
    assert requested.cancellation_status == "requested"

    terminal = await _wait_for_terminal(manager, created.job_id)
    assert terminal.state == "cancelled"
    assert terminal.cancellation_status == "cancelled"
    assert terminal.artifact is None


@pytest.mark.asyncio
async def test_cancel_during_uninterruptible_work_retains_completed_artifact() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner(request, cancellation_requested):  # noqa: ARG001
        started.set()
        await release.wait()
        return _artifact("exact output")

    manager = JobManager(runner)
    created = await manager.create(ProcessRequest(input_source="audio.wav"))
    await started.wait()

    requested = await manager.cancel(created.job_id)
    assert requested.state == "running"
    assert requested.cancellation_status == "requested"

    release.set()
    terminal = await _wait_for_terminal(manager, created.job_id)
    assert terminal.state == "completed"
    assert terminal.cancellation_status == "completed_after_request"
    assert terminal.artifact == _artifact("exact output")


@pytest.mark.asyncio
async def test_blocking_processor_uses_thread_safe_cancel_and_progress_callbacks(
    monkeypatch,
) -> None:
    started = threading.Event()

    async def process(request):
        def blocking_worker():
            request.progress_reporter(
                "converting", 1, 2, "Completed blocking conversion step 1 of 2"
            )
            started.set()
            while not request.cancellation_check():
                time.sleep(0.001)
            raise ProcessingCancelledError("cancelled at blocking worker checkpoint")

        return await asyncio.to_thread(blocking_worker)

    monkeypatch.setattr("aimd.interfaces.api.app.process_core_input", process)
    manager = JobManager(_run_job)
    created = await manager.create(
        ProcessRequest(input_source="document.epub", task_type="convert")
    )
    await asyncio.to_thread(started.wait, 1)

    requested = await manager.cancel(created.job_id)
    assert requested.cancellation_status == "requested"
    terminal = await _wait_for_terminal(manager, created.job_id)

    assert terminal.state == "cancelled"
    assert terminal.cancellation_status == "cancelled"
    events = [event async for event in manager.events(created.job_id)]
    progress = next(event for event in events if event.current == 1)
    assert (progress.stage, progress.current, progress.total) == ("converting", 1, 2)


@pytest.mark.asyncio
async def test_terminal_history_is_bounded() -> None:
    async def runner(request, cancellation_requested):  # noqa: ARG001
        return _artifact()

    manager = JobManager(runner, max_terminal_jobs=2)
    job_ids: list[str] = []
    for source in ("one.md", "two.md", "three.md"):
        created = await manager.create(ProcessRequest(input_source=source))
        job_ids.append(created.job_id)
        await _wait_for_terminal(manager, created.job_id)

    with pytest.raises(JobNotFoundError):
        manager.get(job_ids[0])
    assert manager.get(job_ids[1]).state == "completed"
    assert manager.get(job_ids[2]).state == "completed"


def test_job_api_lifecycle_sse_and_unknown_job(monkeypatch) -> None:
    markdown = "\n# Exact\n"

    async def process(request):  # noqa: ARG001
        return ProcessResult(
            task_type="convert",
            text_context=TextContext(title="Exact", chunk_list=["lossy"]),
            markdown=markdown,
        )

    monkeypatch.setattr("aimd.interfaces.api.app.process_core_input", process)
    with TestClient(create_app()) as client:
        created_response = client.post(
            "/v1/jobs", json={"input_source": "document.md", "task_type": "convert"}
        )
        assert created_response.status_code == 202
        created = created_response.json()
        assert created["state"] == "queued"

        for _ in range(100):
            snapshot_response = client.get(created["status_url"])
            if snapshot_response.json()["state"] == "completed":
                break
        snapshot = snapshot_response.json()
        assert snapshot["artifact"]["markdown"] == markdown

        events_response = client.get(
            created["events_url"], headers={"Last-Event-ID": "1"}
        )
        assert events_response.status_code == 200
        data_lines = [
            line.removeprefix("data: ")
            for line in events_response.text.splitlines()
            if line.startswith("data: ")
        ]
        events = [json.loads(line) for line in data_lines]
        assert [event["sequence"] for event in events] == [2, 3]
        assert events[-1]["state"] == "completed"

        assert client.get("/v1/jobs/unknown").status_code == 404
        assert client.delete("/v1/jobs/unknown").status_code == 404
        assert client.get("/v1/jobs/unknown/events").status_code == 404
