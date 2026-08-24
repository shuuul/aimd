import asyncio
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from aimd.interfaces.api.app import _run_job, create_app
from aimd.interfaces.api.schemas import ProcessRequest
from aimd.core.models import ProcessResult, TextContext
from aimd.core.process import process_input as process_core_input
from aimd.core.errors import (
    BackendUnavailableError,
    ProcessingFailedError,
    UnsupportedInputError,
)


def _make_client(monkeypatch, process_result=None, process_exc=None) -> TestClient:
    async def _fake_process_input(request):  # noqa: ARG001
        if process_exc:
            raise process_exc
        return process_result

    monkeypatch.setattr(
        "aimd.interfaces.api.app.process_core_input",
        _fake_process_input,
    )
    return TestClient(create_app())


def test_healthz(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_process_rejects_unknown_input(monkeypatch) -> None:
    client = _make_client(
        monkeypatch,
        process_exc=UnsupportedInputError("Unsupported input source"),
    )
    response = client.post("/v1/process", json={"input_source": "x"})
    assert response.status_code == 400


def test_process_transcript_success_with_output_file(
    monkeypatch, tmp_path: Path
) -> None:
    result = ProcessResult(
        task_type="transcript",
        text_context=TextContext(
            title="mock-title",
            chunk_list=["hello transcript"],
            split_header_level=None,
        ),
        markdown="# Raw title\n\nhello transcript",
        asset_base_uri="https://example.com/watch",
    )
    client = _make_client(monkeypatch, process_result=result)

    output_file = tmp_path / "out.md"
    response = client.post(
        "/v1/process",
        json={
            "input_source": "input.mp3",
            "output_file": str(output_file),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "transcript"
    assert body["title"] == "mock-title"
    assert body["markdown"] == "# Raw title\n\nhello transcript"
    assert body["asset_base_uri"] == "https://example.com/watch"
    assert body["chunk_list"] == ["hello transcript"]
    assert Path(body["output_file"]).exists()
    assert output_file.read_text(encoding="utf-8") == result.markdown


def test_process_empty_transcript_output_is_processing_failure(
    monkeypatch, tmp_path: Path
) -> None:
    result = ProcessResult(
        task_type="transcript",
        text_context=TextContext(title="empty", chunk_list=[]),
    )
    client = _make_client(monkeypatch, process_result=result)

    output_file = tmp_path / "out.md"
    response = client.post(
        "/v1/process",
        json={"input_source": "input.mp3", "output_file": str(output_file)},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Transcription returned empty content"
    assert not output_file.exists()


def test_process_forwards_model_and_precision(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_process_input(request):
        seen["model"] = request.model
        seen["precision"] = request.precision
        return ProcessResult(
            task_type="transcript",
            text_context=TextContext(title="t", chunk_list=["x"]),
        )

    monkeypatch.setattr(
        "aimd.interfaces.api.app.process_core_input",
        _fake_process_input,
    )
    client = TestClient(create_app())
    response = client.post(
        "/v1/process",
        json={
            "input_source": "input.mp3",
            "model": "qwen3-asr-1.7b",
            "precision": "4bit",
        },
    )

    assert response.status_code == 200
    assert seen == {"model": "qwen3-asr-1.7b", "precision": "4bit"}


def test_process_forwards_remote_settings(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_process_input(request):
        seen["asr_base_url"] = request.asr_base_url
        seen["ocr_base_url"] = request.ocr_base_url
        return ProcessResult(
            task_type="transcript",
            text_context=TextContext(title="t", chunk_list=["x"]),
        )

    monkeypatch.setattr(
        "aimd.interfaces.api.app.process_core_input", _fake_process_input
    )
    response = TestClient(create_app()).post(
        "/v1/process",
        json={
            "input_source": "input.mp3",
            "asr_base_url": "http://asr.example/v1",
            "ocr_base_url": "http://ocr.example/v1",
        },
    )

    assert response.status_code == 200
    assert seen == {
        "asr_base_url": "http://asr.example/v1",
        "ocr_base_url": "http://ocr.example/v1",
    }


def test_process_maps_domain_error_to_http_status(monkeypatch) -> None:
    client = _make_client(
        monkeypatch, process_exc=BackendUnavailableError("unavailable")
    )
    response = client.post("/v1/process", json={"input_source": "audio.wav"})
    assert response.status_code == 422
    assert "unavailable" in response.json()["detail"]


def test_process_maps_processing_failed_error_to_http_500(monkeypatch) -> None:
    client = _make_client(monkeypatch, process_exc=ProcessingFailedError("boom"))
    response = client.post("/v1/process", json={"input_source": "audio.wav"})
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_process_preserves_backend_unavailable_from_processor(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_text("x", encoding="utf-8")

    async def _process_file(*args):  # noqa: ANN002
        raise BackendUnavailableError("ASR backend unavailable on this host")

    async def _process_url(*args):  # noqa: ANN002
        raise AssertionError("should not process URL")

    async def _real_process(request):
        return await process_core_input(
            request,
            process_url=_process_url,
            process_file=_process_file,
            is_supported_file_fn=lambda _: True,
        )

    monkeypatch.setattr(
        "aimd.interfaces.api.app.process_core_input",
        _real_process,
    )
    client = TestClient(create_app())
    response = client.post("/v1/process", json={"input_source": str(audio)})
    assert response.status_code == 422
    assert "ASR backend unavailable" in response.json()["detail"]


def test_job_contract_is_documented_in_openapi(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]

    assert {
        "JobCreated",
        "JobEvent",
        "JobSnapshot",
        "ProcessArtifact",
        "ProcessRequest",
    } <= components.keys()
    assert components["ProcessArtifact"]["example"]["markdown"].endswith("\n")
    assert components["JobEvent"]["example"]["current"] == 2

    events = schema["paths"]["/v1/jobs/{job_id}/events"]["get"]
    assert any(
        parameter["name"] == "Last-Event-ID"
        and parameter["in"] == "header"
        and parameter["required"] is False
        for parameter in events["parameters"]
    )
    event_schema = events["responses"]["200"]["content"]["text/event-stream"]["schema"]
    assert event_schema == {"$ref": "#/components/schemas/JobEvent"}


def test_job_schema_freezes_cancellation_and_progress_contract(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    components = client.get("/openapi.json").json()["components"]["schemas"]

    event = components["JobEvent"]["properties"]
    assert event["state"]["enum"] == [
        "queued",
        "running",
        "completed",
        "failed",
        "cancelled",
    ]
    assert event["stage"]["anyOf"][0]["enum"] == [
        "downloading",
        "extracting",
        "transcribing",
        "ocr",
        "converting",
        "saving",
    ]
    assert event["cancellation_status"]["enum"] == [
        "none",
        "requested",
        "cancelled",
        "completed_after_request",
    ]
    assert event["current"]["anyOf"][0]["minimum"] == 0
    assert event["total"]["anyOf"][0]["minimum"] == 0


@pytest.mark.asyncio
async def test_job_artifact_preserves_plain_markdown_exactly(
    monkeypatch, tmp_path: Path
) -> None:
    markdown = "\n# Exact title\n\nBody without a final newline"
    result = ProcessResult(
        task_type="convert",
        text_context=TextContext(title="Exact title", chunk_list=["lossy body"]),
        markdown=markdown,
        asset_base_uri=f"{tmp_path.resolve().as_uri()}/",
    )

    async def _fake_process_input(request):  # noqa: ARG001
        return result

    monkeypatch.setattr(
        "aimd.interfaces.api.app.process_core_input", _fake_process_input
    )
    output_file = tmp_path / "saved.md"
    artifact = await _run_job(
        ProcessRequest(
            input_source=str(tmp_path / "source.txt"), output_file=str(output_file)
        ),
        asyncio.Event(),
    )

    assert artifact.markdown == markdown
    assert output_file.read_text(encoding="utf-8") == markdown
    assert artifact.source_uri == (tmp_path / "source.txt").resolve().as_uri()
    assert artifact.asset_base_uri == f"{tmp_path.resolve().as_uri()}/"


@pytest.mark.asyncio
async def test_job_artifact_preserves_asset_output_directory(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "document"
    output_dir.mkdir()
    markdown = "# Document\n\n![figure](images/figure.png)\n"
    (output_dir / "document.md").write_text(markdown, encoding="utf-8")
    images = output_dir / "images"
    images.mkdir()
    (images / "figure.png").write_bytes(b"image")
    result = ProcessResult(
        task_type="convert",
        text_context=TextContext(title="Document", chunk_list=["lossy"]),
        markdown=markdown,
        asset_base_uri=f"{output_dir.resolve().as_uri()}/",
        output_dir=output_dir,
    )

    async def _fake_process_input(request):  # noqa: ARG001
        return result

    monkeypatch.setattr(
        "aimd.interfaces.api.app.process_core_input", _fake_process_input
    )
    ignored_output = tmp_path / "ignored.md"
    artifact = await _run_job(
        ProcessRequest(
            input_source=str(tmp_path / "source.docx"),
            output_file=str(ignored_output),
        ),
        asyncio.Event(),
    )

    assert artifact.markdown == markdown
    assert artifact.output_dir == str(output_dir.resolve())
    assert artifact.output_file is None
    assert artifact.asset_base_uri == f"{output_dir.resolve().as_uri()}/"
    assert not ignored_output.exists()
    assert (output_dir / "document.md").read_text(encoding="utf-8") == markdown
    assert (images / "figure.png").read_bytes() == b"image"
