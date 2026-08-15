from pathlib import Path

from fastapi.testclient import TestClient

from aimd.interfaces.api.app import create_app
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
    assert output_file.read_text(encoding="utf-8") == "hello transcript"


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
