from pathlib import Path

from fastapi.testclient import TestClient

from aimd.interfaces.api.app import create_app
from aimd.core.models import ProcessResult, TextContext
from aimd.core.errors import (
    EngineUnavailableError,
    ProcessingFailedError,
    UnsupportedInputError,
)
from aimd.plugins.asr import EngineCapability


def _fake_list_transcription_engines():
    class _Result:
        auto_selected_engine = "qwen"
        engines = {
            "mlx": EngineCapability("mlx", False, "unsupported", None),
            "qwen": EngineCapability("qwen", True, None, None),
        }

    return _Result()


def _make_client(monkeypatch, process_result=None, process_exc=None) -> TestClient:
    async def _fake_process_input(request):  # noqa: ARG001
        if process_exc:
            raise process_exc
        return process_result

    monkeypatch.setattr(
        "aimd.interfaces.api.app.process_core_input",
        _fake_process_input,
    )
    monkeypatch.setattr(
        "aimd.interfaces.api.app.list_transcription_engines", _fake_list_transcription_engines
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


def test_engines_endpoint(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    response = client.get("/v1/engines")
    assert response.status_code == 200
    body = response.json()
    assert "engines" in body
    assert len(body["engines"]) == 2
    assert {engine["name"] for engine in body["engines"]} == {
        "mlx",
        "qwen",
    }


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
    assert body["chunk_list"] == ["hello transcript"]
    assert Path(body["output_file"]).exists()
    assert output_file.read_text(encoding="utf-8") == "hello transcript"


def test_process_maps_domain_error_to_http_status(monkeypatch) -> None:
    client = _make_client(
        monkeypatch, process_exc=EngineUnavailableError("unavailable")
    )
    response = client.post("/v1/process", json={"input_source": "audio.wav"})
    assert response.status_code == 422
    assert "unavailable" in response.json()["detail"]


def test_process_maps_processing_failed_error_to_http_500(monkeypatch) -> None:
    client = _make_client(monkeypatch, process_exc=ProcessingFailedError("boom"))
    response = client.post("/v1/process", json={"input_source": "audio.wav"})
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"
