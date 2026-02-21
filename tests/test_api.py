from pathlib import Path

from fastapi.testclient import TestClient

from aimd.api import app
from aimd.errors import EngineUnavailableError, ProcessingFailedError
from aimd.types import TextContext


client = TestClient(app)


def test_healthz() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_process_rejects_unknown_input() -> None:
    response = client.post(
        "/v1/process",
        json={
            "input_source": "this_is_not_a_file_or_url",
        },
    )
    assert response.status_code == 400
    assert "Unsupported input source" in response.json()["detail"]


def test_engines_endpoint() -> None:
    response = client.get("/v1/engines")
    assert response.status_code == 200
    body = response.json()
    assert "engines" in body
    assert len(body["engines"]) == 4
    assert {engine["name"] for engine in body["engines"]} == {
        "yap",
        "mlx",
        "cuda",
        "cpu",
    }


def test_process_transcript_success_with_output_file(
    monkeypatch, tmp_path: Path
) -> None:
    async def _mock_process_transcript_input(**kwargs):
        return TextContext(
            title="mock-title",
            chunk_list=["hello transcript"],
            split_header_level=None,
        )

    monkeypatch.setattr("aimd.api.ensure_supported_input", lambda _: "transcript")
    monkeypatch.setattr(
        "aimd.api.process_transcript_input", _mock_process_transcript_input
    )

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
    async def _mock_fail(**kwargs):
        raise EngineUnavailableError("Engine 'mlx' unavailable")

    monkeypatch.setattr("aimd.api.ensure_supported_input", lambda _: "transcript")
    monkeypatch.setattr("aimd.api.process_transcript_input", _mock_fail)

    response = client.post(
        "/v1/process",
        json={
            "input_source": "audio.wav",
            "transcribe_engine": "mlx",
        },
    )
    assert response.status_code == 422
    assert "unavailable" in response.json()["detail"]


def test_process_maps_processing_failed_error_to_http_500(monkeypatch) -> None:
    async def _mock_fail(**kwargs):
        raise ProcessingFailedError("boom")

    monkeypatch.setattr("aimd.api.ensure_supported_input", lambda _: "transcript")
    monkeypatch.setattr("aimd.api.process_transcript_input", _mock_fail)

    response = client.post(
        "/v1/process",
        json={
            "input_source": "audio.wav",
            "transcribe_engine": "cpu",
        },
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"
