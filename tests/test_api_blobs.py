from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from aimd.core.models import ProcessResult, TextContext
from aimd.interfaces.api.app import create_app
from aimd.interfaces.api.paths import PathPolicy
from aimd.interfaces.api.schemas import ProcessRequest


def _result() -> ProcessResult:
    return ProcessResult(
        task_type="convert",
        text_context=TextContext(title="notes", chunk_list=["body"]),
        markdown="body",
    )


def test_path_input_source_still_processes(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "source.md"
    source.write_text("source", encoding="utf-8")
    seen: dict[str, str] = {}

    async def process(request):
        seen["input_source"] = request.input_source
        return _result()

    monkeypatch.setattr("aimd.interfaces.api.app.process_core_input", process)
    with TestClient(create_app(allowed_roots=(root,))) as client:
        response = client.post("/v1/process", json={"input_source": str(source)})

    assert response.status_code == 200
    assert seen["input_source"] == str(source.resolve())


def test_upload_blob_and_process_uses_stored_bytes(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    blob_dir = tmp_path / "blobs"
    root.mkdir()
    payload = b"# hello from blob\n"
    seen: dict[str, object] = {}

    async def process(request):
        seen["input_source"] = request.input_source
        seen["bytes"] = Path(request.input_source).read_bytes()
        return _result()

    monkeypatch.setattr("aimd.interfaces.api.app.process_core_input", process)
    with TestClient(create_app(allowed_roots=(root,), blob_dir=blob_dir)) as client:
        upload = client.post(
            "/v1/blobs",
            files={"file": ("notes.md", payload, "text/markdown")},
        )
        assert upload.status_code == 200
        body = upload.json()
        assert body["bytes"] == len(payload)
        assert body["filename"] == "notes.md"
        blob_id = body["blob_id"]
        processed = client.post("/v1/process", json={"blob_id": blob_id})
        job = client.post("/v1/jobs", json={"blob_id": blob_id, "task_type": "convert"})

    assert processed.status_code == 200
    assert processed.json()["markdown"] == "body"
    assert job.status_code == 202
    resolved = Path(str(seen["input_source"])).resolve()
    assert resolved.is_relative_to(blob_dir.resolve())
    assert seen["bytes"] == payload
    assert resolved.name == "notes.md"


def test_job_blob_path_stays_inside_sidecar_blob_dir(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    blob_dir = tmp_path / "blobs"
    root.mkdir()
    seen: list[str] = []

    async def process(request):
        seen.append(request.input_source)
        return _result()

    monkeypatch.setattr("aimd.interfaces.api.app.process_core_input", process)
    with TestClient(create_app(allowed_roots=(root,), blob_dir=blob_dir)) as client:
        blob_id = client.post(
            "/v1/blobs",
            files={"file": ("talk.md", b"talk", "text/markdown")},
        ).json()["blob_id"]
        created = client.post("/v1/jobs", json={"blob_id": blob_id})
        assert created.status_code == 202
        snapshot = client.get(created.json()["status_url"])
        for _ in range(50):
            if snapshot.json()["state"] in {"completed", "failed", "cancelled"}:
                break
            snapshot = client.get(created.json()["status_url"])

    assert snapshot.json()["state"] == "completed"
    assert snapshot.json()["artifact"]["source_uri"] == f"blob:{blob_id}"
    assert seen
    resolved = Path(seen[0]).resolve()
    assert resolved.is_relative_to(blob_dir.resolve())
    assert not resolved.is_relative_to(root.resolve()) or resolved.is_relative_to(
        blob_dir.resolve()
    )


def test_missing_both_and_both_set_are_unprocessable(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with TestClient(create_app(allowed_roots=(root,))) as client:
        missing = client.post("/v1/process", json={})
        both = client.post(
            "/v1/process",
            json={"input_source": str(root / "a.md"), "blob_id": str(uuid4())},
        )

    assert missing.status_code == 422
    assert both.status_code == 422


def test_blobs_requires_bearer_when_token_set(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with TestClient(
        create_app(allowed_roots=(root,), bearer_token="desktop-secret")
    ) as client:
        unauth = client.post(
            "/v1/blobs", files={"file": ("notes.md", b"x", "text/markdown")}
        )
        authorized = client.post(
            "/v1/blobs",
            headers={"Authorization": "Bearer desktop-secret"},
            files={"file": ("notes.md", b"x", "text/markdown")},
        )

    assert unauth.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["bytes"] == 1


def test_path_policy_skips_blob_id_input(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    policy = PathPolicy.from_roots((root,))
    blob_id = str(uuid4())
    request = policy.normalize_request(ProcessRequest(blob_id=blob_id))
    assert request.blob_id == blob_id
    assert request.input_source is None
