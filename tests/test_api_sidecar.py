import asyncio
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from aimd.core.models import ProcessResult, TextContext
from aimd.interfaces.api.app import _validate_sidecar_bind, create_app
from aimd.interfaces.api.jobs import JobCancelledError, JobManager
from aimd.interfaces.api.paths import PathAccessError, PathPolicy
from aimd.interfaces.api.schemas import ProcessArtifact, ProcessRequest


def test_path_policy_normalizes_paths_inside_allowed_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    policy = PathPolicy.from_roots((root,))

    request = policy.normalize_request(
        ProcessRequest(
            input_source=str(root / "folder" / ".." / "source.md"),
            output_file=str(root / "output" / "result.md"),
        )
    )

    assert request.input_source == str((root / "source.md").resolve())
    assert request.output_file == str((root / "output" / "result.md").resolve())


def test_path_policy_rejects_traversal_and_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    secret = outside / "secret.md"
    secret.write_text("secret", encoding="utf-8")
    (root / "escape.md").symlink_to(secret)
    policy = PathPolicy.from_roots((root,))

    with pytest.raises(PathAccessError, match="outside configured allowed roots"):
        policy.normalize(root / ".." / "outside" / "new.md", "output")
    with pytest.raises(PathAccessError, match="outside configured allowed roots"):
        policy.normalize(root / "escape.md", "input")
    with pytest.raises(PathAccessError, match=r"Only HTTP\(S\)"):
        policy.normalize_request(ProcessRequest(input_source="file:///etc/passwd"))


def test_api_enforces_allowed_roots_before_processing(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "source.md"
    source.write_text("source", encoding="utf-8")
    seen: dict[str, str] = {}

    async def process(request):
        seen["input_source"] = request.input_source
        return ProcessResult(
            task_type="convert",
            text_context=TextContext(title="source", chunk_list=["body"]),
            markdown="body",
        )

    monkeypatch.setattr("aimd.interfaces.api.app.process_core_input", process)
    with TestClient(create_app(allowed_roots=(root,))) as client:
        accepted = client.post("/v1/process", json={"input_source": str(source)})
        rejected_input = client.post(
            "/v1/process", json={"input_source": str(tmp_path / "outside.md")}
        )
        rejected_output = client.post(
            "/v1/jobs",
            json={
                "input_source": str(source),
                "output_file": str(tmp_path / "outside.md"),
            },
        )

    assert accepted.status_code == 200
    assert seen["input_source"] == str(source.resolve())
    assert rejected_input.status_code == 403
    assert rejected_output.status_code == 403


def test_sidecar_bearer_auth_protects_health_jobs_and_openapi(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with TestClient(
        create_app(allowed_roots=(root,), bearer_token="desktop-secret")
    ) as client:
        assert client.get("/healthz").status_code == 401
        assert client.get("/readyz").status_code == 401
        assert (
            client.get(
                "/healthz", headers={"Authorization": "Bearer wrong-secret"}
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/v1/jobs", json={"input_source": str(root / "source.md")}
            ).status_code
            == 401
        )
        authorized = {"Authorization": "Bearer desktop-secret"}
        assert client.get("/healthz", headers=authorized).json() == {"status": "ok"}
        assert client.get("/readyz", headers=authorized).json() == {"status": "ok"}
        openapi = client.get("/openapi.json", headers=authorized).json()

    assert openapi["components"]["securitySchemes"]["bearerAuth"] == {
        "type": "http",
        "scheme": "bearer",
    }
    assert openapi["security"] == [{"bearerAuth": []}]


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_sidecar_bind_accepts_loopback(host: str) -> None:
    _validate_sidecar_bind(host, 49152)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com"])
def test_sidecar_bind_rejects_non_loopback(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        _validate_sidecar_bind(host, 8000)


@pytest.mark.parametrize("port", [0, 65536])
def test_sidecar_bind_rejects_invalid_port(port: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 65535"):
        _validate_sidecar_bind("127.0.0.1", port)


@pytest.mark.asyncio
async def test_job_manager_shutdown_cooperatively_stops_active_work() -> None:
    started = asyncio.Event()

    async def runner(request, cancellation_requested):  # noqa: ARG001
        started.set()
        await cancellation_requested.wait()
        raise JobCancelledError

    manager = JobManager(runner)
    created = await manager.create(ProcessRequest(input_source="source.md"))
    await started.wait()
    await manager.shutdown(timeout=1)

    snapshot = manager.get(created.job_id)
    assert snapshot.state == "cancelled"
    assert snapshot.cancellation_status == "cancelled"


@pytest.mark.asyncio
async def test_job_manager_shutdown_forces_task_after_grace_period() -> None:
    started = asyncio.Event()

    async def runner(request, cancellation_requested):  # noqa: ARG001
        started.set()
        await asyncio.Event().wait()
        return ProcessArtifact(
            task_type="convert",
            title="never",
            markdown="never",
            source_uri="file:///never",
        )

    manager = JobManager(runner)
    created = await manager.create(ProcessRequest(input_source="source.md"))
    await started.wait()
    await manager.shutdown(timeout=0)

    assert manager.get(created.job_id).state == "cancelled"


def test_sidecar_contract_documentation_covers_runtime_boundary() -> None:
    contract = (Path(__file__).parents[1] / "docs" / "sidecar.md").read_text(
        encoding="utf-8"
    )

    for required in (
        "AIMD_ALLOWED_ROOTS",
        "AIMD_API_TOKEN",
        "AIMD_HTTP_PORT",
        "GET /readyz",
        "Last-Event-ID",
        "completed_after_request",
        "caller-owned `output_dir`",
    ):
        assert required in contract
