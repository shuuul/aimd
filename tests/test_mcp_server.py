import pytest

from pathlib import Path
import aimd.mcp as mcp_app

from aimd.core.application.models import ProcessResult
from aimd.core.types import TextContext


@pytest.mark.asyncio
async def test_mcp_healthz() -> None:
    assert await mcp_app.healthz() == {"status": "ok"}


@pytest.mark.asyncio
async def test_mcp_process_input_transcript(monkeypatch, tmp_path: Path) -> None:
    class _FakeProcessUseCase:
        async def execute(self, request):  # noqa: ARG002
            return ProcessResult(
                task_type="transcript",
                text_context=TextContext(
                    title="mock-title",
                    chunk_list=["hello"],
                    split_header_level=None,
                ),
            )

    class _FakeContainer:
        process_input_use_case = _FakeProcessUseCase()

    monkeypatch.setattr("aimd.mcp.app.container", _FakeContainer())

    output_file = tmp_path / "out.md"
    result = await mcp_app.process_input("input.mp3", output_file=str(output_file))
    assert result["task_type"] == "transcript"
    assert result["output_file"] is not None
    assert output_file.read_text(encoding="utf-8") == "hello"
