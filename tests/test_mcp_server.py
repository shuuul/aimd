from pathlib import Path

import pytest

from aimd.application.models import ProcessResult
from aimd.types import TextContext


@pytest.mark.asyncio
async def test_mcp_healthz() -> None:
    from aimd.mcp import healthz

    assert await healthz() == {"status": "ok"}


@pytest.mark.asyncio
async def test_mcp_process_input_transcript(monkeypatch, tmp_path: Path) -> None:
    from aimd.mcp import process_input

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

    monkeypatch.setattr("aimd.adapters.mcp.server.container", _FakeContainer())

    output_file = tmp_path / "out.md"
    result = await process_input("input.mp3", output_file=str(output_file))
    assert result["task_type"] == "transcript"
    assert result["output_file"] is not None
    assert output_file.read_text(encoding="utf-8") == "hello"
