from pathlib import Path

import pytest

from aimd.types import TextContext


@pytest.mark.asyncio
async def test_mcp_healthz() -> None:
    from aimd.mcp_server import healthz

    assert await healthz() == {"status": "ok"}


@pytest.mark.asyncio
async def test_mcp_process_input_transcript(monkeypatch, tmp_path: Path) -> None:
    from aimd.mcp_server import process_input

    async def _mock_process_transcript_input(**kwargs):
        return TextContext(
            title="mock-title", chunk_list=["hello"], split_header_level=None
        )

    monkeypatch.setattr(
        "aimd.mcp_server.ensure_supported_input", lambda _: "transcript"
    )
    monkeypatch.setattr(
        "aimd.mcp_server.process_transcript_input",
        _mock_process_transcript_input,
    )

    output_file = tmp_path / "out.md"
    result = await process_input("input.mp3", output_file=str(output_file))
    assert result["task_type"] == "transcript"
    assert result["output_file"] is not None
    assert output_file.read_text(encoding="utf-8") == "hello"
