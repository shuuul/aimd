import pytest

from pathlib import Path
import aimd.interfaces.mcp as mcp_app

from aimd.core.errors import ProcessingFailedError
from aimd.core.models import ProcessResult, TextContext


@pytest.mark.asyncio
async def test_mcp_healthz() -> None:
    assert await mcp_app.healthz() == {"status": "ok"}


@pytest.mark.asyncio
async def test_mcp_process_input_transcript(monkeypatch, tmp_path: Path) -> None:
    async def _fake_process_input(request):  # noqa: ARG001
        return ProcessResult(
            task_type="transcript",
            text_context=TextContext(
                title="mock-title",
                chunk_list=["hello"],
                split_header_level=None,
            ),
            markdown="# Raw\n\nhello",
            asset_base_uri="https://example.com/watch",
        )

    monkeypatch.setattr(
        "aimd.interfaces.mcp.app.process_core_input", _fake_process_input
    )

    output_file = tmp_path / "out.md"
    result = await mcp_app.process_input("input.mp3", output_file=str(output_file))
    assert result["task_type"] == "transcript"
    assert result["markdown"] == "# Raw\n\nhello"
    assert result["asset_base_uri"] == "https://example.com/watch"
    assert result["output_file"] is not None
    assert output_file.read_text(encoding="utf-8") == result["markdown"]


@pytest.mark.asyncio
async def test_mcp_empty_transcript_output_is_processing_failure(
    monkeypatch, tmp_path: Path
) -> None:
    async def _fake_process_input(request):  # noqa: ARG001
        return ProcessResult(
            task_type="transcript",
            text_context=TextContext(title="empty", chunk_list=[]),
        )

    monkeypatch.setattr(
        "aimd.interfaces.mcp.app.process_core_input", _fake_process_input
    )

    output_file = tmp_path / "out.md"
    with pytest.raises(
        ProcessingFailedError, match="Transcription returned empty content"
    ):
        await mcp_app.process_input("input.mp3", output_file=str(output_file))
    assert not output_file.exists()


@pytest.mark.asyncio
async def test_mcp_process_input_accepts_valid_task_type(
    monkeypatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    async def _fake_process_input(request):
        seen["task_type"] = request.task_type
        return ProcessResult(
            task_type="ocr",
            text_context=TextContext(
                title="page",
                chunk_list=["ocr"],
                split_header_level=None,
            ),
            markdown="ocr",
        )

    monkeypatch.setattr(
        "aimd.interfaces.mcp.app.process_core_input", _fake_process_input
    )

    result = await mcp_app.process_input(
        "page.png",
        task_type="ocr",
        output_file=str(tmp_path / "out.md"),
    )
    assert result["task_type"] == "ocr"
    assert seen["task_type"] == "ocr"


@pytest.mark.asyncio
async def test_mcp_process_input_rejects_invalid_task_type() -> None:
    from aimd.core.errors import UnsupportedInputError

    with pytest.raises(UnsupportedInputError, match="Unsupported task"):
        await mcp_app.process_input("input.mp3", task_type="nope")


@pytest.mark.asyncio
async def test_mcp_process_input_forwards_precision(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_process_input(request):
        seen["model"] = request.model
        seen["precision"] = request.precision
        return ProcessResult(
            task_type="transcript",
            text_context=TextContext(title="t", chunk_list=["x"]),
        )

    monkeypatch.setattr(
        "aimd.interfaces.mcp.app.process_core_input", _fake_process_input
    )

    result = await mcp_app.process_input(
        "input.mp3",
        model="qwen3-asr-1.7b",
        precision="4bit",
    )
    assert result["task_type"] == "transcript"
    assert seen == {"model": "qwen3-asr-1.7b", "precision": "4bit"}


@pytest.mark.asyncio
async def test_mcp_process_input_schema_exposes_precision() -> None:
    tools = await mcp_app.mcp.list_tools()
    process_tool = next(tool for tool in tools if tool.name == "process_input")
    assert "precision" in process_tool.input_schema["properties"]


@pytest.mark.asyncio
async def test_mcp_process_input_schema_exposes_task_enum() -> None:
    tools = await mcp_app.mcp.list_tools()
    process_tool = next(tool for tool in tools if tool.name == "process_input")
    task_schema = process_tool.input_schema["properties"]["task_type"]

    assert task_schema["anyOf"][0]["enum"] == ["transcript", "convert", "ocr"]
