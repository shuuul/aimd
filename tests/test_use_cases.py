from pathlib import Path

import pytest

from aimd.application.models import ProcessInput
from aimd.application.use_cases.process_input import ProcessInputUseCase
from aimd.errors import UnsupportedInputError
from aimd.types import TextContext


@pytest.mark.asyncio
async def test_use_case_transcript_flow() -> None:
    async def _transcript(*args):  # noqa: ARG001
        return TextContext(title="a", chunk_list=["t"]), "youtube"

    async def _convert(*args):  # noqa: ARG001
        raise AssertionError("should not call convert")

    use_case = ProcessInputUseCase(
        transcript_processor=_transcript,
        convert_processor=_convert,
        is_supported_file=lambda _: True,
    )

    result = await use_case.execute(
        ProcessInput(input_source="https://example.com/video")
    )
    assert result.task_type == "transcript"
    assert result.text_context.chunk_list == ["t"]
    assert result.platform == "youtube"


@pytest.mark.asyncio
async def test_use_case_local_audio_flow(tmp_path: Path) -> None:
    audio = tmp_path / "a.mp3"
    audio.write_text("x", encoding="utf-8")

    async def _transcript(*args):  # noqa: ARG001
        return TextContext(title="a", chunk_list=["t"]), None

    async def _convert(*args):  # noqa: ARG001
        raise AssertionError("should not call convert")

    use_case = ProcessInputUseCase(
        transcript_processor=_transcript,
        convert_processor=_convert,
        is_supported_file=lambda _: True,
    )

    result = await use_case.execute(ProcessInput(input_source=str(audio)))
    assert result.task_type == "transcript"
    assert result.platform is None


@pytest.mark.asyncio
async def test_use_case_file_convert_flow(tmp_path: Path) -> None:
    doc = tmp_path / "a.txt"
    doc.write_text("x", encoding="utf-8")

    async def _transcript(*args):  # noqa: ARG001
        raise AssertionError("should not call transcript")

    async def _convert(*args):  # noqa: ARG001
        return TextContext(title="d", chunk_list=["c"]), None

    use_case = ProcessInputUseCase(
        transcript_processor=_transcript,
        convert_processor=_convert,
        is_supported_file=lambda _: True,
    )

    result = await use_case.execute(ProcessInput(input_source=str(doc)))
    assert result.task_type == "convert"
    assert result.output_dir is None


@pytest.mark.asyncio
async def test_use_case_unsupported_input_raises() -> None:
    async def _transcript(*args):  # noqa: ARG001
        raise AssertionError("should not call transcript")

    async def _convert(*args):  # noqa: ARG001
        raise AssertionError("should not call convert")

    use_case = ProcessInputUseCase(
        transcript_processor=_transcript,
        convert_processor=_convert,
        is_supported_file=lambda _: False,
    )

    with pytest.raises(UnsupportedInputError):
        await use_case.execute(ProcessInput(input_source="not_supported"))
