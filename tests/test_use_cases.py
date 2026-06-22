from pathlib import Path

import pytest

from aimd.core.application.models import InputRoute, ProcessInput, ProcessResult
from aimd.core.application.use_cases import input_routing
from aimd.core.application.use_cases.input_routing import get_input_route
from aimd.core.application.use_cases.process_input import ProcessInputUseCase
from aimd.core.application.use_cases.processors.convert import ConvertTaskProcessor
from aimd.core.application.use_cases.processors.ocr import OCRTaskProcessor
from aimd.core.errors import UnsupportedInputError
from aimd.core.types import TextContext


class _FakeTaskProcessor:
    def __init__(self, result: ProcessResult):
        self._result = result

    async def process(self, request: ProcessInput, route: InputRoute):  # noqa: ARG002
        return self._result


class _UnexpectedTaskProcessor:
    async def process(self, request: ProcessInput, route: InputRoute):  # noqa: ARG002
        raise AssertionError("should not call processor")


def test_input_route_classifies_url() -> None:
    route = get_input_route(
        "https://example.com/video", is_supported_file=lambda _: False
    )
    assert route.source_kind == "url"
    assert route.task_type == "transcript"


def test_input_route_classifies_audio_video_and_document(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    video = tmp_path / "video.mp4"
    document = tmp_path / "notes.txt"
    for path in (audio, video, document):
        path.write_text("x", encoding="utf-8")

    audio_route = get_input_route(audio.as_posix(), is_supported_file=lambda _: False)
    video_route = get_input_route(video.as_posix(), is_supported_file=lambda _: False)
    document_route = get_input_route(
        document.as_posix(), is_supported_file=lambda _: True
    )

    assert audio_route.source_kind == "audio_file"
    assert audio_route.task_type == "transcript"
    assert video_route.source_kind == "video_file"
    assert video_route.task_type == "transcript"
    assert document_route.source_kind == "document_file"
    assert document_route.task_type == "convert"


def test_input_route_classifies_images_and_explicit_pdf_ocr(
    monkeypatch, tmp_path: Path
) -> None:
    image = tmp_path / "page.png"
    pdf = tmp_path / "scan.pdf"
    image.write_text("x", encoding="utf-8")
    pdf.write_text("x", encoding="utf-8")
    monkeypatch.setattr(input_routing, "_pdf_has_extractable_text", lambda _: True)

    image_route = get_input_route(image.as_posix(), is_supported_file=lambda _: False)
    pdf_convert_route = get_input_route(
        pdf.as_posix(), is_supported_file=lambda _: True
    )
    pdf_ocr_route = get_input_route(
        pdf.as_posix(), is_supported_file=lambda _: True, requested_task_type="ocr"
    )

    assert image_route.source_kind == "image_file"
    assert image_route.task_type == "ocr"
    assert pdf_convert_route.source_kind == "document_file"
    assert pdf_convert_route.task_type == "convert"
    assert pdf_ocr_route.source_kind == "document_file"
    assert pdf_ocr_route.task_type == "ocr"


def test_input_route_classifies_scanned_pdf_as_ocr(monkeypatch, tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_text("x", encoding="utf-8")
    monkeypatch.setattr(input_routing, "_pdf_has_extractable_text", lambda _: False)

    route = get_input_route(pdf.as_posix(), is_supported_file=lambda _: True)

    assert route.source_kind == "document_file"
    assert route.task_type == "ocr"


@pytest.mark.asyncio
async def test_process_convert_passes_temp_dir_to_epub_processor(
    tmp_path: Path,
) -> None:
    epub = tmp_path / "aimd.book.epub"
    epub.write_text("x", encoding="utf-8")
    temp_dir = tmp_path / "tmp"
    output_dir = tmp_path / "book-output"

    async def _process_file(
        input_path: str,
        engine: str,
        language: str | None,
        model: str | None,
        received_temp_dir: Path | None,
    ):
        assert Path(input_path) == epub
        assert engine == "auto"
        assert language is None
        assert model is None
        assert received_temp_dir == temp_dir
        return TextContext(title="book", chunk_list=["c"]), output_dir

    processor = ConvertTaskProcessor(process_file=_process_file)
    result = await processor.process(
        ProcessInput(input_source=epub.as_posix(), temp_dir=temp_dir),
        InputRoute(source_kind="document_file", task_type="convert"),
    )

    assert result.output_dir == output_dir


@pytest.mark.asyncio
async def test_process_ocr_passes_request_options_to_processor(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_text("x", encoding="utf-8")
    temp_dir = tmp_path / "tmp"

    async def _process_file(
        input_path: str,
        engine: str,
        model: str | None,
        language: str | None,
        start: int | None,
        end: int | None,
        received_temp_dir: Path | None,
    ):
        assert Path(input_path) == image
        assert engine == "mlx4ocr"
        assert model == "tiny"
        assert language == "zh"
        assert start == 0
        assert end == 1
        assert received_temp_dir == temp_dir
        return TextContext(title="page", chunk_list=["text"])

    processor = OCRTaskProcessor(process_file=_process_file)
    result = await processor.process(
        ProcessInput(
            input_source=image.as_posix(),
            task_type="ocr",
            transcribe_engine="mlx4ocr",
            model="tiny",
            language="zh",
            start=0,
            end=1,
            temp_dir=temp_dir,
        ),
        InputRoute(source_kind="image_file", task_type="ocr"),
    )

    assert result.task_type == "ocr"
    assert result.text_context.chunk_list == ["text"]


@pytest.mark.asyncio
async def test_use_case_transcript_flow() -> None:
    use_case = ProcessInputUseCase(
        processors={
            "transcript": _FakeTaskProcessor(
                ProcessResult(
                    task_type="transcript",
                    text_context=TextContext(title="a", chunk_list=["t"]),
                    platform="youtube",
                )
            ),
            "convert": _UnexpectedTaskProcessor(),
            "ocr": _UnexpectedTaskProcessor(),
        },
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

    use_case = ProcessInputUseCase(
        processors={
            "transcript": _FakeTaskProcessor(
                ProcessResult(
                    task_type="transcript",
                    text_context=TextContext(title="a", chunk_list=["t"]),
                )
            ),
            "convert": _UnexpectedTaskProcessor(),
            "ocr": _UnexpectedTaskProcessor(),
        },
        is_supported_file=lambda _: True,
    )

    result = await use_case.execute(ProcessInput(input_source=str(audio)))
    assert result.task_type == "transcript"
    assert result.platform is None


@pytest.mark.asyncio
async def test_use_case_file_convert_flow(tmp_path: Path) -> None:
    doc = tmp_path / "a.txt"
    doc.write_text("x", encoding="utf-8")

    use_case = ProcessInputUseCase(
        processors={
            "transcript": _UnexpectedTaskProcessor(),
            "convert": _FakeTaskProcessor(
                ProcessResult(
                    task_type="convert",
                    text_context=TextContext(title="d", chunk_list=["c"]),
                )
            ),
            "ocr": _UnexpectedTaskProcessor(),
        },
        is_supported_file=lambda _: True,
    )

    result = await use_case.execute(ProcessInput(input_source=str(doc)))
    assert result.task_type == "convert"
    assert result.output_dir is None


@pytest.mark.asyncio
async def test_use_case_unsupported_input_raises() -> None:
    use_case = ProcessInputUseCase(
        processors={
            "transcript": _UnexpectedTaskProcessor(),
            "convert": _UnexpectedTaskProcessor(),
            "ocr": _UnexpectedTaskProcessor(),
        },
        is_supported_file=lambda _: False,
    )

    with pytest.raises(UnsupportedInputError):
        await use_case.execute(ProcessInput(input_source="not_supported"))
