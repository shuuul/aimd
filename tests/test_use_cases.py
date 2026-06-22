from pathlib import Path

import pytest

import aimd.core.router as router
from aimd.core.models import ProcessInput, TextContext
from aimd.core.process import process_input
from aimd.core.errors import UnsupportedInputError
from aimd.core.router import get_input_route


async def _unexpected_process_url(*args, **kwargs):  # noqa: ANN002, ANN003
    raise AssertionError("should not process URL")


async def _unexpected_process_file(*args, **kwargs):  # noqa: ANN002, ANN003
    raise AssertionError("should not process file")


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
    monkeypatch.setattr(router, "_pdf_has_extractable_text", lambda _: True)

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
    monkeypatch.setattr(router, "_pdf_has_extractable_text", lambda _: False)

    route = get_input_route(pdf.as_posix(), is_supported_file=lambda _: True)

    assert route.source_kind == "document_file"
    assert route.task_type == "ocr"


@pytest.mark.asyncio
async def test_use_case_convert_passes_temp_dir_to_markitdown(
    tmp_path: Path,
) -> None:
    epub = tmp_path / "document.epub"
    epub.write_text("x", encoding="utf-8")
    temp_dir = tmp_path / "tmp"
    output_dir = tmp_path / "doc-output"

    async def _process_file(
        input_path: str,
        engine: str,
        language: str | None,
        model: str | None,
        received_temp_dir: Path | None,
        task_type: str | None,
        start: int | None,
        end: int | None,
    ):
        assert Path(input_path) == epub
        assert engine == "auto"
        assert language is None
        assert model is None
        assert received_temp_dir == temp_dir
        assert task_type == "convert"
        assert start is None
        assert end is None
        return TextContext(title="doc", chunk_list=["c"]), output_dir

    result = await process_input(
        ProcessInput(input_source=epub.as_posix(), temp_dir=temp_dir),
        process_url=_unexpected_process_url,
        process_file=_process_file,
        resolve_engine=lambda engine: engine,
        is_supported_file_fn=lambda _: True,
    )

    assert result.task_type == "convert"
    assert result.output_dir == output_dir


@pytest.mark.asyncio
async def test_use_case_ocr_passes_options_to_markitdown(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_text("x", encoding="utf-8")
    temp_dir = tmp_path / "tmp"

    async def _process_file(
        input_path: str,
        engine: str,
        language: str | None,
        model: str | None,
        received_temp_dir: Path | None,
        task_type: str | None,
        start: int | None,
        end: int | None,
    ):
        assert Path(input_path) == image
        assert engine == "mlx4ocr"
        assert language == "zh"
        assert model == "tiny"
        assert received_temp_dir == temp_dir
        assert task_type == "ocr"
        assert start == 0
        assert end == 1
        return TextContext(title="page", chunk_list=["text"]), None

    result = await process_input(
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
        process_url=_unexpected_process_url,
        process_file=_process_file,
        resolve_engine=lambda engine: engine,
        is_supported_file_fn=lambda _: True,
    )

    assert result.task_type == "ocr"
    assert result.text_context.chunk_list == ["text"]


@pytest.mark.asyncio
async def test_use_case_transcript_flow() -> None:
    async def _process_url(*args):  # noqa: ANN002
        return TextContext(title="a", chunk_list=["t"]), "youtube"

    result = await process_input(
        ProcessInput(input_source="https://example.com/video"),
        process_url=_process_url,
        process_file=_unexpected_process_file,
        resolve_engine=lambda engine: engine,
        is_supported_file_fn=lambda _: True,
    )
    assert result.task_type == "transcript"
    assert result.text_context.chunk_list == ["t"]
    assert result.platform == "youtube"


@pytest.mark.asyncio
async def test_use_case_local_audio_flow(tmp_path: Path) -> None:
    audio = tmp_path / "a.mp3"
    audio.write_text("x", encoding="utf-8")

    async def _process_file(*args):  # noqa: ANN002
        return TextContext(title="a", chunk_list=["t"]), None

    result = await process_input(
        ProcessInput(input_source=str(audio)),
        process_url=_unexpected_process_url,
        process_file=_process_file,
        resolve_engine=lambda engine: engine,
        is_supported_file_fn=lambda _: True,
    )
    assert result.task_type == "transcript"
    assert result.platform is None


@pytest.mark.asyncio
async def test_use_case_file_convert_flow(tmp_path: Path) -> None:
    doc = tmp_path / "a.txt"
    doc.write_text("x", encoding="utf-8")

    async def _process_file(*args):  # noqa: ANN002
        return TextContext(title="d", chunk_list=["c"]), None

    result = await process_input(
        ProcessInput(input_source=str(doc)),
        process_url=_unexpected_process_url,
        process_file=_process_file,
        resolve_engine=lambda engine: engine,
        is_supported_file_fn=lambda _: True,
    )
    assert result.task_type == "convert"
    assert result.output_dir is None


@pytest.mark.asyncio
async def test_use_case_unsupported_input_raises() -> None:
    with pytest.raises(UnsupportedInputError):
        await process_input(
            ProcessInput(input_source="not_supported"),
            process_url=_unexpected_process_url,
            process_file=_unexpected_process_file,
            resolve_engine=lambda engine: engine,
            is_supported_file_fn=lambda _: False,
        )
