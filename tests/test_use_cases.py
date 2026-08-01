from pathlib import Path
import sys

import pytest
from markitdown import FileConversionException
from markitdown._exceptions import FailedConversionAttempt

import aimd.core.process as process_mod
from aimd.core.models import ProcessInput, TextContext
from aimd.core.process import (
    convert_file_with_markitdown,
    get_input_route,
    process_input,
)
from aimd.core.errors import (
    BackendUnavailableError,
    ProcessingFailedError,
    UnsupportedInputError,
)


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
    monkeypatch.setattr(process_mod, "_pdf_has_extractable_text", lambda _: True)

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
    monkeypatch.setattr(process_mod, "_pdf_has_extractable_text", lambda _: False)

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
        language: str | None,
        model: str | None,
        received_temp_dir: Path | None,
        task_type: str | None,
        start: int | None,
        end: int | None,
        precision: str | None,
    ):
        assert Path(input_path) == epub
        assert language is None
        assert model is None
        assert received_temp_dir == temp_dir
        assert task_type == "convert"
        assert start is None
        assert end is None
        assert precision is None
        return TextContext(title="doc", chunk_list=["c"]), output_dir

    result = await process_input(
        ProcessInput(input_source=epub.as_posix(), temp_dir=temp_dir),
        process_url=_unexpected_process_url,
        process_file=_process_file,
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
        language: str | None,
        model: str | None,
        received_temp_dir: Path | None,
        task_type: str | None,
        start: int | None,
        end: int | None,
        precision: str | None,
    ):
        assert Path(input_path) == image
        assert language == "zh"
        assert model == "tiny"
        assert received_temp_dir == temp_dir
        assert task_type == "ocr"
        assert start == 0
        assert end == 1
        assert precision == "bf16"
        return TextContext(title="page", chunk_list=["text"]), None

    result = await process_input(
        ProcessInput(
            input_source=image.as_posix(),
            task_type="ocr",
            model="tiny",
            language="zh",
            start=0,
            end=1,
            temp_dir=temp_dir,
            precision="bf16",
        ),
        process_url=_unexpected_process_url,
        process_file=_process_file,
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
        is_supported_file_fn=lambda _: True,
    )
    assert result.task_type == "transcript"
    assert result.text_context.chunk_list == ["t"]
    assert result.platform == "youtube"


@pytest.mark.asyncio
async def test_use_case_url_passes_precision_to_url_processor() -> None:
    seen: dict[str, object] = {}

    async def _process_url(*args):  # noqa: ANN002
        seen["precision"] = args[8]
        return TextContext(title="a", chunk_list=["t"]), "youtube"

    result = await process_input(
        ProcessInput(
            input_source="https://example.com/video",
            model="qwen3-asr-1.7b",
            precision="8bit",
        ),
        process_url=_process_url,
        process_file=_unexpected_process_file,
        is_supported_file_fn=lambda _: True,
    )
    assert result.task_type == "transcript"
    assert seen["precision"] == "8bit"


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
            is_supported_file_fn=lambda _: False,
        )


@pytest.mark.asyncio
async def test_process_input_preserves_backend_unavailable_error(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "a.mp3"
    audio.write_text("x", encoding="utf-8")

    async def _process_file(*args):  # noqa: ANN002
        raise BackendUnavailableError("no ASR backend available")

    with pytest.raises(BackendUnavailableError, match="no ASR backend available"):
        await process_input(
            ProcessInput(input_source=str(audio)),
            process_url=_unexpected_process_url,
            process_file=_process_file,
            is_supported_file_fn=lambda _: True,
        )


@pytest.mark.asyncio
async def test_run_markitdown_restores_aimd_error_from_file_conversion() -> None:
    domain_error = BackendUnavailableError("backend missing")

    def _convert() -> None:
        raise FileConversionException(
            attempts=[
                FailedConversionAttempt(
                    converter=object(),
                    exc_info=(RuntimeError, RuntimeError("noise"), None),
                ),
                FailedConversionAttempt(
                    converter=object(),
                    exc_info=(
                        BackendUnavailableError,
                        domain_error,
                        domain_error.__traceback__,
                    ),
                ),
            ]
        )

    with pytest.raises(BackendUnavailableError, match="backend missing") as caught:
        await process_mod._run_markitdown(_convert)

    assert caught.value is domain_error
    assert isinstance(caught.value.__context__, FileConversionException)


@pytest.mark.asyncio
async def test_run_markitdown_preserves_domain_error_cause_and_traceback() -> None:
    def _raise_domain_error() -> None:
        try:
            raise OSError("backend import failed")
        except OSError as cause:
            raise BackendUnavailableError("backend missing") from cause

    try:
        _raise_domain_error()
    except BackendUnavailableError as domain_error:
        original_traceback = domain_error.__traceback__
        try:
            raise RuntimeError("converter wrapper") from domain_error
        except RuntimeError:
            wrapper_exc_info = sys.exc_info()

    def _convert() -> None:
        raise FileConversionException(
            attempts=[
                FailedConversionAttempt(converter=object(), exc_info=wrapper_exc_info)
            ]
        )

    with pytest.raises(BackendUnavailableError, match="backend missing") as caught:
        await process_mod._run_markitdown(_convert)

    assert isinstance(caught.value.__cause__, OSError)
    traceback = caught.value.__traceback__
    while traceback is not None and traceback is not original_traceback:
        traceback = traceback.tb_next
    assert traceback is original_traceback


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suffix", "task_type", "patch_target"),
    [
        (".mp3", "transcript", "aimd.plugins.asr._plugin._transcribe_file_sync"),
        (".png", "ocr", "aimd.plugins.ocr._plugin._recognize_ocr_sync"),
        (".epub", "convert", "aimd.plugins.doc._plugin.process_doc_with_assets"),
    ],
)
async def test_aimd_owned_routes_do_not_fall_through_to_markitdown_builtins(
    monkeypatch,
    tmp_path: Path,
    suffix: str,
    task_type: str,
    patch_target: str,
) -> None:
    source = tmp_path / f"input{suffix}"
    source.write_bytes(b"fake")

    def _raise_backend_unavailable(*args, **kwargs):  # noqa: ANN002, ANN003
        raise BackendUnavailableError("owned converter unavailable")

    monkeypatch.setattr(patch_target, _raise_backend_unavailable)

    with pytest.raises(BackendUnavailableError, match="owned converter unavailable"):
        await convert_file_with_markitdown(source, task_type=task_type)


@pytest.mark.asyncio
async def test_run_markitdown_wraps_unknown_conversion_failure() -> None:
    unknown = ValueError("converter exploded")

    def _convert() -> None:
        raise FileConversionException(
            attempts=[
                FailedConversionAttempt(
                    converter=object(),
                    exc_info=(ValueError, unknown, None),
                )
            ]
        )

    with pytest.raises(ProcessingFailedError, match="converter exploded") as caught:
        await process_mod._run_markitdown(_convert)

    assert isinstance(caught.value.__cause__, FileConversionException)


@pytest.mark.asyncio
async def test_run_markitdown_wraps_plain_unknown_exception() -> None:
    def _convert() -> None:
        raise RuntimeError("unexpected markitdown failure")

    with pytest.raises(
        ProcessingFailedError, match="unexpected markitdown failure"
    ) as caught:
        await process_mod._run_markitdown(_convert)

    assert isinstance(caught.value.__cause__, RuntimeError)
