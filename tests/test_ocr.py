from pathlib import Path
import subprocess

import pytest

from aimd.core.errors import EngineUnavailableError, ProcessingFailedError
from aimd.plugins.ocr.engines import (
    MLX4OCREngine,
    OCRPage,
    OCRResult,
    TransformersOCREngine,
    _resolve_mlx4ocr_model,
    resolve_ocr_engine,
)
from aimd.plugins.ocr.models import create_transformers_ocr_model, resolve_transformers_ocr_model
from aimd.plugins.ocr.models.generic import GenericTransformersOCRModel
from aimd.plugins.ocr.models.got import GOTOCRModel
from aimd.plugins.ocr.models.unlimited import (
    UnlimitedOCRModel,
    normalize_unlimited_ocr_output,
    read_unlimited_ocr_output_files,
)
from aimd.plugins.ocr.processor import process_ocr


def test_resolve_ocr_engine_selects_platform_defaults(monkeypatch) -> None:
    monkeypatch.setattr("aimd.plugins.ocr.engines.platform.system", lambda: "Darwin")
    assert resolve_ocr_engine("auto") == "mlx4ocr"

    monkeypatch.setattr("aimd.plugins.ocr.engines.platform.system", lambda: "Linux")
    assert resolve_ocr_engine("auto") == "transformers"


def test_resolve_ocr_engine_rejects_wrong_platform(monkeypatch) -> None:
    monkeypatch.setattr("aimd.plugins.ocr.engines.platform.system", lambda: "Linux")
    with pytest.raises(EngineUnavailableError):
        resolve_ocr_engine("mlx4ocr")


def test_resolve_mlx4ocr_model_maps_aimd_names() -> None:
    assert _resolve_mlx4ocr_model(None) == ("ppocrv6", "medium")
    assert _resolve_mlx4ocr_model("paddleocr_v6") == ("ppocrv6", "medium")
    assert _resolve_mlx4ocr_model("glm_ocr") == ("glm-ocr", None)
    assert _resolve_mlx4ocr_model("paddleocr_vl") == ("paddleocr-vl", None)


def test_resolve_transformers_ocr_model_maps_vlm_models() -> None:
    assert resolve_transformers_ocr_model(None) == "stepfun-ai/GOT-OCR-2.0-hf"
    assert resolve_transformers_ocr_model("got_ocr") == "stepfun-ai/GOT-OCR-2.0-hf"
    assert resolve_transformers_ocr_model("unlimited_ocr") == "baidu/Unlimited-OCR"
    assert resolve_transformers_ocr_model("baidu/Unlimited-OCR") == "baidu/Unlimited-OCR"
    assert resolve_transformers_ocr_model("glm_ocr") == "zai-org/GLM-OCR"
    assert (
        resolve_transformers_ocr_model("paddleocr_vl")
        == "PaddlePaddle/PaddleOCR-VL-1.5"
    )
    assert resolve_transformers_ocr_model("org/custom-model") == "org/custom-model"


def test_resolve_transformers_ocr_model_rejects_ppocrv6() -> None:
    with pytest.raises(ProcessingFailedError):
        resolve_transformers_ocr_model("paddleocr_v6")


def test_create_transformers_ocr_model_selects_model_adapter() -> None:
    assert isinstance(create_transformers_ocr_model("got_ocr"), GOTOCRModel)
    assert isinstance(create_transformers_ocr_model("unlimited_ocr"), UnlimitedOCRModel)
    generic = create_transformers_ocr_model("zai-org/GLM-OCR")
    assert isinstance(generic, GenericTransformersOCRModel)
    assert generic.model_id == "zai-org/GLM-OCR"


def test_mlx4ocr_pdf_command_uses_default_paddleocr_v6(monkeypatch, tmp_path: Path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_text("x", encoding="utf-8")
    captured: dict[str, list[str]] = {}

    def _run(command, **kwargs):  # noqa: ANN001
        captured["command"] = command
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(command, 0, stdout="recognized", stderr="")

    monkeypatch.setattr("aimd.plugins.ocr.engines.shutil.which", lambda _: "mlx4ocr")
    monkeypatch.setattr("aimd.plugins.ocr.engines.subprocess.run", _run)

    text = MLX4OCREngine()._recognize_pdf_or_document(
        pdf,
        model=None,
        start=0,
        end=2,
        temp_dir=tmp_path,
    )

    assert text == "recognized"
    command = captured["command"]
    assert command[command.index("--engine") + 1] == "ppocrv6"
    assert command[command.index("--variant") + 1] == "medium"
    assert command[command.index("--start") + 1] == "0"
    assert command[command.index("--end") + 1] == "2"


@pytest.mark.parametrize(
    ("model", "mlx4ocr_engine"),
    [("glm_ocr", "glm-ocr"), ("paddleocr_vl", "paddleocr-vl")],
)
def test_mlx4ocr_pdf_command_maps_vlm_models(
    monkeypatch, tmp_path: Path, model: str, mlx4ocr_engine: str
):
    pdf = tmp_path / "scan.pdf"
    pdf.write_text("x", encoding="utf-8")
    captured: dict[str, list[str]] = {}

    def _run(command, **kwargs):  # noqa: ANN001, ARG001
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="recognized", stderr="")

    monkeypatch.setattr("aimd.plugins.ocr.engines.shutil.which", lambda _: "mlx4ocr")
    monkeypatch.setattr("aimd.plugins.ocr.engines.subprocess.run", _run)

    text = MLX4OCREngine()._recognize_pdf_or_document(
        pdf,
        model=model,
        start=None,
        end=None,
        temp_dir=tmp_path,
    )

    assert text == "recognized"
    command = captured["command"]
    assert command[command.index("--engine") + 1] == mlx4ocr_engine
    assert "--variant" not in command


def test_transformers_engine_ocr_image_uses_resolved_model(
    monkeypatch, tmp_path: Path
) -> None:
    image = tmp_path / "scan.png"
    image.write_text("x", encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeModel:
        def recognize_image(self, input_path, *, language, temp_dir):  # noqa: ANN001
            captured["input_path"] = input_path
            captured["language"] = language
            captured["temp_dir"] = temp_dir
            return "recognized text"

        def recognize_images(self, image_paths, *, language, temp_dir):  # noqa: ANN001
            raise AssertionError("not used")

    def _create_model(model):  # noqa: ANN001
        captured["model"] = model
        return _FakeModel()

    monkeypatch.setattr(
        "aimd.plugins.ocr.engines.create_transformers_ocr_model", _create_model
    )

    result = TransformersOCREngine().recognize(
        image,
        model="got_ocr",
        language="zh",
    )

    assert result == OCRResult(
        title="scan",
        pages=(OCRPage(page_index=None, text="recognized text"),),
    )
    assert captured == {
        "model": "got_ocr",
        "input_path": image,
        "language": "zh",
        "temp_dir": None,
    }


def test_transformers_engine_unlimited_ocr_image_uses_model_infer(
    monkeypatch, tmp_path: Path
) -> None:
    image = tmp_path / "scan.png"
    image.write_text("x", encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeModel:
        def infer(self, tokenizer, **kwargs):  # noqa: ANN001
            captured["tokenizer"] = tokenizer
            captured.update(kwargs)
            return "recognized markdown"

    monkeypatch.setattr(
        "aimd.plugins.ocr.models.unlimited.get_cached_model_and_processor",
        lambda model_name, loader: (_FakeModel(), "tokenizer"),  # noqa: ARG005
    )

    text = UnlimitedOCRModel().recognize_image(
        image,
        temp_dir=tmp_path,
    )

    assert text == "recognized markdown"
    assert captured["tokenizer"] == "tokenizer"
    assert captured["prompt"] == "<image>document parsing."
    assert captured["image_file"] == image.as_posix()
    assert captured["base_size"] == 1024
    assert captured["image_size"] == 640
    assert captured["crop_mode"] is True
    assert captured["no_repeat_ngram_size"] == 35
    assert captured["ngram_window"] == 128
    assert captured["save_results"] is True


def test_transformers_engine_unlimited_ocr_pdf_uses_infer_multi(
    monkeypatch, tmp_path: Path
) -> None:
    page_paths = [tmp_path / "page-1.png", tmp_path / "page-2.png"]
    for page_path in page_paths:
        page_path.write_text("x", encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeModel:
        def infer_multi(self, tokenizer, **kwargs):  # noqa: ANN001
            captured["tokenizer"] = tokenizer
            captured.update(kwargs)
            return ["page one", "page two"]

    monkeypatch.setattr(
        "aimd.plugins.ocr.models.unlimited.get_cached_model_and_processor",
        lambda model_name, loader: (_FakeModel(), "tokenizer"),  # noqa: ARG005
    )

    result = UnlimitedOCRModel().recognize_images(
        page_paths,
        temp_dir=tmp_path,
    )

    assert result == ["page one", "page two"]
    assert captured["tokenizer"] == "tokenizer"
    assert captured["prompt"] == "<image>Multi page parsing."
    assert captured["image_files"] == [path.as_posix() for path in page_paths]
    assert captured["image_size"] == 1024
    assert captured["no_repeat_ngram_size"] == 35
    assert captured["ngram_window"] == 1024
    assert captured["save_results"] is True


def test_normalize_unlimited_ocr_output_accepts_common_shapes() -> None:
    assert normalize_unlimited_ocr_output("text", expected_pages=1) == ["text"]
    assert normalize_unlimited_ocr_output(["one", "two"], expected_pages=2) == [
        "one",
        "two",
    ]
    assert normalize_unlimited_ocr_output(
        {"pages": [{"text": "one"}, {"markdown": "two"}]},
        expected_pages=2,
    ) == ["one", "two"]


def test_read_unlimited_ocr_output_files_fallback(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "scan"
    nested.mkdir(parents=True)
    (nested / "page_0002.md").write_text("two", encoding="utf-8")
    (nested / "page_0001.md").write_text("one", encoding="utf-8")

    assert read_unlimited_ocr_output_files(output_dir, expected_pages=2) == [
        "one",
        "two",
    ]


@pytest.mark.asyncio
async def test_process_ocr_wraps_engine_result_as_text_context(
    monkeypatch, tmp_path: Path
) -> None:
    image = tmp_path / "page.png"
    image.write_text("x", encoding="utf-8")

    class _FakeEngine:
        def recognize(self, input_path, **kwargs):
            assert input_path == image
            assert kwargs["model"] == "paddleocr_v6"
            assert kwargs["language"] == "zh"
            assert kwargs["start"] is None
            assert kwargs["end"] is None
            return OCRResult(
                title="page",
                pages=(OCRPage(page_index=None, text="recognized text"),),
            )

    monkeypatch.setattr(
        "aimd.plugins.ocr.processor.create_ocr_engine", lambda engine: _FakeEngine()
    )

    result = await process_ocr(
        image,
        engine="mlx4ocr",
        model="paddleocr_v6",
        language="zh",
    )

    assert result.title == "page"
    assert result.chunk_list == ["recognized text"]
    assert result.split_header_level == 2


@pytest.mark.asyncio
async def test_process_ocr_rejects_invalid_page_range(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_text("x", encoding="utf-8")

    with pytest.raises(ProcessingFailedError):
        await process_ocr(pdf, start=2, end=1)
