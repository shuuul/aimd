from pathlib import Path
import subprocess

import pytest

from aimd.core.errors import EngineUnavailableError, ProcessingFailedError
from aimd.ocr.engines import OCRPage, OCRResult, resolve_ocr_engine
from aimd.ocr.mlx4ocr_engine import MLX4OCREngine, _resolve_mlx4ocr_model
from aimd.ocr.processor import process_ocr
from aimd.ocr.transformers_engine import (
    TransformersOCREngine,
    _resolve_transformers_ocr_model,
)


def test_resolve_ocr_engine_selects_platform_defaults(monkeypatch) -> None:
    monkeypatch.setattr("aimd.ocr.engines.platform.system", lambda: "Darwin")
    assert resolve_ocr_engine("auto") == "mlx4ocr"

    monkeypatch.setattr("aimd.ocr.engines.platform.system", lambda: "Linux")
    assert resolve_ocr_engine("auto") == "transformers"


def test_resolve_ocr_engine_rejects_wrong_platform(monkeypatch) -> None:
    monkeypatch.setattr("aimd.ocr.engines.platform.system", lambda: "Linux")
    with pytest.raises(EngineUnavailableError):
        resolve_ocr_engine("mlx4ocr")


def test_resolve_mlx4ocr_model_maps_aimd_names() -> None:
    assert _resolve_mlx4ocr_model(None) == ("ppocrv6", "medium")
    assert _resolve_mlx4ocr_model("paddleocr_v6") == ("ppocrv6", "medium")
    assert _resolve_mlx4ocr_model("glm_ocr") == ("glm-ocr", None)
    assert _resolve_mlx4ocr_model("paddleocr_vl") == ("paddleocr-vl", None)


def test_resolve_transformers_ocr_model_maps_vlm_models() -> None:
    assert _resolve_transformers_ocr_model(None) == "stepfun-ai/GOT-OCR-2.0-hf"
    assert _resolve_transformers_ocr_model("got_ocr") == "stepfun-ai/GOT-OCR-2.0-hf"
    assert _resolve_transformers_ocr_model("glm_ocr") == "zai-org/GLM-OCR"
    assert (
        _resolve_transformers_ocr_model("paddleocr_vl")
        == "PaddlePaddle/PaddleOCR-VL-1.5"
    )
    assert _resolve_transformers_ocr_model("org/custom-model") == "org/custom-model"


def test_resolve_transformers_ocr_model_rejects_ppocrv6() -> None:
    with pytest.raises(ProcessingFailedError):
        _resolve_transformers_ocr_model("paddleocr_v6")


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

    monkeypatch.setattr("aimd.ocr.mlx4ocr_engine.shutil.which", lambda _: "mlx4ocr")
    monkeypatch.setattr("aimd.ocr.mlx4ocr_engine.subprocess.run", _run)

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

    monkeypatch.setattr("aimd.ocr.mlx4ocr_engine.shutil.which", lambda _: "mlx4ocr")
    monkeypatch.setattr("aimd.ocr.mlx4ocr_engine.subprocess.run", _run)

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

    def _recognize_image(self, input_path, *, model_name, language):  # noqa: ANN001, ARG001
        captured["input_path"] = input_path
        captured["model_name"] = model_name
        captured["language"] = language
        return "recognized text"

    monkeypatch.setattr(TransformersOCREngine, "_recognize_image", _recognize_image)

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
        "input_path": image,
        "model_name": "stepfun-ai/GOT-OCR-2.0-hf",
        "language": "zh",
    }


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
        "aimd.ocr.processor.create_ocr_engine", lambda engine: _FakeEngine()
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
