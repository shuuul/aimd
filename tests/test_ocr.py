from pathlib import Path
import sys
import types

import pytest

from aimd.core.errors import BackendUnavailableError, ProcessingFailedError
from aimd.plugins.ocr.backends import (
    MLXVLMOCRBackend,
    OCRPage,
    OCRResult,
    TransformersOCRBackend,
    select_ocr_backend,
)
from aimd.plugins.ocr.models import (
    create_transformers_ocr_model,
    resolve_transformers_ocr_model,
)
from aimd.plugins.ocr.models.generic import GenericTransformersOCRModel
from aimd.plugins.ocr.models.got import GOTOCRModel
from aimd.plugins.ocr.models.mlx import resolve_mlx_vlm_model
from aimd.plugins.ocr.models.unlimited import (
    UnlimitedOCRModel,
    normalize_unlimited_ocr_output,
    read_unlimited_ocr_output_files,
)
from aimd.plugins.ocr import process_ocr


def test_select_ocr_backend_selects_platform_defaults(monkeypatch) -> None:
    monkeypatch.setattr("aimd.plugins.ocr.backends.platform.system", lambda: "Darwin")
    assert select_ocr_backend() == "mlx-vlm"

    monkeypatch.setattr("aimd.plugins.ocr.backends.platform.system", lambda: "Linux")
    assert select_ocr_backend() == "transformers"


def test_select_ocr_backend_rejects_unsupported_platform(monkeypatch) -> None:
    monkeypatch.setattr("aimd.plugins.ocr.backends.platform.system", lambda: "Windows")
    with pytest.raises(BackendUnavailableError):
        select_ocr_backend()


def test_resolve_mlx_vlm_model_maps_aimd_names() -> None:
    assert resolve_mlx_vlm_model(None) == "mlx-community/GLM-OCR-bf16"
    assert resolve_mlx_vlm_model("glm_ocr") == "mlx-community/GLM-OCR-bf16"
    assert resolve_mlx_vlm_model("org/custom-model") == "org/custom-model"


def test_resolve_mlx_vlm_model_rejects_paddleocr_aliases() -> None:
    with pytest.raises(ProcessingFailedError):
        resolve_mlx_vlm_model("paddleocr_v6")


def test_resolve_transformers_ocr_model_maps_vlm_models() -> None:
    assert resolve_transformers_ocr_model(None) == "stepfun-ai/GOT-OCR-2.0-hf"
    assert resolve_transformers_ocr_model("got_ocr") == "stepfun-ai/GOT-OCR-2.0-hf"
    assert resolve_transformers_ocr_model("unlimited_ocr") == "baidu/Unlimited-OCR"
    assert (
        resolve_transformers_ocr_model("baidu/Unlimited-OCR") == "baidu/Unlimited-OCR"
    )
    assert resolve_transformers_ocr_model("glm_ocr") == "zai-org/GLM-OCR"
    assert resolve_transformers_ocr_model("org/custom-model") == "org/custom-model"


@pytest.mark.parametrize("model", ["paddleocr_v6", "paddleocr_vl"])
def test_resolve_transformers_ocr_model_rejects_paddleocr_aliases(
    model: str,
) -> None:
    with pytest.raises(ProcessingFailedError):
        resolve_transformers_ocr_model(model)


def test_create_transformers_ocr_model_selects_model_adapter() -> None:
    assert isinstance(create_transformers_ocr_model("got_ocr"), GOTOCRModel)
    assert isinstance(create_transformers_ocr_model("unlimited_ocr"), UnlimitedOCRModel)
    generic = create_transformers_ocr_model("zai-org/GLM-OCR")
    assert isinstance(generic, GenericTransformersOCRModel)
    assert generic.model_id == "zai-org/GLM-OCR"


def test_mlx_vlm_image_uses_python_package(monkeypatch, tmp_path: Path):
    image = tmp_path / "scan.png"
    image.write_text("x", encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeModel:
        config = "config"

    def _load(model_id):  # noqa: ANN001
        captured["model_id"] = model_id
        return _FakeModel(), "processor"

    def _apply_chat_template(processor, config, prompt, **kwargs):  # noqa: ANN001
        captured["processor"] = processor
        captured["config"] = config
        captured["prompt"] = prompt
        captured.update(kwargs)
        return "formatted prompt"

    def _generate(model, processor, prompt, **kwargs):  # noqa: ANN001
        captured["generate_model"] = model
        captured["generate_processor"] = processor
        captured["formatted_prompt"] = prompt
        captured.update(kwargs)
        return types.SimpleNamespace(text="recognized")

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        types.SimpleNamespace(load=_load, generate=_generate),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.prompt_utils",
        types.SimpleNamespace(apply_chat_template=_apply_chat_template),
    )
    monkeypatch.setattr("aimd.plugins.ocr.models.mlx._cached_mlx_vlm_model", None)
    monkeypatch.setattr("aimd.plugins.ocr.models.mlx._cached_mlx_vlm_processor", None)
    monkeypatch.setattr("aimd.plugins.ocr.models.mlx._cached_mlx_vlm_model_id", None)

    result = MLXVLMOCRBackend().recognize(image, model="glm_ocr")

    assert result == OCRResult(
        title="scan",
        pages=(OCRPage(page_index=None, text="recognized"),),
    )
    assert captured["model_id"] == "mlx-community/GLM-OCR-bf16"
    assert captured["prompt"] == "Text Recognition:"
    assert captured["num_images"] == 1
    assert captured["image"] == [image.as_posix()]
    assert captured["max_tokens"] == 4096


def test_mlx_vlm_pdf_uses_python_package(monkeypatch, tmp_path: Path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_text("x", encoding="utf-8")
    render_dir = tmp_path / "rendered"
    render_dir.mkdir()
    page_paths = [render_dir / "page-1.png", render_dir / "page-2.png"]
    for page_path in page_paths:
        page_path.write_text("x", encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeModel:
        config = "config"

    def _load(model_id):  # noqa: ANN001
        captured["model_id"] = model_id
        return _FakeModel(), "processor"

    def _generate(model, processor, prompt, **kwargs):  # noqa: ANN001, ARG001
        captured.setdefault("images", []).append(kwargs["image"][0])
        return types.SimpleNamespace(text=f"recognized {len(captured['images'])}")

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        types.SimpleNamespace(load=_load, generate=_generate),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.prompt_utils",
        types.SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt"),
    )
    monkeypatch.setattr("aimd.plugins.ocr.models.mlx._cached_mlx_vlm_model", None)
    monkeypatch.setattr("aimd.plugins.ocr.models.mlx._cached_mlx_vlm_processor", None)
    monkeypatch.setattr("aimd.plugins.ocr.models.mlx._cached_mlx_vlm_model_id", None)
    monkeypatch.setattr(
        "aimd.plugins.ocr.backends._render_pdf_pages",
        lambda input_path, **kwargs: tuple(enumerate(page_paths)),  # noqa: ARG005
    )

    pages = MLXVLMOCRBackend()._recognize_pdf_or_document(
        pdf,
        model=None,
        start=0,
        end=2,
        temp_dir=tmp_path,
    )

    assert pages == (
        OCRPage(page_index=0, text="recognized 1"),
        OCRPage(page_index=1, text="recognized 2"),
    )
    assert captured["images"] == [
        page_paths[0].as_posix(),
        page_paths[1].as_posix(),
    ]
    assert captured["model_id"] == "mlx-community/GLM-OCR-bf16"


def test_transformers_backend_ocr_image_uses_resolved_model(
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
        "aimd.plugins.ocr.backends.create_transformers_ocr_model", _create_model
    )

    result = TransformersOCRBackend().recognize(
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
async def test_process_ocr_wraps_backend_result_as_text_context(
    monkeypatch, tmp_path: Path
) -> None:
    image = tmp_path / "page.png"
    image.write_text("x", encoding="utf-8")

    class _FakeBackend:
        def recognize(self, input_path, **kwargs):
            assert input_path == image
            assert kwargs["model"] == "glm_ocr"
            assert kwargs["language"] == "zh"
            assert kwargs["start"] is None
            assert kwargs["end"] is None
            return OCRResult(
                title="page",
                pages=(OCRPage(page_index=None, text="recognized text"),),
            )

    monkeypatch.setattr(
        "aimd.plugins.ocr._plugin.create_ocr_backend", lambda: _FakeBackend()
    )

    result = await process_ocr(
        image,
        model="glm_ocr",
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
