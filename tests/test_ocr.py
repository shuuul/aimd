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
from aimd.plugins.ocr.models.glm import GLMOCRModel
from aimd.plugins.ocr.models.base import (
    get_cuda_dtype,
    validate_transformers_precision,
)
from aimd.plugins.ocr.models.mlx import (
    MLXVLMOCRModel,
    SlidingWindowNoRepeatNgramProcessor,
    _is_unlimited_ocr_model,
    resolve_mlx_vlm_model,
)
from aimd.plugins.ocr.models.unlimited import (
    UnlimitedOCRModel,
    normalize_unlimited_ocr_markdown,
    normalize_unlimited_ocr_output,
    read_unlimited_ocr_output_files,
)
from aimd.plugins.ocr import process_ocr
from aimd.plugins.ocr.const import (
    IMAGE_FILE_EXTENSIONS,
    OCR_DOCUMENT_EXTENSIONS,
    OCR_EXTENSIONS,
)


def test_ocr_extension_constants_are_centralized() -> None:
    assert IMAGE_FILE_EXTENSIONS == {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".tif",
        ".tiff",
    }
    assert OCR_DOCUMENT_EXTENSIONS == {".pdf"}
    assert OCR_EXTENSIONS == IMAGE_FILE_EXTENSIONS | OCR_DOCUMENT_EXTENSIONS


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
    assert resolve_mlx_vlm_model(None) == "mlx-community/Unlimited-OCR-4bit"
    assert resolve_mlx_vlm_model("unlimited_ocr") == "mlx-community/Unlimited-OCR-4bit"
    assert resolve_mlx_vlm_model("unlimited-ocr") == "mlx-community/Unlimited-OCR-4bit"
    assert (
        resolve_mlx_vlm_model("unlimited_ocr_bf16")
        == "mlx-community/Unlimited-OCR-bf16"
    )
    assert resolve_mlx_vlm_model("glm_ocr") == "mlx-community/GLM-OCR-4bit"
    assert resolve_mlx_vlm_model("glm-ocr") == "mlx-community/GLM-OCR-4bit"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("mlx-community/Unlimited-OCR-4bit", "mlx-community/Unlimited-OCR-4bit"),
        ("mlx-community/Unlimited-OCR-6bit", "mlx-community/Unlimited-OCR-6bit"),
        ("mlx-community/Unlimited-OCR-8bit", "mlx-community/Unlimited-OCR-8bit"),
        ("mlx-community/Unlimited-OCR-bf16", "mlx-community/Unlimited-OCR-bf16"),
        ("mlx-community/GLM-OCR-4bit", "mlx-community/GLM-OCR-4bit"),
        ("mlx-community/GLM-OCR-6bit", "mlx-community/GLM-OCR-6bit"),
        ("mlx-community/GLM-OCR-8bit", "mlx-community/GLM-OCR-8bit"),
        ("mlx-community/GLM-OCR-bf16", "mlx-community/GLM-OCR-bf16"),
    ],
)
def test_resolve_mlx_vlm_model_accepts_explicit_hf_ids(
    model: str, expected: str
) -> None:
    assert resolve_mlx_vlm_model(model) == expected


@pytest.mark.parametrize(
    "model",
    ["paddleocr_v6", "got_ocr", "got-ocr", "org/custom-model", "baidu/Unlimited-OCR"],
)
def test_resolve_mlx_vlm_model_rejects_unsupported_models(model: str) -> None:
    with pytest.raises(ProcessingFailedError):
        resolve_mlx_vlm_model(model)


def test_mlx_vlm_unlimited_ocr_model_detection_splits_adapters() -> None:
    for quantization in ("4bit", "6bit", "8bit", "bf16"):
        model_id = f"mlx-community/Unlimited-OCR-{quantization}"
        assert _is_unlimited_ocr_model(model_id)
        assert not _is_unlimited_ocr_model(f"mlx-community/GLM-OCR-{quantization}")
    # The Transformers checkpoint stays recognized for backwards compatibility.
    assert _is_unlimited_ocr_model("baidu/Unlimited-OCR")


def test_resolve_transformers_ocr_model_maps_vlm_models() -> None:
    assert resolve_transformers_ocr_model(None) == "baidu/Unlimited-OCR"
    assert resolve_transformers_ocr_model("unlimited_ocr") == "baidu/Unlimited-OCR"
    assert (
        resolve_transformers_ocr_model("baidu/Unlimited-OCR") == "baidu/Unlimited-OCR"
    )
    assert resolve_transformers_ocr_model("glm_ocr") == "zai-org/GLM-OCR"
    assert resolve_transformers_ocr_model("glm-ocr") == "zai-org/GLM-OCR"
    assert resolve_transformers_ocr_model("zai-org/GLM-OCR") == "zai-org/GLM-OCR"


@pytest.mark.parametrize(
    "model",
    [
        "paddleocr_v6",
        "paddleocr_vl",
        "got_ocr",
        "got-ocr",
        "got_ocr2",
        "stepfun-ai/GOT-OCR-2.0-hf",
        "org/custom-model",
    ],
)
def test_resolve_transformers_ocr_model_rejects_unsupported_models(
    model: str,
) -> None:
    with pytest.raises(ProcessingFailedError):
        resolve_transformers_ocr_model(model)


def test_create_transformers_ocr_model_selects_model_adapter() -> None:
    assert isinstance(create_transformers_ocr_model(None), UnlimitedOCRModel)
    assert isinstance(create_transformers_ocr_model("unlimited_ocr"), UnlimitedOCRModel)
    glm = create_transformers_ocr_model("glm_ocr")
    assert isinstance(glm, GLMOCRModel)
    assert glm.model_id == "zai-org/GLM-OCR"
    assert isinstance(create_transformers_ocr_model("zai-org/GLM-OCR"), GLMOCRModel)


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
    assert captured["model_id"] == "mlx-community/GLM-OCR-4bit"
    assert captured["prompt"] == "Text Recognition:"
    assert captured["num_images"] == 1
    assert captured["image"] == [image.as_posix()]
    assert captured["max_tokens"] == 4096


def test_mlx_vlm_backend_forwards_precision_to_model(
    monkeypatch, tmp_path: Path
) -> None:
    image = tmp_path / "scan.png"
    image.write_text("x", encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeModel:
        config = "config"

    def _load(model_id):  # noqa: ANN001
        captured["model_id"] = model_id
        return _FakeModel(), "processor"

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        types.SimpleNamespace(
            load=_load,
            generate=lambda *args, **kwargs: types.SimpleNamespace(text="recognized"),  # noqa: ARG005
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.prompt_utils",
        types.SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt"),
    )
    monkeypatch.setattr("aimd.plugins.ocr.models.mlx._cached_mlx_vlm_model", None)
    monkeypatch.setattr("aimd.plugins.ocr.models.mlx._cached_mlx_vlm_processor", None)
    monkeypatch.setattr("aimd.plugins.ocr.models.mlx._cached_mlx_vlm_model_id", None)

    result = MLXVLMOCRBackend().recognize(image, model="glm-ocr", precision="6bit")

    assert result.pages[0].text == "recognized"
    assert captured["model_id"] == "mlx-community/GLM-OCR-6bit"


def test_mlx_vlm_unlimited_ocr_image_uses_gundam_settings(
    monkeypatch, tmp_path: Path
) -> None:
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
        return f"<image>{prompt}"

    def _generate(model, processor, prompt, **kwargs):  # noqa: ANN001
        captured["generate_model"] = model
        captured["generate_processor"] = processor
        captured["formatted_prompt"] = prompt
        captured.update(kwargs)
        return types.SimpleNamespace(
            text="<|det|>text [1, 2, 3, 4]<|/det|>recognized unlimited"
        )

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        types.SimpleNamespace(
            __version__="0.6.4",
            load=_load,
            generate=_generate,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.prompt_utils",
        types.SimpleNamespace(apply_chat_template=_apply_chat_template),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.models.unlimited_ocr",
        types.ModuleType("mlx_vlm.models.unlimited_ocr"),
    )
    monkeypatch.setattr("aimd.plugins.ocr.models.mlx._cached_mlx_vlm_model", None)
    monkeypatch.setattr("aimd.plugins.ocr.models.mlx._cached_mlx_vlm_processor", None)
    monkeypatch.setattr("aimd.plugins.ocr.models.mlx._cached_mlx_vlm_model_id", None)

    result = MLXVLMOCRBackend().recognize(image, model="unlimited_ocr")

    assert result == OCRResult(
        title="scan",
        pages=(OCRPage(page_index=None, text="recognized unlimited"),),
    )
    assert captured["model_id"] == "mlx-community/Unlimited-OCR-4bit"
    assert captured["prompt"] == "document parsing."
    assert captured["num_images"] == 1
    assert captured["image"] == [image.as_posix()]
    assert captured["max_tokens"] == 8192
    assert captured["temperature"] == 0.0
    assert captured["cropping"] is True
    assert captured["image_size"] == 640
    assert captured["base_size"] == 1024
    assert captured.get("logits_processors") is not None
    assert len(captured["logits_processors"]) == 1


def test_mlx_vlm_unlimited_ocr_pdf_runs_page_by_page(
    monkeypatch, tmp_path: Path
) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_text("x", encoding="utf-8")
    render_dir = tmp_path / "rendered"
    render_dir.mkdir()
    page_paths = [render_dir / "page-1.png", render_dir / "page-2.png"]
    for page_path in page_paths:
        page_path.write_text("x", encoding="utf-8")
    captured: dict[str, object] = {"images": [], "prompts": [], "num_images": []}

    class _FakeModel:
        config = "config"

    def _load(model_id):  # noqa: ANN001
        captured["model_id"] = model_id
        return _FakeModel(), "processor"

    def _apply_chat_template(processor, config, prompt, **kwargs):  # noqa: ANN001
        captured["prompts"].append(prompt)
        captured["num_images"].append(kwargs.get("num_images"))
        return f"<image>{prompt}"

    def _generate(model, processor, prompt, **kwargs):  # noqa: ANN001, ARG001
        captured["images"].append(kwargs["image"][0])
        captured.setdefault("generate_kwargs", []).append(
            {
                "max_tokens": kwargs.get("max_tokens"),
                "temperature": kwargs.get("temperature"),
                "cropping": kwargs.get("cropping"),
                "image_size": kwargs.get("image_size"),
                "base_size": kwargs.get("base_size"),
                "logits_processors": kwargs.get("logits_processors"),
            }
        )
        page_no = len(captured["images"])
        return types.SimpleNamespace(text=f"page {page_no} body")

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        types.SimpleNamespace(
            __version__="0.6.8",
            load=_load,
            generate=_generate,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.prompt_utils",
        types.SimpleNamespace(apply_chat_template=_apply_chat_template),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.models.unlimited_ocr",
        types.ModuleType("mlx_vlm.models.unlimited_ocr"),
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
        model="unlimited_ocr",
        start=0,
        end=1,
        temp_dir=tmp_path,
    )

    assert pages == (
        OCRPage(page_index=0, text="page 1 body"),
        OCRPage(page_index=1, text="page 2 body"),
    )
    assert captured["model_id"] == "mlx-community/Unlimited-OCR-4bit"
    assert captured["prompts"] == ["document parsing.", "document parsing."]
    assert captured["num_images"] == [1, 1]
    assert captured["images"] == [path.as_posix() for path in page_paths]
    assert len(captured["generate_kwargs"]) == 2
    for kwargs in captured["generate_kwargs"]:
        assert kwargs["max_tokens"] == 8192
        assert kwargs["temperature"] == 0.0
        assert kwargs["cropping"] is True
        assert kwargs["image_size"] == 640
        assert kwargs["base_size"] == 1024
        assert kwargs["logits_processors"] is not None
        assert len(kwargs["logits_processors"]) == 1


def test_sliding_window_no_repeat_ngram_processor_bans_repeated_tail() -> None:
    mx = pytest.importorskip("mlx.core")
    processor = SlidingWindowNoRepeatNgramProcessor(ngram_size=3, window=16)
    # Sequence ends with prefix (1, 2); historical ngram (1, 2, 9) should ban 9.
    tokens = mx.array([7, 1, 2, 9, 3, 1, 2])
    logits = mx.zeros((20,), dtype=mx.float32)
    out = processor(tokens, logits)
    assert float(out[9]) == float("-inf")
    assert float(out[0]) == 0.0


def test_mlx_vlm_unlimited_ocr_rejects_old_mlx_vlm(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "scan.png"
    image.write_text("x", encoding="utf-8")

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        types.SimpleNamespace(__version__="0.6.3", load=lambda *_a, **_k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.prompt_utils",
        types.SimpleNamespace(apply_chat_template=lambda *_a, **_k: "prompt"),
    )
    monkeypatch.setattr("aimd.plugins.ocr.models.mlx._cached_mlx_vlm_model", None)
    monkeypatch.setattr("aimd.plugins.ocr.models.mlx._cached_mlx_vlm_processor", None)
    monkeypatch.setattr("aimd.plugins.ocr.models.mlx._cached_mlx_vlm_model_id", None)

    with pytest.raises(BackendUnavailableError, match="mlx-vlm>=0.6.4"):
        MLXVLMOCRModel("unlimited_ocr").recognize_image(image)


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
        model="glm_ocr",
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
    assert captured["model_id"] == "mlx-community/GLM-OCR-4bit"


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

    def _create_model(model, precision=None):  # noqa: ANN001
        captured["model"] = model
        captured["precision"] = precision
        return _FakeModel()

    monkeypatch.setattr(
        "aimd.plugins.ocr.backends.create_transformers_ocr_model", _create_model
    )

    result = TransformersOCRBackend().recognize(
        image,
        model="glm_ocr",
        language="zh",
        precision="bf16",
    )

    assert result == OCRResult(
        title="scan",
        pages=(OCRPage(page_index=None, text="recognized text"),),
    )
    assert captured == {
        "model": "glm_ocr",
        "precision": "bf16",
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
            return "<|det|>text [1, 2, 3, 4]<|/det|>recognized markdown"

    monkeypatch.setattr(
        "aimd.plugins.ocr.models.unlimited.get_cached_model_and_processor",
        lambda model_name, loader, precision=None: (_FakeModel(), "tokenizer"),  # noqa: ARG005
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
        lambda model_name, loader, precision=None: (_FakeModel(), "tokenizer"),  # noqa: ARG005
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


def test_normalize_unlimited_ocr_markdown_removes_layout_protocol() -> None:
    raw = (
        "<|det|>title [443, 67, 554, 87]<|/det|># Document title\n"
        "<|det|>text [249, 98, 740, 118]<|/det|>First paragraph.\n"
        "continued line\n"
        "<|det|>image [10, 20, 30, 40]<|/det|>\n"
        "<|det|>text [50, 60, 70, 80]<|/det|>- list item"
    )

    assert normalize_unlimited_ocr_markdown(raw) == (
        "# Document title\n\nFirst paragraph.\ncontinued line\n\n- list item"
    )


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
async def test_process_ocr_preserves_page_oriented_compatibility_result(
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
            assert kwargs["precision"] == "bf16"
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
        precision="bf16",
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


def test_ocr_plugin_markdown_single_page_without_page_headers(
    monkeypatch, tmp_path: Path
) -> None:
    from markitdown import MarkItDown

    image = tmp_path / "scan.png"
    image.write_bytes(b"fake image")

    class _FakeBackend:
        def recognize(self, input_path, **kwargs):  # noqa: ANN001, ARG002
            return OCRResult(
                title="scan",
                pages=(OCRPage(page_index=None, text="single page text"),),
            )

    monkeypatch.setattr(
        "aimd.plugins.ocr._plugin.create_ocr_backend", lambda: _FakeBackend()
    )

    result = MarkItDown(enable_plugins=True).convert(image, task_type="ocr")

    assert result.title == "scan"
    assert result.markdown == "single page text"
    assert "## Page" not in result.markdown


def test_ocr_plugin_markdown_multipage_uses_page_headers(
    monkeypatch, tmp_path: Path
) -> None:
    from markitdown import MarkItDown

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    class _FakeBackend:
        def recognize(self, input_path, **kwargs):  # noqa: ANN001, ARG002
            return OCRResult(
                title="scan",
                pages=(
                    OCRPage(page_index=0, text="page one body"),
                    OCRPage(page_index=1, text="page two body"),
                ),
            )

    monkeypatch.setattr(
        "aimd.plugins.ocr._plugin.create_ocr_backend", lambda: _FakeBackend()
    )

    result = MarkItDown(enable_plugins=True).convert(pdf, task_type="ocr")

    assert result.title == "scan"
    assert result.markdown == (
        "## Page 1\n\npage one body\n\n## Page 2\n\npage two body"
    )


@pytest.mark.asyncio
async def test_process_input_ocr_text_context_owned_by_core(
    monkeypatch, tmp_path: Path
) -> None:
    from aimd.core.models import ProcessInput
    from aimd.core.process import process_input

    image = tmp_path / "scan.png"
    image.write_bytes(b"fake image")
    page_one = "A" * 300
    page_two = "B" * 300

    class _FakeBackend:
        def recognize(self, input_path, **kwargs):  # noqa: ANN001, ARG002
            return OCRResult(
                title="scan",
                pages=(
                    OCRPage(page_index=0, text=page_one),
                    OCRPage(page_index=1, text=page_two),
                ),
            )

    monkeypatch.setattr(
        "aimd.plugins.ocr._plugin.create_ocr_backend", lambda: _FakeBackend()
    )

    async def _process_file_with_small_chunks(
        file_path,
        language=None,
        model=None,
        temp_dir=None,
        task_type=None,
        start=None,
        end=None,
        precision=None,
    ):  # noqa: ANN001
        from aimd.core.process import convert_file_with_markitdown

        return await convert_file_with_markitdown(
            file_path,
            language=language,
            model=model,
            temp_dir=temp_dir,
            task_type=task_type,
            start=start,
            end=end,
            precision=precision,
            max_chunk_size=400,
        )

    result = await process_input(
        ProcessInput(input_source=image.as_posix(), task_type="ocr"),
        process_file=_process_file_with_small_chunks,
    )

    assert result.task_type == "ocr"
    assert result.text_context.title == "scan"
    assert len(result.text_context.chunk_list) >= 2
    assert result.text_context.split_header_level == 2
    assert any("## Page 1" in chunk for chunk in result.text_context.chunk_list)
    assert any("## Page 2" in chunk for chunk in result.text_context.chunk_list)


@pytest.mark.asyncio
async def test_process_input_ocr_short_markdown_has_no_split_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    from aimd.core.models import ProcessInput
    from aimd.core.process import process_input

    image = tmp_path / "page.png"
    image.write_bytes(b"fake image")

    class _FakeBackend:
        def recognize(self, input_path, **kwargs):  # noqa: ANN001, ARG002
            return OCRResult(
                title="page",
                pages=(OCRPage(page_index=None, text="short ocr text"),),
            )

    monkeypatch.setattr(
        "aimd.plugins.ocr._plugin.create_ocr_backend", lambda: _FakeBackend()
    )

    result = await process_input(
        ProcessInput(input_source=image.as_posix(), task_type="ocr"),
    )

    assert result.task_type == "ocr"
    assert result.text_context.title == "page"
    assert result.text_context.chunk_list == ["short ocr text"]
    assert result.text_context.split_header_level is None


@pytest.mark.parametrize("precision", ["4bit", "6bit", "8bit", "bf16"])
def test_resolve_mlx_vlm_model_alias_with_precision(precision: str) -> None:
    assert (
        resolve_mlx_vlm_model("unlimited-ocr", precision)
        == f"mlx-community/Unlimited-OCR-{precision}"
    )
    assert (
        resolve_mlx_vlm_model("glm-ocr", precision)
        == f"mlx-community/GLM-OCR-{precision}"
    )


def test_resolve_mlx_vlm_model_alias_defaults_to_4bit() -> None:
    assert resolve_mlx_vlm_model("unlimited-ocr") == "mlx-community/Unlimited-OCR-4bit"
    assert resolve_mlx_vlm_model("glm-ocr") == "mlx-community/GLM-OCR-4bit"
    assert resolve_mlx_vlm_model(None) == "mlx-community/Unlimited-OCR-4bit"


def test_resolve_mlx_vlm_model_legacy_alias_with_precision() -> None:
    assert (
        resolve_mlx_vlm_model("unlimited_ocr", "8bit")
        == "mlx-community/Unlimited-OCR-8bit"
    )
    assert resolve_mlx_vlm_model("glm_ocr", "bf16") == "mlx-community/GLM-OCR-bf16"
    assert (
        resolve_mlx_vlm_model("unlimited_ocr_6bit")
        == "mlx-community/Unlimited-OCR-6bit"
    )


def test_resolve_mlx_vlm_model_normalizes_dash_precision() -> None:
    assert resolve_mlx_vlm_model("glm-ocr", "4-bit") == "mlx-community/GLM-OCR-4bit"
    assert (
        resolve_mlx_vlm_model("unlimited-ocr", "BF16")
        == "mlx-community/Unlimited-OCR-bf16"
    )


def test_resolve_mlx_vlm_model_full_id_with_matching_precision() -> None:
    assert (
        resolve_mlx_vlm_model("mlx-community/GLM-OCR-8bit", "8bit")
        == "mlx-community/GLM-OCR-8bit"
    )
    assert (
        resolve_mlx_vlm_model("mlx-community/Unlimited-OCR-bf16")
        == "mlx-community/Unlimited-OCR-bf16"
    )


@pytest.mark.parametrize(
    ("model", "precision"),
    [
        ("mlx-community/GLM-OCR-8bit", "4bit"),
        ("mlx-community/Unlimited-OCR-bf16", "8bit"),
        ("unlimited-ocr-4bit", "6bit"),
        ("unlimited_ocr_bf16", "4bit"),
    ],
)
def test_resolve_mlx_vlm_model_rejects_conflicting_precision(
    model: str, precision: str
) -> None:
    with pytest.raises(ProcessingFailedError, match="conflicts"):
        resolve_mlx_vlm_model(model, precision)


def test_resolve_mlx_vlm_model_rejects_unknown_precision() -> None:
    with pytest.raises(ProcessingFailedError, match="Unsupported precision"):
        resolve_mlx_vlm_model("glm-ocr", "fp8")


def test_mlx_vlm_model_adapter_accepts_precision() -> None:
    assert (
        MLXVLMOCRModel("glm-ocr", precision="6bit").model_id
        == "mlx-community/GLM-OCR-6bit"
    )
    assert (
        MLXVLMOCRModel(None, precision="8bit").model_id
        == "mlx-community/Unlimited-OCR-8bit"
    )


def test_create_transformers_ocr_model_passes_precision() -> None:
    unlimited = create_transformers_ocr_model("unlimited-ocr", precision="bf16")
    assert isinstance(unlimited, UnlimitedOCRModel)
    assert unlimited.precision == "bf16"

    glm = create_transformers_ocr_model("glm_ocr", precision="bf16")
    assert isinstance(glm, GLMOCRModel)
    assert glm.precision == "bf16"

    default = create_transformers_ocr_model(None)
    assert default.precision is None


@pytest.mark.parametrize("precision", ["4bit", "6bit", "8bit"])
def test_transformers_ocr_rejects_quantized_precision(precision: str) -> None:
    with pytest.raises(ProcessingFailedError, match="quantized precision"):
        create_transformers_ocr_model("unlimited-ocr", precision=precision)
    with pytest.raises(ProcessingFailedError, match="quantized precision"):
        validate_transformers_precision(precision)


def test_validate_transformers_precision_normalizes_dash_variant() -> None:
    assert validate_transformers_precision(None) is None
    assert validate_transformers_precision("BF16") == "bf16"


def test_get_cuda_dtype_honors_explicit_bf16(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert get_cuda_dtype("bf16") is torch.bfloat16
    assert get_cuda_dtype(None) is torch.bfloat16

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert get_cuda_dtype(None) is torch.float16
    with pytest.raises(BackendUnavailableError, match="bf16"):
        get_cuda_dtype("bf16")
    with pytest.raises(ProcessingFailedError, match="does not support precision"):
        get_cuda_dtype("4bit")
