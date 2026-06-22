"""Linux Transformers OCR backend adapter."""

from pathlib import Path
import shutil
import subprocess
import tempfile

from aimd.core.errors import EngineUnavailableError, ProcessingFailedError

from .engines import OCRPage, OCRResult
from .mlx4ocr_engine import IMAGE_FILE_EXTENSIONS

GOT_OCR_MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"
DEFAULT_TRANSFORMERS_OCR_MODEL = "got_ocr"
TRANSFORMERS_OCR_MODEL_ALIASES = {
    "got_ocr": GOT_OCR_MODEL_ID,
    "got-ocr": GOT_OCR_MODEL_ID,
    "got_ocr2": GOT_OCR_MODEL_ID,
    "got-ocr2": GOT_OCR_MODEL_ID,
    "stepfun-ai/got-ocr-2.0-hf": GOT_OCR_MODEL_ID,
    "glm_ocr": "zai-org/GLM-OCR",
    "glm-ocr": "zai-org/GLM-OCR",
    "zai-org/glm-ocr": "zai-org/GLM-OCR",
    "paddleocr_vl": "PaddlePaddle/PaddleOCR-VL-1.5",
    "paddleocr-vl": "PaddlePaddle/PaddleOCR-VL-1.5",
    "paddlepaddle/paddleocr-vl-1.5": "PaddlePaddle/PaddleOCR-VL-1.5",
}

_cached_model = None
_cached_processor = None
_cached_model_name: str | None = None


def _resolve_transformers_ocr_model(model: str | None) -> str:
    """Resolve aimd OCR model aliases to Hugging Face model IDs."""
    requested = model or DEFAULT_TRANSFORMERS_OCR_MODEL
    normalized = requested.strip().lower().replace(" ", "_")
    if normalized in TRANSFORMERS_OCR_MODEL_ALIASES:
        return TRANSFORMERS_OCR_MODEL_ALIASES[normalized]
    if "/" in requested:
        return requested.strip()
    raise ProcessingFailedError(
        "Unsupported Transformers OCR model. Supported models: got_ocr, "
        "glm_ocr, paddleocr_vl, or an explicit Hugging Face model ID. "
        "PP-OCRv6/paddleocr_v6 is not supported by the Transformers backend."
    )


def _get_model_and_processor(model_name: str):
    """Load or return a cached OCR-capable Transformers model and processor."""
    global _cached_model, _cached_processor, _cached_model_name  # noqa: PLW0603
    if _cached_model is not None and _cached_model_name == model_name:
        return _cached_model, _cached_processor

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise EngineUnavailableError(
            "Transformers OCR requires torch and transformers. Install project "
            "dependencies with `uv sync`, then retry."
        ) from exc

    if not torch.cuda.is_available():
        raise EngineUnavailableError(
            "CUDA is not available. The Transformers OCR backend requires a "
            "CUDA-capable GPU for practical inference."
        )

    dtype = (
        torch.bfloat16
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        else torch.float16
    )
    try:
        _cached_processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        _cached_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=dtype,
        ).to("cuda")
    except Exception as exc:  # noqa: BLE001 - upstream model errors vary widely
        extra = ""
        if model_name == "zai-org/GLM-OCR":
            extra = " GLM-OCR currently requires installing Transformers from GitHub."
        elif model_name == "PaddlePaddle/PaddleOCR-VL-1.5":
            extra = " PaddleOCR-VL may require torchvision with the current model code."
        raise EngineUnavailableError(
            f"Unable to load Transformers OCR model {model_name!r}: {exc}{extra}"
        ) from exc

    _cached_model_name = model_name
    return _cached_model, _cached_processor


def _model_device(model: object):
    """Return the first parameter device for tensor placement."""
    return next(model.parameters()).device


def _inputs_to_model_device(inputs, model: object):
    """Move processor tensors to the model device."""
    import torch

    device = _model_device(model)
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def _build_messages(image_path: Path, language: str | None) -> list[dict[str, object]]:
    instruction = "Text Recognition:"
    if language:
        instruction = f"Text Recognition. Prefer {language} text when ambiguous:"
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path.as_posix()},
                {"type": "text", "text": instruction},
            ],
        }
    ]


def _render_pdf_with_pdftoppm(
    input_path: Path,
    *,
    start: int | None,
    end: int | None,
    output_dir: Path,
) -> list[Path]:
    """Render a PDF to PNG pages with poppler's pdftoppm when available."""
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise EngineUnavailableError(
            "PDF OCR on Linux requires the `pdftoppm` executable from poppler. "
            "Install poppler-utils, or OCR image files directly."
        )

    output_prefix = output_dir / input_path.stem
    command = [pdftoppm, "-png", "-r", "200"]
    if start is not None:
        command.extend(["-f", str(start + 1)])
    if end is not None:
        command.extend(["-l", str(end + 1)])
    command.extend([input_path.as_posix(), output_prefix.as_posix()])

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise ProcessingFailedError(f"pdftoppm failed while rendering PDF: {stderr}")

    pages = sorted(output_dir.glob(f"{input_path.stem}-*.png"))
    if not pages:
        raise ProcessingFailedError("pdftoppm rendered no pages for OCR")
    return pages


class TransformersOCREngine:
    """Run OCR through CUDA-capable Hugging Face Transformers VLM models."""

    def recognize(
        self,
        input_path: Path,
        *,
        model: str | None = None,
        language: str | None = None,
        start: int | None = None,
        end: int | None = None,
        temp_dir: Path | None = None,
    ) -> OCRResult:
        resolved_model = _resolve_transformers_ocr_model(model)
        if input_path.suffix.lower() in IMAGE_FILE_EXTENSIONS:
            text = self._recognize_image(
                input_path,
                model_name=resolved_model,
                language=language,
            )
            pages = (OCRPage(page_index=None, text=text),)
        elif input_path.suffix.lower() == ".pdf":
            with tempfile.TemporaryDirectory(
                prefix="aimd-ocr-pdf-", dir=temp_dir
            ) as output_dir:
                page_paths = _render_pdf_with_pdftoppm(
                    input_path,
                    start=start,
                    end=end,
                    output_dir=Path(output_dir),
                )
                pages = tuple(
                    OCRPage(
                        page_index=(start or 0) + index,
                        text=self._recognize_image(
                            page_path,
                            model_name=resolved_model,
                            language=language,
                        ),
                    )
                    for index, page_path in enumerate(page_paths)
                )
        else:
            raise ProcessingFailedError(
                "Transformers OCR supports image files and PDFs only."
            )

        return OCRResult(title=input_path.stem, pages=pages)

    def _recognize_image(
        self,
        input_path: Path,
        *,
        model_name: str,
        language: str | None,
    ) -> str:
        model, processor = _get_model_and_processor(model_name)
        if model_name == GOT_OCR_MODEL_ID:
            return self._recognize_image_with_got(input_path, model, processor)

        try:
            inputs = processor.apply_chat_template(
                _build_messages(input_path, language),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as exc:  # noqa: BLE001 - processor errors are model-specific
            raise ProcessingFailedError(
                f"Unable to prepare image for Transformers OCR: {exc}"
            ) from exc

        inputs.pop("token_type_ids", None)
        inputs = _inputs_to_model_device(inputs, model)
        input_ids = inputs["input_ids"]

        import torch

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=8192)

        generated_ids = getattr(generated, "sequences", generated)
        decoded = processor.decode(
            generated_ids[0][input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text = decoded.strip()
        if not text:
            raise ProcessingFailedError("Transformers OCR produced empty content")
        return text

    def _recognize_image_with_got(
        self,
        input_path: Path,
        model: object,
        processor: object,
    ) -> str:
        try:
            inputs = processor(input_path.as_posix(), return_tensors="pt")
        except Exception as exc:  # noqa: BLE001 - processor errors are model-specific
            raise ProcessingFailedError(
                f"Unable to prepare image for GOT-OCR: {exc}"
            ) from exc

        inputs = _inputs_to_model_device(inputs, model)
        input_ids = inputs["input_ids"]

        import torch

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                do_sample=False,
                tokenizer=processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=4096,
            )

        generated_ids = getattr(generated, "sequences", generated)
        text = processor.decode(
            generated_ids[0][input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not text:
            raise ProcessingFailedError("GOT-OCR produced empty content")
        return text
