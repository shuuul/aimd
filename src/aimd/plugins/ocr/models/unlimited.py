"""Baidu Unlimited-OCR Transformers model adapter."""

from pathlib import Path
import tempfile

from aimd.core.errors import BackendUnavailableError, ProcessingFailedError

from .base import (
    get_cached_model_and_processor,
    get_cuda_dtype,
    validate_transformers_precision,
)

UNLIMITED_OCR_MODEL_ID = "baidu/Unlimited-OCR"


class UnlimitedOCRModel:
    """Adapter for baidu/Unlimited-OCR's custom infer/infer_multi API."""

    model_id = UNLIMITED_OCR_MODEL_ID

    def __init__(self, precision: str | None = None) -> None:
        self.precision = validate_transformers_precision(precision)

    def recognize_image(
        self,
        input_path: Path,
        *,
        language: str | None = None,  # noqa: ARG002 - upstream prompt is fixed
        temp_dir: Path | None = None,
    ) -> str:
        return self.recognize_images([input_path], temp_dir=temp_dir)[0]

    def recognize_images(
        self,
        image_paths: list[Path],
        *,
        language: str | None = None,  # noqa: ARG002 - upstream prompt is fixed
        temp_dir: Path | None = None,
    ) -> list[str]:
        model, tokenizer = get_cached_model_and_processor(
            self.model_id,
            lambda name: _load_unlimited_ocr_model(name, precision=self.precision),
            precision=self.precision,
        )
        with tempfile.TemporaryDirectory(
            prefix="aimd-unlimited-ocr-", dir=temp_dir
        ) as output_dir:
            try:
                if len(image_paths) == 1:
                    result = model.infer(
                        tokenizer,
                        prompt="<image>document parsing.",
                        image_file=image_paths[0].as_posix(),
                        output_path=output_dir,
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        max_length=32768,
                        no_repeat_ngram_size=35,
                        ngram_window=128,
                        save_results=True,
                    )
                else:
                    result = model.infer_multi(
                        tokenizer,
                        prompt="<image>Multi page parsing.",
                        image_files=[path.as_posix() for path in image_paths],
                        output_path=output_dir,
                        image_size=1024,
                        max_length=32768,
                        no_repeat_ngram_size=35,
                        ngram_window=1024,
                        save_results=True,
                    )
            except Exception as exc:  # noqa: BLE001 - remote model errors are model-specific
                raise ProcessingFailedError(
                    f"Unlimited-OCR inference failed: {exc}"
                ) from exc

            texts = normalize_unlimited_ocr_output(
                result, expected_pages=len(image_paths)
            )
            if texts is None:
                texts = read_unlimited_ocr_output_files(
                    Path(output_dir), expected_pages=len(image_paths)
                )
            if texts is None or any(not text.strip() for text in texts):
                raise ProcessingFailedError("Unlimited-OCR produced empty content")
            return [text.strip() for text in texts]


def normalize_unlimited_ocr_output(
    result: object,
    *,
    expected_pages: int,
) -> list[str] | None:
    """Normalize known Unlimited-OCR return shapes into page text."""
    if result is None:
        return None
    if isinstance(result, str):
        if expected_pages == 1:
            return [result]
        return [part.strip() for part in result.split("\n\n") if part.strip()]
    if isinstance(result, list | tuple):
        texts = [str(item).strip() for item in result]
        return texts if len(texts) == expected_pages else None
    if isinstance(result, dict):
        for key in ("text", "markdown", "content"):
            value = result.get(key)
            if isinstance(value, str):
                return [value]
        pages = result.get("pages")
        if isinstance(pages, list | tuple):
            texts = []
            for page in pages:
                if isinstance(page, dict):
                    text = (
                        page.get("text") or page.get("markdown") or page.get("content")
                    )
                    if text is not None:
                        texts.append(str(text).strip())
                else:
                    texts.append(str(page).strip())
            return texts if len(texts) == expected_pages else None
    return None


def read_unlimited_ocr_output_files(
    output_dir: Path,
    *,
    expected_pages: int,
) -> list[str] | None:
    """Read markdown files if Unlimited-OCR only persisted results to output_path."""
    output_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )
    if not output_files:
        return None
    texts = [path.read_text(encoding="utf-8").strip() for path in output_files]
    if expected_pages == 1:
        return ["\n\n".join(text for text in texts if text)]
    return texts if len(texts) == expected_pages else None


def _load_unlimited_ocr_model(
    model_name: str, precision: str | None = None
) -> tuple[object, object]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise BackendUnavailableError(
            "Transformers OCR requires torch and transformers. Install project "
            "dependencies with `uv sync`, then retry."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        model = (
            AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=get_cuda_dtype(precision),
            )
            .eval()
            .to("cuda")
        )
    except Exception as exc:  # noqa: BLE001 - upstream model errors vary widely
        raise BackendUnavailableError(
            f"Unable to load Transformers OCR model {model_name!r}: {exc} "
            "Unlimited-OCR requires CUDA and trusted remote model code."
        ) from exc
    return model, tokenizer
