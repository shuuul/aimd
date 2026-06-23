"""OCR backend contracts, platform resolution, and backend adapters."""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

from aimd.core.errors import (
    BackendUnavailableError,
    ProcessingFailedError,
)

from .models import create_transformers_ocr_model

IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
DEFAULT_OCR_MODEL = "glm_ocr"
MLX_VLM_MODEL_ALIASES = {
    "glm_ocr": "mlx-community/GLM-OCR-bf16",
    "glm-ocr": "mlx-community/GLM-OCR-bf16",
    "mlx-community/glm-ocr-bf16": "mlx-community/GLM-OCR-bf16",
}
MLX_VLM_DEFAULT_PROMPT = "Text Recognition:"


@dataclass(frozen=True, slots=True)
class OCRPage:
    """OCR text for one image or document page."""

    page_index: int | None
    text: str


@dataclass(frozen=True, slots=True)
class OCRResult:
    """Normalized OCR result returned by backend adapters."""

    title: str
    pages: tuple[OCRPage, ...]


def select_ocr_backend() -> str:
    """Select the platform OCR backend."""
    system = platform.system()
    if system == "Darwin":
        return "mlx-vlm"
    if system == "Linux":
        return "transformers"
    raise BackendUnavailableError(
        "OCR is unavailable on this platform. Supported OCR platforms are macOS "
        "with mlx-vlm and Linux with CUDA-capable Transformers."
    )


def _resolve_mlx_vlm_model(model: str | None) -> str:
    """Resolve aimd OCR model aliases to mlx-vlm model IDs."""
    requested = model or DEFAULT_OCR_MODEL
    normalized = requested.strip().lower().replace(" ", "_")
    if normalized in MLX_VLM_MODEL_ALIASES:
        return MLX_VLM_MODEL_ALIASES[normalized]
    if "/" in requested:
        return requested.strip()
    raise ProcessingFailedError(
        "Unsupported MLX VLM OCR model. Supported models: glm_ocr, "
        "or an explicit mlx-vlm compatible Hugging Face model ID."
    )


class MLXVLMOCRBackend:
    """Run OCR through mlx-vlm on macOS."""

    name = "mlx-vlm"
    runtime = "mlx"

    def recognize(
        self,
        input_path: Path,
        *,
        model: str | None = None,
        language: str | None = None,  # noqa: ARG002 - reserved for backends that support it
        start: int | None = None,
        end: int | None = None,
        temp_dir: Path | None = None,
    ) -> OCRResult:
        if input_path.suffix.lower() in IMAGE_FILE_EXTENSIONS:
            text = self._recognize_image(input_path, model=model)
            pages = (OCRPage(page_index=None, text=text.strip()),)
        else:
            pages = self._recognize_pdf_or_document(
                input_path,
                model=model,
                start=start,
                end=end,
                temp_dir=temp_dir,
            )

        return OCRResult(
            title=input_path.stem,
            pages=pages,
        )

    def _recognize_image(self, input_path: Path, *, model: str | None) -> str:
        return _run_mlx_vlm_ocr(input_path, model_id=_resolve_mlx_vlm_model(model))

    def _recognize_pdf_or_document(
        self,
        input_path: Path,
        *,
        model: str | None,
        start: int | None,
        end: int | None,
        temp_dir: Path | None,
    ) -> tuple[OCRPage, ...]:
        model_id = _resolve_mlx_vlm_model(model)
        if input_path.suffix.lower() != ".pdf":
            raise ProcessingFailedError("MLX VLM OCR supports image files and PDFs only.")

        try:
            rendered_pages = _render_pdf_with_pymupdf(
                input_path,
                start=start,
                end=end,
                temp_dir=temp_dir,
            )
        except ImportError as exc:
            raise BackendUnavailableError(
                "PDF OCR with MLX VLM requires pymupdf. Install project "
                "dependencies with `uv sync`, then retry."
            ) from exc

        try:
            return self._recognize_rendered_pages(
                rendered_pages,
                model_id=model_id,
            )
        finally:
            if rendered_pages:
                shutil.rmtree(rendered_pages[0][1].parent, ignore_errors=True)

    def _recognize_rendered_pages(
        self,
        rendered_pages: tuple[tuple[int, Path], ...],
        *,
        model_id: str,
    ) -> tuple[OCRPage, ...]:
        model, processor = _load_mlx_vlm_model(model_id)
        return tuple(
            OCRPage(
                page_index=page_index,
                text=_generate_mlx_vlm_text(model, processor, image_path),
            )
            for page_index, image_path in rendered_pages
        )


_cached_mlx_vlm_model = None
_cached_mlx_vlm_processor = None
_cached_mlx_vlm_model_id: str | None = None


def _load_mlx_vlm_modules():
    try:
        return import_module("mlx_vlm"), import_module("mlx_vlm.prompt_utils")
    except ImportError as exc:
        raise BackendUnavailableError(
            "MLX VLM OCR requires mlx-vlm. Install OCR dependencies with `uv sync` "
            "on macOS/Python 3.12+, then retry."
        ) from exc


def _load_mlx_vlm_model(model_id: str) -> tuple[object, object]:
    """Load or return cached mlx-vlm model state."""
    global _cached_mlx_vlm_model, _cached_mlx_vlm_processor, _cached_mlx_vlm_model_id  # noqa: PLW0603
    if _cached_mlx_vlm_model is not None and _cached_mlx_vlm_model_id == model_id:
        return _cached_mlx_vlm_model, _cached_mlx_vlm_processor

    mlx_vlm, _prompt_utils = _load_mlx_vlm_modules()
    try:
        _cached_mlx_vlm_model, _cached_mlx_vlm_processor = mlx_vlm.load(model_id)
    except Exception as exc:  # noqa: BLE001 - upstream model load errors vary
        _cached_mlx_vlm_model = None
        _cached_mlx_vlm_processor = None
        _cached_mlx_vlm_model_id = None
        raise BackendUnavailableError(
            f"Unable to load MLX VLM OCR model {model_id!r}: {exc}"
        ) from exc
    _cached_mlx_vlm_model_id = model_id
    return _cached_mlx_vlm_model, _cached_mlx_vlm_processor


def _run_mlx_vlm_ocr(input_path: Path, *, model_id: str) -> str:
    """Run mlx-vlm OCR on one image path."""
    model, processor = _load_mlx_vlm_model(model_id)
    return _generate_mlx_vlm_text(model, processor, input_path)


def _generate_mlx_vlm_text(
    model: object,
    processor: object,
    image_path: Path,
) -> str:
    """Generate OCR text for one image using an already-loaded mlx-vlm model."""
    mlx_vlm, prompt_utils = _load_mlx_vlm_modules()
    config = getattr(model, "config", None)
    formatted_prompt = prompt_utils.apply_chat_template(
        processor,
        config,
        MLX_VLM_DEFAULT_PROMPT,
        num_images=1,
    )
    result = mlx_vlm.generate(
        model,
        processor,
        formatted_prompt,
        image=[image_path.as_posix()],
        max_tokens=4096,
    )
    text = str(getattr(result, "text", result)).strip()
    if not text:
        raise ProcessingFailedError("MLX VLM OCR produced empty content")
    return text


def _render_pdf_with_pymupdf(
    input_path: Path,
    *,
    start: int | None,
    end: int | None,
    temp_dir: Path | None,
) -> tuple[tuple[int, Path], ...]:
    """Render PDF pages to PNG paths using the Python package API."""
    import fitz

    output_dir = Path(
        tempfile.mkdtemp(
            prefix="aimd-mlx-vlm-pdf-",
            dir=temp_dir,
        )
    )
    rendered_pages: list[tuple[int, Path]] = []
    try:
        with fitz.open(input_path) as document:
            page_count = document.page_count
            if page_count == 0:
                raise ProcessingFailedError(f"PDF has no pages: {input_path}")
            first_page = 0 if start is None else start
            last_page = page_count - 1 if end is None else end
            if first_page < 0 or last_page < first_page or last_page >= page_count:
                raise ProcessingFailedError(
                    f"Invalid OCR page range {first_page}-{last_page} for {input_path} "
                    f"with {page_count} pages."
                )
            matrix = fitz.Matrix(2.0, 2.0)
            for page_index in range(first_page, last_page + 1):
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                page_path = output_dir / f"{input_path.stem}_page_{page_index + 1:04d}.png"
                pixmap.save(page_path)
                rendered_pages.append((page_index, page_path))
        return tuple(rendered_pages)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


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
        raise BackendUnavailableError(
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


class TransformersOCRBackend:
    """Run OCR through CUDA-capable Hugging Face Transformers VLM models."""

    name = "transformers"
    runtime = "cuda"

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
        model_adapter = create_transformers_ocr_model(model)
        if input_path.suffix.lower() in IMAGE_FILE_EXTENSIONS:
            text = model_adapter.recognize_image(
                input_path,
                language=language,
                temp_dir=temp_dir,
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
                page_texts = model_adapter.recognize_images(
                    page_paths,
                    language=language,
                    temp_dir=temp_dir,
                )
                pages = tuple(
                    OCRPage(page_index=(start or 0) + index, text=text)
                    for index, text in enumerate(page_texts)
                )
        else:
            raise ProcessingFailedError(
                "Transformers OCR supports image files and PDFs only."
            )

        return OCRResult(title=input_path.stem, pages=pages)


def create_ocr_backend() -> MLXVLMOCRBackend | TransformersOCRBackend:
    """Create the OCR backend adapter for the current platform."""
    resolved = select_ocr_backend()
    if resolved == "mlx-vlm":
        return MLXVLMOCRBackend()

    return TransformersOCRBackend()
