"""OCR backend contracts, platform resolution, and backend adapters."""

from dataclasses import dataclass
from pathlib import Path
import platform
import shutil
import tempfile

from aimd.core.errors import (
    BackendUnavailableError,
    ProcessingFailedError,
)

from .models import create_transformers_ocr_model
from .models.mlx import MLXVLMOCRModel

IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


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
        return MLXVLMOCRModel(model).recognize_image(input_path)

    def _recognize_pdf_or_document(
        self,
        input_path: Path,
        *,
        model: str | None,
        start: int | None,
        end: int | None,
        temp_dir: Path | None,
    ) -> tuple[OCRPage, ...]:
        if input_path.suffix.lower() != ".pdf":
            raise ProcessingFailedError(
                "MLX VLM OCR supports image files and PDFs only."
            )

        rendered_pages = _render_pdf_pages(
            input_path, start=start, end=end, temp_dir=temp_dir
        )
        try:
            return self._recognize_rendered_pages(
                rendered_pages,
                model=model,
            )
        finally:
            _cleanup_rendered_pages(rendered_pages)

    def _recognize_rendered_pages(
        self,
        rendered_pages: tuple[tuple[int, Path], ...],
        *,
        model: str | None,
    ) -> tuple[OCRPage, ...]:
        model_adapter = MLXVLMOCRModel(model)
        page_texts = model_adapter.recognize_images(_page_paths(rendered_pages))
        return tuple(
            OCRPage(
                page_index=page_index,
                text=text,
            )
            for (page_index, _image_path), text in zip(
                rendered_pages, page_texts, strict=True
            )
        )


def _render_pdf_pages(
    input_path: Path,
    *,
    start: int | None,
    end: int | None,
    temp_dir: Path | None,
) -> tuple[tuple[int, Path], ...]:
    """Render PDF pages to PNG paths with PyMuPDF."""
    try:
        import fitz
    except ImportError as exc:
        raise BackendUnavailableError(
            "PDF OCR requires pymupdf. Install project dependencies with `uv sync`, "
            "then retry."
        ) from exc

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
                page_path = (
                    output_dir / f"{input_path.stem}_page_{page_index + 1:04d}.png"
                )
                pixmap.save(page_path)
                rendered_pages.append((page_index, page_path))
        return tuple(rendered_pages)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def _cleanup_rendered_pages(rendered_pages: tuple[tuple[int, Path], ...]) -> None:
    if rendered_pages:
        shutil.rmtree(rendered_pages[0][1].parent, ignore_errors=True)


def _page_paths(rendered_pages: tuple[tuple[int, Path], ...]) -> list[Path]:
    return [path for _page_index, path in rendered_pages]


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
            rendered_pages = _render_pdf_pages(
                input_path, start=start, end=end, temp_dir=temp_dir
            )
            try:
                page_texts = model_adapter.recognize_images(
                    _page_paths(rendered_pages),
                    language=language,
                    temp_dir=temp_dir,
                )
                pages = tuple(
                    OCRPage(page_index=page_index, text=text)
                    for (page_index, _page_path), text in zip(
                        rendered_pages, page_texts, strict=True
                    )
                )
            finally:
                _cleanup_rendered_pages(rendered_pages)
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
