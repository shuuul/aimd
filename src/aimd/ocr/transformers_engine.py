"""Linux Transformers OCR backend adapter."""

from pathlib import Path
import shutil
import subprocess
import tempfile

from aimd.core.errors import EngineUnavailableError, ProcessingFailedError

from .engines import OCRPage, OCRResult
from .mlx4ocr_engine import IMAGE_FILE_EXTENSIONS
from .models import create_transformers_ocr_model


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
