"""OCR backend contracts, platform resolution, and backend adapters."""

from dataclasses import dataclass
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Protocol

from aimd.core.errors import (
    BackendUnavailableError,
    ProcessingFailedError,
)

from .models import create_transformers_ocr_model

IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
DEFAULT_OCR_MODEL = "paddleocr_v6"
PPOCRV6_VARIANTS = {"tiny", "small", "medium"}


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


class OCRBackend(Protocol):
    """OCR backend adapter."""

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
        """Recognize text from an image or PDF."""
        ...


def select_ocr_backend() -> str:
    """Select the platform OCR backend."""
    system = platform.system().lower()
    if system == "darwin":
        return "mlx4ocr"
    if system == "linux":
        return "transformers"
    raise BackendUnavailableError(
        "OCR is unavailable on this platform. Supported OCR platforms are macOS "
        "with mlx4ocr and Linux with CUDA-capable Transformers."
    )


def _resolve_mlx4ocr_model(model: str | None) -> tuple[str, str | None]:
    """Map aimd OCR model names to mlx4ocr backend/variant options."""
    normalized = (model or DEFAULT_OCR_MODEL).lower().replace("-", "_")
    if normalized in {"paddleocr_v6", "ppocrv6", "pp_ocrv6"}:
        return "ppocrv6", "medium"
    if normalized == "glm_ocr":
        return "glm-ocr", None
    if normalized == "paddleocr_vl":
        return "paddleocr-vl", None
    if normalized in PPOCRV6_VARIANTS:
        return "ppocrv6", normalized
    raise ProcessingFailedError(
        "Unsupported mlx4ocr model. Supported models: "
        "paddleocr_v6, glm_ocr, paddleocr_vl. "
        "PP-OCRv6 variants tiny, small, and medium are also accepted."
    )


class MLX4OCRBackend:
    """Run OCR through the mlx4ocr macOS runtime."""

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
        else:
            text = self._recognize_pdf_or_document(
                input_path,
                model=model,
                start=start,
                end=end,
                temp_dir=temp_dir,
            )

        return OCRResult(
            title=input_path.stem,
            pages=(OCRPage(page_index=None, text=text.strip()),),
        )

    def _recognize_image(self, input_path: Path, *, model: str | None) -> str:
        mlx4ocr_backend, variant = _resolve_mlx4ocr_model(model)
        if mlx4ocr_backend != "ppocrv6":
            return self._recognize_image_with_vlm(
                input_path, mlx4ocr_backend=mlx4ocr_backend
            )

        try:
            import cv2
            from mlx4ocr import PP_OCRv6
        except ImportError as exc:
            raise BackendUnavailableError(
                "mlx4ocr is not installed. Install OCR dependencies with `uv sync` "
                "on macOS/Python 3.12+, then retry."
            ) from exc

        image = cv2.imread(input_path.as_posix())
        if image is None:
            raise ProcessingFailedError(f"Unable to read image for OCR: {input_path}")

        ocr = PP_OCRv6.from_hub(variant)
        try:
            result = ocr.predict(image)
            return str(result.result.text).strip()
        finally:
            ocr.close()

    def _recognize_image_with_vlm(
        self, input_path: Path, *, mlx4ocr_backend: str
    ) -> str:
        try:
            from mlx4ocr import VLMOCR
        except ImportError as exc:
            raise BackendUnavailableError(
                "mlx4ocr VLM OCR requires the optional mlx4ocr[vlm] dependencies. "
                "Install them before using glm_ocr or paddleocr_vl."
            ) from exc

        ocr = VLMOCR.from_hub(engine=mlx4ocr_backend)
        try:
            result = ocr.predict_path(input_path.as_posix())
            return str(result.text).strip()
        finally:
            ocr.close()

    def _recognize_pdf_or_document(
        self,
        input_path: Path,
        *,
        model: str | None,
        start: int | None,
        end: int | None,
        temp_dir: Path | None,
    ) -> str:
        executable = Path(sys.executable).with_name("mlx4ocr")
        mlx4ocr_command = (
            executable.as_posix() if executable.exists() else shutil.which("mlx4ocr")
        )
        if mlx4ocr_command is None:
            raise BackendUnavailableError(
                "mlx4ocr is not installed. Install OCR dependencies with `uv sync` "
                "on macOS/Python 3.12+, then retry."
            )
        mlx4ocr_backend, variant = _resolve_mlx4ocr_model(model)

        with tempfile.TemporaryDirectory(
            prefix="aimd-ocr-", dir=temp_dir
        ) as output_root:
            command = [
                mlx4ocr_command,
                "--path",
                input_path.as_posix(),
                "--format",
                "markdown",
                "--output",
                output_root,
            ]
            command.extend(["--engine", mlx4ocr_backend])
            if variant:
                command.extend(["--variant", variant])
            if start is not None:
                command.extend(["--start", str(start)])
            if end is not None:
                command.extend(["--end", str(end)])

            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                stderr = completed.stderr.strip() or completed.stdout.strip()
                if "No module named mlx4ocr" in stderr:
                    raise BackendUnavailableError(
                        "mlx4ocr is not installed. Install OCR dependencies with `uv sync` "
                        "on macOS/Python 3.12+, then retry."
                    )
                raise ProcessingFailedError(f"mlx4ocr failed: {stderr}")

            output_path = (
                Path(output_root) / input_path.stem / "ocr" / f"{input_path.stem}.md"
            )
            if output_path.exists():
                return output_path.read_text(encoding="utf-8").strip()
            return completed.stdout.strip()


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


def create_ocr_backend() -> OCRBackend:
    """Create the OCR backend adapter for the current platform."""
    resolved = select_ocr_backend()
    if resolved == "mlx4ocr":
        return MLX4OCRBackend()

    return TransformersOCRBackend()
