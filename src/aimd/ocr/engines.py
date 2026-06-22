"""OCR engine contracts and platform resolution."""

from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Protocol

from aimd.core.errors import EngineUnavailableError, UnsupportedEngineError


@dataclass(frozen=True, slots=True)
class OCRPage:
    """OCR text for one image or document page."""

    page_index: int | None
    text: str


@dataclass(frozen=True, slots=True)
class OCRResult:
    """Normalized OCR result returned by engine adapters."""

    title: str
    pages: tuple[OCRPage, ...]


class OCREngine(Protocol):
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


def resolve_ocr_engine(engine: str) -> str:
    """Resolve and validate an OCR engine name for the current platform."""
    normalized = engine.lower().strip() if engine else "auto"
    if normalized == "auto":
        system = platform.system().lower()
        if system == "darwin":
            return "mlx4ocr"
        if system == "linux":
            return "transformers"
        raise EngineUnavailableError(
            "OCR auto engine is unavailable on this platform. "
            "Supported OCR platforms are macOS (mlx4ocr) and Linux (transformers)."
        )

    if normalized not in {"mlx4ocr", "transformers"}:
        raise UnsupportedEngineError(
            "Unsupported OCR engine. Supported OCR engines: auto, mlx4ocr, transformers."
        )

    system = platform.system().lower()
    if normalized == "mlx4ocr" and system != "darwin":
        raise EngineUnavailableError("mlx4ocr OCR is only available on macOS.")
    if normalized == "transformers" and system != "linux":
        raise EngineUnavailableError("Transformers OCR is only available on Linux.")
    return normalized


def create_ocr_engine(engine: str) -> OCREngine:
    """Create an OCR backend adapter after platform resolution."""
    resolved = resolve_ocr_engine(engine)
    if resolved == "mlx4ocr":
        from .mlx4ocr_engine import MLX4OCREngine

        return MLX4OCREngine()

    from .transformers_engine import TransformersOCREngine

    return TransformersOCREngine()
