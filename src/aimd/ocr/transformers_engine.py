"""Linux Transformers OCR backend boundary."""

from pathlib import Path

from aimd.core.errors import EngineUnavailableError

from .engines import OCRResult


class TransformersOCREngine:
    """Placeholder for the Linux Transformers OCR implementation."""

    def recognize(
        self,
        input_path: Path,  # noqa: ARG002
        *,
        model: str | None = None,  # noqa: ARG002
        language: str | None = None,  # noqa: ARG002
        start: int | None = None,  # noqa: ARG002
        end: int | None = None,  # noqa: ARG002
        temp_dir: Path | None = None,  # noqa: ARG002
    ) -> OCRResult:
        raise EngineUnavailableError(
            "Transformers OCR is not implemented yet. Use macOS mlx4ocr for the "
            "first OCR pass, or select a Linux OCR model before enabling this backend."
        )
