"""macOS mlx4ocr backend adapter."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from aimd.core.errors import EngineUnavailableError, ProcessingFailedError

from .engines import OCRPage, OCRResult

IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
DEFAULT_OCR_MODEL = "paddleocr_v6"
PPOCRV6_VARIANTS = {"tiny", "small", "medium"}


def _resolve_mlx4ocr_model(model: str | None) -> tuple[str, str | None]:
    """Map aimd OCR model names to mlx4ocr engine/variant options."""
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


class MLX4OCREngine:
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
        mlx4ocr_engine, variant = _resolve_mlx4ocr_model(model)
        if mlx4ocr_engine != "ppocrv6":
            return self._recognize_image_with_vlm(input_path, engine=mlx4ocr_engine)

        try:
            import cv2
            from mlx4ocr import PP_OCRv6
        except ImportError as exc:
            raise EngineUnavailableError(
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

    def _recognize_image_with_vlm(self, input_path: Path, *, engine: str) -> str:
        try:
            from mlx4ocr import VLMOCR
        except ImportError as exc:
            raise EngineUnavailableError(
                "mlx4ocr VLM OCR requires the optional mlx4ocr[vlm] dependencies. "
                "Install them before using glm_ocr or paddleocr_vl."
            ) from exc

        ocr = VLMOCR.from_hub(engine=engine)
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
            raise EngineUnavailableError(
                "mlx4ocr is not installed. Install OCR dependencies with `uv sync` "
                "on macOS/Python 3.12+, then retry."
            )
        mlx4ocr_engine, variant = _resolve_mlx4ocr_model(model)

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
            command.extend(["--engine", mlx4ocr_engine])
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
                    raise EngineUnavailableError(
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
