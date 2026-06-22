"""Input classification and support checks for processing use-cases."""

from pathlib import Path
from typing import Callable

from ...errors import InputNotFoundError, UnsupportedInputError
from ...utils import is_url
from ..models import InputRoute, TaskType

from aimd.media.const import AUDIO_FILE_EXTENSIONS, VIDEO_FILE_EXTENSIONS

FileSupportChecker = Callable[[str | Path], bool]

IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
OCR_DOCUMENT_EXTENSIONS = {".pdf"}
VALID_TASK_TYPES: set[TaskType] = {"transcript", "convert", "ocr"}


def _pdf_has_extractable_text(file_path: Path) -> bool:
    """Best-effort check for text-layer PDFs before falling back to OCR."""
    try:
        import pymupdf
    except ImportError:
        return True

    try:
        with pymupdf.open(file_path) as document:
            for page in document:
                if page.get_text("text").strip():
                    return True
    except Exception:
        return True
    return False


def get_input_route(
    input_source: str,
    is_supported_file: FileSupportChecker,
    requested_task_type: TaskType | None = None,
) -> InputRoute:
    """Classify a source and select the processing task."""
    if is_url(input_source):
        if requested_task_type and requested_task_type != "transcript":
            return InputRoute(source_kind="url", task_type=None)
        return InputRoute(source_kind="url", task_type="transcript")

    try:
        file_path = Path(input_source)
        if file_path.exists():
            suffix = file_path.suffix.lower()
            if requested_task_type == "ocr":
                if suffix in IMAGE_FILE_EXTENSIONS:
                    return InputRoute(source_kind="image_file", task_type="ocr")
                if suffix in OCR_DOCUMENT_EXTENSIONS:
                    return InputRoute(source_kind="document_file", task_type="ocr")
                return InputRoute(source_kind="unknown", task_type=None)
            if suffix in IMAGE_FILE_EXTENSIONS:
                return InputRoute(source_kind="image_file", task_type="ocr")
            if suffix in OCR_DOCUMENT_EXTENSIONS and not _pdf_has_extractable_text(
                file_path
            ):
                return InputRoute(source_kind="document_file", task_type="ocr")
            if suffix in AUDIO_FILE_EXTENSIONS:
                if requested_task_type and requested_task_type != "transcript":
                    return InputRoute(source_kind="audio_file", task_type=None)
                return InputRoute(source_kind="audio_file", task_type="transcript")
            if suffix in VIDEO_FILE_EXTENSIONS:
                if requested_task_type and requested_task_type != "transcript":
                    return InputRoute(source_kind="video_file", task_type=None)
                return InputRoute(source_kind="video_file", task_type="transcript")
            if is_supported_file(file_path):
                if requested_task_type and requested_task_type != "convert":
                    return InputRoute(source_kind="document_file", task_type=None)
                return InputRoute(source_kind="document_file", task_type="convert")
    except (OSError, ValueError):
        pass

    return InputRoute(source_kind="unknown", task_type=None)


def ensure_supported_input(
    input_source: str,
    is_supported_file: FileSupportChecker,
    requested_task_type: TaskType | None = None,
) -> InputRoute:
    """Validate and return supported input route, else raise domain error."""
    if requested_task_type is not None and requested_task_type not in VALID_TASK_TYPES:
        raise UnsupportedInputError(
            "Unsupported task. Supported tasks: transcript, convert, ocr."
        )

    route = get_input_route(input_source, is_supported_file, requested_task_type)
    if route.task_type is None:
        input_path = Path(input_source)
        if not is_url(input_source) and input_path.suffix and not input_path.exists():
            raise InputNotFoundError(f"Input file not found: {input_source}")
        if requested_task_type == "ocr":
            raise UnsupportedInputError(
                "Unsupported OCR input. OCR supports image files "
                "(.png, .jpg, .jpeg, .webp, .tif, .tiff) and PDF files."
            )
        raise UnsupportedInputError(
            "Unsupported input source. Supported inputs: audio/video files, "
            "video URLs, supported document files, and OCR-capable images/PDFs."
        )
    return route
