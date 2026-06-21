"""Input classification and support checks for processing use-cases."""

from pathlib import Path
from typing import Callable

from ...errors import InputNotFoundError, UnsupportedInputError
from ...utils import is_url
from ..models import InputRoute

from aimd_media.const import AUDIO_FILE_EXTENSIONS, VIDEO_FILE_EXTENSIONS

FileSupportChecker = Callable[[str | Path], bool]


def get_input_route(
    input_source: str, is_supported_file: FileSupportChecker
) -> InputRoute:
    """Classify a source and select the processing task."""
    if is_url(input_source):
        return InputRoute(source_kind="url", task_type="transcript")

    try:
        file_path = Path(input_source)
        if file_path.exists():
            suffix = file_path.suffix.lower()
            if suffix in AUDIO_FILE_EXTENSIONS:
                return InputRoute(source_kind="audio_file", task_type="transcript")
            if suffix in VIDEO_FILE_EXTENSIONS:
                return InputRoute(source_kind="video_file", task_type="transcript")
            if is_supported_file(file_path):
                return InputRoute(source_kind="document_file", task_type="convert")
    except (OSError, ValueError):
        pass

    return InputRoute(source_kind="unknown", task_type=None)


def ensure_supported_input(
    input_source: str, is_supported_file: FileSupportChecker
) -> InputRoute:
    """Validate and return supported input route, else raise domain error."""
    route = get_input_route(input_source, is_supported_file)
    if route.task_type is None:
        input_path = Path(input_source)
        if not is_url(input_source) and input_path.suffix and not input_path.exists():
            raise InputNotFoundError(f"Input file not found: {input_source}")
        raise UnsupportedInputError(
            "Unsupported input source. Supported inputs: audio/video files, "
            "video URLs, and supported document files."
        )
    return route
