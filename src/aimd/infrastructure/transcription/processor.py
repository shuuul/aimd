"""Audio/video transcription orchestration."""

from pathlib import Path

from logly import logger

from ...const import AUDIO_EXTENSIONS
from ...errors import InputNotFoundError, ProcessingFailedError, UnsupportedInputError
from ...types import TextContext
from ..capabilities.detector import resolve_engine_with_preflight
from .mlx_engine import transcribe_audio_mlx
from .qwen_engine import transcribe_audio_qwen


async def get_text_from_audio(
    file_path: str | Path,
    engine: str = "auto",
    language: str | None = None,
    model: str | None = None,
    temp_dir: Path | None = None,
) -> TextContext:
    """Extract text from audio or video file using a transcription engine."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise InputNotFoundError(f"Audio/video file not found: {file_path}")

    file_ext = file_path.suffix.lower()
    if file_ext not in AUDIO_EXTENSIONS:
        raise UnsupportedInputError(
            f"Unsupported file format: {file_ext}. "
            f"Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )

    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".ts", ".m4v"}
    is_video = file_ext in video_extensions
    if is_video:
        logger.info(f"Processing video file: {file_path} (audio will be extracted)")
    else:
        logger.info(f"Processing audio file: {file_path}")

    actual_engine = resolve_engine_with_preflight(engine)
    logger.info(f"Using transcription engine: {actual_engine}")

    try:
        if actual_engine == "mlx":
            transcribed_text = await transcribe_audio_mlx(
                file_path,
                model=model,
                language=language,
                temp_dir=temp_dir,
            )
        elif actual_engine == "qwen":
            transcribed_text = await transcribe_audio_qwen(
                file_path,
                model=model,
                language=language,
                temp_dir=temp_dir,
            )
        else:
            raise UnsupportedInputError(f"Unsupported engine: {actual_engine}")
    except Exception as e:
        if isinstance(
            e, (InputNotFoundError, UnsupportedInputError, ProcessingFailedError)
        ):
            raise
        error_msg = str(e)
        if "format" in error_msg.lower() or "codec" in error_msg.lower():
            raise ProcessingFailedError(
                f"Transcription failed due to format/codec issue: {error_msg}. "
                "Try converting the file to a standard format like mp3 or wav."
            ) from e
        raise ProcessingFailedError(f"Transcription failed: {error_msg}") from e

    return TextContext(title=file_path.stem, chunk_list=[transcribed_text])
