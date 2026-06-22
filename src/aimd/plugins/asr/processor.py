"""Audio/video transcription orchestration."""

import asyncio
from pathlib import Path

from logly import logger

from .capabilities import select_transcription_backend
from .const import AUDIO_EXTENSIONS
from .errors import ProcessingFailedError, UnsupportedInputError
from .models.mlx import transcribe_audio_mlx
from .models.qwen import transcribe_audio_qwen


async def transcribe_file(
    file_path: str | Path,
    language: str | None = None,
    model: str | None = None,
    temp_dir: Path | None = None,
) -> str:
    """Transcribe an audio or video file with the platform ASR backend."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise UnsupportedInputError(f"Audio/video file not found: {file_path}")

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

    backend = select_transcription_backend()
    logger.info(f"Using transcription backend: {backend}")

    try:
        if backend == "mlx":
            transcribed_text = await transcribe_audio_mlx(
                file_path,
                model=model,
                language=language,
                temp_dir=temp_dir,
            )
        elif backend == "qwen":
            transcribed_text = await transcribe_audio_qwen(
                file_path,
                model=model,
                language=language,
                temp_dir=temp_dir,
            )
        else:
            raise UnsupportedInputError(f"Unsupported transcription backend: {backend}")
    except Exception as e:
        if isinstance(e, (UnsupportedInputError, ProcessingFailedError)):
            raise
        error_msg = str(e)
        if "format" in error_msg.lower() or "codec" in error_msg.lower():
            raise ProcessingFailedError(
                f"Transcription failed due to format/codec issue: {error_msg}. "
                "Try converting the file to a standard format like mp3 or wav."
            ) from e
        raise ProcessingFailedError(f"Transcription failed: {error_msg}") from e

    return transcribed_text


def transcribe_file_sync(
    file_path: str | Path,
    language: str | None = None,
    model: str | None = None,
    temp_dir: Path | None = None,
) -> str:
    """Synchronous MarkItDown boundary for ASR transcription."""
    return asyncio.run(
        transcribe_file(
            file_path,
            language=language,
            model=model,
            temp_dir=temp_dir,
        )
    )
