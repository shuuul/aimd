"""Audio/video transcription orchestration."""

from pathlib import Path

from logly import logger

from ...const import AUDIO_EXTENSIONS, LANGUAGE_TO_YAP_LOCALE
from ...errors import InputNotFoundError, ProcessingFailedError, UnsupportedInputError
from ...types import TextContext
from .mlx_engine import transcribe_audio_mlx
from .resolver import resolve_engine_with_preflight
from .whisper_engine import transcribe_audio_whisper
from .yap_engine import transcribe_audio_yap


def _language_to_yap_locale(language: str | None) -> str:
    if language is None:
        return "zh_CN"

    lang = language.lower()
    if lang in LANGUAGE_TO_YAP_LOCALE:
        return LANGUAGE_TO_YAP_LOCALE[lang]

    raise UnsupportedInputError(
        f"Unsupported language for yap engine: '{language}'. "
        f"Supported: {list(LANGUAGE_TO_YAP_LOCALE.keys())}"
    )


async def get_text_from_audio(
    file_path: str | Path,
    engine: str = "auto",
    language: str | None = None,
    model_size: str = "large-v3-turbo",
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
        if actual_engine == "yap":
            transcribed_text = await transcribe_audio_yap(
                file_path,
                _language_to_yap_locale(language),
            )
        elif actual_engine == "mlx":
            transcribed_text = await transcribe_audio_mlx(
                file_path,
                model_size,
                language,
            )
        elif actual_engine in ("cuda", "cpu"):
            transcribed_text = await transcribe_audio_whisper(
                file_path,
                model_size,
                actual_engine,
                language,
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
