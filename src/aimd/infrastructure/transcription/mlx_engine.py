"""mlx-whisper transcription engine implementation."""

import asyncio
import platform
from pathlib import Path

from logly import logger

from ...const import MLX_MODEL_MAPPINGS
from ...errors import ProcessingFailedError, UnsupportedInputError
from ...platform_utils import is_apple_silicon


async def transcribe_audio_mlx(
    file_path: Path,
    model_size: str = "large-v3-turbo",
    language: str | None = None,
) -> str:
    """Transcribe audio using mlx-whisper (Apple Silicon only)."""
    if platform.system() != "Darwin":
        raise ProcessingFailedError("mlx engine is only available on macOS")

    if not is_apple_silicon():
        raise ProcessingFailedError("mlx engine requires Apple Silicon (M1/M2/M3/M4)")

    try:
        import mlx_whisper
    except ImportError:
        raise ProcessingFailedError(
            "mlx-whisper library is not installed. Please install it: "
            "pip install mlx-whisper"
        )

    if model_size not in MLX_MODEL_MAPPINGS:
        raise UnsupportedInputError(
            f"Unsupported model size: {model_size}. Supported: {list(MLX_MODEL_MAPPINGS.keys())}"
        )

    mlx_model_path = f"mlx-community/{MLX_MODEL_MAPPINGS[model_size]}"
    logger.info(f"Transcribing with MLX model: {mlx_model_path}")

    try:
        def _transcribe():
            return mlx_whisper.transcribe(
                str(file_path),
                path_or_hf_repo=mlx_model_path,
                language=language,
                word_timestamps=True,
            )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _transcribe)

        transcribed_text = result["text"].strip()
        if not transcribed_text:
            raise ProcessingFailedError("MLX Whisper produced empty transcription")

        logger.info(
            f"Successfully transcribed {len(transcribed_text)} characters with MLX"
        )
        return transcribed_text
    except Exception as e:
        raise ProcessingFailedError(f"MLX transcription failed: {e}") from e
