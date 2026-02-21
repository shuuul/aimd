"""faster-whisper CPU/CUDA engine implementation."""

import asyncio
from pathlib import Path

from logly import logger

from ...const import WHISPER_MODEL_SIZES
from ...errors import ProcessingFailedError, UnsupportedInputError


async def transcribe_audio_whisper(
    file_path: Path,
    model_size: str = "large-v3-turbo",
    device: str = "cpu",
    language: str | None = None,
) -> str:
    """Transcribe audio using faster-whisper with CPU or CUDA."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ProcessingFailedError(
            "faster-whisper library is not installed. Please install it: "
            "pip install faster-whisper"
        )

    if model_size not in WHISPER_MODEL_SIZES:
        raise UnsupportedInputError(
            f"Unsupported model size: {model_size}. Supported: {WHISPER_MODEL_SIZES}"
        )

    if device == "cuda":
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
        except ImportError:
            logger.warning("PyTorch not installed, falling back to CPU")
            device = "cpu"

    logger.info(f"Transcribing with faster-whisper: model={model_size}, device={device}")

    try:
        compute_type = "float16" if device == "cuda" else "int8"

        def _transcribe():
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            segments, info = model.transcribe(str(file_path), language=language, beam_size=5)
            transcribed_text = ""
            for segment in segments:
                transcribed_text += segment.text + "\n"
            return transcribed_text.strip(), info

        loop = asyncio.get_event_loop()
        transcribed_text, info = await loop.run_in_executor(None, _transcribe)
        if not transcribed_text:
            raise ProcessingFailedError("Whisper produced empty transcription")

        logger.info(
            f"Successfully transcribed {len(transcribed_text)} characters, "
            f"detected language: {info.language} ({info.language_probability:.2f})"
        )
        return transcribed_text
    except Exception as e:
        raise ProcessingFailedError(f"Whisper transcription failed: {e}") from e
