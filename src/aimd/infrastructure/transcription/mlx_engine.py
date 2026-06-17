"""mlx-audio STT transcription engine (Apple Silicon only)."""

import asyncio
import inspect
import platform
from pathlib import Path

from logly import logger

from ...const import MLX_AUDIO_DEFAULT_MODEL, MLX_AUDIO_MODELS
from ...errors import ProcessingFailedError, UnsupportedInputError
from ...platform_utils import is_apple_silicon
from .audio_utils import convert_to_wav_if_needed

LANGUAGE_CODE_TO_NAME = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
}


def _is_qwen3_asr_model(model: str) -> bool:
    """Qwen3-ASR models expect a language hint; other mlx-audio STT models often auto-detect."""
    return "Qwen3-ASR" in model


def _resolve_language(model: str, language: str | None) -> str | None:
    """Map short language codes to full names accepted by mlx-audio models."""
    if language is None:
        return "Chinese" if _is_qwen3_asr_model(model) else None
    lang = language.lower()
    if lang in LANGUAGE_CODE_TO_NAME:
        return LANGUAGE_CODE_TO_NAME[lang]
    for full_name in LANGUAGE_CODE_TO_NAME.values():
        if lang == full_name.lower():
            return full_name
    raise UnsupportedInputError(
        f"Unsupported language for mlx engine: '{language}'. "
        f"Supported: {list(LANGUAGE_CODE_TO_NAME.keys())}"
    )


async def transcribe_audio_mlx(
    file_path: Path,
    model: str | None = None,
    language: str | None = None,
    temp_dir: Path | None = None,
) -> str:
    """Transcribe audio using mlx-audio STT (Apple Silicon only)."""
    if platform.system() != "Darwin":
        raise ProcessingFailedError("mlx engine is only available on macOS")

    if not is_apple_silicon():
        raise ProcessingFailedError("mlx engine requires Apple Silicon (M1/M2/M3/M4)")

    try:
        from mlx_audio.stt import load as load_stt
    except ImportError:
        raise ProcessingFailedError(
            "mlx-audio library is not installed. Install it: pip install mlx-audio"
        )

    resolved_model = model or MLX_AUDIO_DEFAULT_MODEL
    if resolved_model not in MLX_AUDIO_MODELS:
        raise UnsupportedInputError(
            f"Unknown mlx-audio model: {resolved_model}. "
            f"Available: {list(MLX_AUDIO_MODELS.keys())}"
        )

    resolved_language = _resolve_language(resolved_model, language)
    logger.info(
        f"Transcribing with mlx-audio model: {resolved_model}, language: {resolved_language or 'auto'}"
    )

    wav_path: Path | None = None
    try:
        wav_path = convert_to_wav_if_needed(file_path, temp_dir=temp_dir)
        audio_path = wav_path or file_path

        def _transcribe() -> str:
            stt_model = load_stt(resolved_model)
            generate_kwargs = {}
            if resolved_language is not None:
                signature = inspect.signature(stt_model.generate)
                if "language" in signature.parameters:
                    generate_kwargs["language"] = resolved_language
            result = stt_model.generate(str(audio_path), **generate_kwargs)
            return result.text.strip()

        loop = asyncio.get_event_loop()
        transcribed_text = await loop.run_in_executor(None, _transcribe)

        if not transcribed_text:
            raise ProcessingFailedError("mlx-audio produced empty transcription")

        logger.info(
            f"Successfully transcribed {len(transcribed_text)} characters with mlx-audio"
        )
        return transcribed_text
    except (ProcessingFailedError, UnsupportedInputError):
        raise
    except Exception as e:
        raise ProcessingFailedError(f"mlx-audio transcription failed: {e}") from e
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
