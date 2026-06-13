"""Qwen3-ASR transcription engine implementation (Linux/CUDA)."""

import asyncio
import warnings
from pathlib import Path

from logly import logger

from ...const import QWEN_ASR_DEFAULT_MODEL, QWEN_ASR_MODELS
from ...errors import ProcessingFailedError, UnsupportedInputError
from .audio_utils import convert_to_wav_if_needed

# Silence noisy upstream warnings emitted while loading nagisa / qwen-asr /
# transformers. These are benign and would otherwise pollute CLI output.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"nagisa(\..*)?")
warnings.filterwarnings(
    "ignore",
    message=r"The following generation flags are not valid.*",
    category=UserWarning,
)

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
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "ms": "Malay",
}

_cached_model = None
_cached_model_name: str | None = None


def _resolve_language(language: str | None) -> str | None:
    """Map short language codes to full names expected by qwen-asr, or None for auto."""
    if language is None:
        return None
    lang = language.lower()
    if lang in LANGUAGE_CODE_TO_NAME:
        return LANGUAGE_CODE_TO_NAME[lang]
    for full_name in LANGUAGE_CODE_TO_NAME.values():
        if lang == full_name.lower():
            return full_name
    raise UnsupportedInputError(
        f"Unsupported language for qwen engine: '{language}'. "
        f"Supported: {list(LANGUAGE_CODE_TO_NAME.keys())}"
    )


def _get_model(model_name: str):
    """Load or return cached Qwen3ASRModel."""
    global _cached_model, _cached_model_name  # noqa: PLW0603
    if _cached_model is not None and _cached_model_name == model_name:
        return _cached_model

    import torch
    from qwen_asr import Qwen3ASRModel  # type: ignore[import-untyped]

    _cached_model = Qwen3ASRModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=4096,
    )
    _ensure_pad_token(_cached_model)
    _cached_model_name = model_name
    return _cached_model


def _ensure_pad_token(model: object) -> None:
    """Set generation_config.pad_token_id to eos_token_id once.

    Avoids the per-call `Setting pad_token_id to eos_token_id` log line
    transformers prints during open-ended generation when a pad token is unset.
    """
    inner = getattr(model, "model", model)
    gen_config = getattr(inner, "generation_config", None)
    if gen_config is None:
        return
    eos = getattr(gen_config, "eos_token_id", None)
    if eos is not None and getattr(gen_config, "pad_token_id", None) is None:
        gen_config.pad_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos


async def transcribe_audio_qwen(
    file_path: Path,
    model: str | None = None,
    language: str | None = None,
    temp_dir: Path | None = None,
) -> str:
    """Transcribe audio using Qwen3-ASR (requires Linux + CUDA)."""
    try:
        import torch  # type: ignore
    except ImportError:
        raise ProcessingFailedError(
            "PyTorch is not installed. Required for qwen engine."
        )

    if not torch.cuda.is_available():
        raise ProcessingFailedError(
            "CUDA is not available. qwen engine requires a CUDA-capable GPU."
        )

    try:
        import qwen_asr  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        raise ProcessingFailedError(
            "qwen-asr library is not installed. Install it: pip install qwen-asr"
        )

    resolved_model = model or QWEN_ASR_DEFAULT_MODEL
    if resolved_model not in QWEN_ASR_MODELS:
        raise UnsupportedInputError(
            f"Unknown qwen-asr model: {resolved_model}. "
            f"Available: {list(QWEN_ASR_MODELS.keys())}"
        )

    resolved_language = _resolve_language(language)
    logger.info(
        f"Transcribing with qwen-asr model: {resolved_model}, language: {resolved_language or 'auto'}"
    )

    wav_path: Path | None = None
    try:
        wav_path = convert_to_wav_if_needed(file_path, temp_dir=temp_dir)
        audio_path = wav_path or file_path

        def _transcribe() -> str:
            qwen_model = _get_model(resolved_model)
            results = qwen_model.transcribe(
                audio=str(audio_path),
                language=resolved_language,
            )
            return results[0].text.strip()

        loop = asyncio.get_event_loop()
        transcribed_text = await loop.run_in_executor(None, _transcribe)

        if not transcribed_text:
            raise ProcessingFailedError("qwen-asr produced empty transcription")

        logger.info(
            f"Successfully transcribed {len(transcribed_text)} characters with qwen-asr"
        )
        return transcribed_text
    except (ProcessingFailedError, UnsupportedInputError):
        raise
    except Exception as e:
        raise ProcessingFailedError(f"qwen-asr transcription failed: {e}") from e
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
