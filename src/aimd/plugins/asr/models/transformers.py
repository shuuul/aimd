"""Transformers ASR backend implementation (Linux/CUDA)."""

import asyncio
import warnings
from pathlib import Path

from logly import logger

from ..audio_utils import convert_to_wav_if_needed
from ..const import (
    TRANSFORMERS_ASR_DEFAULT_MODEL,
    TRANSFORMERS_ASR_MODELS,
)
from ..errors import ProcessingFailedError, UnsupportedInputError

# Silence noisy upstream transformers generation warnings. These are benign and
# would otherwise pollute CLI output.
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
_cached_processor = None
_cached_model_name: str | None = None


def _resolve_language(language: str | None) -> str | None:
    """Map short language codes to full names expected by Qwen3-ASR, or None for auto."""
    if language is None:
        return None
    lang = language.lower()
    if lang in LANGUAGE_CODE_TO_NAME:
        return LANGUAGE_CODE_TO_NAME[lang]
    for full_name in LANGUAGE_CODE_TO_NAME.values():
        if lang == full_name.lower():
            return full_name
    raise UnsupportedInputError(
        f"Unsupported language for Transformers ASR backend: '{language}'. "
        f"Supported: {list(LANGUAGE_CODE_TO_NAME.keys())}"
    )


def _get_model_and_processor(model_name: str):
    """Load or return cached Qwen3-ASR Transformers model and processor."""
    global _cached_model, _cached_processor, _cached_model_name  # noqa: PLW0603
    if _cached_model is not None and _cached_model_name == model_name:
        return _cached_model, _cached_processor

    import torch
    from transformers import AutoModel, AutoProcessor

    dtype = (
        torch.bfloat16
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        else torch.float16
    )
    _cached_processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    _cached_model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=dtype,
    ).to("cuda")
    _ensure_pad_token(_cached_model)
    _cached_model_name = model_name
    return _cached_model, _cached_processor


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


def _model_device_and_dtype(model: object):
    """Return the first parameter device/dtype for tensor placement."""
    first_parameter = next(model.parameters())
    return first_parameter.device, first_parameter.dtype


def _inputs_to_model_device(inputs, model: object):
    """Move processor tensors to model device without casting token ids."""
    import torch

    device, dtype = _model_device_and_dtype(model)
    return {
        key: (
            (
                value.to(device=device, dtype=dtype)
                if torch.is_floating_point(value)
                else value.to(device=device)
            )
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in inputs.items()
    }


def _load_audio_array(audio_path: Path):
    """Load a mono 16 kHz float32 waveform for Qwen3-ASR processor input."""
    import torchaudio
    import torchaudio.functional as F

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = F.resample(waveform, sample_rate, 16000)
    waveform = waveform.squeeze(0).clamp(-1, 1).float()
    return waveform.cpu().numpy()


def _build_prompt(processor: object, language: str | None) -> str:
    """Build Qwen3-ASR audio chat prompt."""
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    if language is not None:
        prompt += f"language {language}<asr_text>"
    return prompt


def _parse_qwen_output(output: str, forced_language: str | None) -> str:
    """Extract transcription text from Qwen3-ASR generated output."""
    text = output.strip()
    if forced_language is not None:
        return text
    marker = "<asr_text>"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text


def _transcribe_qwen(audio_path: Path, model_name: str, language: str | None) -> str:
    """Run Qwen3-ASR through Hugging Face Transformers."""
    import torch

    resolved_language = _resolve_language(language)
    asr_model, processor = _get_model_and_processor(model_name)
    audio = _load_audio_array(audio_path)
    prompt = _build_prompt(processor, resolved_language)
    inputs = processor(
        text=[prompt],
        audio=[audio],
        return_tensors="pt",
        padding=True,
    )
    inputs = _inputs_to_model_device(inputs, asr_model)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        generated = asr_model.generate(**inputs, max_new_tokens=4096)

    generated_ids = getattr(generated, "sequences", generated)
    decoded = processor.batch_decode(
        generated_ids[:, input_ids.shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return _parse_qwen_output(decoded[0], resolved_language).strip()


async def transcribe_audio_transformers(
    file_path: Path,
    model: str | None = None,
    language: str | None = None,
    temp_dir: Path | None = None,
) -> str:
    """Transcribe audio using Transformers ASR models (requires Linux + CUDA)."""
    try:
        import torch  # type: ignore
    except ImportError:
        raise ProcessingFailedError(
            "PyTorch is not installed. Required for Transformers ASR backend."
        )

    if not torch.cuda.is_available():
        raise ProcessingFailedError(
            "CUDA is not available. Transformers ASR backend requires a CUDA-capable GPU."
        )

    try:
        import torchaudio  # type: ignore[import-untyped]  # noqa: F401
        import transformers  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        raise ProcessingFailedError(
            "transformers and torchaudio are required for Transformers ASR backend."
        )

    resolved_model = model or TRANSFORMERS_ASR_DEFAULT_MODEL
    if resolved_model not in TRANSFORMERS_ASR_MODELS:
        raise UnsupportedInputError(
            f"Unknown Transformers ASR model: {resolved_model}. "
            f"Available: {list(TRANSFORMERS_ASR_MODELS.keys())}"
        )

    logger.info(
        "Transcribing with Qwen3-ASR model on Transformers backend: "
        f"{resolved_model}, language: {language or 'auto'}"
    )

    wav_path: Path | None = None
    try:
        wav_path = convert_to_wav_if_needed(file_path, temp_dir=temp_dir)
        audio_path = wav_path or file_path

        def _transcribe() -> str:
            return _transcribe_qwen(audio_path, resolved_model, language)

        loop = asyncio.get_event_loop()
        transcribed_text = await loop.run_in_executor(None, _transcribe)

        if not transcribed_text:
            raise ProcessingFailedError("Qwen3-ASR produced empty transcription")

        logger.info(
            f"Successfully transcribed {len(transcribed_text)} characters with Qwen3-ASR"
        )
        return transcribed_text
    except (ProcessingFailedError, UnsupportedInputError):
        raise
    except Exception as e:
        raise ProcessingFailedError(f"Qwen3-ASR transcription failed: {e}") from e
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
