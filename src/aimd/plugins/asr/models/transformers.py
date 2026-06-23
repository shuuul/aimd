"""Transformers ASR backend implementation for Qwen3-ASR."""

import asyncio
import shutil
import subprocess
import threading
import warnings
from pathlib import Path

from logly import logger

from aimd.core.errors import ProcessingFailedError, UnsupportedInputError

from ..audio_utils import convert_to_wav_if_needed
from ..const import TRANSFORMERS_ASR_DEFAULT_MODEL

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
QWEN3_ASR_PAD_TOKEN_ID = 151645


class TransformersASRModel:
    """Qwen3-ASR Transformers model adapter."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or TRANSFORMERS_ASR_DEFAULT_MODEL

    async def transcribe(
        self,
        file_path: Path,
        *,
        language: str | None = None,
        temp_dir: Path | None = None,
    ) -> str:
        """Transcribe audio using Qwen3-ASR through local Transformers classes."""
        try:
            import torch  # type: ignore
        except ImportError:
            raise ProcessingFailedError(
                "PyTorch is not installed. Required for Transformers ASR backend."
            )

        try:
            import transformers  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            raise ProcessingFailedError(
                "transformers is required for Transformers ASR backend."
            )

        device = _select_device(torch)
        logger.info(
            "Transcribing with Qwen3-ASR model on Transformers backend: "
            f"{self.model_id}, device: {device}, language: {language or 'auto'}"
        )

        wav_path: Path | None = None
        try:
            wav_path = convert_to_wav_if_needed(file_path, temp_dir=temp_dir)
            audio_path = wav_path or file_path

            def _transcribe() -> str:
                return _transcribe_qwen(audio_path, self.model_id, language)

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


_cached_model = None
_cached_processor = None
_cached_model_name: str | None = None
_cached_device: str | None = None
_model_cache_lock = threading.Lock()
_generation_lock = threading.Lock()


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


def _select_device(torch) -> str:  # noqa: ANN001
    """Select the best available PyTorch device for local Transformers ASR."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_model_and_processor(model_name: str):
    """Load or return cached Qwen3-ASR Transformers model and processor."""
    global _cached_model, _cached_processor, _cached_model_name, _cached_device  # noqa: PLW0603

    import torch

    device = _select_device(torch)
    if (
        _cached_model is not None
        and _cached_model_name == model_name
        and _cached_device == device
    ):
        return _cached_model, _cached_processor

    with _model_cache_lock:
        if (
            _cached_model is not None
            and _cached_model_name == model_name
            and _cached_device == device
        ):
            return _cached_model, _cached_processor

        from transformers import AutoModel, AutoProcessor

        from aimd.plugins.asr.qwen3_asr_transformers import (
            register_qwen3_asr_transformers,
        )

        register_qwen3_asr_transformers()
        dtype = (
            torch.bfloat16
            if device == "cuda"
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            else torch.float16
            if device in {"cuda", "mps"}
            else torch.float32
        )
        _cached_processor = AutoProcessor.from_pretrained(
            model_name,
            fix_mistral_regex=True,
        )
        _cached_model = AutoModel.from_pretrained(
            model_name,
            dtype=dtype,
        ).to(device)
        _cached_model.eval()
        _ensure_pad_token(_cached_model)
        _cached_model_name = model_name
        _cached_device = device
        return _cached_model, _cached_processor


def _ensure_pad_token(model: object) -> None:
    """Set generation_config.pad_token_id to eos_token_id once.

    Avoids the per-call `Setting pad_token_id to eos_token_id` log line
    transformers prints during open-ended generation when a pad token is unset.
    """
    candidates = [model, getattr(model, "thinker", None), getattr(model, "model", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        gen_config = getattr(candidate, "generation_config", None)
        if gen_config is None:
            continue
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
    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ProcessingFailedError(
            "Qwen3-ASR audio loading requires ffmpeg to produce 16 kHz mono PCM."
        )
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise ProcessingFailedError(
            f"ffmpeg audio decode failed: {result.stderr.decode(errors='replace')}"
        )
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ProcessingFailedError("Decoded audio is empty")
    return np.clip(audio, -1.0, 1.0)


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

    with _generation_lock, torch.no_grad():
        generated = asr_model.generate(
            **inputs,
            max_new_tokens=4096,
            pad_token_id=QWEN3_ASR_PAD_TOKEN_ID,
        )

    generated_ids = getattr(generated, "sequences", generated)
    decoded = processor.batch_decode(
        generated_ids[:, input_ids.shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return _parse_qwen_output(decoded[0], resolved_language).strip()
