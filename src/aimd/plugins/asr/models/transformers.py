"""Transformers ASR backend implementation for Qwen3-ASR."""

import asyncio
import threading
from pathlib import Path

from logly import logger

from aimd.core.errors import (
    BackendUnavailableError,
    ProcessingFailedError,
    UnsupportedInputError,
)
from aimd.core.precision import normalize_transformers_precision
from aimd.core.version import parse_version_tuple

from ..audio_utils import convert_to_wav_if_needed
from ..const import (
    TRANSFORMERS_ASR_DEFAULT_MODEL,
    resolve_transformers_asr_model,
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
    "yue": "Cantonese",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}


class TransformersASRModel:
    """Qwen3-ASR Transformers model adapter (native HF backend)."""

    def __init__(
        self, model_id: str | None = None, precision: str | None = None
    ) -> None:
        # Fail fast on quantized precisions; bf16 device support is validated
        # at load time when torch device capabilities are known.
        self.precision = normalize_transformers_precision(precision)
        self.model_id = resolve_transformers_asr_model(
            model_id or TRANSFORMERS_ASR_DEFAULT_MODEL
        )

    async def transcribe(
        self,
        file_path: Path,
        *,
        language: str | None = None,
        temp_dir: Path | None = None,
        context: str | None = None,
    ) -> str:
        """Transcribe audio using native Transformers Qwen3-ASR."""
        try:
            import torch  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ProcessingFailedError(
                "PyTorch is not installed. Required for Transformers ASR backend."
            ) from exc

        try:
            import transformers  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:
            raise ProcessingFailedError(
                "transformers is required for Transformers ASR backend."
            ) from exc

        device = _select_device()
        logger.info(
            "Transcribing with Qwen3-ASR model on Transformers backend: "
            f"{self.model_id}, device: {device}, language: {language or 'auto'}"
        )

        wav_path: Path | None = None
        try:
            wav_path = convert_to_wav_if_needed(file_path, temp_dir=temp_dir)
            audio_path = wav_path or file_path

            def _transcribe() -> str:
                return _transcribe_qwen(
                    audio_path, self.model_id, language, self.precision, context
                )

            transcribed_text = await asyncio.to_thread(_transcribe)

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
_cached_precision: str | None = None
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


def _select_device() -> str:
    """Select the best available PyTorch device for local Transformers ASR."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _require_native_qwen3_asr() -> None:
    """Fail fast when the installed Transformers build lacks native Qwen3-ASR."""
    import transformers

    if parse_version_tuple(transformers.__version__) < (5, 14, 1):
        raise ProcessingFailedError(
            "Native Qwen3-ASR requires transformers>=5.14.1 "
            f"(installed {transformers.__version__})."
        )
    try:
        from transformers import AutoModelForMultimodalLM  # noqa: F401
    except ImportError as exc:
        raise ProcessingFailedError(
            "Installed transformers build lacks AutoModelForMultimodalLM "
            "required for native Qwen3-ASR."
        ) from exc


def _resolve_torch_dtype(device: str, precision: str | None):
    """Pick the torch dtype for a device, honoring an explicit precision."""
    import torch

    if precision == "bf16":
        if (
            device == "cuda"
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        ):
            return torch.bfloat16
        raise BackendUnavailableError(
            "precision=bf16 requires a CUDA device with bf16 support. "
            "Omit --precision to use automatic dtype selection."
        )
    if precision is not None:
        raise ProcessingFailedError(
            f"Transformers ASR backend does not support precision {precision!r}. "
            "Use bf16 or omit precision for automatic dtype selection."
        )
    if device == "cuda" and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def _get_model_and_processor(model_name: str, precision: str | None = None):
    """Load or return cached native Qwen3-ASR Transformers model and processor."""
    global _cached_model, _cached_processor, _cached_model_name, _cached_device, _cached_precision  # noqa: PLW0603

    device = _select_device()
    if (
        _cached_model is not None
        and _cached_model_name == model_name
        and _cached_device == device
        and _cached_precision == precision
    ):
        return _cached_model, _cached_processor

    with _model_cache_lock:
        if (
            _cached_model is not None
            and _cached_model_name == model_name
            and _cached_device == device
            and _cached_precision == precision
        ):
            return _cached_model, _cached_processor

        _require_native_qwen3_asr()
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        dtype = _resolve_torch_dtype(device, precision)
        _cached_processor = AutoProcessor.from_pretrained(model_name)
        _cached_model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            dtype=dtype,
        ).to(device)
        _cached_model.eval()
        _ensure_pad_token(_cached_model)
        _cached_model_name = model_name
        _cached_device = device
        _cached_precision = precision
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


def _inputs_to_model_device(inputs, model: object):
    """Move processor tensors to model device without casting token ids."""
    import torch

    first_parameter = next(model.parameters())
    device = first_parameter.device
    dtype = first_parameter.dtype
    moved = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _parse_qwen_output(output: str) -> str:
    """Extract transcription text from Qwen3-ASR generated output."""
    text = output.strip()
    marker = "<asr_text>"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    if "assistant\n" in text:
        text = text.split("assistant\n", 1)[-1].strip()
    return text


def _build_transcription_inputs(
    processor,
    audio_path: Path,
    resolved_language: str | None,
    context: str | None,
):
    """Build processor inputs, injecting context as a system message when given.

    Qwen3-ASR biases transcription toward free-form context (vocabulary, names,
    background information) provided in the system turn. Without context, use
    the convenience `apply_transcription_request` wrapper.
    """
    if not context:
        return processor.apply_transcription_request(
            audio=str(audio_path),
            language=resolved_language,
        )

    messages: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": context}]},
        {
            "role": "user",
            "content": [{"type": "audio", "path": str(audio_path)}],
        },
    ]
    chat_kwargs: dict[str, object] = {"tokenize": True, "return_dict": True}
    if resolved_language is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"language {resolved_language}<asr_text>"}
                ],
            }
        )
        chat_kwargs["continue_final_message"] = True
    else:
        chat_kwargs["add_generation_prompt"] = True

    return processor.apply_chat_template([messages], **chat_kwargs)


def _transcribe_qwen(
    audio_path: Path,
    model_name: str,
    language: str | None,
    precision: str | None = None,
    context: str | None = None,
) -> str:
    """Run native Hugging Face Qwen3-ASR transcription."""
    import torch

    resolved_language = _resolve_language(language)
    asr_model, processor = _get_model_and_processor(model_name, precision)

    if not hasattr(processor, "apply_transcription_request"):
        raise ProcessingFailedError(
            "Installed transformers processor lacks apply_transcription_request; "
            "upgrade to transformers>=5.14.1 for native Qwen3-ASR."
        )

    inputs = _build_transcription_inputs(
        processor, audio_path, resolved_language, context
    )
    inputs = _inputs_to_model_device(inputs, asr_model)
    input_ids = inputs["input_ids"]

    with _generation_lock, torch.inference_mode():
        generated = asr_model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )

    generated_ids = getattr(generated, "sequences", generated)
    new_tokens = generated_ids[:, input_ids.shape[1] :]

    if hasattr(processor, "decode"):
        try:
            decoded = processor.decode(
                new_tokens,
                return_format="transcription_only",
            )
            if isinstance(decoded, list):
                return str(decoded[0]).strip()
            return str(decoded).strip()
        except TypeError:
            # Older processor.decode signatures without return_format.
            pass

    if hasattr(processor, "batch_decode"):
        text = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    elif hasattr(processor, "tokenizer"):
        text = processor.tokenizer.decode(
            new_tokens[0],
            skip_special_tokens=True,
        )
    else:
        raise ProcessingFailedError("Qwen3-ASR processor cannot decode outputs")
    return _parse_qwen_output(text).strip()
