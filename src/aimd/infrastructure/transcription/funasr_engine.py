"""FunASR transcription engine implementation (CPU/CUDA)."""

import asyncio
from pathlib import Path

from logly import logger

from ...const import FUNASR_DEFAULT_MODEL, FUNASR_MODELS
from ...errors import ProcessingFailedError, UnsupportedInputError

_SENSEVOICE_MODELS = {"FunAudioLLM/SenseVoiceSmall"}
_NO_SPEECH_CONTROL_TOKENS = ("<|no|>", "<|nospeech|>")
_NO_SPEECH_FALLBACK_MODEL = "FunAudioLLM/SenseVoiceSmall"

_cached_model = None
_cached_model_name: str | None = None
_cached_device: str | None = None


def _resolve_device() -> str:
    """Pick cuda:0 when available, otherwise cpu."""
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _get_model(model_name: str, device: str):
    """Load or return a cached FunASR AutoModel."""
    global _cached_model, _cached_model_name, _cached_device  # noqa: PLW0603
    if (
        _cached_model is not None
        and _cached_model_name == model_name
        and _cached_device == device
    ):
        return _cached_model

    from funasr import AutoModel  # type: ignore[import-untyped]

    _cached_model = AutoModel(
        model=model_name,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        hub="hf",
    )
    _cached_model_name = model_name
    _cached_device = device
    return _cached_model


def _contains_control_tokens(text: str) -> bool:
    """Return True when transcription output contains known no-speech tokens."""
    lowered = text.lower()
    return any(token in lowered for token in _NO_SPEECH_CONTROL_TOKENS)


def _clean_transcribed_text(text: str) -> str:
    """Remove control tokens that should not surface in user-facing transcripts."""
    cleaned = text
    for token in _NO_SPEECH_CONTROL_TOKENS:
        cleaned = cleaned.replace(token, " ")
    return " ".join(cleaned.split())


def _should_retry_with_sensevoice(exc: Exception, model_name: str) -> bool:
    """Retry only for the specific no-speech control-token failure mode."""
    if model_name in _SENSEVOICE_MODELS:
        return False
    return _contains_control_tokens(str(exc))


async def transcribe_audio_funasr(
    file_path: Path,
    model: str | None = None,
    language: str | None = None,
) -> str:
    """Transcribe audio using FunASR (Fun-ASR-Nano / SenseVoiceSmall)."""
    try:
        import funasr  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        raise ProcessingFailedError(
            "funasr library is not installed. Install it: pip install funasr"
        )

    resolved_model = model or FUNASR_DEFAULT_MODEL
    if resolved_model not in FUNASR_MODELS:
        raise UnsupportedInputError(
            f"Unknown FunASR model: {resolved_model}. "
            f"Available: {list(FUNASR_MODELS.keys())}"
        )

    device = _resolve_device()
    logger.info(f"Transcribing with FunASR model: {resolved_model}, device: {device}")

    try:

        def _transcribe_with_model(model_name: str) -> str:
            funasr_model = _get_model(model_name, device)

            is_sensevoice = model_name in _SENSEVOICE_MODELS
            if is_sensevoice:
                from funasr.utils.postprocess_utils import (
                    rich_transcription_postprocess,
                )  # type: ignore[import-untyped]

                res = funasr_model.generate(
                    input=str(file_path),
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15,
                )
                return rich_transcription_postprocess(res[0]["text"]).strip()

            res = funasr_model.generate(
                input=[str(file_path)],
                cache={},
                batch_size_s=0,
            )
            return res[0]["text"].strip()

        async def _run_transcription(model_name: str) -> str:
            return await loop.run_in_executor(
                None, lambda: _transcribe_with_model(model_name)
            )

        loop = asyncio.get_event_loop()
        try:
            raw_transcribed_text = await _run_transcription(resolved_model)
        except Exception as exc:
            if not _should_retry_with_sensevoice(exc, resolved_model):
                raise
            logger.warning(
                "Fun-ASR-Nano hit a no-speech control-token decode failure; "
                "retrying with SenseVoiceSmall"
            )
            raw_transcribed_text = await _run_transcription(_NO_SPEECH_FALLBACK_MODEL)

        transcribed_text = _clean_transcribed_text(raw_transcribed_text)
        if (
            not transcribed_text
            and resolved_model not in _SENSEVOICE_MODELS
            and _contains_control_tokens(raw_transcribed_text)
        ):
            logger.warning(
                "Fun-ASR-Nano produced only no-speech control tokens; "
                "retrying with SenseVoiceSmall"
            )
            raw_transcribed_text = await _run_transcription(_NO_SPEECH_FALLBACK_MODEL)
            transcribed_text = _clean_transcribed_text(raw_transcribed_text)

        if not transcribed_text:
            raise ProcessingFailedError("FunASR produced empty transcription")

        logger.info(
            f"Successfully transcribed {len(transcribed_text)} characters with FunASR"
        )
        return transcribed_text
    except (ProcessingFailedError, UnsupportedInputError):
        raise
    except Exception as e:
        raise ProcessingFailedError(f"FunASR transcription failed: {e}") from e
