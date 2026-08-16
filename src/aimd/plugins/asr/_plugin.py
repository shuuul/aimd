"""MarkItDown plugin for local audio/video transcription."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from markitdown import (
    DocumentConverter,
    DocumentConverterResult,
    MarkItDown,
    StreamInfo,
)
from logly import logger

from aimd.core.errors import (
    InputNotFoundError,
    ProcessingCancelledError,
    ProcessingFailedError,
    UnsupportedInputError,
)

from .capabilities import select_transcription_backend
from .const import AUDIO_EXTENSIONS, TRANSFORMERS_ASR_MODELS, VIDEO_FILE_EXTENSIONS
from .models.base import ASRModel
from .models.mlx import MLXAudioASRModel
from .models.transformers import TransformersASRModel

__plugin_interface_version__ = 1


async def _transcribe_segment_with_fallback(
    asr_model: ASRModel,
    segment_path: Path,
    backend: str,
    language: str | None = None,
    temp_dir: Path | None = None,
    context: str | None = None,
) -> str:
    """Transcribes a single audio path, checking for repetition and falling back to 8-bit."""
    try:
        text = await asr_model.transcribe(
            segment_path, language=language, temp_dir=temp_dir, context=context
        )
    except ProcessingFailedError as e:
        if "empty transcription" in str(e).lower():
            return ""
        raise

    from .audio_utils import detect_repetition_loop, get_8bit_fallback_model

    if detect_repetition_loop(text):
        fallback_model_id = get_8bit_fallback_model(asr_model.model_id)
        if fallback_model_id:
            logger.warning(
                f"Repetition loop detected with 4-bit model ({asr_model.model_id}). "
                f"Switching to 8-bit fallback model: {fallback_model_id}"
            )
            if backend == "mlx":
                fallback_model = MLXAudioASRModel(fallback_model_id)
            else:
                fallback_model = TransformersASRModel(fallback_model_id)

            try:
                text = await fallback_model.transcribe(
                    segment_path, language=language, temp_dir=temp_dir, context=context
                )
            except ProcessingFailedError as e:
                if "empty transcription" in str(e).lower():
                    return ""
                raise
        else:
            logger.warning(
                "Repetition loop detected, but no 8-bit fallback model is available."
            )

    return text


async def transcribe_file(
    file_path: str | Path,
    language: str | None = None,
    model: str | None = None,
    temp_dir: Path | None = None,
    precision: str | None = None,
    cancellation_check: Callable[[], bool] | None = None,
    progress_reporter: Callable[[str, int | None, int | None, str | None], None]
    | None = None,
    context: str | None = None,
) -> str:
    """Transcribe an audio or video file with the platform ASR backend."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise InputNotFoundError(f"Audio/video file not found: {file_path}")

    file_ext = file_path.suffix.lower()
    if file_ext not in AUDIO_EXTENSIONS:
        raise UnsupportedInputError(
            f"Unsupported file format: {file_ext}. "
            f"Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )

    is_video = file_ext in VIDEO_FILE_EXTENSIONS
    if is_video:
        logger.info(f"Processing video file: {file_path} (audio will be extracted)")
    else:
        logger.info(f"Processing audio file: {file_path}")

    backend = _select_backend_for_model(model)
    logger.info(f"Using transcription backend: {backend}")

    asr_model: ASRModel
    if backend == "mlx":
        asr_model = MLXAudioASRModel(model, precision=precision)
    else:
        asr_model = TransformersASRModel(model, precision=precision)

    from .audio_utils import get_audio_duration, segment_audio

    try:
        duration = get_audio_duration(file_path)
        logger.info(f"Audio/video duration is {duration:.2f} seconds")
    except Exception as e:
        logger.warning(
            f"Could not determine audio/video duration: {e}. Processing without segmentation."
        )
        duration = 0.0

    if duration > 600.0:
        logger.info(
            "Audio/video duration exceeds 10 minutes. Segmenting for transcription..."
        )
        segments = segment_audio(file_path, segment_time_secs=600.0, temp_dir=temp_dir)
        try:
            results = []
            for i, segment_path in enumerate(segments):
                if cancellation_check is not None and cancellation_check():
                    raise ProcessingCancelledError(
                        "Transcription cancelled between segments"
                    )
                if progress_reporter is not None:
                    progress_reporter(
                        "transcribing",
                        i + 1,
                        len(segments),
                        f"Transcribing segment {i + 1} of {len(segments)}",
                    )
                logger.info(
                    f"Transcribing segment {i + 1}/{len(segments)}: {segment_path.name}"
                )
                text = await _transcribe_segment_with_fallback(
                    asr_model,
                    segment_path,
                    backend,
                    language=language,
                    temp_dir=temp_dir,
                    context=context,
                )
                if text:
                    results.append(text)

            transcribed_text = " ".join(results).strip()
            if not transcribed_text:
                raise ProcessingFailedError(
                    "ASR produced empty transcription for all segments"
                )
            return transcribed_text
        finally:
            for segment_path in segments:
                segment_path.unlink(missing_ok=True)
    else:
        return await _transcribe_segment_with_fallback(
            asr_model,
            file_path,
            backend,
            language=language,
            temp_dir=temp_dir,
            context=context,
        )


def _select_backend_for_model(model: str | None) -> str:
    """Select ASR backend, honoring explicit Transformers Qwen3-ASR model IDs."""
    if model in TRANSFORMERS_ASR_MODELS or (model or "").startswith("Qwen/Qwen3-ASR-"):
        return "transformers"
    return select_transcription_backend()


def _transcribe_file_sync(
    file_path: Path,
    *,
    language: str | None = None,
    model: str | None = None,
    temp_dir: Path | None = None,
    precision: str | None = None,
    cancellation_check: Callable[[], bool] | None = None,
    progress_reporter: Callable[[str, int | None, int | None, str | None], None]
    | None = None,
    context: str | None = None,
) -> str:
    """Synchronous MarkItDown boundary for ASR transcription."""
    return asyncio.run(
        transcribe_file(
            file_path,
            language=language,
            model=model,
            temp_dir=temp_dir,
            precision=precision,
            cancellation_check=cancellation_check,
            progress_reporter=progress_reporter,
            context=context,
        )
    )


def register_converters(markitdown: MarkItDown, **kwargs: Any) -> None:
    """Register the ASR converter with MarkItDown."""
    markitdown.register_converter(AimdASRConverter(), priority=-1.0)


class AimdASRConverter(DocumentConverter):
    """Convert local audio/video files to markdown transcripts."""

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        return (stream_info.extension or "").lower() in AUDIO_EXTENSIONS

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        input_path = Path(stream_info.local_path)
        transcript = _transcribe_file_sync(
            input_path,
            language=kwargs.get("language"),
            model=kwargs.get("model"),
            temp_dir=kwargs.get("temp_dir"),
            precision=kwargs.get("precision"),
            cancellation_check=kwargs.get("cancellation_check"),
            progress_reporter=kwargs.get("progress_reporter"),
            context=kwargs.get("context"),
        )

        return DocumentConverterResult(
            title=input_path.stem,
            markdown=transcript,
        )
