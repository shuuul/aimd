"""MarkItDown plugin for local audio/video transcription."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, BinaryIO

from markitdown import (
    DocumentConverter,
    DocumentConverterResult,
    MarkItDown,
    StreamInfo,
)
from logly import logger

from aimd.core.errors import UnsupportedInputError

from .capabilities import select_transcription_backend
from .const import AUDIO_EXTENSIONS, TRANSFORMERS_ASR_MODELS, VIDEO_FILE_EXTENSIONS
from .models.base import ASRModel
from .models.mlx import MLXAudioASRModel
from .models.transformers import TransformersASRModel

__plugin_interface_version__ = 1


async def transcribe_file(
    file_path: str | Path,
    language: str | None = None,
    model: str | None = None,
    temp_dir: Path | None = None,
) -> str:
    """Transcribe an audio or video file with the platform ASR backend."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise UnsupportedInputError(f"Audio/video file not found: {file_path}")

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
        asr_model = MLXAudioASRModel(model)
    else:
        asr_model = TransformersASRModel(model)

    return await asr_model.transcribe(
        file_path,
        language=language,
        temp_dir=temp_dir,
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
) -> str:
    """Synchronous MarkItDown boundary for ASR transcription."""
    return asyncio.run(
        transcribe_file(
            file_path,
            language=language,
            model=model,
            temp_dir=temp_dir,
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
        )

        return DocumentConverterResult(
            title=input_path.stem,
            markdown=transcript,
        )
