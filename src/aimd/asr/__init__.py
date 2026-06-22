"""ASR package for local audio/video transcription."""

from .capabilities import (
    EngineCapability,
    get_engine_capabilities,
    resolve_engine_with_preflight,
)
from .const import (
    AUDIO_EXTENSIONS,
    AUDIO_FILE_EXTENSIONS,
    MLX_AUDIO_MODELS,
    QWEN_ASR_MODELS,
    VIDEO_FILE_EXTENSIONS,
)
from .processor import transcribe_file, transcribe_file_sync

__all__ = [
    "AUDIO_EXTENSIONS",
    "AUDIO_FILE_EXTENSIONS",
    "EngineCapability",
    "MLX_AUDIO_MODELS",
    "QWEN_ASR_MODELS",
    "VIDEO_FILE_EXTENSIONS",
    "get_engine_capabilities",
    "resolve_engine_with_preflight",
    "transcribe_file",
    "transcribe_file_sync",
]
