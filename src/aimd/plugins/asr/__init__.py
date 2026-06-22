"""ASR package for local audio/video transcription."""

from ._plugin import AimdASRConverter, __plugin_interface_version__, register_converters
from .capabilities import (
    BackendCapability,
    get_backend_capabilities,
    select_transcription_backend,
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
    "AimdASRConverter",
    "BackendCapability",
    "MLX_AUDIO_MODELS",
    "QWEN_ASR_MODELS",
    "VIDEO_FILE_EXTENSIONS",
    "__plugin_interface_version__",
    "get_backend_capabilities",
    "register_converters",
    "select_transcription_backend",
    "transcribe_file",
    "transcribe_file_sync",
]
