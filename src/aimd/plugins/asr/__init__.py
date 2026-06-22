"""ASR package for local audio/video transcription."""

from ._plugin import AimdASRConverter, __plugin_interface_version__, register_converters
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
from .engines import ListEnginesResult, list_engines, list_transcription_engines
from .processor import transcribe_file, transcribe_file_sync

__all__ = [
    "AUDIO_EXTENSIONS",
    "AUDIO_FILE_EXTENSIONS",
    "AimdASRConverter",
    "EngineCapability",
    "ListEnginesResult",
    "MLX_AUDIO_MODELS",
    "QWEN_ASR_MODELS",
    "VIDEO_FILE_EXTENSIONS",
    "__plugin_interface_version__",
    "get_engine_capabilities",
    "list_engines",
    "list_transcription_engines",
    "register_converters",
    "resolve_engine_with_preflight",
    "transcribe_file",
    "transcribe_file_sync",
]
