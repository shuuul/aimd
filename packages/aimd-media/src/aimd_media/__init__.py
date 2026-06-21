"""Media package for aimd."""

from .capabilities import (
    EngineCapability,
    get_engine_capabilities,
    resolve_engine_with_preflight,
)
from .const import AUDIO_EXTENSIONS, MLX_AUDIO_MODELS
from .processor import transcribe_file, transcribe_file_sync
from .url import MediaTextResult, get_text_from_url
from ._plugin import (
    AimdMediaConverter,
    __plugin_interface_version__,
    register_converters,
)

__all__ = [
    "AUDIO_EXTENSIONS",
    "AimdMediaConverter",
    "EngineCapability",
    "MLX_AUDIO_MODELS",
    "MediaTextResult",
    "__plugin_interface_version__",
    "get_text_from_url",
    "get_engine_capabilities",
    "register_converters",
    "resolve_engine_with_preflight",
    "transcribe_file",
    "transcribe_file_sync",
]
