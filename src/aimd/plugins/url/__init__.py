"""URL-to-Markdown processing and MarkItDown plugin."""

from ._plugin import (
    AimdReadableHtmlConverter,
    AimdUrlTranscriptConverter,
    __plugin_interface_version__,
    register_converters,
)
from .formatter import detect_platform
from .router import is_url

__all__ = [
    "AimdReadableHtmlConverter",
    "AimdUrlTranscriptConverter",
    "__plugin_interface_version__",
    "detect_platform",
    "is_url",
    "register_converters",
]
