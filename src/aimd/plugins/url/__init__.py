"""URL-to-Markdown processing and MarkItDown plugin."""

from ._plugin import (
    AimdReadableHtmlConverter,
    AimdUrlTranscriptConverter,
    __plugin_interface_version__,
    register_converters,
)
from .router import is_url
from .transcript import detect_platform

__all__ = [
    "AimdReadableHtmlConverter",
    "AimdUrlTranscriptConverter",
    "__plugin_interface_version__",
    "detect_platform",
    "is_url",
    "register_converters",
]
