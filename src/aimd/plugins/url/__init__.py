"""URL-to-Markdown processing and MarkItDown plugin."""

from urllib.parse import urlparse

from ._plugin import (
    AimdReadableHtmlConverter,
    AimdUrlTranscriptConverter,
    __plugin_interface_version__,
    detect_platform,
    register_converters,
)


def is_url(value: str) -> bool:
    """Return whether a string is an HTTP(S) URL."""
    try:
        result = urlparse(value)
    except ValueError:
        return False
    return result.scheme in {"http", "https"} and bool(result.netloc)


__all__ = [
    "AimdReadableHtmlConverter",
    "AimdUrlTranscriptConverter",
    "__plugin_interface_version__",
    "detect_platform",
    "is_url",
    "register_converters",
]
