"""Media package for core."""

from .url import MediaTextResult, get_text_from_url
from ._plugin import (
    AimdMediaConverter,
    __plugin_interface_version__,
    register_converters,
)

__all__ = [
    "AimdMediaConverter",
    "MediaTextResult",
    "__plugin_interface_version__",
    "get_text_from_url",
    "register_converters",
]
