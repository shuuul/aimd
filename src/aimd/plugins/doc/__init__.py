"""Pandoc-backed document conversion package and MarkItDown plugin."""

from ._plugin import __plugin_interface_version__, register_converters
from .processor import (
    PANDOC_DOCUMENT_EXTENSIONS,
)

__all__ = [
    "PANDOC_DOCUMENT_EXTENSIONS",
    "__plugin_interface_version__",
    "register_converters",
]
