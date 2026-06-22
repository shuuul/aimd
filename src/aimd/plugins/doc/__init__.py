"""Pandoc-backed document conversion package and MarkItDown plugin."""

from .cleaner import clean_markdown
from .processor import (
    DocConversion,
    PANDOC_DOCUMENT_EXTENSIONS,
    PANDOC_INPUT_FORMAT_BY_EXTENSION,
    process_doc_with_assets,
)
from ._plugin import (
    AimdDocConverter,
    __plugin_interface_version__,
    register_converters,
)

__all__ = [
    "AimdDocConverter",
    "DocConversion",
    "PANDOC_DOCUMENT_EXTENSIONS",
    "PANDOC_INPUT_FORMAT_BY_EXTENSION",
    "__plugin_interface_version__",
    "clean_markdown",
    "process_doc_with_assets",
    "register_converters",
]
