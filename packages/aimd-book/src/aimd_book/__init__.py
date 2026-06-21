"""Ebook package for aimd."""

from .cleaner import clean_markdown
from .processor import BookConversion, process_book_with_images
from ._plugin import (
    AimdBookConverter,
    __plugin_interface_version__,
    register_converters,
)

__all__ = [
    "AimdBookConverter",
    "BookConversion",
    "__plugin_interface_version__",
    "clean_markdown",
    "process_book_with_images",
    "register_converters",
]
