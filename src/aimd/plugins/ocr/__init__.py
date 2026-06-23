"""OCR package and MarkItDown plugin."""

from ._plugin import (
    AimdOCRConverter,
    __plugin_interface_version__,
    process_ocr,
    process_ocr_sync,
    register_converters,
)

__all__ = [
    "AimdOCRConverter",
    "__plugin_interface_version__",
    "process_ocr",
    "process_ocr_sync",
    "register_converters",
]
