"""OCR package and MarkItDown plugin."""

from .const import (
    IMAGE_FILE_EXTENSIONS,
    OCR_DOCUMENT_EXTENSIONS,
    OCR_EXTENSIONS,
)
from ._plugin import (
    AimdOCRConverter,
    __plugin_interface_version__,
    process_ocr,
    process_ocr_sync,
    register_converters,
)

__all__ = [
    "AimdOCRConverter",
    "IMAGE_FILE_EXTENSIONS",
    "OCR_DOCUMENT_EXTENSIONS",
    "OCR_EXTENSIONS",
    "__plugin_interface_version__",
    "process_ocr",
    "process_ocr_sync",
    "register_converters",
]
