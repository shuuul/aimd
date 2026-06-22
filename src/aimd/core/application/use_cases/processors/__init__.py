"""Task processors used by the process input facade."""

from ._base import TaskProcessor
from .convert import ConvertProcessor, ConvertTaskProcessor
from .ocr import OCRProcessor, OCRTaskProcessor
from .transcript import TranscriptProcessor, TranscriptTaskProcessor

__all__ = [
    "ConvertProcessor",
    "ConvertTaskProcessor",
    "OCRProcessor",
    "OCRTaskProcessor",
    "TaskProcessor",
    "TranscriptProcessor",
    "TranscriptTaskProcessor",
]
