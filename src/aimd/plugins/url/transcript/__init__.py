"""URL transcript extraction package."""

from .platforms import detect_platform
from .processor import UrlTextResult, get_text_from_url

__all__ = [
    "UrlTextResult",
    "detect_platform",
    "get_text_from_url",
]
