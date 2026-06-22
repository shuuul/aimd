"""URL-to-Markdown processing and MarkItDown plugin."""

from ._plugin import (
    AimdReadableHtmlConverter,
    AimdUrlTranscriptConverter,
    __plugin_interface_version__,
    register_converters,
)
from .cookies import (
    AUTH_REQUIRED_PLATFORMS,
    build_cookie_sources,
    is_auth_required_error,
    parse_cookies_from_browser,
)
from .audio_download import download_audio
from .defuddle import DefuddleResult, extract_html_with_defuddle
from .formatter import detect_platform
from .processor import UrlTextResult, get_text_from_url
from .router import is_url

__all__ = [
    "AUTH_REQUIRED_PLATFORMS",
    "AimdReadableHtmlConverter",
    "AimdUrlTranscriptConverter",
    "DefuddleResult",
    "UrlTextResult",
    "__plugin_interface_version__",
    "build_cookie_sources",
    "detect_platform",
    "download_audio",
    "extract_html_with_defuddle",
    "get_text_from_url",
    "is_auth_required_error",
    "is_url",
    "parse_cookies_from_browser",
    "register_converters",
]
