"""URL processing infrastructure."""

from .cookies import (
    AUTH_REQUIRED_PLATFORMS,
    build_cookie_sources,
    is_auth_required_error,
    parse_cookies_from_browser,
)
from .audio_download import download_audio
from .processor import MediaTextResult, get_text_from_url

__all__ = [
    "AUTH_REQUIRED_PLATFORMS",
    "build_cookie_sources",
    "download_audio",
    "get_text_from_url",
    "is_auth_required_error",
    "MediaTextResult",
    "parse_cookies_from_browser",
]
