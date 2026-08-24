"""yt-dlp client construction for URL transcript extraction."""

from functools import lru_cache
from typing import Any

import yt_dlp

# YouTube n-challenge solving needs an external JS runtime. yt-dlp defaults to
# deno only; enable node as well so macOS/Homebrew Node setups work without Deno.
_YOUTUBE_JS_RUNTIMES: dict[str, dict[str, str]] = {
    "deno": {},
    "node": {},
}


class _QuietYtDlpLogger:
    """Suppress yt-dlp's direct stderr logging; AIMD maps and logs errors itself."""

    def debug(self, msg: str) -> None:  # noqa: ARG002
        pass

    def warning(self, msg: str) -> None:  # noqa: ARG002
        pass

    def error(self, msg: str) -> None:  # noqa: ARG002
        pass


@lru_cache(maxsize=1)
def impersonation_available() -> bool:
    """Return True when yt-dlp impersonation dependencies are available."""
    try:
        with yt_dlp.YoutubeDL(
            {
                "quiet": True,
                "no_warnings": True,
                "ignoreconfig": True,
                "logger": _QuietYtDlpLogger(),
                "impersonate": "chrome",
            }
        ):
            return True
    except Exception:
        return False


def apply_cookie_source(
    ydl_opts: dict[str, Any],
    cookie_source: dict[str, Any],
) -> None:
    """Apply a selected cookie source to yt-dlp options."""
    if not cookie_source.get("use_cookies", False):
        return

    if cookie_source.get("cookiefile"):
        ydl_opts["cookiefile"] = cookie_source["cookiefile"]
    elif cookie_source.get("cookiesfrombrowser"):
        ydl_opts["cookiesfrombrowser"] = cookie_source["cookiesfrombrowser"]


def _base_ydl_opts(platform: str) -> dict[str, Any]:
    """Return common yt-dlp options for URL operations.

    Args:
        platform: Detected URL platform name (for example ``youtube``).

    Returns:
        yt-dlp option dict shared by metadata, subtitle, and audio clients.
    """
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "ignoreconfig": True,
        "logger": _QuietYtDlpLogger(),
    }

    if platform == "youtube":
        ydl_opts["js_runtimes"] = dict(_YOUTUBE_JS_RUNTIMES)
        if impersonation_available():
            ydl_opts["impersonate"] = "chrome"

    return ydl_opts


def create_info_ydl(
    *,
    platform: str,
    cookie_source: dict[str, Any],
) -> yt_dlp.YoutubeDL:
    """Create a YoutubeDL client for metadata/subtitle listing."""
    ydl_opts = _base_ydl_opts(platform)
    ydl_opts.update(
        {
            "writesubtitles": True,
            "writeautomaticsub": True,
            "skip_download": True,
        }
    )
    apply_cookie_source(ydl_opts, cookie_source)

    return yt_dlp.YoutubeDL(ydl_opts)


# yt-dlp's networking default is 20s. YouTube timedtext conversion of long
# ASR tracks (especially ``fmt=srt``) can stall past that before succeeding
# or returning 502, so subtitle fetches wait a bit longer.
SUBTITLE_SOCKET_TIMEOUT = 40.0


def create_subtitle_ydl(
    *,
    platform: str,
    cookie_source: dict[str, Any],
) -> yt_dlp.YoutubeDL:
    """Create a YoutubeDL client for direct subtitle download."""
    ydl_opts = _base_ydl_opts(platform)
    ydl_opts["socket_timeout"] = SUBTITLE_SOCKET_TIMEOUT
    apply_cookie_source(ydl_opts, cookie_source)

    return yt_dlp.YoutubeDL(ydl_opts)
