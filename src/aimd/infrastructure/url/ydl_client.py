"""yt-dlp client construction helpers."""

from typing import Any

import yt_dlp


def impersonation_available() -> bool:
    """Return True when yt-dlp impersonation dependencies are available."""
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "impersonate": "chrome"}):
            return True
    except Exception:
        return False


def create_ydl(
    *,
    platform: str,
    cookie_source: dict[str, Any],
    for_subtitles: bool,
) -> yt_dlp.YoutubeDL:
    """Create a fresh YoutubeDL client for a single operation."""
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
    }

    if not for_subtitles:
        ydl_opts.update(
            {
                "listsubtitles": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "skip_download": True,
            }
        )

    if platform == "youtube" and impersonation_available():
        ydl_opts["impersonate"] = "chrome"

    if cookie_source.get("use_cookies", False):
        if cookie_source.get("cookiefile"):
            ydl_opts["cookiefile"] = cookie_source["cookiefile"]
        elif cookie_source.get("cookiesfrombrowser"):
            ydl_opts["cookiesfrombrowser"] = cookie_source["cookiesfrombrowser"]

    return yt_dlp.YoutubeDL(ydl_opts)
