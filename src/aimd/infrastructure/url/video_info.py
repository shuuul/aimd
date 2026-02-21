"""Video metadata extraction for URL processing."""

import asyncio
from typing import Any

from logly import logger

from ...errors import ProcessingFailedError, UnsupportedInputError
from .cookies import (
    AUTH_REQUIRED_PLATFORMS,
    build_cookie_sources,
    is_auth_required_error,
    is_keyring_error,
    is_unsupported_url_error,
)
from .ydl_client import create_ydl


async def extract_video_info(
    *,
    url: str,
    platform: str,
    cookies_file: str | None,
    cookies_from_browser: str | None,
) -> dict[str, Any]:
    """Extract video information using yt-dlp with cookie-source fallback chain."""
    loop = asyncio.get_running_loop()
    sources = build_cookie_sources(
        platform=platform,
        cookies_file=cookies_file,
        cookies_from_browser=cookies_from_browser,
    )

    if not sources:
        raise ProcessingFailedError("No available cookie source configuration")

    last_error: Exception | None = None
    auth_required_seen = False

    for source in sources:

        def _extract_with_source() -> dict[str, Any]:
            with create_ydl(
                platform=platform,
                cookie_source=source,
                for_subtitles=False,
            ) as ydl:
                return ydl.extract_info(url, download=False)

        try:
            info_dict = await loop.run_in_executor(None, _extract_with_source)
            if info_dict:
                logger.debug(f"Video info extracted with source: {source['name']}")
                return info_dict
        except Exception as exc:
            last_error = exc
            logger.warning(f"Video info extraction failed with {source['name']}: {exc}")

            if is_unsupported_url_error(exc):
                raise UnsupportedInputError(f"Unsupported URL: {url}") from exc

            if is_auth_required_error(exc):
                auth_required_seen = True
                if (
                    platform in AUTH_REQUIRED_PLATFORMS
                    and source["name"] == "no-cookie"
                ):
                    break
                continue

            if is_keyring_error(exc):
                continue

            continue

    if auth_required_seen:
        raise ProcessingFailedError(
            "Authenticated cookies are required for this URL. "
            "Provide --cookies (Netscape file) or --cookies-from-browser."
        ) from last_error

    if last_error is not None:
        raise ProcessingFailedError(
            f"Failed to extract video information: {last_error}"
        ) from last_error

    raise ProcessingFailedError("Failed to extract video information")
