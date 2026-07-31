"""Media metadata extraction for URL transcript processing."""

import asyncio
from functools import partial
from typing import Any

from logly import logger

from aimd.core.errors import ProcessingFailedError, UnsupportedInputError

from .cookies import (
    AUTH_REQUIRED_PLATFORMS,
    build_cookie_sources,
    is_auth_required_error,
    is_cookie_source_unavailable_error,
    is_keyring_error,
    is_unsupported_url_error,
)
from .ydl import create_info_ydl


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
    cookie_source_issue_seen = False

    def _extract_with_source(cookie_source: dict[str, Any]) -> dict[str, Any]:
        with create_info_ydl(
            platform=platform,
            cookie_source=cookie_source,
        ) as ydl:
            return ydl.extract_info(url, download=False)

    for source in sources:
        try:
            info_dict = await loop.run_in_executor(
                None, partial(_extract_with_source, source)
            )
            if info_dict:
                info_dict["_aimd_cookie_source"] = source
                logger.debug(f"Video info extracted with source: {source['name']}")
                return info_dict
        except Exception as exc:
            last_error = exc
            logger.debug(f"Video info extraction failed with {source['name']}: {exc}")

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

            if is_keyring_error(exc) or is_cookie_source_unavailable_error(exc):
                cookie_source_issue_seen = True
                continue

            continue

    if platform in AUTH_REQUIRED_PLATFORMS and (
        auth_required_seen or cookie_source_issue_seen
    ):
        raise ProcessingFailedError(
            "Authenticated cookies are required for this URL. "
            "Provide --cookies (Netscape file) or --cookies-from-browser."
        ) from last_error

    if last_error is not None:
        raise ProcessingFailedError(
            f"Failed to extract video information: {last_error}"
        ) from last_error

    raise ProcessingFailedError("Failed to extract video information")
