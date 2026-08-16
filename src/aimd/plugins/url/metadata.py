"""Media metadata extraction for URL transcript processing."""

import asyncio
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

_MAX_DESCRIPTION_CHARS = 800
_MAX_TAG_COUNT = 20
_MAX_CHAPTER_COUNT = 30
_MAX_CONTEXT_CHARS = 2000


def build_metadata_context(info_dict: dict[str, Any]) -> str | None:
    """Build an ASR biasing context from extracted URL metadata.

    The returned free-form text is injected into the ASR model's system prompt
    so proper nouns, names, and domain terminology mentioned in the page
    metadata are recognized more accurately. Returns None when no useful
    metadata is available.
    """
    parts: list[str] = []

    title = info_dict.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(f"Title: {title.strip()}")

    uploader = info_dict.get("uploader") or info_dict.get("channel")
    if isinstance(uploader, str) and uploader.strip():
        parts.append(f"Author: {uploader.strip()}")

    description = info_dict.get("description")
    if isinstance(description, str) and description.strip():
        parts.append(f"Description: {description.strip()[:_MAX_DESCRIPTION_CHARS]}")

    tags = [
        tag.strip()
        for tag in info_dict.get("tags") or []
        if isinstance(tag, str) and tag.strip()
    ]
    if tags:
        parts.append(f"Tags: {', '.join(tags[:_MAX_TAG_COUNT])}")

    chapters = [
        chapter["title"].strip()
        for chapter in info_dict.get("chapters") or []
        if isinstance(chapter, dict)
        and isinstance(chapter.get("title"), str)
        and chapter["title"].strip()
    ]
    if chapters:
        parts.append(f"Chapters: {'; '.join(chapters[:_MAX_CHAPTER_COUNT])}")

    if not parts:
        return None

    body = "\n".join(parts)
    context = (
        "The following background information describes the audio being "
        "transcribed. Use it to recognize proper nouns, names, and domain "
        "terminology accurately.\n"
        f"{body}"
    )
    return context[:_MAX_CONTEXT_CHARS]


async def extract_video_info(
    *,
    url: str,
    platform: str,
    cookies_file: str | None,
    cookies_from_browser: str | None,
) -> dict[str, Any]:
    """Extract video information using yt-dlp with cookie-source fallback chain."""
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
            info_dict = await asyncio.to_thread(_extract_with_source, source)
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
