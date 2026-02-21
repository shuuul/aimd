"""Subtitle extraction helpers."""

import asyncio

from logly import logger

from ...const import (
    CHINESE_SUBTITLE_LANGUAGES,
    ENGLISH_SUBTITLE_LANGUAGES,
    FORBIDDEN_SUBTITLE_LANGUAGES,
)
from .ydl_client import create_ydl


def get_preferred_languages(language: str | None) -> list[str]:
    """Get preferred subtitle languages based on language code."""
    english_languages = ENGLISH_SUBTITLE_LANGUAGES
    chinese_languages = CHINESE_SUBTITLE_LANGUAGES

    if language:
        lang = language.lower()
        if lang in ("zh", "chinese", "zh-hans", "zh-hant"):
            return chinese_languages + english_languages
        if lang in ("en", "english"):
            return english_languages + chinese_languages

    return chinese_languages + english_languages


async def download_subtitle(url: str, platform: str) -> str | None:
    """Download subtitle content from URL using yt-dlp without cookies."""

    def _download() -> str:
        with create_ydl(
            platform=platform,
            cookie_source={"use_cookies": False},
            for_subtitles=True,
        ) as ydl:
            response = ydl.urlopen(url)
            return response.read().decode("utf-8")

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _download)
    except Exception as exc:
        logger.error(f"Failed to download subtitle from {url}: {exc}")
        return None


async def extract_subtitles(
    info_dict: dict[str, object],
    platform: str,
    language: str | None,
) -> str | None:
    """Extract subtitles from video metadata with platform-specific handling."""
    subtitles = info_dict.get("subtitles", {})
    auto_subtitles = info_dict.get("automatic_captions", {})

    if not subtitles and not auto_subtitles:
        logger.info("No subtitles available")
        return None

    preferred_languages = get_preferred_languages(language)
    selected_lang = None
    selected_sub = None
    is_manual = False

    for lang in preferred_languages:
        if lang in subtitles:
            selected_lang = lang
            selected_sub = subtitles[lang]
            is_manual = True
            break

    if not selected_sub:
        for lang in preferred_languages:
            if lang in auto_subtitles:
                selected_lang = lang
                selected_sub = auto_subtitles[lang]
                is_manual = False
                break

    if not selected_sub:
        for lang in subtitles:
            if lang not in FORBIDDEN_SUBTITLE_LANGUAGES:
                selected_lang = lang
                selected_sub = subtitles[selected_lang]
                is_manual = True
                break

        if not selected_sub:
            for lang in auto_subtitles:
                if lang not in FORBIDDEN_SUBTITLE_LANGUAGES:
                    selected_lang = lang
                    selected_sub = auto_subtitles[selected_lang]
                    is_manual = False
                    break

    if not selected_sub:
        logger.info("No suitable subtitles found")
        return None

    logger.info(
        f"Selected subtitle language: {selected_lang} ({'manual' if is_manual else 'auto'})"
    )

    try:
        if platform == "bilibili":
            return selected_sub[0]["data"]

        if platform in ("youtube", "unknown"):
            sub_url = None
            preferred_formats = ["srt", "vtt", "ttml"]
            for fmt in preferred_formats:
                for sub in selected_sub:
                    if sub.get("ext") == fmt:
                        sub_url = sub["url"]
                        break
                if sub_url:
                    break

            if not sub_url:
                logger.warning(
                    "No suitable subtitle format found (tried SRT, VTT, TTML)"
                )
                return None

            subtitle_content = await download_subtitle(sub_url, platform)
            if subtitle_content:
                logger.info(f"Successfully extracted subtitles using {platform} format")
            return subtitle_content

        logger.error(f"Unsupported platform for subtitles: {platform}")
        return None
    except Exception as exc:
        logger.error(f"Failed to extract subtitles using {platform} format: {exc}")
        return None
