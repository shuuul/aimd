"""Subtitle extraction helpers."""

import asyncio
import re
from collections.abc import Iterable
from typing import Any, Literal

from logly import logger

from .ydl import create_subtitle_ydl

ENGLISH_SUBTITLE_LANGUAGES = [
    "en-orig",
    "en",
    "en-US",
    "en-GB",
    "ai-en",
]

CHINESE_SUBTITLE_LANGUAGES = [
    "zh-orig",
    "zh-Hans",
    "zh-Hant",
    "zh-CN",
    "zh",
    "ai-zh",
    "zh-TW",
    "zh-HK",
]

FORBIDDEN_SUBTITLE_LANGUAGES = ["danmaku"]

ContentLanguage = Literal["zh", "en"]

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_CONTENT_LANGUAGE_SAMPLE_CHARS = 4000


def _is_original_language(lang: str) -> bool:
    """Return True when a subtitle language code represents the original language."""
    normalized = lang.lower()
    return (
        normalized == "orig" or normalized.endswith("-orig") or "original" in normalized
    )


def detect_content_language(*texts: str | None) -> ContentLanguage | None:
    """Infer zh/en from title/description via CJK vs Latin script signals.

    Chinese titles often include Latin brand tokens, so any meaningful CJK
    presence prefers Chinese. Returns None when the signal is inconclusive.

    Args:
        *texts: Metadata fragments such as title and description.

    Returns:
        ``"zh"``, ``"en"``, or None when language cannot be inferred.
    """
    combined = " ".join(text.strip() for text in texts if text and text.strip())
    if not combined:
        return None

    sample = combined[:_CONTENT_LANGUAGE_SAMPLE_CHARS]
    cjk_count = len(_CJK_RE.findall(sample))
    latin_count = len(_LATIN_RE.findall(sample))

    if cjk_count >= 2:
        return "zh"
    if cjk_count == 1 and latin_count < 20:
        return "zh"
    if latin_count >= 3:
        return "en"
    return None


def resolve_subtitle_language(
    language: str | None,
    *,
    title: str | None = None,
    description: str | None = None,
) -> str | None:
    """Resolve an explicit language preference or infer one from metadata.

    Args:
        language: Explicit user/API language preference, if any.
        title: Video title used when language is unspecified.
        description: Video description used when language is unspecified.

    Returns:
        Explicit language, inferred ``zh``/``en``, or None.
    """
    if language:
        return language
    return detect_content_language(title, description)


def _dedupe_preserve_order(languages: Iterable[str]) -> list[str]:
    """Return languages in their first-seen order without duplicates."""
    seen: set[str] = set()
    ordered: list[str] = []
    for lang in languages:
        if lang not in seen:
            seen.add(lang)
            ordered.append(lang)
    return ordered


def get_preferred_languages(
    language: str | None,
    available_languages: Iterable[str] | None = None,
) -> list[str]:
    """Get preferred subtitle languages based on language code."""
    english_languages = ENGLISH_SUBTITLE_LANGUAGES
    chinese_languages = CHINESE_SUBTITLE_LANGUAGES
    available = _dedupe_preserve_order(available_languages or [])
    original_languages = [lang for lang in available if _is_original_language(lang)]
    english_available = [
        lang
        for lang in available
        if lang in english_languages and not _is_original_language(lang)
    ]
    chinese_available = [
        lang
        for lang in available
        if lang in chinese_languages and not _is_original_language(lang)
    ]
    remaining_available = [
        lang
        for lang in available
        if lang not in original_languages
        and lang not in english_available
        and lang not in chinese_available
    ]

    if available:
        default_priority = (
            original_languages
            + english_available
            + chinese_available
            + remaining_available
        )
    else:
        default_priority = _dedupe_preserve_order(english_languages + chinese_languages)

    if language:
        lang = language.lower()
        if lang in ("orig", "original"):
            return default_priority
        if lang in ("zh", "chinese", "zh-hans", "zh-hant"):
            if available:
                return (
                    chinese_available
                    + original_languages
                    + english_available
                    + remaining_available
                )
            return _dedupe_preserve_order(chinese_languages + english_languages)
        if lang in ("en", "english"):
            if available:
                return (
                    english_available
                    + original_languages
                    + chinese_available
                    + remaining_available
                )
            return _dedupe_preserve_order(english_languages + chinese_languages)

    return default_priority


async def download_subtitle(
    url: str,
    platform: str,
    cookie_source: dict[str, Any] | None = None,
) -> str | None:
    """Download subtitle content from URL using yt-dlp and the selected cookies."""

    def _download() -> str:
        with create_subtitle_ydl(
            platform=platform,
            cookie_source=cookie_source or {"use_cookies": False},
        ) as ydl:
            response = ydl.urlopen(url)
            return response.read().decode("utf-8")

    try:
        return await asyncio.to_thread(_download)
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
    cookie_source = info_dict.get("_aimd_cookie_source")
    if not isinstance(cookie_source, dict):
        cookie_source = None

    if not subtitles and not auto_subtitles:
        logger.info("No subtitles available")
        return None

    title = info_dict.get("title")
    description = info_dict.get("description")
    resolved_language = resolve_subtitle_language(
        language,
        title=title if isinstance(title, str) else None,
        description=description if isinstance(description, str) else None,
    )
    if language is None and resolved_language is not None:
        logger.info(
            f"Inferred content language from title/description: {resolved_language}"
        )

    available_languages = list(subtitles) + list(auto_subtitles)
    preferred_languages = get_preferred_languages(
        resolved_language, available_languages
    )
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

            subtitle_content = await download_subtitle(
                sub_url,
                platform,
                cookie_source=cookie_source,
            )
            if subtitle_content:
                logger.info(f"Successfully extracted subtitles using {platform} format")
            return subtitle_content

        logger.error(f"Unsupported platform for subtitles: {platform}")
        return None
    except Exception as exc:
        logger.error(f"Failed to extract subtitles using {platform} format: {exc}")
        return None
