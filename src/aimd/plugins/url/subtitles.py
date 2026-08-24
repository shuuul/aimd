"""Subtitle extraction helpers."""

from __future__ import annotations

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

from logly import logger
from yt_dlp.networking.exceptions import HTTPError, TransportError

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
SubtitleEntry = Mapping[str, Any]

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_CONTENT_LANGUAGE_SAMPLE_CHARS = 4000


def _is_original_language(lang: str) -> bool:
    """Return True when a subtitle language code represents the original language."""
    normalized = lang.lower()
    return (
        normalized == "orig" or normalized.endswith("-orig") or "original" in normalized
    )


def _normalize_lang_code(lang: str) -> str:
    """Normalize a language code for family comparisons."""
    return lang.lower().replace("_", "-")


def _language_family(lang: str) -> ContentLanguage | None:
    """Map a language code to the zh/en family AIMD prioritizes."""
    normalized = _normalize_lang_code(lang)
    if (
        normalized in {"en", "english"}
        or normalized.startswith("en-")
        or normalized.startswith("ai-en")
    ):
        return "en"
    if (
        normalized in {"zh", "chinese"}
        or normalized.startswith("zh-")
        or normalized.startswith("ai-zh")
    ):
        return "zh"
    return None


def normalize_metadata_language(language: str | None) -> ContentLanguage | None:
    """Normalize yt-dlp metadata language values to zh/en when possible.

    Args:
        language: Raw metadata language such as ``en``, ``en-US``, or ``zh-CN``.

    Returns:
        ``"zh"``, ``"en"``, or None when the value is missing/unsupported.
    """
    if not language or not language.strip():
        return None
    return _language_family(language.strip())


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
    metadata_language: str | None = None,
) -> str | None:
    """Resolve an explicit language preference or infer one from metadata.

    Preference order when ``language`` is unset:
    1. yt-dlp metadata language (spoken/original language)
    2. title/description script heuristics

    Args:
        language: Explicit user/API language preference, if any.
        title: Video title used when language is unspecified.
        description: Video description used when language is unspecified.
        metadata_language: yt-dlp ``language`` field, if present.

    Returns:
        Explicit language, inferred ``zh``/``en``, or None.
    """
    if language:
        return language
    metadata = normalize_metadata_language(metadata_language)
    if metadata is not None:
        return metadata
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


def _ordered_family_matches(
    available: Sequence[str],
    *,
    family: ContentLanguage,
    originals_first: bool,
) -> list[str]:
    """Return available codes in a language family, optionally originals first."""
    family_langs = [lang for lang in available if _language_family(lang) == family]
    if not originals_first:
        return family_langs

    originals = [lang for lang in family_langs if _is_original_language(lang)]
    non_originals = [lang for lang in family_langs if not _is_original_language(lang)]
    # Keep curated priority for known codes, then any other family matches.
    curated = (
        ENGLISH_SUBTITLE_LANGUAGES if family == "en" else CHINESE_SUBTITLE_LANGUAGES
    )
    curated_originals = [lang for lang in curated if lang in originals]
    curated_non_originals = [lang for lang in curated if lang in non_originals]
    other_originals = [lang for lang in originals if lang not in curated_originals]
    other_non_originals = [
        lang for lang in non_originals if lang not in curated_non_originals
    ]
    return _dedupe_preserve_order(
        curated_originals
        + other_originals
        + curated_non_originals
        + other_non_originals
    )


def get_preferred_languages(
    language: str | None,
    available_languages: Iterable[str] | None = None,
) -> list[str]:
    """Get preferred subtitle languages based on language code.

    Default (``language`` unset/``orig``) prioritizes ``*-orig`` tracks first, then
    English, Chinese, and remaining languages.
    """
    available = _dedupe_preserve_order(available_languages or [])
    original_languages = [lang for lang in available if _is_original_language(lang)]
    english_available = _ordered_family_matches(
        available, family="en", originals_first=True
    )
    chinese_available = _ordered_family_matches(
        available, family="zh", originals_first=True
    )
    remaining_available = [
        lang
        for lang in available
        if lang not in original_languages
        and lang not in english_available
        and lang not in chinese_available
    ]

    if available:
        default_priority = _dedupe_preserve_order(
            original_languages
            + english_available
            + chinese_available
            + remaining_available
        )
    else:
        default_priority = _dedupe_preserve_order(
            ENGLISH_SUBTITLE_LANGUAGES + CHINESE_SUBTITLE_LANGUAGES
        )

    if language:
        lang = language.lower()
        if lang in ("orig", "original"):
            return default_priority
        if lang in ("zh", "chinese", "zh-hans", "zh-hant"):
            if available:
                return _dedupe_preserve_order(
                    chinese_available
                    + original_languages
                    + english_available
                    + remaining_available
                )
            return _dedupe_preserve_order(
                CHINESE_SUBTITLE_LANGUAGES + ENGLISH_SUBTITLE_LANGUAGES
            )
        if lang in ("en", "english"):
            if available:
                # Keep Chinese behind other leftovers so English auto-captions win
                # even when YouTube lists zh-Hans before en/en-orig.
                non_chinese_remaining = [
                    item
                    for item in remaining_available
                    if _language_family(item) != "zh"
                ]
                return _dedupe_preserve_order(
                    english_available
                    + original_languages
                    + non_chinese_remaining
                    + chinese_available
                )
            return _dedupe_preserve_order(
                ENGLISH_SUBTITLE_LANGUAGES + CHINESE_SUBTITLE_LANGUAGES
            )

    return default_priority


def _subtitle_translation_target(entry: SubtitleEntry) -> str | None:
    """Return the timedtext ``tlang`` target when a track is a translation."""
    url = entry.get("url")
    if not isinstance(url, str) or not url:
        return None
    query = parse_qs(urlparse(url).query)
    targets = query.get("tlang")
    if not targets:
        return None
    return targets[0]


def _entries_match_resolved_language(
    lang_code: str,
    entries: Sequence[SubtitleEntry],
    resolved_language: str | None,
    *,
    accept_original_tracks: bool,
) -> bool:
    """Return whether subtitle entries are acceptable for the resolved language.

    Translation tracks (YouTube ``tlang=``) must match the resolved family. Source
    tracks without ``tlang`` are accepted when their language code matches.
    When ``accept_original_tracks`` is true, ``*-orig`` tracks are always allowed so
    default selection can prefer the spoken-language original captions.
    """
    if lang_code in FORBIDDEN_SUBTITLE_LANGUAGES:
        return False
    if accept_original_tracks and _is_original_language(lang_code):
        return True

    family = (
        _language_family(resolved_language) if resolved_language is not None else None
    )
    if family is None:
        return True

    code_family = _language_family(lang_code)
    translation_targets = {
        target
        for entry in entries
        if (target := _subtitle_translation_target(entry)) is not None
    }

    if translation_targets:
        return any(_language_family(target) == family for target in translation_targets)

    return code_family == family


def _iter_subtitle_candidates(
    *,
    preferred_languages: Sequence[str],
    subtitles: Mapping[str, Sequence[SubtitleEntry]],
    auto_subtitles: Mapping[str, Sequence[SubtitleEntry]],
    resolved_language: str | None,
    accept_original_tracks: bool,
) -> list[tuple[str, Sequence[SubtitleEntry], bool]]:
    """Build ordered subtitle candidates: manual first, then auto, per preference."""
    candidates: list[tuple[str, Sequence[SubtitleEntry], bool]] = []
    seen: set[str] = set()

    for lang in preferred_languages:
        if lang in seen or lang in FORBIDDEN_SUBTITLE_LANGUAGES:
            continue
        if lang in subtitles:
            entries = subtitles[lang]
            if _entries_match_resolved_language(
                lang,
                entries,
                resolved_language,
                accept_original_tracks=accept_original_tracks,
            ):
                candidates.append((lang, entries, True))
                seen.add(lang)

    for lang in preferred_languages:
        if lang in seen or lang in FORBIDDEN_SUBTITLE_LANGUAGES:
            continue
        if lang in auto_subtitles:
            entries = auto_subtitles[lang]
            if _entries_match_resolved_language(
                lang,
                entries,
                resolved_language,
                accept_original_tracks=accept_original_tracks,
            ):
                candidates.append((lang, entries, False))
                seen.add(lang)

    if candidates:
        return candidates

    # Last resort: ignore resolved-language filtering but still skip forbidden codes.
    for source, is_manual in ((subtitles, True), (auto_subtitles, False)):
        for lang, entries in source.items():
            if lang in seen or lang in FORBIDDEN_SUBTITLE_LANGUAGES:
                continue
            candidates.append((lang, entries, is_manual))
            seen.add(lang)
    return candidates


# YouTube auto-captions advertise json3/srv*/ttml/srt/vtt. Native json3 and
# srv1 are served directly; ``fmt=srt``/srv2/srv3 are converted on demand and
# often stall or 502 on long ASR tracks. json3/srv1 are normalized to SRT
# after download so stripping and ``raw_transcript`` keep working.
_PREFERRED_SUBTITLE_FORMATS = ("json3", "ttml", "vtt", "srv1", "srt")
_SUBTITLE_DOWNLOAD_ATTEMPTS = 3
_SUBTITLE_RETRY_BACKOFF_SECONDS = (0.5, 1.5)
_RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
_DEFAULT_CUE_DURATION_MS = 1000


def _iter_subtitle_urls(entries: Sequence[SubtitleEntry]) -> list[str]:
    """Return subtitle URLs in download-preference order."""
    urls: list[str] = []
    seen: set[str] = set()
    for fmt in _PREFERRED_SUBTITLE_FORMATS:
        for entry in entries:
            if entry.get("ext") != fmt:
                continue
            url = entry.get("url")
            if isinstance(url, str) and url and url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def _pick_subtitle_url(entries: Sequence[SubtitleEntry]) -> str | None:
    """Pick a preferred subtitle URL from yt-dlp subtitle entries."""
    urls = _iter_subtitle_urls(entries)
    return urls[0] if urls else None


def _ms_to_srt_timestamp(ms: int) -> str:
    """Format milliseconds as an SRT timestamp (HH:MM:SS,mmm)."""
    if ms < 0:
        ms = 0
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _cues_to_srt(cues: list[tuple[int, int, str]]) -> str | None:
    """Render (start_ms, end_ms, text) cues as SRT, or None when empty."""
    if not cues:
        return None
    blocks: list[str] = []
    for index, (start_ms, end_ms, text) in enumerate(cues, start=1):
        end = end_ms if end_ms > start_ms else start_ms + _DEFAULT_CUE_DURATION_MS
        blocks.append(
            f"{index}\n{_ms_to_srt_timestamp(start_ms)} --> {_ms_to_srt_timestamp(end)}\n{text}"
        )
    return "\n\n".join(blocks) + "\n"


def _json3_to_srt(text: str) -> str | None:
    """Convert YouTube json3 timedtext into SRT."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    events = data.get("events")
    if not isinstance(events, list):
        return None

    cues: list[tuple[int, int, str]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        segs = event.get("segs")
        if not isinstance(segs, list):
            continue
        parts: list[str] = []
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            utf8 = seg.get("utf8")
            if isinstance(utf8, str) and utf8 not in {"\n", "\r\n"}:
                parts.append(utf8)
        body = "".join(parts).replace("\n", " ").strip()
        if not body:
            continue
        start = event.get("tStartMs")
        if not isinstance(start, int | float):
            continue
        start_ms = int(start)
        duration = event.get("dDurationMs")
        duration_ms = (
            int(duration)
            if isinstance(duration, int | float) and duration > 0
            else _DEFAULT_CUE_DURATION_MS
        )
        cues.append((start_ms, start_ms + duration_ms, body))
    return _cues_to_srt(cues)


def _srv1_to_srt(text: str) -> str | None:
    """Convert YouTube srv1 ``<transcript>`` timedtext into SRT."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None

    cues: list[tuple[int, int, str]] = []
    for element in root.iter("text"):
        body = "".join(element.itertext()).replace("\n", " ").strip()
        if not body:
            continue
        try:
            start_ms = int(float(element.attrib.get("start", "0")) * 1000)
            duration_s = float(element.attrib.get("dur", "0") or 0)
        except ValueError:
            continue
        duration_ms = (
            int(duration_s * 1000) if duration_s > 0 else _DEFAULT_CUE_DURATION_MS
        )
        cues.append((start_ms, start_ms + duration_ms, body))
    return _cues_to_srt(cues)


def _srv3_to_srt(text: str) -> str | None:
    """Convert YouTube srv3 ``<timedtext>`` XML into SRT."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None

    cues: list[tuple[int, int, str]] = []
    for element in root.iter("p"):
        body = "".join(element.itertext()).replace("\n", " ").strip()
        if not body:
            continue
        try:
            start_ms = int(float(element.attrib.get("t", "0")))
            duration_ms = int(float(element.attrib.get("d", "0") or 0))
        except ValueError:
            continue
        if duration_ms <= 0:
            duration_ms = _DEFAULT_CUE_DURATION_MS
        cues.append((start_ms, start_ms + duration_ms, body))
    return _cues_to_srt(cues)


def _normalize_subtitle_payload(text: str) -> str | None:
    """Normalize native YouTube captions to SRT; leave SRT/VTT/TTML unchanged.

    Returns None when the payload looks like a native caption format but yields
    no cues, so the caller can try the next advertised format.
    """
    stripped = text.strip()
    if not stripped:
        return None

    head = stripped[:2000]
    if stripped.startswith("{") and ('"events"' in head or '"wireMagic"' in head):
        return _json3_to_srt(stripped)
    if "<transcript" in head:
        return _srv1_to_srt(stripped)
    if "<timedtext" in head:
        return _srv3_to_srt(stripped)
    return stripped


def _is_retryable_subtitle_error(exc: BaseException) -> bool:
    """Return True for transient timedtext timeouts and gateway failures."""
    if isinstance(exc, TimeoutError | TransportError):
        return True
    if isinstance(exc, HTTPError):
        return exc.status in _RETRYABLE_HTTP_STATUS
    return False


async def download_subtitle(
    url: str,
    platform: str,
    cookie_source: dict[str, Any] | None = None,
) -> str | None:
    """Download subtitle content from URL using yt-dlp and the selected cookies.

    YouTube's on-demand SRT conversion for auto-captions is flaky (timeouts and
    502s on long videos). Retry the same URL a few times before giving up so a
    later format/language fallback is not forced by a single blip.
    """

    def _download() -> str:
        with create_subtitle_ydl(
            platform=platform,
            cookie_source=cookie_source or {"use_cookies": False},
        ) as ydl:
            response = ydl.urlopen(url)
            return response.read().decode("utf-8")

    for attempt in range(1, _SUBTITLE_DOWNLOAD_ATTEMPTS + 1):
        try:
            return await asyncio.to_thread(_download)
        except Exception as exc:
            if attempt < _SUBTITLE_DOWNLOAD_ATTEMPTS and _is_retryable_subtitle_error(
                exc
            ):
                delay = _SUBTITLE_RETRY_BACKOFF_SECONDS[
                    min(attempt - 1, len(_SUBTITLE_RETRY_BACKOFF_SECONDS) - 1)
                ]
                logger.warning(
                    "Subtitle download failed "
                    f"(attempt {attempt}/{_SUBTITLE_DOWNLOAD_ATTEMPTS}): {exc}; "
                    f"retrying in {delay}s"
                )
                if delay:
                    await asyncio.sleep(delay)
                continue
            logger.error(f"Failed to download subtitle from {url}: {exc}")
            return None
    return None


async def extract_subtitles(
    info_dict: dict[str, object],
    platform: str,
    language: str | None,
) -> str | None:
    """Extract subtitles from video metadata with platform-specific handling.

    Args:
        info_dict: yt-dlp metadata including subtitle maps.
        platform: Detected platform key such as ``youtube``.
        language: Explicit user language preference. When unset, ``*-orig`` tracks
            are preferred by default; metadata/title/description still guide
            translation-track filtering and logging.
    """
    raw_subtitles = info_dict.get("subtitles", {})
    raw_auto_subtitles = info_dict.get("automatic_captions", {})
    subtitles: dict[str, Sequence[SubtitleEntry]] = (
        raw_subtitles if isinstance(raw_subtitles, dict) else {}
    )
    auto_subtitles: dict[str, Sequence[SubtitleEntry]] = (
        raw_auto_subtitles if isinstance(raw_auto_subtitles, dict) else {}
    )
    cookie_source = info_dict.get("_aimd_cookie_source")
    if not isinstance(cookie_source, dict):
        cookie_source = None

    if not subtitles and not auto_subtitles:
        logger.info("No subtitles available")
        return None

    title = info_dict.get("title")
    description = info_dict.get("description")
    metadata_language = info_dict.get("language")
    content_language = resolve_subtitle_language(
        language,
        title=title if isinstance(title, str) else None,
        description=description if isinstance(description, str) else None,
        metadata_language=(
            metadata_language if isinstance(metadata_language, str) else None
        ),
    )
    preference_language = language
    accept_original_tracks = (
        preference_language is None
        or preference_language.lower()
        in (
            "orig",
            "original",
        )
    )
    if language is None and content_language is not None:
        logger.info(
            "Inferred content language from metadata/title/description: "
            f"{content_language}"
        )
    if accept_original_tracks:
        logger.info("Using default subtitle priority: *-orig first")

    available_languages = list(subtitles) + list(auto_subtitles)
    preferred_languages = get_preferred_languages(
        preference_language, available_languages
    )
    candidates = _iter_subtitle_candidates(
        preferred_languages=preferred_languages,
        subtitles=subtitles,
        auto_subtitles=auto_subtitles,
        resolved_language=content_language,
        accept_original_tracks=accept_original_tracks,
    )
    if not candidates:
        logger.info("No suitable subtitles found")
        return None

    try:
        for selected_lang, selected_sub, is_manual in candidates:
            logger.info(
                "Trying subtitle language: "
                f"{selected_lang} ({'manual' if is_manual else 'auto'})"
            )

            if platform == "bilibili":
                if not selected_sub:
                    continue
                data = selected_sub[0].get("data")
                if isinstance(data, str) and data.strip():
                    logger.info(
                        f"Selected subtitle language: {selected_lang} "
                        f"({'manual' if is_manual else 'auto'})"
                    )
                    return data
                continue

            if platform not in ("youtube", "unknown"):
                logger.error(f"Unsupported platform for subtitles: {platform}")
                return None

            sub_urls = _iter_subtitle_urls(selected_sub)
            if not sub_urls:
                logger.warning(
                    "No suitable subtitle format found for "
                    f"{selected_lang} (tried json3, TTML, VTT, srv1, SRT)"
                )
                continue

            for sub_url in sub_urls:
                subtitle_content = await download_subtitle(
                    sub_url,
                    platform,
                    cookie_source=cookie_source,
                )
                if not subtitle_content or not subtitle_content.strip():
                    continue
                normalized = _normalize_subtitle_payload(subtitle_content)
                if normalized and normalized.strip():
                    logger.info(
                        f"Selected subtitle language: {selected_lang} "
                        f"({'manual' if is_manual else 'auto'})"
                    )
                    logger.info(
                        f"Successfully extracted subtitles using {platform} format"
                    )
                    return normalized

        logger.info("No suitable subtitles found")
        return None
    except Exception as exc:
        logger.error(f"Failed to extract subtitles using {platform} format: {exc}")
        return None
