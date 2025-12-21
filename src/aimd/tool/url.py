"""URL processing tools for extracting content from video URLs using yt-dlp."""

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import yt_dlp
from logly import logger

from ..types import TextContext
from ..utils import is_valid_url, is_supported_url
from ..const import (
    ENGLISH_SUBTITLE_LANGUAGES,
    CHINESE_SUBTITLE_LANGUAGES,
    FORBIDDEN_SUBTITLE_LANGUAGES,
)
from .audio import get_text_from_audio

# Module-level cached YoutubeDL instance with cookies for reuse across all extractions
_cached_ydl_instance = None


def _get_ydl_instance(use_cookies: bool = True, platform: str = "unknown"):
    """Get or create a YoutubeDL instance optimized for info extraction and subtitle downloads.

    Args:
        use_cookies: Whether to use browser cookies (required for Bilibili)
        platform: Platform name for platform-specific optimizations

    Returns:
        YoutubeDL instance
    """
    global _cached_ydl_instance

    # Create a new instance if cookies setting changed or no instance exists
    if (
        _cached_ydl_instance is None
        or getattr(_cached_ydl_instance, "_use_cookies", True) != use_cookies
    ):
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "listsubtitles": True,
            # Optimize for info extraction and subtitle downloads
            "writesubtitles": True,
            "writeautomaticsub": True,
            "skip_download": True,  # Don't download video files by default
        }

        # Add browser impersonation for better compatibility (recommended by yt-dlp docs)
        # Only enable if impersonation dependencies are available
        if platform == "youtube":
            try:
                # Test if impersonation is available by creating a test instance
                test_opts = {"quiet": True, "impersonate": "chrome"}
                test_ydl = yt_dlp.YoutubeDL(test_opts)
                test_ydl.close()  # Clean up test instance
                ydl_opts["impersonate"] = "chrome"
                logger.debug("Browser impersonation enabled for YouTube")
            except Exception as e:
                logger.debug(
                    f"Browser impersonation not available, continuing without: {e}"
                )
                # Continue without impersonation

        # Add cookies for platforms that need them (especially Bilibili)
        if use_cookies:
            try:
                ydl_opts["cookiesfrombrowser"] = ("chrome", "default")
            except Exception as e:
                logger.warning(f"Failed to load cookies from browser: {e}")

        _cached_ydl_instance = yt_dlp.YoutubeDL(ydl_opts)
        _cached_ydl_instance._use_cookies = use_cookies  # Track cookies setting

    return _cached_ydl_instance


async def get_text_from_url(
    url: str,
    transcribe_engine: str = "auto",
    locale: str | None = None,
) -> TextContext:
    """Extract text content from video URLs using yt-dlp.

    First attempts to extract subtitles. If no subtitles are available,
    downloads audio and transcribes it.

    Args:
        url: Video URL from any supported platform
        transcribe_engine: Transcription engine for audio fallback
        locale: Language locale for audio transcription

    Returns:
        TextContext with title and content

    Raises:
        ValueError: If URL is invalid or unsupported
        RuntimeError: If content extraction fails
    """
    if not is_valid_url(url):
        raise ValueError(f"Invalid URL: {url}")

    if not is_supported_url(url):
        raise ValueError(f"Unsupported URL: {url}")

    logger.info(f"Processing video URL: {url}")
    platform = _detect_platform(url)

    try:
        # Extract video information
        info_dict = await _extract_video_info(url, platform)
        title = info_dict.get("title", "Unknown Title")

        # Try to get subtitles
        subtitle_content = await _extract_subtitles(info_dict, platform, locale)

        if subtitle_content and subtitle_content.strip():
            logger.info("Successfully extracted subtitles")
            content = _format_content(info_dict, subtitle_content)
            return TextContext(title=title, chunk_list=[content])

        # Fallback: extract content from audio
        logger.info("No subtitles available, extracting content from audio")
        audio_content = await _extract_content_from_audio(
            info_dict, url, transcribe_engine, locale
        )

        if audio_content and audio_content.strip():
            logger.info("Successfully extracted content from audio")
            content = _format_content(info_dict, audio_content)
            return TextContext(title=title, chunk_list=[content])

        # If all else fails, return basic video info
        logger.warning("Could not extract content, returning basic video info")
        content = _format_content(info_dict, None)
        return TextContext(title=title, chunk_list=[content])

    except Exception as e:
        logger.error(f"Failed to extract content from URL {url}: {e}")
        raise RuntimeError(f"URL content extraction failed: {e}") from e


async def _extract_video_info(url: str, platform: str) -> dict[str, Any]:
    """Extract video information using yt-dlp with graceful fallback.

    Args:
        url: Video URL
        platform: Detected platform name

    Returns:
        Video information dictionary

    Raises:
        RuntimeError: If extraction fails with all methods
    """

    def _extract_with_config(use_cookies: bool, platform: str):
        ydl = _get_ydl_instance(use_cookies=use_cookies, platform=platform)
        return ydl.extract_info(url, download=False)

    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()

    # First try with cookies (required for Bilibili, preferred for others)
    try:
        logger.debug("Attempting video info extraction with cookies")
        info_dict = await loop.run_in_executor(
            None, _extract_with_config, True, platform
        )
        if info_dict:
            return info_dict
    except Exception as e:
        logger.warning(f"Failed to extract video info with cookies: {e}")

        # For YouTube, try fallback without cookies if the error suggests authentication issues
        if platform == "youtube" and (
            "not available on this app" in str(e).lower()
            or "watch on the latest version" in str(e).lower()
        ):
            logger.info("YouTube access issue detected, trying without cookies")
            try:
                info_dict = await loop.run_in_executor(
                    None, _extract_with_config, False, platform
                )
                if info_dict:
                    logger.info("Successfully extracted video info without cookies")
                    return info_dict
            except Exception as fallback_error:
                logger.error(f"Fallback extraction also failed: {fallback_error}")
                raise RuntimeError(
                    f"Failed to extract video information: {fallback_error}"
                ) from fallback_error
        else:
            # For non-YouTube platforms or other errors, don't retry without cookies
            raise RuntimeError(f"Failed to extract video information: {e}") from e

    raise RuntimeError("Failed to extract video information")


async def _extract_subtitles(
    info_dict: dict[str, Any], platform: str, locale: str | None
) -> str | None:
    """Extract subtitles from video with platform-specific handling and language preferences.

    Args:
        info_dict: Video information from yt-dlp
        platform: Detected platform name
        locale: Language locale preference

    Returns:
        Subtitle text if available, None otherwise
    """
    # Check if subtitles are available in info_dict
    subtitles = info_dict.get("subtitles", {})
    auto_subtitles = info_dict.get("automatic_captions", {})

    if not subtitles and not auto_subtitles:
        logger.info("No subtitles available")
        return None

    # Get priority order for subtitle languages based on locale
    preferred_languages = _get_preferred_languages(locale)
    logger.debug(f"Subtitle language preference order: {preferred_languages[:5]}...")

    # Find best available subtitle - prioritize manual subtitles over auto subtitles
    selected_lang = None
    selected_sub = None
    is_manual = False

    # First try manual subtitles
    for lang in preferred_languages:
        if lang in subtitles:
            selected_lang = lang
            selected_sub = subtitles[lang]
            is_manual = True
            break

    # Then try auto subtitles if no manual ones found
    if not selected_sub:
        for lang in preferred_languages:
            if lang in auto_subtitles:
                selected_lang = lang
                selected_sub = auto_subtitles[lang]
                is_manual = False
                break

    # If no preferred language found, use first available (excluding forbidden)
    if not selected_sub:
        # Try manual subtitles first
        for lang in subtitles:
            if lang not in FORBIDDEN_SUBTITLE_LANGUAGES:
                selected_lang = lang
                selected_sub = subtitles[selected_lang]
                is_manual = True
                break

        # Then try auto subtitles
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

    # Extract subtitle content based on platform
    try:
        if platform == "bilibili":
            # Bilibili uses direct data in subtitle dictionary
            subtitle_content = selected_sub[0]["data"]
            logger.info(f"Successfully extracted subtitles using {platform} format")
            return subtitle_content
        elif platform in ("youtube", "unknown"):
            # YouTube and other platforms use URL-based subtitles
            sub_url = None

            # Try preferred formats in order: SRT, VTT, TTML
            preferred_formats = ["srt", "vtt", "ttml"]
            for fmt in preferred_formats:
                for sub in selected_sub:
                    if sub.get("ext") == fmt:
                        sub_url = sub["url"]
                        logger.info(f"Found {fmt.upper()} subtitle format")
                        break
                if sub_url:
                    break

            if sub_url:
                subtitle_content = await _download_subtitle(sub_url)
                logger.info(f"Successfully extracted subtitles using {platform} format")
                return subtitle_content
            else:
                logger.warning(
                    "No suitable subtitle format found (tried SRT, VTT, TTML)"
                )
                return None
        else:
            logger.error(f"We do not support platform: {platform} for subtitles")
            return None
    except Exception as e:
        logger.error(f"Failed to extract subtitles using {platform} format: {e}")
        return None


async def _download_subtitle(url: str) -> str | None:
    """Download subtitle content from URL using yt-dlp.

    Args:
        url: Direct URL to subtitle file

    Returns:
        Subtitle content as string, None if failed
    """
    try:

        def _download():
            ydl = _get_ydl_instance()
            # Use yt-dlp's internal URL handling to download subtitle content
            response = ydl.urlopen(url)
            content = response.read().decode("utf-8")
            return content

        # Run download in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, _download)

        logger.info(f"Successfully downloaded subtitle from {url}")
        return content

    except Exception as e:
        logger.error(f"Failed to download subtitle from {url}: {e}")
        return None


async def _extract_content_from_audio(
    info_dict: dict[str, Any],
    url: str,
    transcribe_engine: str,
    locale: str | None,
) -> str | None:
    """Extract content by downloading audio and transcribing it.

    Args:
        info_dict: Video information from yt-dlp
        url: Original video URL
        transcribe_engine: Transcription engine to use
        locale: Language locale for transcription

    Returns:
        Transcribed text if successful, None otherwise
    """
    logger.info("Attempting to extract content from audio")

    # Create temporary directory for audio file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # Download audio using yt-dlp
            audio_file_path = await _download_audio(info_dict, url, temp_path)

            if not audio_file_path or not audio_file_path.exists():
                logger.warning("Failed to download audio file")
                return None

            # Transcribe audio to text
            logger.info(f"Transcribing audio file: {audio_file_path}")
            text_context = await get_text_from_audio(
                audio_file_path,
                engine=transcribe_engine,
                locale=locale or "zh_CN",
            )

            if text_context.chunk_list and text_context.chunk_list[0]:
                content = text_context.chunk_list[0]
                if len(content) > 10:
                    logger.info("Successfully transcribed audio to text")
                    return content
                else:
                    logger.warning("Audio transcription returned empty text")
                    return None
            else:
                logger.warning("Audio transcription returned no content")
                return None

        except Exception as e:
            logger.error(f"Failed to extract content from audio: {e}")
            return None


async def _download_audio(
    info_dict: dict[str, Any], url: str, temp_path: Path
) -> Path | None:
    """Download audio from video using yt-dlp.

    Args:
        info_dict: Video information from yt-dlp
        url: Original video URL
        temp_path: Temporary directory path

    Returns:
        Path to downloaded audio file, None if failed
    """
    try:
        # Generate audio filename
        audio_filename = f"audio_{info_dict.get('id', 'unknown')}"

        def _download():
            # Create a separate YoutubeDL instance for audio download with specific options
            # Following deepwiki recommendations for audio-only downloads
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "format": "bestaudio/best",  # Select best audio format
                "outtmpl": str(temp_path / audio_filename),
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "m4a",
                        "preferredquality": "192",
                    }
                ],
            }

            # Detect platform for audio download
            platform = _detect_platform(url)

            # Add browser impersonation for better compatibility (if available)
            if platform == "youtube":
                try:
                    # Test if impersonation is available
                    test_opts = {"quiet": True, "impersonate": "chrome"}
                    test_ydl = yt_dlp.YoutubeDL(test_opts)
                    test_ydl.close()  # Clean up test instance
                    ydl_opts["impersonate"] = "chrome"
                    logger.debug(
                        "Browser impersonation enabled for YouTube audio download"
                    )
                except Exception as e:
                    logger.debug(
                        f"Browser impersonation not available for audio download: {e}"
                    )
                    # Continue without impersonation

            # Add cookies for all platforms (required for Bilibili)
            try:
                ydl_opts["cookiesfrombrowser"] = ("chrome", "default")
            except Exception as e:
                logger.warning(f"Failed to load cookies for audio download: {e}")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        # Run download in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _download)

        # Find the actual downloaded file (yt-dlp might change extension)
        audio_files = list(temp_path.glob("audio_*.*"))
        logger.info(f"Audio files: {audio_files}")

        if audio_files:
            return audio_files[0]
        else:
            logger.error("No audio file found after download")
            return None

    except Exception as e:
        logger.error(f"Failed to download audio: {e}")
        return None


def _format_content(info_dict: dict[str, Any], content: str | None) -> str:
    """Format video information with content.

    Args:
        info_dict: Video information dictionary from yt-dlp
        content: Subtitle or transcription content (None for basic info only)

    Returns:
        Formatted content string
    """
    title = info_dict.get("title", "Unknown Title")
    description = info_dict.get("description", "No description available")
    uploader = info_dict.get("uploader", info_dict.get("channel", "Unknown Uploader"))
    duration = info_dict.get("duration", 0)
    view_count = info_dict.get("view_count", 0)
    upload_date = info_dict.get("upload_date", "")
    webpage_url = info_dict.get("webpage_url", "")

    formatted_content = f"""# {title}

**Uploader:** {uploader}
**Duration:** {duration} seconds
**Upload Date:** {upload_date}
**View Count:** {view_count:,} views
**URL:** {webpage_url}

## Description

{description}

## Content

"""

    if content and content.strip():
        formatted_content += content
    else:
        formatted_content += "*No subtitles or transcription available for this video.*"

    formatted_content += """

---

*This content was extracted using yt-dlp via aimd*"""

    return formatted_content


def _detect_platform(url: str) -> str:
    """Detect the platform from URL.

    Args:
        url: Video URL

    Returns:
        Platform name (youtube, bilibili, xiaohongshu, or unknown)
    """
    url_lower = url.lower()
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    elif "bilibili.com" in url_lower:
        return "bilibili"
    elif "xiaohongshu.com" in url_lower or "xhslink.com" in url_lower:
        return "xiaohongshu"
    else:
        return "unknown"


def _get_preferred_languages(locale: str | None) -> list[str]:
    """Get preferred subtitle languages based on locale.

    Args:
        locale: Language locale preference

    Returns:
        List of language codes in priority order
    """
    # Use predefined language groups
    english_languages = ENGLISH_SUBTITLE_LANGUAGES
    chinese_languages = CHINESE_SUBTITLE_LANGUAGES

    # Determine locale preference
    if locale:
        locale_lower = locale.lower()

        # Check for Chinese locale variants
        if any(x in locale_lower for x in ["zh", "cn", "chinese", "hans", "hant"]):
            return chinese_languages + english_languages

        # Check for English locale variants
        elif any(x in locale_lower for x in ["en", "us", "gb", "english"]):
            return english_languages + chinese_languages

    # Default order (Chinese first)
    return chinese_languages + english_languages
