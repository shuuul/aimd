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

# Module-level cookies file path (set via get_text_from_url's cookies_file parameter)
_cookies_file: str | None = None


def _is_keyring_error(error: Exception) -> bool:
    """Check if an error is related to keyring/cookie decryption issues.

    These errors occur when yt-dlp tries to read browser cookies but the
    system keyring (e.g., GNOME Keyring, KWallet) is unavailable or locked.

    Args:
        error: The exception to check

    Returns:
        True if the error is keyring-related
    """
    error_str = str(error).lower()
    keyring_indicators = [
        "keyring",
        "secretservice",
        "secret service",
        "secret-service",
        "failed to decrypt",
        "could not decrypt",
        "dbus",
        "org.freedesktop.secret",
        "gnome-keyring",
        "kwallet",
    ]
    return any(indicator in error_str for indicator in keyring_indicators)


def _get_ydl_instance(use_cookies: bool = True, platform: str = "unknown"):
    """Get or create a YoutubeDL instance optimized for info extraction and subtitle downloads.

    Args:
        use_cookies: Whether to use browser cookies (required for Bilibili)
        platform: Platform name for platform-specific optimizations

    Returns:
        YoutubeDL instance
    """
    global _cached_ydl_instance

    # Create a new instance if cookies/platform setting changed or no instance exists
    if (
        _cached_ydl_instance is None
        or getattr(_cached_ydl_instance, "_use_cookies", True) != use_cookies
        or getattr(_cached_ydl_instance, "_cookies_file", None) != _cookies_file
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
            if _cookies_file:
                # Use explicit cookies file (avoids keyring issues entirely)
                ydl_opts["cookiefile"] = _cookies_file
                logger.debug(f"Using cookies file: {_cookies_file}")
            else:
                try:
                    ydl_opts["cookiesfrombrowser"] = ("chrome", "default")
                except Exception as e:
                    logger.warning(f"Failed to load cookies from browser: {e}")

        _cached_ydl_instance = yt_dlp.YoutubeDL(ydl_opts)
        _cached_ydl_instance._use_cookies = use_cookies  # Track cookies setting
        _cached_ydl_instance._cookies_file = _cookies_file  # Track cookies file

    return _cached_ydl_instance


async def get_text_from_url(
    url: str,
    transcribe_engine: str = "auto",
    language: str | None = None,
    save_original_path: Path | None = None,
    cookies_file: str | None = None,
) -> TextContext:
    """Extract text content from video URLs using yt-dlp.

    First attempts to extract subtitles. If no subtitles are available,
    downloads audio and transcribes it.

    Args:
        url: Video URL from any supported platform
        transcribe_engine: Transcription engine for audio fallback
        language: Whisper language code (e.g. "zh", "en"). None for auto-detection.
        save_original_path: Path to save the original downloaded audio file.
            If a directory, the audio will be saved there with auto-generated name.
            If a file path, the audio will be saved to that exact path.
            If None, the audio is stored in a temporary directory and deleted after processing.
        cookies_file: Path to a Netscape-format cookies file. When provided,
            this is used instead of extracting cookies from the browser (avoids
            keyring issues). Can be exported with browser extensions or yt-dlp.

    Returns:
        TextContext with title and content

    Raises:
        ValueError: If URL is invalid or unsupported
        RuntimeError: If content extraction fails
    """
    global _cookies_file
    _cookies_file = cookies_file

    if not is_valid_url(url):
        raise ValueError(f"Invalid URL: {url}")

    if not is_supported_url(url):
        raise ValueError(f"Unsupported URL: {url}")

    logger.info(f"Processing video URL: {url}")
    if cookies_file:
        logger.info(f"Using cookies file: {cookies_file}")
    platform = _detect_platform(url)

    try:
        # Extract video information
        info_dict = await _extract_video_info(url, platform)
        title = info_dict.get("title", "Unknown Title")

        # Try to get subtitles
        subtitle_content = await _extract_subtitles(info_dict, platform, language)

        if subtitle_content and subtitle_content.strip():
            logger.info("Successfully extracted subtitles")
            content = _format_content(info_dict, subtitle_content)
            return TextContext(title=title, chunk_list=[content])

        # Fallback: extract content from audio
        logger.info("No subtitles available, extracting content from audio")
        audio_content = await _extract_content_from_audio(
            info_dict, url, transcribe_engine, language, save_original_path
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

        # Determine if we should retry without cookies
        should_retry_without_cookies = False

        # Keyring errors: retry without cookies on ANY platform
        if _is_keyring_error(e):
            logger.info("Keyring error detected, retrying without browser cookies")
            should_retry_without_cookies = True

        # YouTube-specific access issues
        elif platform == "youtube" and (
            "not available on this app" in str(e).lower()
            or "watch on the latest version" in str(e).lower()
        ):
            logger.info("YouTube access issue detected, trying without cookies")
            should_retry_without_cookies = True

        if should_retry_without_cookies:
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
            # For other errors, don't retry without cookies
            raise RuntimeError(f"Failed to extract video information: {e}") from e

    raise RuntimeError("Failed to extract video information")


async def _extract_subtitles(
    info_dict: dict[str, Any], platform: str, language: str | None
) -> str | None:
    """Extract subtitles from video with platform-specific handling and language preferences.

    Args:
        info_dict: Video information from yt-dlp
        platform: Detected platform name
        language: Whisper language code preference (e.g. "zh", "en")

    Returns:
        Subtitle text if available, None otherwise
    """
    # Check if subtitles are available in info_dict
    subtitles = info_dict.get("subtitles", {})
    auto_subtitles = info_dict.get("automatic_captions", {})

    if not subtitles and not auto_subtitles:
        logger.info("No subtitles available")
        return None

    # Get priority order for subtitle languages based on language preference
    preferred_languages = _get_preferred_languages(language)
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

    Subtitles are publicly accessible, so cookies are intentionally disabled
    to avoid triggering keyring checks or authentication issues.

    Args:
        url: Direct URL to subtitle file

    Returns:
        Subtitle content as string, None if failed
    """
    try:

        def _download():
            # Force disable cookies for subtitle downloads (subtitles are public)
            ydl = _get_ydl_instance(use_cookies=False)
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
    language: str | None,
    save_original_path: Path | None = None,
) -> str | None:
    """Extract content by downloading audio and transcribing it.

    Args:
        info_dict: Video information from yt-dlp
        url: Original video URL
        transcribe_engine: Transcription engine to use
        language: Whisper language code (e.g. "zh", "en"). None for auto-detection.
        save_original_path: Path to save the original audio file (directory or file path)

    Returns:
        Transcribed text if successful, None otherwise
    """
    logger.info("Attempting to extract content from audio")

    async def _process_audio(download_path: Path) -> str | None:
        """Process audio download and transcription."""
        try:
            # Download audio using yt-dlp
            audio_file_path = await _download_audio(
                info_dict, url, download_path, save_original_path
            )

            if not audio_file_path or not audio_file_path.exists():
                logger.warning("Failed to download audio file")
                return None

            # Transcribe audio to text
            logger.info(f"Transcribing audio file: {audio_file_path}")
            text_context = await get_text_from_audio(
                audio_file_path,
                engine=transcribe_engine,
                language=language,
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

    # If save_original_path is specified, use it (or its parent) as download directory
    if save_original_path is not None:
        # Determine the download directory
        if save_original_path.is_dir() or (
            not save_original_path.exists() and save_original_path.suffix == ""
        ):
            # It's a directory path
            download_dir = save_original_path
        else:
            # It's a file path, use its parent directory
            download_dir = save_original_path.parent

        download_dir.mkdir(parents=True, exist_ok=True)
        return await _process_audio(download_dir)
    else:
        # Use temporary directory (deleted after processing)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            return await _process_audio(temp_path)


async def _download_audio(
    info_dict: dict[str, Any],
    url: str,
    download_path: Path,
    save_original_path: Path | None = None,
) -> Path | None:
    """Download audio from video using yt-dlp with format fallbacks.

    Tries format strategies in order:
    1. Best format with audio track (extracts audio via FFmpeg)
    2. Best combined format as fallback

    Audio-only selectors (bestaudio) are avoided because YouTube's SABR
    streaming returns HTTP 403 for direct audio-only downloads.

    Args:
        info_dict: Video information from yt-dlp
        url: Original video URL
        download_path: Directory path to download audio to
        save_original_path: If specified as a file path, use this exact filename.
            If None or a directory, auto-generate filename.

    Returns:
        Path to downloaded audio file, None if failed
    """
    # Determine output filename
    if save_original_path is not None and save_original_path.suffix != "":
        # save_original_path is a file path, use the stem as filename
        audio_filename = save_original_path.stem
    else:
        # Auto-generate filename from video title or ID
        video_title = info_dict.get("title", "")
        video_id = info_dict.get("id", "unknown")
        if video_title:
            # Sanitize title for filename
            safe_title = "".join(
                c if c.isalnum() or c in " -_" else "_" for c in video_title
            )[:100]
            audio_filename = safe_title.strip() or f"audio_{video_id}"
        else:
            audio_filename = f"audio_{video_id}"

    # Define format strategies to try in order
    # Each strategy is a tuple of (format_selector, preferred_codec, description)
    # NOTE: Audio-only selectors like "bestaudio" cause HTTP 403 on YouTube due to
    # SABR streaming restrictions. Use combined formats and extract audio via FFmpeg.
    # However, Bilibili uses DASH (separate audio/video streams only), so "bestaudio"
    # is required there since no combined formats exist.
    platform = _detect_platform(url)

    if platform == "bilibili":
        # Bilibili only provides separate DASH streams (audio-only + video-only).
        # "bestaudio" works fine here; the YouTube SABR issue doesn't apply.
        format_strategies = [
            ("bestaudio", "m4a", "best audio-only stream"),
            ("bestaudio[ext=m4a]", "m4a", "best m4a audio stream"),
            ("best[acodec!=none]", "m4a", "best format with audio track"),
            ("best", "m4a", "best combined format"),
        ]
    else:
        format_strategies = [
            ("best[acodec!=none]", "m4a", "best format with audio track"),
            ("best", "m4a", "best combined format"),
        ]
    last_error = None

    for format_selector, preferred_codec, description in format_strategies:
        try:
            logger.info(f"Trying format strategy: {description}")
            audio_file = await _try_download_with_format(
                url=url,
                download_path=download_path,
                audio_filename=audio_filename,
                format_selector=format_selector,
                preferred_codec=preferred_codec,
                platform=platform,
            )

            if audio_file and audio_file.exists():
                if save_original_path is not None:
                    logger.info(f"Original audio saved to: {audio_file}")
                logger.info(
                    f"Successfully downloaded audio using strategy: {description}"
                )
                return audio_file

        except Exception as e:
            last_error = e
            logger.warning(f"Format strategy '{description}' failed: {e}")
            continue

    # All strategies failed
    logger.error(f"All download strategies failed. Last error: {last_error}")
    return None


async def _try_download_with_format(
    url: str,
    download_path: Path,
    audio_filename: str,
    format_selector: str,
    preferred_codec: str,
    platform: str,
) -> Path | None:
    """Attempt to download audio with specific format settings.

    Args:
        url: Video URL to download from
        download_path: Directory to save the file
        audio_filename: Base filename (without extension)
        format_selector: yt-dlp format selector string
        preferred_codec: Preferred audio codec for extraction
        platform: Detected platform name

    Returns:
        Path to downloaded file, or None if failed
    """

    def _build_download_opts(use_cookies: bool = True):
        """Build yt-dlp download options."""
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": format_selector,
            "outtmpl": str(download_path / audio_filename),
        }

        # Add audio extraction postprocessor
        # 'best' codec means keep original format if possible
        if preferred_codec == "best":
            ydl_opts["postprocessors"] = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "best",
                }
            ]
        else:
            ydl_opts["postprocessors"] = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": preferred_codec,
                    "preferredquality": "192",
                }
            ]

        # Add browser impersonation for YouTube (if available)
        if platform == "youtube":
            try:
                test_opts = {"quiet": True, "impersonate": "chrome"}
                test_ydl = yt_dlp.YoutubeDL(test_opts)
                test_ydl.close()
                ydl_opts["impersonate"] = "chrome"
                logger.debug("Browser impersonation enabled for audio download")
            except Exception as e:
                logger.debug(f"Browser impersonation not available: {e}")

        # Add cookies only for platforms that require them (NOT YouTube)
        # YouTube downloads fail with 403 when cookies are used due to SABR streaming restrictions
        # See: https://github.com/yt-dlp/yt-dlp/issues/12482
        if use_cookies and platform != "youtube":
            if _cookies_file:
                # Use explicit cookies file (avoids keyring issues entirely)
                ydl_opts["cookiefile"] = _cookies_file
                logger.debug(f"Using cookies file for download: {_cookies_file}")
            else:
                try:
                    ydl_opts["cookiesfrombrowser"] = ("chrome", "default")
                except Exception as e:
                    logger.warning(f"Failed to load cookies: {e}")

        return ydl_opts

    def _download(use_cookies: bool = True):
        ydl_opts = _build_download_opts(use_cookies=use_cookies)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    # Run download in thread pool
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _download, True)
    except Exception as e:
        # Auto-fallback: retry without cookies on keyring errors
        if _is_keyring_error(e):
            logger.warning(
                f"Keyring error during audio download, retrying without cookies: {e}"
            )
            await loop.run_in_executor(None, _download, False)
        else:
            raise

    # Find the downloaded file
    # yt-dlp might change the extension based on codec/format
    audio_files = list(download_path.glob(f"{audio_filename}.*"))
    if not audio_files:
        # Fallback: try finding any recently created audio/video file
        audio_files = list(download_path.glob("audio_*.*"))

    logger.debug(f"Found files after download: {audio_files}")

    if audio_files:
        # Return the first matching file
        return audio_files[0]

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


def _get_preferred_languages(language: str | None) -> list[str]:
    """Get preferred subtitle languages based on language code.

    Args:
        language: Whisper language code (e.g. "zh", "en")

    Returns:
        List of subtitle language codes in priority order
    """
    # Use predefined language groups
    english_languages = ENGLISH_SUBTITLE_LANGUAGES
    chinese_languages = CHINESE_SUBTITLE_LANGUAGES

    # Determine language preference
    if language:
        lang = language.lower()

        # Check for Chinese language
        if lang in ("zh", "chinese"):
            return chinese_languages + english_languages

        # Check for English language
        elif lang in ("en", "english"):
            return english_languages + chinese_languages

    # Default order (Chinese first)
    return chinese_languages + english_languages
