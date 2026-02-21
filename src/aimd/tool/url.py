"""URL processing tools for extracting content from video URLs using yt-dlp."""

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import yt_dlp
from logly import logger

from ..const import (
    CHINESE_SUBTITLE_LANGUAGES,
    ENGLISH_SUBTITLE_LANGUAGES,
    FORBIDDEN_SUBTITLE_LANGUAGES,
)
from ..errors import ProcessingFailedError, UnsupportedInputError
from ..types import TextContext
from ..utils import is_valid_url
from .audio import get_text_from_audio


def _is_keyring_error(error: Exception) -> bool:
    """Check if an error is related to keyring/cookie decryption issues."""
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


def _is_unsupported_url_error(error: Exception) -> bool:
    """Best-effort check for yt-dlp unsupported URL errors."""
    message = str(error).lower()
    return "unsupported url" in message or "no suitable extractor" in message


def _impersonation_available() -> bool:
    """Return True when yt-dlp impersonation dependencies are available."""
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "impersonate": "chrome"}):
            return True
    except Exception:
        return False


def _create_ydl(
    *,
    use_cookies: bool,
    platform: str,
    cookies_file: str | None,
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

    if platform == "youtube" and _impersonation_available():
        ydl_opts["impersonate"] = "chrome"

    if use_cookies:
        if cookies_file:
            ydl_opts["cookiefile"] = cookies_file
        else:
            ydl_opts["cookiesfrombrowser"] = ("chrome", "default")

    return yt_dlp.YoutubeDL(ydl_opts)


async def get_text_from_url(
    url: str,
    transcribe_engine: str = "auto",
    language: str | None = None,
    save_original_path: Path | None = None,
    cookies_file: str | None = None,
) -> TextContext:
    """Extract text content from video URLs using yt-dlp."""
    if not is_valid_url(url):
        raise UnsupportedInputError(f"Invalid URL: {url}")

    logger.info(f"Processing video URL: {url}")
    if cookies_file:
        logger.info(f"Using cookies file: {cookies_file}")

    platform = _detect_platform(url)

    try:
        info_dict = await _extract_video_info(url, platform, cookies_file)
        title = info_dict.get("title", "Unknown Title")

        subtitle_content = await _extract_subtitles(info_dict, platform, language)
        if subtitle_content and subtitle_content.strip():
            logger.info("Successfully extracted subtitles")
            content = _format_content(info_dict, subtitle_content)
            return TextContext(title=title, chunk_list=[content])

        logger.info("No subtitles available, extracting content from audio")
        audio_content = await _extract_content_from_audio(
            info_dict=info_dict,
            url=url,
            transcribe_engine=transcribe_engine,
            language=language,
            save_original_path=save_original_path,
            cookies_file=cookies_file,
        )

        if audio_content and audio_content.strip():
            logger.info("Successfully extracted content from audio")
            content = _format_content(info_dict, audio_content)
            return TextContext(title=title, chunk_list=[content])

        logger.warning("Could not extract content, returning basic video info")
        content = _format_content(info_dict, None)
        return TextContext(title=title, chunk_list=[content])

    except UnsupportedInputError:
        raise
    except ProcessingFailedError:
        raise
    except Exception as exc:
        logger.error(f"Failed to extract content from URL {url}: {exc}")
        raise ProcessingFailedError(f"URL content extraction failed: {exc}") from exc


async def _extract_video_info(
    url: str, platform: str, cookies_file: str | None
) -> dict[str, Any]:
    """Extract video information using yt-dlp with graceful fallback."""

    def _extract_with_config(use_cookies: bool) -> dict[str, Any]:
        with _create_ydl(
            use_cookies=use_cookies,
            platform=platform,
            cookies_file=cookies_file,
            for_subtitles=False,
        ) as ydl:
            return ydl.extract_info(url, download=False)

    loop = asyncio.get_running_loop()

    try:
        info_dict = await loop.run_in_executor(None, _extract_with_config, True)
        if info_dict:
            return info_dict
    except Exception as exc:
        logger.warning(f"Failed to extract video info with cookies: {exc}")

        if _is_unsupported_url_error(exc):
            raise UnsupportedInputError(f"Unsupported URL: {url}") from exc

        should_retry_without_cookies = False
        if _is_keyring_error(exc):
            should_retry_without_cookies = True
        elif platform == "youtube" and (
            "not available on this app" in str(exc).lower()
            or "watch on the latest version" in str(exc).lower()
        ):
            should_retry_without_cookies = True

        if should_retry_without_cookies:
            try:
                info_dict = await loop.run_in_executor(
                    None, _extract_with_config, False
                )
                if info_dict:
                    logger.info("Successfully extracted video info without cookies")
                    return info_dict
            except Exception as fallback_exc:
                if _is_unsupported_url_error(fallback_exc):
                    raise UnsupportedInputError(
                        f"Unsupported URL: {url}"
                    ) from fallback_exc
                raise ProcessingFailedError(
                    f"Failed to extract video information: {fallback_exc}"
                ) from fallback_exc
        else:
            raise ProcessingFailedError(
                f"Failed to extract video information: {exc}"
            ) from exc

    raise ProcessingFailedError("Failed to extract video information")


async def _extract_subtitles(
    info_dict: dict[str, Any], platform: str, language: str | None
) -> str | None:
    """Extract subtitles from video with platform-specific handling."""
    subtitles = info_dict.get("subtitles", {})
    auto_subtitles = info_dict.get("automatic_captions", {})

    if not subtitles and not auto_subtitles:
        logger.info("No subtitles available")
        return None

    preferred_languages = _get_preferred_languages(language)

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

            subtitle_content = await _download_subtitle(sub_url, platform)
            if subtitle_content:
                logger.info(f"Successfully extracted subtitles using {platform} format")
            return subtitle_content

        logger.error(f"Unsupported platform for subtitles: {platform}")
        return None
    except Exception as exc:
        logger.error(f"Failed to extract subtitles using {platform} format: {exc}")
        return None


async def _download_subtitle(url: str, platform: str) -> str | None:
    """Download subtitle content from URL using yt-dlp without cookies."""

    def _download() -> str:
        with _create_ydl(
            use_cookies=False,
            platform=platform,
            cookies_file=None,
            for_subtitles=True,
        ) as ydl:
            response = ydl.urlopen(url)
            return response.read().decode("utf-8")

    try:
        loop = asyncio.get_running_loop()
        content = await loop.run_in_executor(None, _download)
        return content
    except Exception as exc:
        logger.error(f"Failed to download subtitle from {url}: {exc}")
        return None


async def _extract_content_from_audio(
    info_dict: dict[str, Any],
    url: str,
    transcribe_engine: str,
    language: str | None,
    save_original_path: Path | None = None,
    cookies_file: str | None = None,
) -> str | None:
    """Extract content by downloading audio and transcribing it."""

    async def _process_audio(download_path: Path) -> str | None:
        try:
            audio_file_path = await _download_audio(
                info_dict=info_dict,
                url=url,
                download_path=download_path,
                save_original_path=save_original_path,
                cookies_file=cookies_file,
            )
            if not audio_file_path or not audio_file_path.exists():
                return None

            text_context = await get_text_from_audio(
                audio_file_path,
                engine=transcribe_engine,
                language=language,
            )
            if text_context.chunk_list and text_context.chunk_list[0]:
                content = text_context.chunk_list[0]
                if len(content) > 10:
                    return content
            return None
        except Exception as exc:
            logger.error(f"Failed to extract content from audio: {exc}")
            return None

    if save_original_path is not None:
        if save_original_path.is_dir() or (
            not save_original_path.exists() and save_original_path.suffix == ""
        ):
            download_dir = save_original_path
        else:
            download_dir = save_original_path.parent
        download_dir.mkdir(parents=True, exist_ok=True)
        return await _process_audio(download_dir)

    with tempfile.TemporaryDirectory() as temp_dir:
        return await _process_audio(Path(temp_dir))


async def _download_audio(
    info_dict: dict[str, Any],
    url: str,
    download_path: Path,
    save_original_path: Path | None = None,
    cookies_file: str | None = None,
) -> Path | None:
    """Download audio from video using yt-dlp with format fallbacks."""
    if save_original_path is not None and save_original_path.suffix != "":
        audio_filename = save_original_path.stem
    else:
        video_title = info_dict.get("title", "")
        video_id = info_dict.get("id", "unknown")
        if video_title:
            safe_title = "".join(
                c if c.isalnum() or c in " -_" else "_" for c in video_title
            )[:100]
            audio_filename = safe_title.strip() or f"audio_{video_id}"
        else:
            audio_filename = f"audio_{video_id}"

    platform = _detect_platform(url)
    if platform == "bilibili":
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
                cookies_file=cookies_file,
            )
            if audio_file and audio_file.exists():
                return audio_file
        except Exception as exc:
            logger.warning(f"Format strategy '{description}' failed: {exc}")

    return None


async def _try_download_with_format(
    url: str,
    download_path: Path,
    audio_filename: str,
    format_selector: str,
    preferred_codec: str,
    platform: str,
    cookies_file: str | None,
) -> Path | None:
    """Attempt to download audio with specific format settings."""

    def _build_download_opts(use_cookies: bool) -> dict[str, Any]:
        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "format": format_selector,
            "outtmpl": str(download_path / audio_filename),
        }

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

        if platform == "youtube" and _impersonation_available():
            ydl_opts["impersonate"] = "chrome"

        if use_cookies and platform != "youtube":
            if cookies_file:
                ydl_opts["cookiefile"] = cookies_file
            else:
                ydl_opts["cookiesfrombrowser"] = ("chrome", "default")

        return ydl_opts

    def _download(use_cookies: bool) -> None:
        with yt_dlp.YoutubeDL(_build_download_opts(use_cookies)) as ydl:
            ydl.download([url])

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _download, True)
    except Exception as exc:
        if _is_keyring_error(exc):
            logger.warning(
                f"Keyring error during audio download, retrying without cookies: {exc}"
            )
            await loop.run_in_executor(None, _download, False)
        else:
            raise

    audio_files = list(download_path.glob(f"{audio_filename}.*"))
    if not audio_files:
        audio_files = list(download_path.glob("audio_*.*"))
    return audio_files[0] if audio_files else None


def _format_content(info_dict: dict[str, Any], content: str | None) -> str:
    """Format video information with content."""
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
    """Detect the platform from URL."""
    url_lower = url.lower()
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    if "bilibili.com" in url_lower:
        return "bilibili"
    if "xiaohongshu.com" in url_lower or "xhslink.com" in url_lower:
        return "xiaohongshu"
    return "unknown"


def _get_preferred_languages(language: str | None) -> list[str]:
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
