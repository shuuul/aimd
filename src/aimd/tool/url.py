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

AUTH_REQUIRED_PLATFORMS = {"bilibili", "xiaohongshu"}


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


def _is_auth_required_error(error: Exception) -> bool:
    """Best-effort check for login-required/private-content errors."""
    message = str(error).lower()
    indicators = [
        "login required",
        "sign in",
        "private",
        "members only",
        "premium",
        "watchlater",
        "supporter-only",
        "cookies are required",
        "authentication",
        "403",
        "-403",
        "-101",
    ]
    return any(indicator in message for indicator in indicators)


def _impersonation_available() -> bool:
    """Return True when yt-dlp impersonation dependencies are available."""
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "impersonate": "chrome"}):
            return True
    except Exception:
        return False


def _parse_cookies_from_browser(
    spec: str,
) -> tuple[str, str | None, str | None, str | None]:
    """Parse browser cookie source spec to yt-dlp tuple.

    Supported input examples:
    - chrome
    - chrome:default
    - chrome+gnomekeyring:default
    - firefox::container-1
    - firefox:default::container-1
    """
    raw = spec.strip()
    if not raw:
        raise ValueError("cookies_from_browser cannot be empty")

    browser_profile, sep, container = raw.partition("::")
    container_name = container.strip() if sep and container.strip() else None

    browser_keyring, has_profile, profile = browser_profile.partition(":")
    profile_name = profile.strip() if has_profile and profile.strip() else None

    browser_name, has_keyring, keyring = browser_keyring.partition("+")
    keyring_name = keyring.strip() if has_keyring and keyring.strip() else None
    browser_name = browser_name.strip().lower()

    if not browser_name:
        raise ValueError(f"Invalid cookies_from_browser value: {spec}")

    return browser_name, profile_name, keyring_name, container_name


def _build_cookie_sources(
    *,
    platform: str,
    cookies_file: str | None,
    cookies_from_browser: str | None,
) -> list[dict[str, Any]]:
    """Build ordered cookie source attempts for yt-dlp operations."""
    sources: list[dict[str, Any]] = []

    if cookies_file:
        sources.append(
            {
                "name": "cookiefile",
                "use_cookies": True,
                "cookiefile": cookies_file,
                "cookiesfrombrowser": None,
            }
        )

    if cookies_from_browser:
        try:
            browser_tuple = _parse_cookies_from_browser(cookies_from_browser)
            sources.append(
                {
                    "name": f"cookiesfrombrowser:{cookies_from_browser}",
                    "use_cookies": True,
                    "cookiefile": None,
                    "cookiesfrombrowser": browser_tuple,
                }
            )
        except ValueError as exc:
            logger.warning(str(exc))

    if not cookies_file and not cookies_from_browser:
        # Default fallback chain for common user environments.
        default_browser_specs = ("chrome:default", "firefox")
        for spec in default_browser_specs:
            sources.append(
                {
                    "name": f"cookiesfrombrowser:{spec}",
                    "use_cookies": True,
                    "cookiefile": None,
                    "cookiesfrombrowser": _parse_cookies_from_browser(spec),
                }
            )

    # On platforms that usually require authentication, avoid early no-cookie fallback.
    if platform not in AUTH_REQUIRED_PLATFORMS:
        sources.append(
            {
                "name": "no-cookie",
                "use_cookies": False,
                "cookiefile": None,
                "cookiesfrombrowser": None,
            }
        )

    return sources


def _create_ydl(
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

    if platform == "youtube" and _impersonation_available():
        ydl_opts["impersonate"] = "chrome"

    if cookie_source.get("use_cookies", False):
        if cookie_source.get("cookiefile"):
            ydl_opts["cookiefile"] = cookie_source["cookiefile"]
        elif cookie_source.get("cookiesfrombrowser"):
            ydl_opts["cookiesfrombrowser"] = cookie_source["cookiesfrombrowser"]

    return yt_dlp.YoutubeDL(ydl_opts)


async def get_text_from_url(
    url: str,
    transcribe_engine: str = "auto",
    language: str | None = None,
    save_original_path: Path | None = None,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
) -> TextContext:
    """Extract text content from video URLs using yt-dlp."""
    if not is_valid_url(url):
        raise UnsupportedInputError(f"Invalid URL: {url}")

    logger.info(f"Processing video URL: {url}")
    if cookies_file:
        logger.info(f"Using cookies file: {cookies_file}")
    if cookies_from_browser:
        logger.info(f"Using browser cookies source: {cookies_from_browser}")

    platform = _detect_platform(url)

    try:
        info_dict = await _extract_video_info(
            url=url,
            platform=platform,
            cookies_file=cookies_file,
            cookies_from_browser=cookies_from_browser,
        )
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
            cookies_from_browser=cookies_from_browser,
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
    *,
    url: str,
    platform: str,
    cookies_file: str | None,
    cookies_from_browser: str | None,
) -> dict[str, Any]:
    """Extract video information using yt-dlp with cookie-source fallback chain."""

    loop = asyncio.get_running_loop()
    sources = _build_cookie_sources(
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
            with _create_ydl(
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

            if _is_unsupported_url_error(exc):
                raise UnsupportedInputError(f"Unsupported URL: {url}") from exc

            if _is_auth_required_error(exc):
                auth_required_seen = True

                # For auth-heavy platforms, never downgrade to no-cookie attempt here.
                if (
                    platform in AUTH_REQUIRED_PLATFORMS
                    and source["name"] == "no-cookie"
                ):
                    break

                continue

            if _is_keyring_error(exc):
                continue

            # Non-auth, non-keyring failures can still be transient; continue fallback chain.
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
            platform=platform,
            cookie_source={"use_cookies": False},
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
    cookies_from_browser: str | None = None,
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
                cookies_from_browser=cookies_from_browser,
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
    cookies_from_browser: str | None = None,
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
            ("bestaudio", None, "best audio-only stream"),
            ("bestaudio[ext=m4a]", "m4a", "best m4a audio stream"),
            ("best[acodec!=none]", "m4a", "best format with audio track"),
            ("best", "m4a", "best combined format"),
        ]
    else:
        format_strategies = [
            (
                "bestaudio[acodec^=opus][abr<=128]/bestaudio[abr<=128]/bestaudio",
                None,
                "audio-only stream (prefer opus <=128kbps)",
            ),
            ("bestaudio[ext=m4a]/bestaudio", "m4a", "audio-only m4a stream"),
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
                cookies_from_browser=cookies_from_browser,
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
    preferred_codec: str | None,
    platform: str,
    cookies_file: str | None,
    cookies_from_browser: str | None,
) -> Path | None:
    """Attempt to download audio with specific format settings."""

    def _build_download_opts(cookie_source: dict[str, Any]) -> dict[str, Any]:
        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "format": format_selector,
            "outtmpl": str(download_path / audio_filename),
        }

        if preferred_codec:
            ydl_opts["postprocessors"] = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": preferred_codec,
                    "preferredquality": "192",
                }
            ]

        if platform == "youtube" and _impersonation_available():
            ydl_opts["impersonate"] = "chrome"

        if cookie_source.get("use_cookies") and platform != "youtube":
            if cookie_source.get("cookiefile"):
                ydl_opts["cookiefile"] = cookie_source["cookiefile"]
            elif cookie_source.get("cookiesfrombrowser"):
                ydl_opts["cookiesfrombrowser"] = cookie_source["cookiesfrombrowser"]

        return ydl_opts

    def _download_with_source(cookie_source: dict[str, Any]) -> None:
        with yt_dlp.YoutubeDL(_build_download_opts(cookie_source)) as ydl:
            ydl.download([url])

    sources = _build_cookie_sources(
        platform=platform,
        cookies_file=cookies_file,
        cookies_from_browser=cookies_from_browser,
    )

    loop = asyncio.get_running_loop()
    last_error: Exception | None = None

    for source in sources:
        try:
            await loop.run_in_executor(None, _download_with_source, source)
            break
        except Exception as exc:
            last_error = exc
            if _is_keyring_error(exc) or _is_auth_required_error(exc):
                continue
            continue

    if (
        last_error
        and platform in AUTH_REQUIRED_PLATFORMS
        and _is_auth_required_error(last_error)
    ):
        raise ProcessingFailedError(
            "Authenticated cookies are required for this download. "
            "Provide --cookies or --cookies-from-browser."
        ) from last_error

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
