"""Audio fallback download and transcription for URL processing."""

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import yt_dlp
from logly import logger

from ...errors import ProcessingFailedError
from ..transcription.processor import get_text_from_audio
from .cookies import (
    AUTH_REQUIRED_PLATFORMS,
    build_cookie_sources,
    is_auth_required_error,
    is_keyring_error,
)
from .formatter import detect_platform
from .ydl_client import impersonation_available


_MAX_FILENAME_BYTES = 200


def _truncate_to_bytes(name: str, max_bytes: int = _MAX_FILENAME_BYTES) -> str:
    """Truncate a filename stem so its UTF-8 encoding stays within *max_bytes*.

    Avoids splitting multi-byte characters by encoding one character at a time.
    """
    encoded = name.encode()
    if len(encoded) <= max_bytes:
        return name
    truncated = bytearray()
    for ch in name:
        ch_bytes = ch.encode()
        if len(truncated) + len(ch_bytes) > max_bytes:
            break
        truncated.extend(ch_bytes)
    return truncated.decode()


async def extract_content_from_audio(
    info_dict: dict[str, Any],
    url: str,
    transcribe_engine: str,
    language: str | None,
    model: str | None = None,
    save_original_path: Path | None = None,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    temp_dir: Path | None = None,
) -> str | None:
    """Extract content by downloading audio and transcribing it."""

    async def _process_audio(download_path: Path) -> str | None:
        try:
            audio_file_path = await download_audio(
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
                model=model,
                temp_dir=temp_dir,
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

    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp:
        return await _process_audio(Path(tmp))


async def download_audio(
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
            )
            audio_filename = (
                _truncate_to_bytes(safe_title.strip()) or f"audio_{video_id}"
            )
        else:
            audio_filename = f"audio_{video_id}"

    platform = detect_platform(url)
    if platform == "bilibili":
        format_strategies = [
            ("bestaudio", None, "best audio-only stream"),
            ("bestaudio[ext=m4a]", "m4a", "best m4a audio stream"),
            ("best[acodec!=none]", "m4a", "best format with audio track"),
            ("best", "m4a", "best combined format"),
        ]
    elif platform in ("xiaoyuzhoufm",):
        format_strategies = [
            ("best", None, "best available format"),
            ("bestaudio", None, "best audio-only stream"),
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
            audio_file = await try_download_with_format(
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


async def try_download_with_format(
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
            "outtmpl": str(download_path / audio_filename) + ".%(ext)s",
        }

        if preferred_codec:
            ydl_opts["postprocessors"] = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": preferred_codec,
                    "preferredquality": "192",
                }
            ]

        if platform == "youtube" and impersonation_available():
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

    sources = build_cookie_sources(
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
            if is_keyring_error(exc) or is_auth_required_error(exc):
                continue
            continue

    if (
        last_error
        and platform in AUTH_REQUIRED_PLATFORMS
        and is_auth_required_error(last_error)
    ):
        raise ProcessingFailedError(
            "Authenticated cookies are required for this download. "
            "Provide --cookies or --cookies-from-browser."
        ) from last_error

    audio_files = list(download_path.glob(f"{audio_filename}.*"))
    if not audio_files:
        audio_files = list(download_path.glob("audio_*.*"))
    return audio_files[0] if audio_files else None
