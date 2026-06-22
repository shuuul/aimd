"""URL processing orchestration."""

from dataclasses import dataclass
from pathlib import Path

from logly import logger

from .errors import ProcessingFailedError, UnsupportedInputError
from .audio_download import extract_content_from_audio
from .formatter import detect_platform, format_content, strip_subtitle_formatting
from .subtitles import extract_subtitles
from .video_info import extract_video_info


@dataclass(slots=True)
class UrlTextResult:
    """Markdown extracted from a URL."""

    title: str
    markdown: str
    platform: str


async def get_text_from_url(
    url: str,
    language: str | None = None,
    model: str | None = None,
    save_original_path: Path | None = None,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    temp_dir: Path | None = None,
    raw_transcript: bool = False,
) -> UrlTextResult:
    """Extract text content from transcript-capable URLs using yt-dlp."""
    if not url.startswith(("http://", "https://")):
        raise UnsupportedInputError(f"Invalid URL: {url}")

    logger.info(f"Processing URL: {url}")
    if cookies_file:
        logger.info(f"Using cookies file: {cookies_file}")
    if cookies_from_browser:
        logger.info(f"Using browser cookies source: {cookies_from_browser}")

    platform = detect_platform(url)

    try:
        info_dict = await extract_video_info(
            url=url,
            platform=platform,
            cookies_file=cookies_file,
            cookies_from_browser=cookies_from_browser,
        )
        title = str(info_dict.get("title", "Unknown Title"))

        subtitle_content = await extract_subtitles(info_dict, platform, language)
        if subtitle_content and subtitle_content.strip():
            logger.info("Successfully extracted subtitles")
            if not raw_transcript:
                subtitle_content = strip_subtitle_formatting(subtitle_content)
            content = format_content(info_dict, subtitle_content, platform)
            return UrlTextResult(title=title, markdown=content, platform=platform)

        logger.info("No subtitles available, extracting content from audio")
        audio_content = await extract_content_from_audio(
            info_dict=info_dict,
            url=url,
            language=language,
            model=model,
            save_original_path=save_original_path,
            cookies_file=cookies_file,
            cookies_from_browser=cookies_from_browser,
            temp_dir=temp_dir,
        )

        if audio_content and audio_content.strip():
            logger.info("Successfully extracted content from audio")
            content = format_content(info_dict, audio_content, platform)
            return UrlTextResult(title=title, markdown=content, platform=platform)

        logger.warning("Could not extract content, returning basic video info")
        content = format_content(info_dict, None, platform)
        return UrlTextResult(title=title, markdown=content, platform=platform)
    except UnsupportedInputError:
        raise
    except ProcessingFailedError:
        raise
    except Exception as exc:
        logger.error(f"Failed to extract content from URL {url}: {exc}")
        raise ProcessingFailedError(f"URL content extraction failed: {exc}") from exc
