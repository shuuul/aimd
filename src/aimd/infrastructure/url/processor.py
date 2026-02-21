"""URL processing orchestration."""

from pathlib import Path

from logly import logger

from ...errors import ProcessingFailedError, UnsupportedInputError
from ...types import TextContext
from ...utils import is_valid_url
from .audio_download import extract_content_from_audio
from .formatter import detect_platform, format_content
from .subtitles import extract_subtitles
from .video_info import extract_video_info


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
            content = format_content(info_dict, subtitle_content)
            return TextContext(title=title, chunk_list=[content])

        logger.info("No subtitles available, extracting content from audio")
        audio_content = await extract_content_from_audio(
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
            content = format_content(info_dict, audio_content)
            return TextContext(title=title, chunk_list=[content])

        logger.warning("Could not extract content, returning basic video info")
        content = format_content(info_dict, None)
        return TextContext(title=title, chunk_list=[content])
    except UnsupportedInputError:
        raise
    except ProcessingFailedError:
        raise
    except Exception as exc:
        logger.error(f"Failed to extract content from URL {url}: {exc}")
        raise ProcessingFailedError(f"URL content extraction failed: {exc}") from exc
