"""MarkItDown plugin for AIMD URL inputs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from logly import logger

from markitdown import (
    DocumentConverter,
    DocumentConverterResult,
    MarkItDown,
    StreamInfo,
)

from aimd.core.errors import (
    ProcessingCancelledError,
    ProcessingFailedError,
    UnsupportedInputError,
)

from .audio import detect_platform, extract_content_from_audio
from .markdown import format_content, strip_subtitle_formatting
from .metadata import build_metadata_context, extract_video_info
from .defuddle import extract_html_with_defuddle
from .subtitles import extract_subtitles, resolve_subtitle_language

HTML_EXTENSIONS = {".html", ".htm"}

__plugin_interface_version__ = 1


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
    precision: str | None = None,
    cancellation_check: Callable[[], bool] | None = None,
    progress_reporter: Callable[[str, int | None, int | None, str | None], None]
    | None = None,
    context: str | None = None,
    metadata_context: bool = True,
    asr_base_url: str | None = None,
    asr_model: str | None = None,
    asr_api_key: str | None = None,
) -> UrlTextResult:
    """Extract text content from transcript-capable URLs using yt-dlp."""
    if not url.startswith(("http://", "https://")):
        raise UnsupportedInputError(f"Invalid URL: {url}")

    logger.info(f"Processing URL: {url}")
    if progress_reporter is not None:
        progress_reporter("extracting", None, None, "Extracting URL metadata")
    if cookies_file:
        logger.info(f"Using cookies file: {cookies_file}")
    if cookies_from_browser:
        logger.info(f"Using browser cookies source: {cookies_from_browser}")

    platform = detect_platform(url)
    info_dict = await extract_video_info(
        url=url,
        platform=platform,
        cookies_file=cookies_file,
        cookies_from_browser=cookies_from_browser,
    )
    title = str(info_dict.get("title", "Unknown Title"))
    description = info_dict.get("description")
    metadata_language = info_dict.get("language")
    effective_language = language
    if effective_language is None:
        effective_language = resolve_subtitle_language(
            None,
            title=title,
            description=description if isinstance(description, str) else None,
            metadata_language=(
                metadata_language if isinstance(metadata_language, str) else None
            ),
        )
        if effective_language is not None:
            logger.info(
                "Inferred content language from metadata/title/description: "
                f"{effective_language}"
            )

    subtitle_content = await extract_subtitles(info_dict, platform, language)
    if cancellation_check is not None and cancellation_check():
        raise ProcessingCancelledError("URL processing cancelled after subtitle lookup")
    if subtitle_content and subtitle_content.strip():
        logger.info("Successfully extracted subtitles")
        if not raw_transcript:
            subtitle_content = strip_subtitle_formatting(subtitle_content)
        content = format_content(info_dict, subtitle_content, platform)
        return UrlTextResult(title=title, markdown=content, platform=platform)

    logger.info("No subtitles available, extracting content from audio")
    if progress_reporter is not None:
        progress_reporter("downloading", None, None, "Downloading source audio")
    asr_context = context
    if metadata_context:
        built_context = build_metadata_context(info_dict)
        if built_context:
            asr_context = f"{context}\n\n{built_context}" if context else built_context
            logger.info("Injecting URL metadata as ASR context for transcription")
    audio_content = await extract_content_from_audio(
        info_dict=info_dict,
        url=url,
        language=effective_language,
        model=model,
        save_original_path=save_original_path,
        cookies_file=cookies_file,
        cookies_from_browser=cookies_from_browser,
        temp_dir=temp_dir,
        precision=precision,
        cancellation_check=cancellation_check,
        progress_reporter=progress_reporter,
        context=asr_context,
        asr_base_url=asr_base_url,
        asr_model=asr_model,
        asr_api_key=asr_api_key,
    )

    if audio_content and audio_content.strip():
        logger.info("Successfully extracted content from audio")
        content = format_content(info_dict, audio_content, platform)
        return UrlTextResult(title=title, markdown=content, platform=platform)

    raise ProcessingFailedError(
        "Failed to extract subtitles or transcribe audio for URL"
    )


def register_converters(markitdown: MarkItDown, **kwargs: Any) -> None:
    """Register AIMD URL converters with MarkItDown."""
    markitdown.register_converter(AimdUrlTranscriptConverter(), priority=-1.0)
    markitdown.register_converter(AimdReadableHtmlConverter(), priority=-1.0)


class AimdUrlTranscriptConverter(DocumentConverter):
    """Convert HTTP(S) transcript URLs to markdown through AIMD's URL flow."""

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        return bool(
            stream_info.url
            and stream_info.url.startswith(("http://", "https://"))
            and kwargs.get("task_type") == "transcript"
        )

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        result = asyncio.run(
            get_text_from_url(
                stream_info.url,
                language=kwargs.get("language"),
                model=kwargs.get("model"),
                save_original_path=kwargs.get("save_original_path"),
                cookies_file=kwargs.get("cookies_file"),
                cookies_from_browser=kwargs.get("cookies_from_browser"),
                temp_dir=kwargs.get("temp_dir"),
                raw_transcript=kwargs.get("raw_transcript", False),
                precision=kwargs.get("precision"),
                cancellation_check=kwargs.get("cancellation_check"),
                progress_reporter=kwargs.get("progress_reporter"),
                context=kwargs.get("context"),
                metadata_context=kwargs.get("metadata_context", True),
                asr_base_url=kwargs.get("asr_base_url"),
                asr_model=kwargs.get("asr_model"),
                asr_api_key=kwargs.get("asr_api_key"),
            )
        )

        return DocumentConverterResult(
            title=result.title,
            markdown=result.markdown,
        )


class AimdReadableHtmlConverter(DocumentConverter):
    """Extract readable HTML content through Defuddle when explicitly requested."""

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        if (
            kwargs.get("task_type") != "readable_html"
            and kwargs.get("defuddle") is not True
        ):
            return False

        if stream_info.url and stream_info.url.startswith(("http://", "https://")):
            return True

        return (stream_info.extension or "").lower() in HTML_EXTENSIONS

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        source = stream_info.url or stream_info.local_path
        result = extract_html_with_defuddle(
            Path(source) if stream_info.local_path else source,
            markdown=True,
            npx_command=kwargs.get("npx_command", "npx"),
        )

        return DocumentConverterResult(
            title=result.title,
            markdown=result.content,
        )
