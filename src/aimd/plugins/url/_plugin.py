"""MarkItDown plugin for AIMD URL inputs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, BinaryIO

from markitdown import (
    DocumentConverter,
    DocumentConverterResult,
    FailedConversionAttempt,
    MarkItDown,
    StreamInfo,
)

from .defuddle import extract_html_with_defuddle
from .processor import get_text_from_url

HTML_EXTENSIONS = {".html", ".htm"}

__plugin_interface_version__ = 1


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
        if not stream_info.url:
            raise FailedConversionAttempt("aimd.plugins.url requires a URL")

        try:
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
                )
            )
        except Exception as exc:
            raise FailedConversionAttempt(
                f"AIMD URL transcript conversion failed: {exc}"
            ) from exc

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

        extension = (stream_info.extension or "").lower()
        return extension in HTML_EXTENSIONS

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        source = stream_info.url or stream_info.local_path
        if not source:
            raise FailedConversionAttempt(
                "aimd.plugins.url Defuddle conversion requires a URL or local file path"
            )

        try:
            result = extract_html_with_defuddle(
                Path(source) if stream_info.local_path else source,
                markdown=True,
                npx_command=kwargs.get("npx_command", "npx"),
            )
        except Exception as exc:
            raise FailedConversionAttempt(f"Defuddle conversion failed: {exc}") from exc

        return DocumentConverterResult(
            title=result.title,
            markdown=result.content,
        )
