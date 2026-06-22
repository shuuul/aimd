"""aimd wrappers around the bundled media module."""

from pathlib import Path

from aimd.media.errors import ProcessingFailedError as MediaProcessingFailedError
from aimd.media.errors import UnsupportedInputError as MediaUnsupportedInputError
from aimd.media.url import get_text_from_url

from ..errors import ProcessingFailedError, UnsupportedInputError
from ..types import TextContext
from .markitdown_processor import _text_context_from_markdown


async def get_text_context_from_media_url(
    url: str,
    transcribe_engine: str = "auto",
    language: str | None = None,
    model: str | None = None,
    save_original_path: Path | None = None,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    temp_dir: Path | None = None,
    raw_transcript: bool = False,
) -> tuple[TextContext, str]:
    """Extract a media URL through media and wrap it as TextContext."""
    try:
        result = await get_text_from_url(
            url=url,
            transcribe_engine=transcribe_engine,
            language=language,
            model=model,
            save_original_path=save_original_path,
            cookies_file=cookies_file,
            cookies_from_browser=cookies_from_browser,
            temp_dir=temp_dir,
            raw_transcript=raw_transcript,
        )
    except MediaUnsupportedInputError as exc:
        raise UnsupportedInputError(str(exc)) from exc
    except MediaProcessingFailedError as exc:
        raise ProcessingFailedError(str(exc)) from exc

    return (
        _text_context_from_markdown(
            result.markdown,
            fallback_title=result.title,
            title=result.title,
            max_chunk_size=40000,
        ),
        result.platform,
    )
