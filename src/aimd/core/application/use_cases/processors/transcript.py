"""Transcript task processor."""

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from ....errors import InputNotFoundError
from ....types import TextContext
from ....utils import is_url
from ...models import InputRoute, ProcessInput, ProcessResult

TranscriptProcessor = Callable[
    [
        str,
        str,
        str | None,
        str | None,
        Path | None,
        Path | None,
        str | None,
        Path | None,
        bool,
    ],
    Awaitable[tuple[TextContext, str | None]],
]


@dataclass(slots=True)
class TranscriptTaskProcessor:
    """Run transcript tasks for URLs and local audio/video files."""

    process_url: Callable[
        [
            str,
            str,
            str | None,
            str | None,
            Path | None,
            str | None,
            str | None,
            Path | None,
            bool,
        ],
        Awaitable[tuple[TextContext, str]],
    ]
    process_file: Callable[
        [str, str, str | None, str | None, Path | None],
        Awaitable[tuple[TextContext, Path | None]],
    ]
    resolve_engine: Callable[[str], str]

    async def process(
        self,
        request: ProcessInput,
        route: InputRoute,  # noqa: ARG002
    ) -> ProcessResult:
        if is_url(request.input_source):
            if request.transcribe_engine != "auto":
                self.resolve_engine(request.transcribe_engine)
            text_context, platform = await self.process_url(
                request.input_source,
                request.transcribe_engine,
                request.language,
                request.model,
                request.save_original,
                str(request.cookies) if request.cookies else None,
                request.cookies_from_browser,
                request.temp_dir,
                request.raw_transcript,
            )
        else:
            input_path = Path(request.input_source)
            if not input_path.exists():
                raise InputNotFoundError(
                    f"Input file not found: {request.input_source}"
                )

            resolved_engine = self.resolve_engine(request.transcribe_engine)
            text_context, _ = await self.process_file(
                input_path.as_posix(),
                resolved_engine,
                request.language,
                request.model,
                request.temp_dir,
            )
            platform = None

        return ProcessResult(
            task_type="transcript",
            text_context=text_context,
            platform=platform,
        )
