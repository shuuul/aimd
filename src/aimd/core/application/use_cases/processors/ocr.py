"""OCR task processor."""

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from ....types import TextContext
from ...models import InputRoute, ProcessInput, ProcessResult

OCRProcessor = Callable[
    [str, str, str | None, str | None, int | None, int | None, Path | None],
    Awaitable[TextContext],
]


@dataclass(slots=True)
class OCRTaskProcessor:
    """Run OCR tasks for image files and scanned PDFs."""

    process_file: OCRProcessor

    async def process(
        self,
        request: ProcessInput,
        route: InputRoute,  # noqa: ARG002
    ) -> ProcessResult:
        input_path = Path(request.input_source)
        text_context = await self.process_file(
            input_path.as_posix(),
            request.transcribe_engine,
            request.model,
            request.language,
            request.start,
            request.end,
            request.temp_dir,
        )
        return ProcessResult(task_type="ocr", text_context=text_context)
