"""Document conversion task processor."""

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from ....types import TextContext
from ...models import InputRoute, ProcessInput, ProcessResult

ConvertProcessor = Callable[
    [str, str, str | None, str | None, Path | None],
    Awaitable[tuple[TextContext, Path | None]],
]


@dataclass(slots=True)
class ConvertTaskProcessor:
    """Run document conversion tasks."""

    process_file: ConvertProcessor

    async def process(
        self,
        request: ProcessInput,
        route: InputRoute,  # noqa: ARG002
    ) -> ProcessResult:
        input_path = Path(request.input_source)
        text_context, output_dir = await self.process_file(
            input_path.as_posix(), "auto", None, None, request.temp_dir
        )
        return ProcessResult(
            task_type="convert",
            text_context=text_context,
            output_dir=output_dir,
        )
