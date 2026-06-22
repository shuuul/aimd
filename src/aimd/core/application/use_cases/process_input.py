"""Use-case for input processing orchestration."""

from dataclasses import dataclass
from typing import Mapping

from ...errors import InputNotFoundError, ProcessingFailedError, UnsupportedInputError
from ..models import InputRoute, ProcessInput, ProcessResult, TaskType
from .input_routing import FileSupportChecker, ensure_supported_input
from .processors import TaskProcessor


@dataclass(slots=True)
class ProcessInputUseCase:
    """Core facade/router for processing routed inputs."""

    processors: Mapping[TaskType, TaskProcessor]
    is_supported_file: FileSupportChecker

    def ensure_supported_input(self, input_source: str) -> InputRoute:
        """Validate and return the source/task route for a source."""
        return ensure_supported_input(input_source, self.is_supported_file)

    async def execute(self, request: ProcessInput) -> ProcessResult:
        route = self.ensure_supported_input(request.input_source)
        task_type = route.task_type
        if task_type is None:
            raise UnsupportedInputError("Unsupported input source.")

        processor = self.processors.get(task_type)
        if processor is None:
            raise UnsupportedInputError(
                f"No processor configured for task: {task_type}"
            )

        try:
            return await processor.process(request, route)
        except (InputNotFoundError, UnsupportedInputError, ProcessingFailedError):
            raise
        except Exception as exc:
            raise ProcessingFailedError(str(exc)) from exc
