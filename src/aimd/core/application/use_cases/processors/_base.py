"""Base task processor contract."""

from typing import Protocol

from ...models import InputRoute, ProcessInput, ProcessResult


class TaskProcessor(Protocol):
    """Process one routed task into a canonical result."""

    async def process(
        self,
        request: ProcessInput,
        route: InputRoute,
    ) -> ProcessResult:
        """Run task-specific processing for a routed input."""
        ...
