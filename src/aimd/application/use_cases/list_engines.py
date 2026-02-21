"""Use-case for engine capability introspection."""

from dataclasses import dataclass
from typing import Callable

from ...errors import EngineUnavailableError
from ...infrastructure.capabilities.detector import (
    EngineCapability,
    get_engine_capabilities,
    resolve_engine_with_preflight,
)


@dataclass(slots=True)
class ListEnginesResult:
    auto_selected_engine: str | None
    engines: dict[str, EngineCapability]


@dataclass(slots=True)
class ListEnginesUseCase:
    """List available transcription engines and auto-selected preference."""

    get_capabilities: Callable[[], dict[str, EngineCapability]] = (
        get_engine_capabilities
    )
    resolve_engine: Callable[[str], str] = resolve_engine_with_preflight

    def execute(self) -> ListEnginesResult:
        capabilities = self.get_capabilities()
        auto_selected_engine: str | None = None
        try:
            auto_selected_engine = self.resolve_engine("auto")
        except EngineUnavailableError:
            auto_selected_engine = None

        return ListEnginesResult(
            auto_selected_engine=auto_selected_engine,
            engines=capabilities,
        )
