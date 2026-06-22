"""Transcription engine capability introspection."""

from dataclasses import dataclass
from typing import Callable

from .capabilities import (
    EngineCapability,
    get_engine_capabilities,
    resolve_engine_with_preflight,
)
from .errors import EngineUnavailableError


@dataclass(slots=True)
class ListEnginesResult:
    auto_selected_engine: str | None
    engines: dict[str, EngineCapability]


def list_engines(
    *,
    get_capabilities: Callable[[], dict[str, EngineCapability]],
    resolve_engine: Callable[[str], str],
) -> ListEnginesResult:
    """List available transcription engines and auto-selected preference."""
    capabilities = get_capabilities()
    auto_selected_engine: str | None = None
    try:
        auto_selected_engine = resolve_engine("auto")
    except EngineUnavailableError:
        auto_selected_engine = None

    return ListEnginesResult(
        auto_selected_engine=auto_selected_engine,
        engines=capabilities,
    )


def list_transcription_engines() -> ListEnginesResult:
    """List transcription engine capabilities using default ASR dependencies."""
    return list_engines(
        get_capabilities=get_engine_capabilities,
        resolve_engine=resolve_engine_with_preflight,
    )
