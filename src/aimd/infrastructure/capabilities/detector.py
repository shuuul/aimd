"""Environment capability detection for transcription engines."""

from dataclasses import dataclass
import importlib.util
import platform

from ...const import TRANSCRIPTION_ENGINES
from ...errors import EngineUnavailableError, UnsupportedEngineError
from ...platform_utils import is_apple_silicon


@dataclass
class EngineCapability:
    name: str
    available: bool
    reason: str | None = None
    fix_hint: str | None = None
    deprecated: bool = False


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def get_engine_capabilities() -> dict[str, EngineCapability]:
    """Return availability details for each supported transcription engine."""
    is_macos = platform.system() == "Darwin"
    apple_silicon = is_apple_silicon() if is_macos else False
    has_mlx_audio = _module_available("mlx_audio")

    capabilities: dict[str, EngineCapability] = {}

    mlx_available = is_macos and apple_silicon and has_mlx_audio
    mlx_reason = None
    if not is_macos:
        mlx_reason = "mlx is only supported on macOS."
    elif not apple_silicon:
        mlx_reason = "mlx requires Apple Silicon (M1/M2/M3/M4)."
    elif not has_mlx_audio:
        mlx_reason = "mlx_audio module is not installed."

    capabilities["mlx"] = EngineCapability(
        name="mlx",
        available=mlx_available,
        reason=mlx_reason,
        fix_hint=None if mlx_available else "Install dependency: mlx-audio-plus",
    )

    has_funasr = _module_available("funasr")
    capabilities["funasr"] = EngineCapability(
        name="funasr",
        available=has_funasr,
        reason=None if has_funasr else "funasr module is not installed.",
        fix_hint=None if has_funasr else "Install dependency: funasr",
    )

    return capabilities


def resolve_engine_with_preflight(engine: str) -> str:
    """Resolve requested engine with fail-fast capability checks."""
    if engine not in TRANSCRIPTION_ENGINES:
        raise UnsupportedEngineError(
            f"Invalid engine '{engine}'. Valid options: {sorted(TRANSCRIPTION_ENGINES)}"
        )

    capabilities = get_engine_capabilities()

    if engine != "auto":
        capability = capabilities[engine]
        if not capability.available:
            reason = capability.reason or "Engine unavailable."
            fix_hint = f" {capability.fix_hint}" if capability.fix_hint else ""
            raise EngineUnavailableError(
                f"Engine '{engine}' unavailable: {reason}{fix_hint}"
            )
        return engine

    if platform.system() == "Darwin":
        priority = ("mlx", "funasr")
    else:
        priority = ("funasr",)

    for candidate in priority:
        if capabilities[candidate].available:
            return candidate

    reasons = "; ".join(
        f"{name}: {capabilities[name].reason or 'unavailable'}" for name in priority
    )
    raise EngineUnavailableError(
        f"No available transcription engine for this environment. {reasons}"
    )
