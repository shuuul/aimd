"""Environment capability detection for transcription engines."""

from dataclasses import dataclass
import importlib.util
import platform
import shutil
import subprocess

from .const import TRANSCRIPTION_ENGINES
from .errors import EngineUnavailableError, UnsupportedEngineError


@dataclass
class EngineCapability:
    name: str
    available: bool
    reason: str | None = None
    fix_hint: str | None = None


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _is_apple_silicon() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        cpu_info = result.stdout.strip().lower()
        return "apple" in cpu_info and any(
            chip in cpu_info for chip in ("m1", "m2", "m3", "m4")
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def get_engine_capabilities() -> dict[str, EngineCapability]:
    """Return availability details for each supported transcription engine."""
    is_macos = platform.system() == "Darwin"
    apple_silicon = _is_apple_silicon() if is_macos else False
    has_faster_whisper = _module_available("faster_whisper")
    has_mlx_whisper = _module_available("mlx_whisper")
    has_torch = _module_available("torch")
    torch_cuda_available = _torch_cuda_available() if has_torch else False

    capabilities: dict[str, EngineCapability] = {}

    capabilities["yap"] = EngineCapability(
        name="yap",
        available=is_macos and shutil.which("yap") is not None,
        reason=(
            None
            if (is_macos and shutil.which("yap") is not None)
            else (
                "yap is only supported on macOS."
                if not is_macos
                else "yap CLI binary not found in PATH."
            )
        ),
        fix_hint=(
            None
            if (is_macos and shutil.which("yap") is not None)
            else "Install yap: https://github.com/finnvoor/yap"
        ),
    )

    mlx_available = is_macos and apple_silicon and has_mlx_whisper
    mlx_reason = None
    if not is_macos:
        mlx_reason = "mlx is only supported on macOS."
    elif not apple_silicon:
        mlx_reason = "mlx requires Apple Silicon (M1/M2/M3/M4)."
    elif not has_mlx_whisper:
        mlx_reason = "mlx_whisper module is not installed."

    capabilities["mlx"] = EngineCapability(
        name="mlx",
        available=mlx_available,
        reason=mlx_reason,
        fix_hint=None if mlx_available else "Install dependency: mlx-whisper",
    )

    cuda_available = has_faster_whisper and has_torch and torch_cuda_available
    cuda_reason = None
    if not has_faster_whisper:
        cuda_reason = "faster_whisper module is not installed."
    elif not has_torch:
        cuda_reason = "torch module is not installed."
    elif not torch_cuda_available:
        cuda_reason = "CUDA is not available in the current PyTorch runtime."

    capabilities["cuda"] = EngineCapability(
        name="cuda",
        available=cuda_available,
        reason=cuda_reason,
        fix_hint=(
            None
            if cuda_available
            else "Install CUDA-enabled PyTorch and verify torch.cuda.is_available()."
        ),
    )

    capabilities["cpu"] = EngineCapability(
        name="cpu",
        available=has_faster_whisper,
        reason=None
        if has_faster_whisper
        else "faster_whisper module is not installed.",
        fix_hint=None if has_faster_whisper else "Install dependency: faster-whisper",
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
        priority = ("yap", "mlx", "cpu")
    else:
        priority = ("cuda", "cpu")

    for candidate in priority:
        if capabilities[candidate].available:
            return candidate

    reasons = "; ".join(
        f"{name}: {capabilities[name].reason or 'unavailable'}" for name in priority
    )
    raise EngineUnavailableError(
        f"No available transcription engine for this environment. {reasons}"
    )
